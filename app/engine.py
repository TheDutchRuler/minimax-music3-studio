"""MiniMax Music 3 inference engine.

Wraps the diffusers ModularPipeline behind a single-worker job queue, since the
GPU can only render one song at a time. Loading is lazy and happens on the
worker thread so the HTTP server stays responsive during the ~1-2 min load.
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import soundfile as sf
import torch

log = logging.getLogger("music3.engine")

ROOT = Path(__file__).resolve().parent.parent
# Weights live wherever MUSIC3_MODELS points; keep them on the fastest drive.
# Loading 26GB off a SATA spinning disk costs ~340s vs roughly a minute on NVMe.
MODEL_DIR = Path(os.environ.get("MUSIC3_MODELS") or (ROOT / "models"))
LIBRARY_DIR = ROOT / "library"
INDEX_FILE = LIBRARY_DIR / "index.json"

# The model tops out at 9000 acoustic frames @ 25fps = 360s. The card advertises
# five minutes, so we keep a little headroom below the hard ceiling.
MAX_DURATION = 300.0
MIN_DURATION = 10.0

# The pipeline emits float32 samples. Writing 16-bit PCM quantises them on the
# way out for no reason, so archive the master bit-exact and let the user
# convert on export. FLOAT | PCM_24 | PCM_16.
WAV_SUBTYPE = os.environ.get("MUSIC3_WAV_SUBTYPE", "FLOAT")

# Flow-matching steps PER 200-FRAME CHUNK (the pipeline runs
# chunks x steps x 2 CFG transformer passes). The pipeline default of 30 is the
# reference recipe the checkpoint was tuned for; None means "use that default".
# Raising it multiplies DiT time linearly (a 3-min song has ~44 chunks) with no
# demonstrated quality gain, so it is opt-in via the UI, not a default.
DEFAULT_STEPS = None
PIPELINE_DEFAULT_STEPS = 30


@dataclass
class Job:
    id: str
    title: str
    prompt: str
    lyrics: str
    instrumental: bool
    duration: float
    seed: int
    steps: int | None = None
    group: str | None = None
    status: str = "queued"  # queued | loading | running | done | error
    progress: float = 0.0
    stage: str = ""
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    audio: str | None = None  # filename inside library/
    elapsed: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        return d


class Engine:
    def __init__(self, offload: str = "auto", turbo: bool = True):
        """Memory strategy. Measured on a 24GB RTX 4090 with 61GB system RAM:

        'auto'  whole-component CPU offload  peak ~18GB VRAM, ~26GB RAM (default)
        'gpu'   all weights resident on GPU  peak ~22.5GB VRAM, ~2GB RAM (fastest,
                needs the GPU otherwise idle - desktop apps alone can take 4GB)

        Do NOT apply per-layer (group) offloading to `language_model`. It is an
        autoregressive model running one forward pass per audio frame at 25fps,
        so per-layer offload re-streams 16.4GB across PCIe every frame. Measured
        at 10% GPU utilisation and 76W with no meaningful progress.
        """
        self.offload = offload
        self.turbo = turbo
        self._undo_turbo = None
        self.pipe = None
        self.sampling_rate = 32000
        self._supported: set[str] | None = None
        # Leave headroom for the desktop: renders at ~22.5GB of 24GB starve
        # dwm/Chrome of VRAM (observed: Chrome GPU-process crash, desktop
        # stutter that presents exactly like a GPU crash — no TDR involved).
        # Capping our allocator keeps other apps alive; on OOM we degrade
        # gracefully via the turbo-retry path instead of taking the desktop down.
        # 0.80 leaves ~4.8GB for Windows/dwm/browsers — measured desktop use on
        # this machine is 2.5-4GB, and exceeding the card total causes the
        # eviction thrash that presents as "GPU crashes". For unattended max
        # speed (overnight renders), set MUSIC3_VRAM_FRACTION=0.92.
        frac = float(os.environ.get("MUSIC3_VRAM_FRACTION", "0.80"))
        if torch.cuda.is_available() and 0.5 <= frac < 1.0:
            try:
                torch.cuda.set_per_process_memory_fraction(frac)
                log.info("VRAM cap: %.0f%% of device memory", frac * 100)
            except Exception as exc:
                log.warning("could not set VRAM cap: %s", exc)

        self.load_state = "unloaded"  # unloaded | loading | ready | error
        self.load_error: str | None = None
        self._warmed = False  # first render pays the VRAM migration cost

        self.jobs: dict[str, Job] = {}
        self.groups: dict[str, list[str]] = {}
        self.writes: dict[str, dict] = {}  # songwriter requests (share the GPU worker)
        self.order: list[str] = []
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []

        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        self._load_library()

        self._worker = threading.Thread(target=self._run, daemon=True, name="music3-worker")
        self._worker.start()

    # ---------- events ----------

    def subscribe(self, cb: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(cb)

        def unsub():
            with self._lock:
                if cb in self._listeners:
                    self._listeners.remove(cb)

        return unsub

    def _emit(self, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(event)
            except Exception:
                pass

    def _push_job(self, job: Job) -> None:
        self._emit({"type": "job", "job": job.public()})

    # ---------- library ----------

    def _load_library(self) -> None:
        if not INDEX_FILE.exists():
            self.library: list[dict] = []
            return
        try:
            self.library = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            log.warning("library index unreadable, starting fresh")
            self.library = []
        # Drop entries whose audio vanished from disk.
        self.library = [t for t in self.library if (LIBRARY_DIR / t["audio"]).exists()]

    def _save_library(self) -> None:
        tmp = INDEX_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.library, indent=2), encoding="utf-8")
        tmp.replace(INDEX_FILE)

    def delete_track(self, track_id: str) -> bool:
        with self._lock:
            match = next((t for t in self.library if t["id"] == track_id), None)
            if not match:
                return False
            self.library = [t for t in self.library if t["id"] != track_id]
            self._save_library()
        try:
            (LIBRARY_DIR / match["audio"]).unlink(missing_ok=True)
        except Exception:
            pass
        self._emit({"type": "library"})
        return True

    # ---------- queue ----------

    def submit(
        self,
        prompt: str,
        lyrics: str,
        title: str,
        instrumental: bool,
        duration: float,
        seed: int | None,
        count: int = 2,
        steps: int | None = None,
    ) -> list[Job]:
        duration = max(MIN_DURATION, min(MAX_DURATION, float(duration)))
        count = max(1, min(4, count))
        # Same-prompt variations render as ONE batched AR pass (ensemble):
        # the memory-bandwidth cost of the 8B+depth reads amortizes across
        # variations, so K songs cost barely more than one AR stage.
        # Group size is VRAM- AND duration-gated: the batched AR keeps the
        # ~17.2GB LLM+depth residency plus a KV cache of
        # bucket_tokens x 2K rows x ~0.141MB under the cap — long songs eat the
        # headroom fast (a 3-min song needs ~0.7GB per song of KV alone).
        # A too-ambitious group still degrades safely (group -> solo -> eager).
        max_group = 1
        if torch.cuda.is_available():
            frac = float(os.environ.get("MUSIC3_VRAM_FRACTION", "0.80"))
            budget_gb = torch.cuda.get_device_properties(0).total_memory * frac / 1024**3
            frames = duration * 25.0
            bucket = ((int(frames) + 700 + 8 + 1023) // 1024) * 1024  # ~700 prompt tokens
            kv_gb_per_row = bucket * 0.1406 / 1024  # MB per token-row -> GB
            headroom = budget_gb - 17.2 - 0.5  # residency + graphs/workspace
            rows = int(headroom / kv_gb_per_row) if kv_gb_per_row > 0 else 2
            max_group = max(1, min(4, rows // 2))
        batchable = min(count, max_group)
        gid = uuid.uuid4().hex[:8] if (batchable > 1 and self.turbo) else None
        created = []
        for i in range(count):
            s = (
                int(torch.randint(0, 2**31 - 1, (1,)).item())
                if seed is None
                else int(seed) + i
            )
            # Only the first `batchable` variations share the batched pass;
            # the rest render solo afterwards instead of blowing the VRAM gate.
            in_group = gid is not None and i < batchable
            job = Job(
                id=uuid.uuid4().hex[:12],
                title=title.strip() or "Untitled",
                prompt=prompt.strip(),
                lyrics="" if instrumental else lyrics.strip(),
                instrumental=instrumental,
                duration=duration,
                seed=s,
                steps=steps,
                group=gid if in_group else None,
            )
            with self._lock:
                self.jobs[job.id] = job
                self.order.append(job.id)
            self._push_job(job)
            created.append(job)
        if gid is not None:
            grouped = [j for j in created if j.group == gid]
            with self._lock:
                self.groups[gid] = [j.id for j in grouped]
            self._q.put(f"group:{gid}")
            for job in created:
                if job.group is None:
                    self._q.put(job.id)
        else:
            for job in created:
                self._q.put(job.id)
        return created

    def cancel(self, job_id: str) -> bool:
        """Only queued jobs can be cancelled; the GPU pass is not interruptible."""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job or job.status != "queued":
                return False
            job.status = "error"
            job.error = "Cancelled"
            job.finished = time.time()
        self._push_job(job)
        return True

    # ---------- model ----------

    def _ensure_loaded(self) -> None:
        if self.pipe is not None:
            return

        try:
            # Free the songwriter's 3.4GB before the 25GB load spike.
            import writer

            writer.release()
        except Exception:  # pragma: no cover
            pass

        from diffusers import ComponentsManager, ModularPipeline

        self.load_state = "loading"
        self._emit({"type": "model", "state": "loading"})
        t0 = time.time()

        src = str(MODEL_DIR)
        log.info("loading MiniMax Music 3 from %s (offload=%s)", src, self.offload)

        if self.offload == "auto":
            # Components are onloaded whole on first use and cached, so the LLM
            # becomes resident for its entire decode phase. The first song pays
            # a one-off warm-up while it migrates to VRAM.
            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(device="cuda")
            pipe = ModularPipeline.from_pretrained(src, components_manager=manager)
            pipe.load_components(dtype=torch.bfloat16)
        else:  # "gpu"
            pipe = ModularPipeline.from_pretrained(src)
            pipe.load_components(dtype=torch.bfloat16)
            pipe.to("cuda")

        if self.turbo:
            try:
                from turbo import install_all

                self._undo_turbo = install_all(pipe)
            except Exception as exc:
                log.warning("turbo unavailable, using reference implementation: %s", exc)
                self._undo_turbo = None

        self.pipe = pipe
        self.sampling_rate = int(getattr(pipe, "sampling_rate", 32000) or 32000)
        self._supported = self._discover_kwargs(pipe)
        self.load_state = "ready"
        log.info(
            "model ready in %.1fs, sr=%d, extra kwargs=%s",
            time.time() - t0,
            self.sampling_rate,
            sorted(self._supported),
        )
        self._emit({"type": "model", "state": "ready"})

    @staticmethod
    def _discover_kwargs(pipe) -> set[str]:
        """Figure out which optional knobs this pipeline build accepts.

        The modular pipeline is experimental and its input list has shifted
        between commits, so we probe rather than hard-code.
        """
        names: set[str] = set()
        blocks = getattr(pipe, "blocks", None)
        for attr in ("input_names", "inputs"):
            vals = getattr(blocks, attr, None) if blocks is not None else None
            if not vals:
                continue
            for v in vals:
                name = getattr(v, "name", v)
                if isinstance(name, str):
                    names.add(name)
        if not names:
            try:
                names = set(inspect.signature(pipe.__call__).parameters)
            except (TypeError, ValueError):
                pass
        return names

    def _accepts(self, key: str) -> bool:
        return bool(self._supported) and key in self._supported

    # ---------- worker ----------

    def _writer_progress(self, stage: str, detail: str) -> None:
        self._emit({"type": "writer", "state": stage, "detail": detail})

    def enhance(self, brief: str, instrumental: bool, timeout: float = 600.0) -> dict:
        """Run the songwriter. If the GPU worker is busy rendering, write on
        CPU immediately instead of queueing behind a multi-minute render."""
        with self._lock:
            busy = bool(self.groups) or any(
                self.jobs[j].status in ("queued", "loading", "running") for j in self.order
            )
        if busy:
            self._writer_progress("writing", "GPU is rendering — writing on CPU instead")
            import writer

            try:
                result = writer.write_song(
                    brief, instrumental, force_cpu=True, progress=self._writer_progress
                )
                self._writer_progress("done", "")
                return result
            except Exception as exc:
                self._writer_progress("error", str(exc))
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

        wid = uuid.uuid4().hex[:12]
        req = {"brief": brief, "instrumental": instrumental,
               "event": threading.Event(), "result": None, "error": None}
        self.writes[wid] = req
        self._writer_progress("queued", "Starting the songwriter")
        self._q.put(f"write:{wid}")
        if not req["event"].wait(timeout):
            self.writes.pop(wid, None)
            self._writer_progress("error", "timed out")
            raise TimeoutError("songwriter timed out (a render may be occupying the GPU)")
        self.writes.pop(wid, None)
        if req["error"]:
            raise RuntimeError(req["error"])
        return req["result"]

    def _run_write(self, wid: str) -> None:
        req = self.writes.get(wid)
        if req is None:
            return
        try:
            # The writer uses its own small companion model — no need to load
            # (or wait for) the 22GB music pipeline just to write lyrics.
            import writer

            req["result"] = writer.write_song(
                req["brief"], req["instrumental"], progress=self._writer_progress
            )
            self._writer_progress("done", "")
        except Exception as exc:
            log.error("songwriter failed: %s", traceback.format_exc())
            req["error"] = f"{type(exc).__name__}: {exc}"
            self._writer_progress("error", str(exc))
        finally:
            req["event"].set()

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            if job_id.startswith("group:"):
                self._run_group(job_id[6:])
                continue
            if job_id.startswith("write:"):
                self._run_write(job_id[6:])
                continue
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                self._process(job)
            except Exception as exc:
                log.error("job %s failed: %s", job.id, traceback.format_exc())
                # If the turbo path is active, retry the job once on the
                # reference implementation before surfacing an error — a
                # compile/CUDA-graph edge case should cost speed, not a song.
                if self._undo_turbo is not None:
                    log.warning("disabling turbo and retrying job %s", job.id)
                    try:
                        self._undo_turbo()
                    finally:
                        self._undo_turbo = None
                    try:
                        from turbo import purge_runtime_state

                        purge_runtime_state(self.pipe)
                    except Exception:
                        pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    job.status = "queued"
                    job.stage = "Retrying without turbo"
                    job.progress = 0.0
                    self._push_job(job)
                    self._q.put(job.id)
                    continue
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished = time.time()
                self._push_job(job)
                # soundfile opens the file and writes the header before the
                # sample data, so a write failure leaves a 44-byte header-only
                # orphan (not 0 bytes) that nothing else would clean up.
                stale = LIBRARY_DIR / f"{job.id}.wav"
                if stale.exists() and stale.stat().st_size < 1024:
                    stale.unlink(missing_ok=True)
                # A failed run can leave the allocator fragmented; reclaim before
                # the next job so one OOM does not cascade.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @staticmethod
    def _effective_inputs(job: Job) -> tuple[str, str]:
        prompt = job.prompt
        lyrics = job.lyrics
        if job.instrumental:
            prompt = f"{prompt.rstrip('. ')}. Instrumental only, no vocals, no singing."
            lyrics = "[instrumental]"
        elif not lyrics.strip():
            lyrics = "[verse]\n[chorus]"
        return prompt, lyrics

    def _run_group(self, gid: str) -> None:
        """Render all still-queued members of a variation group in one batched
        ensemble pass. On any failure, members fall back to the solo path."""
        with self._lock:
            members = [
                self.jobs[j]
                for j in self.groups.pop(gid, [])
                if j in self.jobs and self.jobs[j].status == "queued"
            ]
        if not members:
            return
        if len(members) == 1 or (self.pipe is not None and self._undo_turbo is None):
            for job in members:
                self._q.put(job.id)
            return

        try:
            if self.pipe is None:
                for job in members:
                    job.status = "loading"
                    job.stage = "Loading model (first run takes a minute)"
                    self._push_job(job)
                self._ensure_loaded()

            import ensemble

            lead = members[0]
            prompt, lyrics = self._effective_inputs(lead)
            for job in members:
                job.status = "running"
                job.started = time.time()
                job.stage = f"Composing {len(members)} variations together"
                job.progress = 0.01
                self._push_job(job)

            def progress(frac: float, text: str) -> None:
                for job in members:
                    job.progress = min(0.97, frac)
                    job.stage = text
                    self._push_job(job)

            arrays = ensemble.generate_group(
                self.pipe,
                prompt=prompt,
                lyrics=lyrics,
                duration=float(lead.duration),
                seeds=[j.seed for j in members],
                steps=lead.steps or DEFAULT_STEPS,
                progress=progress,
            )
            self._warmed = True

            for job, data in zip(members, arrays):
                filename = f"{job.id}.wav"
                sf.write(str(LIBRARY_DIR / filename), data, self.sampling_rate, subtype=WAV_SUBTYPE)
                real_dur = data.shape[0] / float(self.sampling_rate)
                job.audio = filename
                job.status = "done"
                job.progress = 1.0
                job.stage = "Done"
                job.finished = time.time()
                job.elapsed = round(job.finished - (job.started or job.finished), 1)
                self._push_job(job)
                track = {
                    "id": job.id,
                    "title": job.title,
                    "prompt": job.prompt,
                    "lyrics": job.lyrics,
                    "instrumental": job.instrumental,
                    "seed": job.seed,
                    "steps": (lead.steps or DEFAULT_STEPS) or PIPELINE_DEFAULT_STEPS,
                    "format": f"{self.sampling_rate}Hz {WAV_SUBTYPE}",
                    "duration": round(real_dur, 1),
                    "audio": filename,
                    "created": job.finished,
                    # Per-song effective cost: the group renders together, so
                    # each song's share is the group wall clock over K.
                    "render_seconds": round((job.elapsed or 0) / len(members), 1),
                    "group_size": len(members),
                }
                with self._lock:
                    self.library.insert(0, track)
                    self._save_library()
            self._emit({"type": "library"})
        except Exception:
            log.error("ensemble group %s failed: %s", gid, traceback.format_exc())
            log.warning("group %s falling back to solo renders", gid)
            try:
                from turbo import purge_runtime_state

                purge_runtime_state(self.pipe)
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for job in members:
                job.status = "queued"
                job.stage = "Retrying individually"
                job.progress = 0.0
                self._push_job(job)
                self._q.put(job.id)

    def _process(self, job: Job) -> None:
        if self.pipe is None:
            job.status = "loading"
            job.stage = "Loading model (first run takes a minute)"
            self._push_job(job)
            try:
                self._ensure_loaded()
            except Exception as exc:
                self.load_state = "error"
                self.load_error = str(exc)
                self._emit({"type": "model", "state": "error", "error": str(exc)})
                raise

        job.status = "running"
        job.started = time.time()
        job.stage = "Composing"
        job.progress = 0.02
        self._push_job(job)

        prompt, lyrics = self._effective_inputs(job)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": float(job.duration),
            "generator": torch.Generator("cuda").manual_seed(job.seed),
            "output": "audios",
        }
        steps = job.steps or DEFAULT_STEPS
        if steps and self._accepts("num_inference_steps"):
            kwargs["num_inference_steps"] = int(steps)

        ticker = _ProgressTicker(
            job,
            self._push_job,
            expected=self._estimate(job.duration, steps or PIPELINE_DEFAULT_STEPS),
            warmed=self._warmed,
        )
        ticker.start()
        try:
            result = self.pipe(**kwargs)
        finally:
            ticker.stop()
            self._warmed = True

        audio = _unwrap_audio(result)

        job.stage = "Writing file"
        job.progress = 0.97
        self._push_job(job)

        data = _to_numpy(audio)
        filename = f"{job.id}.wav"
        sf.write(str(LIBRARY_DIR / filename), data, self.sampling_rate, subtype=WAV_SUBTYPE)

        real_dur = data.shape[0] / float(self.sampling_rate)
        job.audio = filename
        job.status = "done"
        job.progress = 1.0
        job.stage = "Done"
        job.finished = time.time()
        job.elapsed = round(job.finished - (job.started or job.finished), 1)
        self._push_job(job)

        track = {
            "id": job.id,
            "title": job.title,
            "prompt": job.prompt,
            "lyrics": job.lyrics,
            "instrumental": job.instrumental,
            "seed": job.seed,
            "steps": steps or PIPELINE_DEFAULT_STEPS,
            "format": f"{self.sampling_rate}Hz {WAV_SUBTYPE}",
            "duration": round(real_dur, 1),
            "audio": filename,
            "created": job.finished,
            "render_seconds": job.elapsed,
        }
        with self._lock:
            self.library.insert(0, track)
            self._save_library()
        self._emit({"type": "library"})

    def _estimate(self, duration: float, steps: int) -> float:
        """Wall-clock estimate used only to animate the progress bar.

        Stage model measured on a 24GB RTX 4090 in 'auto' mode. Turbo (compiled
        AR + batched-CFG DiT): AR ~1.1s per audio second, DiT ~0.075s per step.
        Reference path: AR ~1.87s/s, DiT ~0.107s/step. The first render adds
        one-off component migration (and compilation, in turbo).
        """
        frames = duration * 25.0
        chunks = 1 if frames <= 200 else max(1, len(range(0, int(frames) - 100, 100)))
        if self._undo_turbo is not None:
            est = 6.0 + duration * 1.1 + chunks * steps * 0.075
            warmup_extra = 290.0
        else:
            est = 6.0 + duration * 1.87 + chunks * steps * 0.107
            warmup_extra = 216.0
        if not self._warmed:
            est += warmup_extra
        return est

    # ---------- status ----------

    def status(self) -> dict:
        vram = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram = {
                "free_gb": round(free / 1024**3, 1),
                "total_gb": round(total / 1024**3, 1),
                "used_gb": round((total - free) / 1024**3, 1),
            }
        with self._lock:
            active = [
                self.jobs[j].public()
                for j in self.order
                if self.jobs[j].status in ("queued", "loading", "running")
            ]
        return {
            "model_state": self.load_state,
            "model_error": self.load_error,
            "turbo": self._undo_turbo is not None,
            "offload": self.offload,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram": vram,
            "active": active,
            "max_duration": MAX_DURATION,
        }


def _to_numpy(audio) -> "np.ndarray":
    """Return float32 samples shaped (frames, channels) for soundfile.

    Normalises rather than assuming a layout: the model card's snippet expects a
    torch tensor and a bare `.T`, but this pipeline has returned numpy arrays as
    (channels, frames) and with an extra leading batch axis — (1, 2, N) — which a
    blind transpose turns into an unwritable (N, 2, 1).
    """
    import numpy as np

    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().to(torch.float32).cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)

    arr = np.squeeze(arr)  # drop batch/singleton axes
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected audio shape {arr.shape}")
    # Channels are the short axis; soundfile wants them last.
    return arr.T if arr.shape[0] < arr.shape[1] else arr


def _unwrap_audio(result: Any):
    """Pull the waveform tensor out of whatever the modular pipeline returned.

    The experimental pipeline has returned a bare tensor, a list, and an object
    with an `.audios` attribute across different commits, so accept all three.
    """
    if hasattr(result, "audios"):
        result = result.audios
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("Pipeline returned no audio")
        result = result[0]
    if hasattr(result, "audios"):
        result = result.audios[0]
    if not hasattr(result, "T"):
        raise RuntimeError(f"Unexpected pipeline output: {type(result).__name__}")
    return result


class _ProgressTicker(threading.Thread):
    """Animates progress between 2% and 95% while the (opaque) pipeline runs.

    The modular pipeline exposes no step callback, so this is a time-based
    approximation. It never reports completion — only _process does that.
    """

    def __init__(self, job: Job, push: Callable[[Job], None], expected: float, warmed: bool = True):
        super().__init__(daemon=True)
        self.job = job
        self.push = push
        self.expected = max(5.0, expected)
        self.warmed = warmed
        self._stop = threading.Event()

    def run(self) -> None:
        t0 = time.time()
        while not self._stop.wait(1.5):
            frac = (time.time() - t0) / self.expected
            # Asymptotic approach so an underestimate never pins the bar at 95%.
            self.job.progress = min(0.95, 0.02 + 0.93 * (1 - pow(2.718, -2.2 * frac)))
            elapsed = time.time() - t0
            if not self.warmed and elapsed < 216:
                self.job.stage = "Warming up GPU (first song only)"
            elif elapsed > 12:
                self.job.stage = "Rendering vocals & arrangement"
            else:
                self.job.stage = "Composing"
            self.push(self.job)

    def stop(self) -> None:
        self._stop.set()

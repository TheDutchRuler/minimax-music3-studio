"""Paired A/B verification of the batched-CFG claim for diffusers issue #14486.

Question from the diffusers maintainers: batching CFG (one batch-2 forward
instead of two sequential batch-1 forwards) shows no real gain on typical
image-diffusion models — is the MiniMax Music 3 DiT genuinely different?

Design: ONE process, ONE model load. Per duration, alternate the denoise inner
block between the reference implementation (guider, sequential branches) and
the batched one, rendering the same seed each time. The DiT compile flag is
forced OFF so batching is the only variable in the denoise stage. Stage times
come from block-level timers; GPU utilization is sampled during the DiT stage
of each arm via nvidia-smi to expose the mechanism (batch-1 underutilization).

A->B->A->B ordering guards against warm-up drift crediting either arm.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ["MUSIC3_TURBO_DIT"] = "0"  # eager DiT in BOTH arms — batching only

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import numpy as np
import torch

MODEL = os.environ.get("MUSIC3_MODELS") or str(Path(__file__).parent.parent / "models_bf16")
SEED = 7

BLOCK_TIMES = {}


def install_timers():
    from diffusers.modular_pipelines.minimax_music3 import denoise as dn, encoders as enc

    for name, cls in {"ar": enc.MiniMaxMusic3SemanticGenerationStep,
                      "dit": dn.MiniMaxMusic3ChunkDenoiseStep}.items():
        orig = cls.__call__

        def timed(self, c, s, _o=orig, _n=name):
            torch.cuda.synchronize()
            t0 = time.time()
            out = _o(self, c, s)
            torch.cuda.synchronize()
            BLOCK_TIMES[_n] = time.time() - t0
            return out

        cls.__call__ = timed


class UtilSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(0.5):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                self.samples.append(int(out))
            except Exception:
                pass

    def stop(self):
        self._stop.set()


def render(pipe, duration):
    BLOCK_TIMES.clear()
    sampler = UtilSampler()
    sampler.start()
    audio = pipe(
        prompt=("Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. "
                "Vocals: soft female lead. Arrangement: fingerpicked guitar and piano."),
        lyrics="[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe",
        audio_duration=float(duration),
        generator=torch.Generator("cuda").manual_seed(SEED),
        output="audios",
    )[0]
    sampler.stop()
    arr = np.squeeze(np.asarray(audio, dtype=np.float32))
    rms = float(np.sqrt((arr ** 2).mean()))
    # utilization during the DiT tail of the render (last portion of samples)
    tail = sampler.samples[-max(4, int(BLOCK_TIMES.get("dit", 5) * 2)):]
    util = sum(tail) / max(len(tail), 1)
    return BLOCK_TIMES.get("ar", 0), BLOCK_TIMES.get("dit", 0), util, rms


def main():
    print(f"models={MODEL}  DiT compile disabled in both arms", flush=True)
    from diffusers import ComponentsManager, ModularPipeline
    import turbo

    t0 = time.time()
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    print(f"LOAD {time.time()-t0:.1f}s", flush=True)

    undo_all = turbo.install_all(pipe)  # AR fast in both arms (held constant)
    from diffusers.modular_pipelines.minimax_music3 import denoise as dn

    batched_call = dn.MiniMaxMusic3ChunkDenoiseInner.__call__  # installed by turbo
    # Recover the true reference inner from a fresh import is impossible after
    # patching, so keep a handle before any undo churn: turbo's undo restores it.
    install_timers()

    print("--- warm-up (compiles AR, batched inner active) ---", flush=True)
    render(pipe, 20)

    for duration in (20, 60):
        results = {"reference": [], "batched": []}
        for arm in ("reference", "batched", "reference", "batched"):
            if arm == "reference":
                # restore the reference inner (undo returns turbo's patches;
                # re-install AR-only pieces right after)
                undo_all()
                turbo_undo_ar = turbo.install_fast_ar()
                ref_mode = True
            else:
                dn.MiniMaxMusic3ChunkDenoiseInner.__call__ = batched_call
                ref_mode = False
            ar, dit, util, rms = render(pipe, duration)
            results[arm].append((dit, util))
            print(f"[{duration}s] {arm:9s}  dit={dit:6.2f}s  gpu~{util:3.0f}%  (ar={ar:.1f}s rms={rms:.3f})", flush=True)
            if ref_mode:
                turbo_undo_ar()
                undo_all = turbo.install_all(pipe)

        ref = min(d for d, _ in results["reference"])
        bat = min(d for d, _ in results["batched"])
        print(f"== {duration}s VERDICT: reference {ref:.2f}s vs batched {bat:.2f}s "
              f"-> {ref/bat:.2f}x from CFG batching alone ==", flush=True)

    print("=== VERIFY DONE ===", flush=True)


if __name__ == "__main__":
    main()

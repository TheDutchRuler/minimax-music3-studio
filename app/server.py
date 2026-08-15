"""HTTP layer for the local MiniMax Music 3 studio."""

from __future__ import annotations

import os

# Must be set before CUDA initializes: expandable segments avoid the ~0.8GB of
# fragmentation that pushed 3-variation groups over the desktop-safe VRAM cap.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
import json
import logging
import queue
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import LIBRARY_DIR, MAX_DURATION, MODEL_DIR, Engine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("music3.server")

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="MiniMax Music 3 Studio")
engine: Engine | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", max_length=8000)
    lyrics: str = Field(default="", max_length=8000)
    title: str = Field(default="", max_length=200)
    instrumental: bool = False
    duration: float = 90.0
    seed: int | None = None
    # 3 variations is the sweet spot: they render as one batched pass, so the
    # third song is nearly free and per-song speed goes sub-realtime (0.88x).
    count: int = 3
    steps: int | None = Field(default=None, ge=1, le=200)


def _engine() -> Engine:
    if engine is None:
        raise HTTPException(503, "engine not started")
    return engine


@app.get("/api/status")
def status():
    return _engine().status()


@app.get("/api/library")
def library():
    return {"tracks": _engine().library}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip() and not req.lyrics.strip():
        raise HTTPException(400, "Describe the music you want, or supply lyrics.")

    title = req.title.strip()
    if not title:
        # Fall back to the first few words of the prompt, like Suno does.
        source = req.prompt.strip() or req.lyrics.strip()
        title = " ".join(source.replace("\n", " ").split()[:5])[:60] or "Untitled"

    jobs = _engine().submit(
        prompt=req.prompt,
        lyrics=req.lyrics,
        title=title,
        instrumental=req.instrumental,
        duration=req.duration,
        seed=req.seed,
        count=req.count,
        steps=req.steps,
    )
    return {"jobs": [j.public() for j in jobs]}


class EnhanceRequest(BaseModel):
    prompt: str = Field(max_length=2000)
    instrumental: bool = False


@app.post("/api/enhance")
def enhance(req: EnhanceRequest):
    """Songwriter: expand a short brief into a structured caption + tagged
    lyrics using the resident 8B. Waits behind any in-flight render."""
    if not req.prompt.strip():
        raise HTTPException(400, "Describe the song first.")
    try:
        return _engine().enhance(req.prompt.strip(), req.instrumental)
    except TimeoutError as exc:
        raise HTTPException(503, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, f"Songwriter failed: {exc}")


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str):
    if not _engine().cancel(job_id):
        raise HTTPException(400, "Job is already running or finished.")
    return {"ok": True}


@app.delete("/api/track/{track_id}")
def delete_track(track_id: str):
    if not _engine().delete_track(track_id):
        raise HTTPException(404, "No such track")
    return {"ok": True}


@app.get("/api/audio/{track_id}")
def audio(track_id: str):
    # Resolve through the library rather than trusting the id as a path.
    eng = _engine()
    track = next((t for t in eng.library if t["id"] == track_id), None)
    name = track["audio"] if track else f"{track_id}.wav"
    path = (LIBRARY_DIR / name).resolve()
    if LIBRARY_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "No such audio")
    return FileResponse(path, media_type="audio/wav", filename=f"{(track or {}).get('title', track_id)}.wav")


@app.get("/api/events")
async def events():
    """Server-sent events: job progress and library changes."""
    eng = _engine()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)

    def on_event(evt: dict) -> None:
        try:
            loop.call_soon_threadsafe(q.put_nowait, json.dumps(evt))
        except (asyncio.QueueFull, RuntimeError):
            pass  # slow client; dropping a frame is fine, UI re-syncs on poll

    unsub = eng.subscribe(on_event)

    async def stream():
        try:
            yield f"data: {json.dumps({'type': 'hello'})}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {payload}\n\n"
        finally:
            unsub()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    global engine
    ap = argparse.ArgumentParser(description="MiniMax Music 3 Studio")
    ap.add_argument(
        "--offload",
        choices=["auto", "gpu"],
        default="auto",
        help="auto (default): whole-component CPU offload, peak ~18GB VRAM. "
        "gpu: all weights resident, peak ~22.5GB VRAM - fastest, but needs the "
        "GPU otherwise idle.",
    )
    ap.add_argument(
        "--no-turbo",
        action="store_true",
        help="Disable the verified speed optimizations (compiled AR decode + "
        "batched-CFG DiT, measured 1.63x on this machine) and run the exact "
        "reference implementation instead.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7878)
    args = ap.parse_args()

    if not (MODEL_DIR / "modular_model_index.json").exists():
        raise SystemExit(
            f"Model weights not found in {MODEL_DIR}.\n"
            "Run:  .venv\\Scripts\\python.exe fetch_weights.py"
        )

    engine = Engine(offload=args.offload, turbo=not args.no_turbo)
    log.info("MiniMax Music 3 Studio -> http://%s:%d", args.host, args.port)
    log.info("offload=%s  max duration=%.0fs", args.offload, MAX_DURATION)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""Measure load time and cold-vs-warm generation on this machine.

Run:  .venv\\Scripts\\python.exe bench.py [mode] [compile]
      mode    = auto | gpu      (default auto)
      compile = compile | nocompile (default nocompile)

The first render pays a one-off cost while components migrate into VRAM.
What matters for the app is the SECOND render, when everything is resident.

Baseline measured on this machine (auto, nocompile): LOAD 342.6s,
cold 265.9s, warm 50.9s for 20s of audio (2.54x realtime).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL = os.environ.get("MUSIC3_MODELS") or str(Path(__file__).parent / "models")
MODE = sys.argv[1] if len(sys.argv) > 1 else "auto"
COMPILE = (sys.argv[2] if len(sys.argv) > 2 else "nocompile") == "compile"
OUT = Path(__file__).parent
DUR = 20.0


def vram(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"  [{tag}] VRAM {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB", flush=True)


def to_np(audio):
    if hasattr(audio, "detach"):
        audio = audio.detach().to(torch.float32).cpu().numpy()
    arr = np.squeeze(np.asarray(audio, dtype=np.float32))
    if arr.ndim == 1:
        return arr
    return arr.T if arr.shape[0] < arr.shape[1] else arr


def main():
    print(f"mode={MODE} compile={COMPILE} models={MODEL}", flush=True)
    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.time()
    if MODE == "auto":
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
    else:
        pipe = ModularPipeline.from_pretrained(MODEL)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to("cuda")
    print(f"LOAD: {time.time()-t0:.1f}s", flush=True)
    vram("loaded")

    if COMPILE:
        try:
            tc = time.time()
            pipe.language_model = torch.compile(
                pipe.language_model, mode="reduce-overhead", fullgraph=False
            )
            print(f"COMPILE wrapped in {time.time()-tc:.1f}s (builds lazily)", flush=True)
        except Exception as e:
            print(f"COMPILE FAILED: {type(e).__name__}: {e}", flush=True)

    sr = int(getattr(pipe, "sampling_rate", 44100) or 44100)
    prompt = ("Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. "
              "Vocals: soft female lead, close and breathy. Arrangement: "
              "fingerpicked guitar and soft piano.")
    lyrics = "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe"

    for run in (1, 2, 3):
        t = time.time()
        try:
            audio = pipe(
                prompt=prompt, lyrics=lyrics, audio_duration=DUR,
                generator=torch.Generator("cuda").manual_seed(run), output="audios",
            )[0]
        except Exception as e:
            print(f"RUN {run} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            return
        dt = time.time() - t
        data = to_np(audio)
        secs = data.shape[0] / sr
        sf.write(str(OUT / f"bench_{MODE}_{run}.wav"), data, sr, subtype="FLOAT")
        label = {1: "COLD (warm-up + any compile)", 2: "WARM", 3: "WARM (confirm)"}[run]
        print(f"RUN {run} {label}: {dt:.1f}s for {secs:.1f}s ({dt/secs:.2f}x realtime)", flush=True)
        vram(f"run{run}")


if __name__ == "__main__":
    main()

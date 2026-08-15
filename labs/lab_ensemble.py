"""Ensemble lab: validate batched variations and measure the amortization win.

Phases:
  1. Load + turbo install (shared compiled DiT).
  2. K=1 ensemble, seed 7  (baseline through the new path; compiles batch-2 decode)
  3. K=1 warm re-run       (timed solo reference)
  4. K=2 ensemble, seeds (7, 8)  (compiles batch-4 decode)
  5. K=2 warm re-run       (timed — the headline)
  6. Checks: durations, levels, and how long the K=2 seed-7 song tracks the
     K=1 seed-7 song (row-independence is exact math; kernel tiling for batch 4
     vs 2 may still diverge trajectories ulp-style — measure, don't assume).

Solo turbo v2 reference (same machine): ~31s warm for a 20s song.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent / "app"))

MODEL = os.environ.get("MUSIC3_MODELS") or str(Path(__file__).parent / "models_bf16")
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
OUT = Path(__file__).parent

PROMPT = ("Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. "
          "Vocals: soft female lead, close and breathy. Arrangement: "
          "fingerpicked guitar and soft piano.")
LYRICS = "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe"


def run(pipe, tag, seeds):
    import ensemble

    torch.cuda.synchronize()
    t0 = time.time()
    outs = ensemble.generate_group(pipe, PROMPT, LYRICS, DUR, seeds)
    torch.cuda.synchronize()
    total = time.time() - t0
    for i, (seed, arr) in enumerate(zip(seeds, outs)):
        sf.write(str(OUT / f"ens_{tag}_s{seed}.wav"), arr, 44100, subtype="FLOAT")
        secs = arr.shape[0] / 44100
        rms = float(np.sqrt((arr ** 2).mean()))
        print(f"[{tag}] seed {seed}: {secs:.1f}s audio  rms={rms:.3f}", flush=True)
    per = total / len(seeds)
    print(f"[{tag}] TOTAL {total:.1f}s for {len(seeds)} song(s) -> {per:.1f}s/song "
          f"({per/DUR:.2f}x RT per song)", flush=True)
    free, tot = torch.cuda.mem_get_info()
    print(f"  VRAM {(tot-free)/1024**3:.1f}/{tot/1024**3:.1f} GB", flush=True)
    return outs


def main():
    print(f"duration={DUR}s models={MODEL}", flush=True)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MUSIC3_VRAM_FRACTION", "0.88")))

    import turbo
    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.time()
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    print(f"LOAD {time.time()-t0:.1f}s", flush=True)
    turbo.install_all(pipe)

    only_k2 = len(sys.argv) > 2 and sys.argv[2] == "k2"
    if only_k2:
        import soundfile as _sf
        solo, _ = _sf.read(str(OUT / "ens_k1_s7.wav"), dtype="float32")
    else:
        print("--- K=1 compile ---", flush=True)
        run(pipe, "k1_compile", [7])
        print("--- K=1 warm ---", flush=True)
        solo = run(pipe, "k1", [7])[0]

    print("--- K=2 compile ---", flush=True)
    run(pipe, "k2_compile", [7, 8])
    print("--- K=2 warm ---", flush=True)
    duo = run(pipe, "k2", [7, 8])

    a, b = solo, duo[0]
    n = min(len(a), len(b))
    if n:
        same = np.array_equal(a[:n], b[:n])
        d = a[:n] - b[:n]
        # First sample where they stop being bit-identical.
        neq = np.nonzero((d != 0).any(axis=-1) if d.ndim > 1 else d != 0)[0]
        div = int(neq[0]) if neq.size else n
        print(f"K1-vs-K2 seed 7: identical={same}  first divergence sample {div}/{n} "
              f"({div/44100:.2f}s)  rms(diff)={float(np.sqrt((d**2).mean())):.4f}", flush=True)
    print("=== ENSEMBLE LAB DONE ===", flush=True)


if __name__ == "__main__":
    main()

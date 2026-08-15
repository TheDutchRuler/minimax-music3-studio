"""Turbo v2 lab: validate + measure slim-head, fused-sampling AR and compiled DiT.

Order of proof:
  0. CPU unit test — Gumbel-argmax sampling vs the reference softmax+multinomial
     on identical logits: the two must agree in distribution.
  1. Load once, install turbo v2, warm-up (pays compilation).
  2. Two measured warm renders (same seed), per-block timings.
  3. Output sanity: duration, levels.

Reference numbers (same machine/seed): eager 50.5s, turbo v1 31.0s.
"""

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent / "app"))

MODEL = os.environ.get("MUSIC3_MODELS") or str(Path(__file__).parent / "models_bf16")
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
SEED = 7
OUT = Path(__file__).parent


def unit_test_gumbel():
    """Both samplers draw from softmax(masked logits); compare frequencies."""
    from diffusers.modular_pipelines.minimax_music3 import encoders as enc
    import turbo

    torch.manual_seed(123)
    logits = (torch.randn(1, 400) * 3.0)
    N = 200_000

    gen = torch.Generator().manual_seed(9)
    ref = torch.stack([enc._sample_top_k(logits, gen) for _ in range(N)]).flatten()

    gen2 = torch.Generator().manual_seed(10)
    vals = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    thr = torch.topk(vals, enc._AR_SAMPLING_TOP_K, dim=-1).values[..., -1, None]
    vals = vals.masked_fill(vals < thr, -float("inf"))
    g = turbo._gumbel((N, 400), gen2, vals.device)
    gum = torch.argmax(vals + g, dim=-1)

    ref_hist = torch.bincount(ref, minlength=400).float() / N
    gum_hist = torch.bincount(gum, minlength=400).float() / N
    l1 = (ref_hist - gum_hist).abs().sum().item()
    top_ref = set(torch.topk(ref_hist, 10).indices.tolist())
    top_gum = set(torch.topk(gum_hist, 10).indices.tolist())
    print(f"GUMBEL UNIT TEST: L1(dist diff)={l1:.4f} (expect < ~0.02), "
          f"top-10 overlap={len(top_ref & top_gum)}/10", flush=True)
    if l1 > 0.05:
        raise SystemExit("Gumbel sampler does not match reference distribution")


BLOCK_TIMES: dict[str, float] = {}


def install_block_timers():
    from diffusers.modular_pipelines.minimax_music3 import (
        decoders as dec, denoise as dn, encoders as enc_mod,
    )

    targets = {
        "semantic_ar": enc_mod.MiniMaxMusic3SemanticGenerationStep,
        "denoise_dit": dn.MiniMaxMusic3ChunkDenoiseStep,
        "vocoder": dec.MiniMaxMusic3VocoderDecodeStep,
    }
    for name, cls in targets.items():
        orig = cls.__call__

        def timed(self, components, state, _orig=orig, _name=name):
            torch.cuda.synchronize()
            t0 = time.time()
            out = _orig(self, components, state)
            torch.cuda.synchronize()
            BLOCK_TIMES[_name] = BLOCK_TIMES.get(_name, 0.0) + (time.time() - t0)
            return out

        cls.__call__ = timed


def run_song(pipe, tag):
    BLOCK_TIMES.clear()
    torch.cuda.synchronize()
    t0 = time.time()
    audio = pipe(
        prompt=("Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. "
                "Vocals: soft female lead, close and breathy. Arrangement: "
                "fingerpicked guitar and soft piano."),
        lyrics="[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe",
        audio_duration=DUR,
        generator=torch.Generator("cuda").manual_seed(SEED),
        output="audios",
    )[0]
    torch.cuda.synchronize()
    total = time.time() - t0

    arr = np.squeeze(np.asarray(audio, dtype=np.float32))
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    sf.write(str(OUT / f"lab2_{tag}.wav"), arr, 44100, subtype="FLOAT")
    blocks = "  ".join(f"{k}={v:.1f}s" for k, v in BLOCK_TIMES.items())
    secs = arr.shape[0] / 44100
    rms = float(np.sqrt((arr ** 2).mean()))
    print(f"[{tag}] total={total:.1f}s ({total/DUR:.2f}x RT)  {blocks}  "
          f"audio={secs:.1f}s rms={rms:.3f}", flush=True)
    free, tot = torch.cuda.mem_get_info()
    print(f"  VRAM {(tot-free)/1024**3:.1f}/{tot/1024**3:.1f} GB", flush=True)


def main():
    print(f"duration={DUR}s seed={SEED} models={MODEL}", flush=True)
    # Same desktop-headroom cap the engine applies (~21.1GB of 24GB).
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MUSIC3_VRAM_FRACTION", "0.88")))
    unit_test_gumbel()

    import turbo
    print(f"flags: slim_head={turbo.V2_SLIM_HEAD} fused={turbo.V2_FUSED} dit={turbo.V2_DIT}", flush=True)

    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.time()
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    print(f"LOAD {time.time()-t0:.1f}s", flush=True)

    turbo.install_all(pipe)
    install_block_timers()

    print("--- warm-up (compiles) ---", flush=True)
    run_song(pipe, "v2_compile")
    print("--- v2 warm 1 ---", flush=True)
    run_song(pipe, "v2_warm1")
    print("--- v2 warm 2 ---", flush=True)
    run_song(pipe, "v2_warm2")
    print("=== V2 LAB DONE ===", flush=True)


if __name__ == "__main__":
    main()

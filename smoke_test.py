"""Standalone check that the model loads and renders on this GPU.

Run:  .venv\\Scripts\\python.exe smoke_test.py [mode] [seconds]

modes:
  gpu     all weights resident on GPU      ~22.5GB VRAM, ~2GB RAM   (fastest)
  auto    whole-component CPU offload      ~18GB VRAM, ~26GB RAM    (default)
  hybrid  per-layer offload of the LLM     UNUSABLE - see below
  cpu     auto + per-layer LLM offload     UNUSABLE - see below

'hybrid' and 'cpu' apply per-LAYER (group) offloading to the language model.
That is fine for diffusion models but catastrophic here: the Global LLM is
autoregressive and runs one forward pass per audio frame (25/sec), so every
frame re-streams 16.4GB across PCIe. Measured: 10% GPU utilisation, 76W, no
progress. Keep the LLM resident; only whole-component offload is viable.

Note: use_stream=False everywhere. use_stream=True allocates pinned host memory
for async H2D copies, which duplicates the 16.4GB language model and pushed this
machine to 31-38GB RSS. Unrelated to the model card's "no streaming generation"
note, which is about incremental audio output.
"""

import sys
import threading
import time
from pathlib import Path

import soundfile as sf
import torch

MODEL = str(Path(__file__).parent / "models")
MODE = sys.argv[1] if len(sys.argv) > 1 else "auto"
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
OUT = Path(__file__).parent / f"smoke_{MODE}.wav"


def vram(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"  [{tag}] VRAM {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB", flush=True)


def watchdog():
    """Abort rather than drive the machine into swap."""
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong)
            for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                      "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                      "ullAvailExtendedVirtual")
        ]

    while True:
        st = MEMSTAT()
        st.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        avail = st.ullAvailPhys / 1024**3
        if avail < 3.0:
            print(f"\n!! ABORT: only {avail:.1f} GB RAM left — {MODE} mode is too "
                  f"memory hungry on this machine.", flush=True)
            import os
            os._exit(9)
        time.sleep(2)


def main():
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"mode={MODE}  duration={DURATION:.0f}s", flush=True)
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    vram("start")

    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.time()
    print(f"loading ({MODE})...", flush=True)

    if MODE == "auto":
        # Whole-component offload: each component is moved to the GPU once when
        # its stage runs, then evicted. Critically, the 8B LLM stays resident for
        # its entire autoregressive phase (one forward pass per audio frame), so
        # there is no per-token PCIe traffic. Peak VRAM ~= largest component.
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
    elif MODE == "cpu":
        from diffusers.hooks import apply_group_offloading

        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
        apply_group_offloading(
            pipe.language_model, onload_device=torch.device("cuda"),
            offload_type="leaf_level", use_stream=True,
        )
    elif MODE == "hybrid":
        from diffusers.hooks import apply_group_offloading

        pipe = ModularPipeline.from_pretrained(MODEL)
        pipe.load_components(dtype=torch.bfloat16)
        # Offload only the 8B LLM (the 16.4GB component); everything else is
        # small enough to stay resident on the GPU. No pinned buffers.
        apply_group_offloading(
            pipe.language_model,
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )
        for name in ("transformer", "rvq_depth_decoder", "condition_encoder", "vocoder"):
            comp = getattr(pipe, name, None)
            if comp is not None and hasattr(comp, "to"):
                comp.to("cuda")
    else:  # gpu
        pipe = ModularPipeline.from_pretrained(MODEL)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to("cuda")

    print(f"loaded in {time.time()-t0:.1f}s  sr={getattr(pipe,'sampling_rate',None)}", flush=True)
    vram("loaded")

    blocks = getattr(pipe, "blocks", None)
    for attr in ("input_names", "inputs"):
        v = getattr(blocks, attr, None) if blocks is not None else None
        if v:
            print(f"blocks.{attr}: {[getattr(x,'name',x) for x in v]}", flush=True)
            break

    lyrics = ("[verse]\nMorning light filtering through the pine\n"
              "Every quiet street is yours and mine\n"
              "[chorus]\nSoftly the world begins to breathe")
    prompt = (
        "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently "
        "into the chorus. Vocals: soft female lead, close and breathy. Arrangement: "
        "fingerpicked guitar and soft piano; brushed drums enter in the chorus."
    )

    t1 = time.time()
    print(f"generating {DURATION:.0f}s...", flush=True)
    audio = pipe(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=DURATION,
        generator=torch.Generator("cuda").manual_seed(7),
        output="audios",
    )[0]
    gen = time.time() - t1
    vram("generated")

    sr = int(getattr(pipe, "sampling_rate", 32000) or 32000)
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().to(torch.float32).cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    data = arr.T if arr.ndim > 1 else arr
    sf.write(str(OUT), data, sr)
    secs = data.shape[0] / sr
    print(f"\nOK -> {OUT}", flush=True)
    print(f"   {secs:.1f}s audio @ {sr}Hz shape={data.shape}", flush=True)
    print(f"   render {gen:.1f}s ({gen/max(secs,1e-6):.2f}x realtime)", flush=True)


if __name__ == "__main__":
    main()

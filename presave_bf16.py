"""One-off: re-save every pipeline component in bf16 safetensors.

Why: model load measured ~340s and is CPU-bound — the transformer ships fp32
(9.3GB) and gets converted to bf16 on every load; the rest re-serializes too.
Pre-saving in bf16 makes loading a straight mmap-and-go.

Loads components one at a time (peak RAM = largest component, ~16GB) so it can
run beside a living system. Quality-neutral by construction: bf16 is exactly
what inference runs in either way.

Run:  .venv\\Scripts\\python.exe presave_bf16.py [src] [dst]
"""

import shutil
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "models"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else _ROOT / "models_bf16"


def convert(name, loader):
    t0 = time.time()
    print(f"[{name}] loading...", flush=True)
    model = loader()
    model = model.to(torch.bfloat16)
    out = DST / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    del model
    print(f"[{name}] saved in {time.time()-t0:.0f}s", flush=True)


def main():
    DST.mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3ForCausalLM

    from diffusers import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )

    convert("transformer", lambda: MiniMaxMusic3Transformer1DModel.from_pretrained(
        str(SRC), subfolder="transformer", torch_dtype=torch.bfloat16))
    convert("language_model", lambda: Qwen3ForCausalLM.from_pretrained(
        str(SRC), subfolder="language_model", torch_dtype=torch.bfloat16))
    convert("rvq_depth_decoder", lambda: MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        str(SRC), subfolder="rvq_depth_decoder", torch_dtype=torch.bfloat16))
    convert("condition_encoder", lambda: MiniMaxMusic3ConditionEncoder.from_pretrained(
        str(SRC), subfolder="condition_encoder", torch_dtype=torch.bfloat16))
    convert("vocoder", lambda: MiniMaxMusic3Vocoder.from_pretrained(
        str(SRC), subfolder="vocoder", torch_dtype=torch.bfloat16))

    for item in ("tokenizer", "scheduler"):
        shutil.copytree(SRC / item, DST / item, dirs_exist_ok=True)
        print(f"[{item}] copied", flush=True)
    for f in ("modular_model_index.json", "config.json", "LICENSE"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DST / f)

    total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1024**3
    print(f"DONE -> {DST}  ({total:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()

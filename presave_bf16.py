"""One-off: re-save the pipeline in bf16 and make the install fully offline.

Three jobs, after which first launch touches the network zero times:
1. Re-save every component as bf16 safetensors (the transformer ships fp32).
2. Rewrite modular_model_index.json to point at the local folder — as shipped
   it records the HF repo id, which makes every model load phone home for
   tokenizer files even with all weights on disk.
3. Pre-fetch the songwriter companion model (~3.4GB) into the app's own cache
   (.hfcache, the same location start.bat uses).

Loads components one at a time (peak RAM = largest component, ~16GB).
Quality-neutral by construction: bf16 is exactly what inference runs in.

Run:  .venv\\Scripts\\python.exe presave_bf16.py [src] [dst]
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
# Must be set before any HF import so the songwriter prefetch lands in the
# same cache start.bat points the server at.
os.environ.setdefault("HF_HOME", str(_ROOT / ".hfcache"))

import torch

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
    for f in ("config.json", "LICENSE"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, DST / f)

    # Point every component at THIS folder instead of the HF repo id, so
    # loading never resolves anything against the Hub.
    index = json.loads((SRC / "modular_model_index.json").read_text(encoding="utf-8"))
    for name, entry in index.items():
        if (
            not name.startswith("_")
            and isinstance(entry, list)
            and len(entry) == 3
            and isinstance(entry[2], dict)
            and "pretrained_model_name_or_path" in entry[2]
        ):
            entry[2]["pretrained_model_name_or_path"] = str(DST.resolve())
    (DST / "modular_model_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("[index] rewritten to local paths (offline-clean loads)", flush=True)

    print("[songwriter] pre-fetching Qwen/Qwen3-1.7B (~3.4GB)...", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen3-1.7B", token=False)
    print("[songwriter] cached", flush=True)

    total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1024**3
    print(f"DONE -> {DST}  ({total:.2f} GB)  — first launch is now fully offline", flush=True)


if __name__ == "__main__":
    main()

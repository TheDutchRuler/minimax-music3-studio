---
license: apache-2.0
tags:
  - diffusers
  - modular-diffusers
  - custom-blocks
  - minimax-music3
  - text-to-audio
  - music-generation
---

# MiniMax Music 3 — Ensemble Blocks

**Render K variations of one prompt in a single batched pass — the marginal
variation is nearly free.**

The MiniMax Music 3 autoregressive stage reads the full 8B language model plus
seven 0.6B depth-decoder passes (~23GB of weights) for *every* audio frame at
25fps: it is memory-bandwidth bound, and that read costs the same whether it
serves one variation or four. These blocks decode K variations in lockstep
(batch 2K rows), sharing the dominant cost. The flow-matching stage batches
across variations too (grouped by exact frame count — no padding).

Measured with THIS block on an RTX 4090 (24GB, bf16, diffusers main, warm,
15s songs, same prompt/seeds):

| variations | total | per variation |
|---|---|---|
| 1 | 28.9s | 28.9s |
| 2 | 34.9s | 17.5s (1.65x) |
| 3 | 41.5s | **13.8s (2.1x)** |

Each additional variation costs ~6s on a ~29s base — the marginal take is
~21% of a solo render. Discussion:
[diffusers#14486](https://github.com/huggingface/diffusers/issues/14486).

## Usage

```python
import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentsManager

blocks = ModularPipelineBlocks.from_pretrained(
    "Rreitsma/minimax-music3-ensemble-blocks", trust_remote_code=True
)
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")
pipe = blocks.init_pipeline("MiniMaxAI/MiniMax-Music3", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)

out = pipe(
    prompt="Genre: acoustic pop. BPM: 96. Key: C major. Warm female vocals, fingerpicked guitar.",
    lyrics="[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe",
    audio_duration=60.0,
    num_variations=3,
    seed=7,            # variation i uses seed + i; omit for random seeds
    output="audios",
)
# out: list of float32 stereo waveforms, one per variation, 44.1kHz, (channels, samples)
```

## Notes & honest caveats

- **Quality:** every variation's math is row-independent and draws from its
  own seeded generator using the reference sampling recipe — identical in
  distribution to solo generation. Batched kernels differ from solo kernels at
  the floating-point-ulp level, so a given seed may take a different (equally
  valid) trajectory than it would solo.
- **Guidance** is standard CFG with the checkpoint's guidance scale. For other
  guidance techniques, use the default MiniMax Music 3 blocks (with the
  guider) instead.
- **VRAM** scales with `duration x num_variations` (KV cache). On 24GB:
  3 variations up to ~1 minute is comfortable; use fewer variations for longer
  songs. An early EOS in one variation freezes its rows at no cost.
- Deliberately **plain eager PyTorch** — no torch.compile, no extra
  dependencies — so it runs wherever diffusers runs. A further-optimized local
  studio (compiled AR decode, ~2.9x total on a 4090) lives at
  [minimax-music3-studio](https://github.com/TheDutchRuler/minimax-music3-studio).

## License & credits

Apache-2.0 (portions derived from the diffusers MiniMax Music 3 modular
pipeline, Copyright 2026 The MiniMax Team and The HuggingFace Team). Model
weights are MiniMax's, CC BY 4.0, fetched separately from
[MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

Built with Claude (Fable 5 Max) following a request from the diffusers
maintainers in [#14486](https://github.com/huggingface/diffusers/issues/14486).

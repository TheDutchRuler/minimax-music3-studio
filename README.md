# MiniMax Music 3 — Turbo Studio

**Full songs on your own GPU, faster than realtime.** A local, Suno-style studio
around [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) with a
re-engineered inference path measured **2.9x faster per song** than the reference
implementation on a single RTX 4090 — no API key, no cloud, no per-song cost.

## The headline

Measured end-to-end in the app (20-second songs, warm server, RTX 4090 24GB,
desktop-safe VRAM profile — every number below is a real render, not a projection):

| Configuration | Per song | Speed |
|---|---|---|
| Reference diffusers implementation | 50.5s | 2.52x realtime |
| + turbo (compiled AR, batched CFG, slim head, fused sampling) | ~31s | 1.55x |
| + ensemble, 2 variations per pass | 21.2s | 1.06x |
| **+ ensemble, 3 variations (the default)** | **17.7s** | **0.88x — faster than realtime** |

A 3-minute song: **~13 minutes → ~4.5 minutes**, and the default click gives you
**three takes** of it. Quality is untouched: bf16 reference precision everywhere,
guidance math identical, sampling proven distribution-identical (details below).

## What's in the box

- **Suno-style web UI** — dark studio interface: create panel with structured-caption
  fields, lyrics editor with section-tag buttons, instrumental toggle, track library
  with players, waveform seek bar, one-click WAV download (32-bit float masters).
- **AI songwriter** — a small companion model (Qwen3-1.7B, ~3.4GB, fetched on
  first use) expands a one-line brief into a title, a three-part structured
  caption, and fully tagged lyrics. (Fun finding: the music checkpoint's own 8B
  cannot write text anymore — the music fine-tune re-adapted its embedding and
  output layers, and chat prompting yields gibberish. Tried, documented.)
- **Ensemble rendering** — variations of a prompt render as ONE batched pass.
  This is the trick that beats the hardware wall (below).
- **Self-healing engine** — any optimized path that fails automatically retries on
  the reference implementation. A bug costs speed, never a song.
- **Desktop-safe by default** — VRAM capped so Chrome and your desktop keep
  working while you render.

## Quick start

Requirements: Windows 11 (WSL not needed), a 24GB CUDA GPU (RTX 3090/4090-class),
~64GB RAM, Python 3.12 via [uv](https://docs.astral.sh/uv/), ~50GB disk.

```bat
uv venv --python 3.12 .venv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install "git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d" transformers accelerate soundfile fastapi "uvicorn[standard]" triton-windows huggingface_hub

.venv\Scripts\python.exe fetch_weights.py      REM ~26.5GB — only the components diffusers uses
.venv\Scripts\python.exe presave_bf16.py       REM one-off bf16 re-save (~22GB)
start.bat
```

Open <http://127.0.0.1:7878>. First launch loads the model (~4 min) and the first
song compiles the CUDA graphs (a few extra minutes, cached on disk in
`.inductor_cache` for every later session). Then it's 0.88x realtime per song.

## How it got 2.9x faster (and why it can't get much more)

The full experiment log lives in this repo's history; the short version:

**1. The AR stage was CPU-dispatch bound.** Music 3 decodes 25 audio frames per
second; each frame is one 8B forward plus seven 0.6B depth-decoder forwards —
thousands of tiny kernel launches. `app/turbo.py` rebuilds the loop with a
transformers StaticCache and `torch.compile(mode="reduce-overhead")` (CUDA
graphs) on the decode step and depth chain. Two subtleties cost days if you miss
them: the pipeline calls `language_model.model(...)` — compiling the parent
module compiles a callable the loop never invokes — and a hand-built StaticCache
must have its tensors passed through `torch._dynamo.mark_static_address`, or
cudagraph trees silently refuse to capture the in-graph cache mutation.

**2. The DiT ran CFG as two sequential passes.** One batch=2 forward computes the
identical guidance formula. Also: `lm_head` reads a 1.38GB matrix per frame when
only 16,389 rows (audio codes + EOS) can ever be sampled — slicing them is
provably identical and ~10x cheaper. And sampling was rebuilt on the Gumbel-max
trick (`argmax(logits + gumbel)` ≡ top-k softmax multinomial — verified by a
200k-draw distribution test, L1=0.015) so the whole depth chain compiles into
graphs with zero eager glue.

**3. Then the wall: memory bandwidth.** Five successively heavier AR builds all
converged on ~40ms/frame — exactly the time needed to read the 8B + 7×0.6B
weights (~23GB/frame) at realistic small-batch GEMV efficiency on Ada. Cross-check:
ComfyUI's RTX 5090 numbers scale to the same wall by bandwidth ratio.

**4. The wall doesn't move — so drive through it sideways.** The weight read costs
the same whether it serves one song or four. `app/ensemble.py` decodes K
same-prompt variations in lockstep (batch 2K rows) and batches the DiT across
songs too (grouped by exact frame count — padding never enters the math). Three
variations cost barely more than one; per-song speed goes **sub-realtime**.

**Negative results, documented so you don't repeat them:**
- Per-layer ("group") offloading of the LLM — the model card suggests it for 8GB
  cards — re-streams 16.4GB over PCIe *per frame* on an autoregressive model:
  10% GPU utilization, effectively hung.
- FP8 weight-only (torchao) measured **2.1x slower** on Windows/torch 2.11: the
  subclass never engages fused dequant kernels, so every forward dequantizes to
  bf16 (more traffic) and breaks the CUDA-graph path. Flag remains (`MUSIC3_FP8=1`).
- NVMe vs HDD for weights: identical load time (CPU-materialization-bound).
  Parallel shard loading: no change. bf16 pre-save: right format, no load win.
- DiT CUDA graphs: no speedup, gigabytes of capture pools. Kernel-fusion-only
  compile kept.

## Quality: what "same" means here, precisely

- Same bf16 precision, same guidance formulas, same sampling parameters as the
  reference recipe; 32-bit float WAV masters (no 16-bit quantization on write).
- The Gumbel-max sampler draws from the *identical distribution* as the
  reference sampler (unit-tested), and every song in an ensemble is
  row-independent with its own seeded noise.
- One honest caveat: compiled kernels change float reduction order by ~ulps, so
  a given seed takes a *different equally-valid trajectory* than eager (and
  across batch sizes). Seeds reproduce within a configuration, not across them.
  `--no-turbo` runs the exact reference path.

## Stability: the "GPU crashes" that weren't

Heavy renders at ~22.5GB starved Windows/Chrome of VRAM; eviction thrash froze
the desktop in ways indistinguishable from a GPU crash — while event logs showed
**zero** driver resets or hardware errors across the whole project. Defaults now:
**80% VRAM cap** (`MUSIC3_VRAM_FRACTION=0.92` for unattended max speed),
**expandable_segments allocator** (the ~0.8GB fragmentation fix that lets
3-variation groups fit under the cap), no DiT capture pools. The safe profile
measured *as fast or faster* than the aggressive one.

## Configuration

| Env / flag | Default | Meaning |
|---|---|---|
| `MUSIC3_MODELS` | `models_bf16` (via start.bat) | Weights directory |
| `MUSIC3_VRAM_FRACTION` | `0.80` | GPU memory cap; `0.92` for unattended runs |
| `MUSIC3_WAV_SUBTYPE` | `FLOAT` | `PCM_24`/`PCM_16` for smaller files |
| `--no-turbo` | off | Exact reference implementation |
| `MUSIC3_TURBO_*`, `MUSIC3_FP8` | see `app/turbo.py` | Per-optimization kill switches / experiments |

Songs render one group at a time; extra requests queue. Each distinct variation
count compiles its own graphs on first use (one-off, disk-cached).

## Repo layout

```
app/
  engine.py      Job queue, group batching, library, VRAM profile, auto-fallback
  turbo.py       The verified single-song optimizations (+ experiment flags)
  ensemble.py    Batched multi-variation generation (the sub-realtime path)
  writer.py      AI songwriter on the resident 8B
  server.py      FastAPI + SSE
  static/        The UI
fetch_weights.py    Download the ~26.5GB the diffusers path actually uses
presave_bf16.py     One-off bf16 re-save
smoke_test.py       Standalone load + render check
labs/               The experiment harnesses behind every number above
make_ab.py          Blind listening-test builder (used for the FP8 verdict)
start.bat           Launcher
```

## Findings that may interest upstream

For **MiniMax** ([model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3)):
the low-VRAM `apply_group_offloading(..., use_stream=True)` snippet is
counter-productive for the autoregressive LLM (per-frame PCIe re-streaming; and
`use_stream=True` pins host memory that doubled RSS on a 61GB machine). The
pipeline also reports/produces 44.1kHz where the card says 32kHz.

For **diffusers** (modular pipeline, [PR #14456](https://github.com/huggingface/diffusers/pull/14456)):
the AR block's submodule calls make naive `torch.compile(pipe.language_model)`
a silent no-op; StaticCache needs `mark_static_address` for cudagraph capture;
batched CFG + the slim head are drop-in wins; and ensemble batching may be
worth first-class support — it is the single biggest speedup available.

## Credits

- **MiniMax** — the Music 3 model (weights [CC BY 4.0](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE); attribute MiniMax when publishing generated audio commercially).
- **Hugging Face diffusers** — the modular pipeline this builds on (pinned commit `dafe3733`).
- **triton-windows** — made `torch.compile` possible on Windows.
- Engineered end-to-end with **Claude (Fable 5 Max)** by Anthropic — profiling,
  the optimization ladder, the ensemble architecture, and this documentation.

App code: MIT license (see `LICENSE`). This is a community project, not
affiliated with MiniMax.

## Troubleshooting

- **CUDA out of memory** — lower variation count, or check `MUSIC3_VRAM_FRACTION`.
  The engine auto-falls back (group → solo → reference) rather than failing a song.
- **Model won't load / missing files** — rerun `fetch_weights.py` (resumes), then
  `presave_bf16.py`.
- **First song is slow** — that's the one-off graph compilation; it's cached for
  later sessions.
- **Verify outside the app** — `.venv\Scripts\python.exe smoke_test.py auto 20`.
- **Port in use** — `app\server.py --port 7900`.

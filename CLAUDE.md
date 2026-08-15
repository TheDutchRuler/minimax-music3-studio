# MiniMax Music 3 Turbo Studio — contributor notes

Local Suno-style studio around MiniMax Music 3 with a re-engineered inference
path (2.9x measured on an RTX 4090; sub-realtime per song via ensemble
batching). Read README.md first — it records every measurement and every
negative result. Do not re-try an approach the README documents as failed
without new evidence (per-layer LLM offload, FP8-via-torchao on Windows,
DiT CUDA graphs, load-time "fixes").

## Architecture in one breath

`app/server.py` (FastAPI+SSE) → `app/engine.py` (single GPU worker: job queue,
variation groups, songwriter requests, VRAM profile, auto-fallback) →
`app/turbo.py` (runtime replacement of the diffusers blocks: compiled AR
decode + depth chain, slim lm_head, Gumbel-max sampling, batched-CFG DiT) and
`app/ensemble.py` (K same-prompt variations decoded in one batched AR pass +
cross-song batched DiT). `app/writer.py` uses the resident 8B for lyrics.

## Hard-won invariants

- diffusers is PINNED to commit `dafe3733` (the pre-merge modular pipeline).
  turbo/ensemble monkeypatch its block classes and port its math line-for-line;
  upgrading diffusers means re-verifying both against the renamed blocks.
- NEVER per-layer/group-offload `language_model`: it decodes autoregressively
  (25 fwd/s), so per-layer offload re-streams 16.4GB per frame.
- Compile targets the callables the loop actually invokes
  (`language_model.model`, not `language_model`) and the StaticCache tensors
  must go through `torch._dynamo.mark_static_address` after prefill, or CUDA
  graphs silently never engage.
- Sampling must mirror `encoders.py` ops exactly (mask -> CFG -> CFG-top-k ->
  sample-top-k); the Gumbel argmax replaces only the final multinomial and is
  distribution-identical (unit test in labs/turbo_lab2.py).
- Quality bar for changes: bf16 everywhere, formula-identical math, or a
  measured distribution-equivalence proof. "Should be fine" is not a proof.
- VRAM: default cap 0.80 of device memory + expandable_segments. Peak-stage
  residency is LLM+depth (~17.6GB); group size is derived from capacity in
  `engine.submit`. Failures degrade group -> solo -> eager reference.
- Every speed claim gets measured warm (second render onward), same seed,
  through the real server. First renders carry one-off compile costs.

## Testing

`labs/` holds the harnesses that produced the README numbers
(`turbo_lab2.py` = unit test + warm timings; `lab_ensemble.py` = batching).
`smoke_test.py` is the standalone load+render check. The engine's own
fallback chain is the safety net — keep it intact.

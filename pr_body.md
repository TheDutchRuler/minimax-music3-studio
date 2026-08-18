## What does this PR do?

Requested by @yiyixuxu in #14486: slice `lm_head` to the sampleable rows in
the MiniMax Music 3 autoregressive decode loop.

The loop samples only the 16,384 semantic codes plus the end-of-audio token;
every other vocabulary row was masked to `-inf` *after* a full-vocabulary head
matmul. With a 200k-row head at bf16 that is ~1.64GB of weight read per audio
frame, paid 25 times per second of audio. Slicing the head to the contiguous
sampleable range `[_AUDIO_END_TOKEN_ID, _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE)`
reads ~134MB instead.

**Measured (RTX 4090, bf16, real checkpoint weights):**
- head op: 5.31ms → 0.54ms per frame (9.9x on the op)
- ≈ 2.4s saved per 20s of generated audio; scales linearly with duration

**Numerical characterization** (`labs/verify_slim_head.py` in
[our repo](https://github.com/TheDutchRuler/minimax-music3-studio), 3 seeds ×
200 steps against the real head weights): logits on sampleable rows match the
full-head path to within **one bf16 ulp** (0.03125) — the GEMM tile shape
changes accumulation order. The sampling distribution is unchanged; as with
any kernel-shape change, an individual seed may take a different
(equally valid) trajectory than before.

The masking, CFG, top-k restriction, and sampling ops are byte-for-byte the
reference sequence, only re-indexed into the sliced range (the sampled local
index gets `head_start` added back, so everything downstream is untouched).

## Before submitting
- [x] Read contributor guidelines
- [x] Discussed in #14486 (maintainer-requested)

## Who can review?
@yiyixuxu @asomoza

Developed and verified with Claude (Fable 5 Max).

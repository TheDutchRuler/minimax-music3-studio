# Portions of this file are faithful batched ports of the diffusers MiniMax
# Music 3 modular pipeline blocks (encoders/before_denoise/denoise/decoders,
# commit dafe3733): Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Licensed under the Apache License, Version 2.0 — see LICENSES/Apache-2.0.txt.
# Modifications Copyright 2026 Xtraworks: generation batched across K
# variations (lockstep AR decode, per-song seeded sampling, cross-song
# batched CFG denoise grouped by exact frame count).

"""Ensemble generation: render K same-prompt variations in ONE batched pass.

Why this beats the single-song wall: the AR stage is memory-bandwidth bound —
every audio frame reads the full 8B LLM plus seven 0.6B depth-decoder passes
(~23GB) from VRAM regardless of batch size. Solo generation pays that read for
2 CFG rows; ensemble generation pays the SAME read for 2K rows (K songs), so K
variations cost barely more than one. Measured solo floor: ~40ms/frame. The
marginal cost of extra rows is activations only (megabytes).

Quality: exact. Each song occupies its own (cond, uncond) row pair; every op —
attention, depth chain, CFG, sampling — is row-independent, and each song draws
its Gumbel noise from its own seeded generator on the same schedule as the solo
path. A song that ends early (EOS) simply freezes its rows; the batch runs at
the pace of the longest song either way, so frozen rows are free.

The AR stage is a faithful batched port of the reference semantic-generation
block (diffusers @ dafe3733, encoders.py) with the same turbo v2 machinery
(slim head, Gumbel-max sampling, compiled decode). The DiT + vocoder stages are
faithful ports of denoise.py / decoders.py, run per song with batched CFG and
the shared compiled transformer forward.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

import turbo

log = logging.getLogger("music3.ensemble")


def _tokenize_pair(components, prompt: str, lyrics: str) -> torch.Tensor:
    """Reference prompt assembly (encoders.MiniMaxMusic3TextEncoderStep)."""
    from diffusers.modular_pipelines.minimax_music3 import encoders as enc

    text = (
        f"{enc._IM_START}{enc._CAPTION_START}{enc._clean_caption(prompt)}{enc._CAPTION_END}"
        f"{enc._LYRICS_START}{enc._normalize_lyrics(lyrics)}{enc._LYRICS_END}"
        f"{enc._IM_END}{enc._AUDIO_START}"
    )
    input_ids = components.tokenizer(text, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] > enc._MAX_PROMPT_TOKENS:
        raise ValueError(
            f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {enc._MAX_PROMPT_TOKENS}"
        )
    unconditional = input_ids.clone()
    unconditional[:, 1:-2] = enc._AUDIO_CFG_TOKEN_ID
    return torch.cat((input_ids, unconditional), dim=0)


@torch.no_grad()
def generate_group(
    pipe,
    prompt: str,
    lyrics: str,
    duration: float,
    seeds: list[int],
    steps: int | None = None,
    progress=None,
) -> list[np.ndarray]:
    """Render len(seeds) variations of one prompt in a single batched AR pass.

    progress: optional callable(fraction, stage_text).
    Returns one float32 (frames, channels) array per seed.
    """
    from diffusers.modular_pipelines.minimax_music3 import encoders as enc
    from transformers import StaticCache

    components = pipe
    K = len(seeds)
    if K < 1:
        raise ValueError("need at least one seed")
    device = components._execution_device
    language_model = components.language_model
    rvq = components.rvq_depth_decoder

    def report(frac, text):
        if progress is not None:
            progress(frac, text)

    # ---- placement (same manual hook trigger as the reference block) ----
    def _place():
        for m in (language_model, rvq):
            if getattr(m, "_hf_hook", None) is not None:
                m._hf_hook.pre_forward(m)

    _place()
    if language_model.model.embed_tokens.weight.device.type != "cuda":
        # Stale per-K caches/graph pools can crowd the VRAM cap enough that the
        # offload manager silently fails to place the 16GB LLM. Purge and retry.
        store0 = getattr(pipe, "_ensemble_state", None)
        if store0 is not None:
            store0["caches"] = {k: v for k, v in store0["caches"].items() if k[0] == K}
            store0["decodes"] = {k: v for k, v in store0["decodes"].items() if k[0] == K}
        torch.cuda.empty_cache()
        _place()
        if language_model.model.embed_tokens.weight.device.type != "cuda":
            raise RuntimeError(
                "could not place the language model on the GPU — free VRAM or "
                "raise MUSIC3_VRAM_FRACTION"
            )

    # ---- geometry (mirrors turbo v2) ----
    vocab_size = int(language_model.config.vocab_size)
    if turbo.V2_SLIM_HEAD:
        slice_start = enc._AUDIO_END_TOKEN_ID
        slice_end = enc._AUDIO_CODE_OFFSET + enc._SEMANTIC_VOCAB_SIZE
    else:
        slice_start, slice_end = 0, vocab_size
    width = slice_end - slice_start
    eos_local = enc._AUDIO_END_TOKEN_ID - slice_start
    sem_local = enc._AUDIO_CODE_OFFSET - slice_start
    num_codebooks = int(components.num_codebooks)
    audio_vocab = int(components.audio_vocab_size)

    max_frames = min(int(duration * components.frame_rate), enc._MAX_AUDIO_FRAMES)
    if max_frames == 0:
        raise ValueError("audio_duration is shorter than one audio frame")

    text_pair = _tokenize_pair(components, prompt, lyrics).to(device)  # [2, L]
    prompt_len = text_pair.shape[1]
    text_ids = text_pair.repeat(K, 1)  # [2K, L] rows: s0c,s0u,s1c,s1u,...

    head_weight = language_model.lm_head.weight
    head_bias = language_model.lm_head.bias

    def head(hidden):
        w = head_weight[slice_start:slice_end]
        b = head_bias[slice_start:slice_end] if head_bias is not None else None
        return torch.nn.functional.linear(hidden, w, b)

    # ---- static cache for batch 2K, bucketed like turbo v2 ----
    need = prompt_len + max_frames + 8
    bucket = ((need + 1023) // 1024) * 1024
    store = getattr(pipe, "_ensemble_state", None)
    if store is None:
        store = pipe._ensemble_state = {"caches": {}, "decodes": {}}
    # Keep artifacts for one batch size only — a changed K frees the old
    # cache tensors so the LLM always has room to come back.
    if any(k[0] != K for k in store["caches"]):
        store["caches"] = {k: v for k, v in store["caches"].items() if k[0] == K}
        store["decodes"] = {k: v for k, v in store["decodes"].items() if k[0] == K}
        torch.cuda.empty_cache()
    cache = store["caches"].get((K, bucket))
    fresh_cache = cache is None
    if fresh_cache:
        kw = dict(
            config=language_model.config, max_cache_len=bucket,
            device=device, dtype=language_model.dtype,
        )
        try:
            cache = StaticCache(max_batch_size=2 * K, **kw)
        except TypeError:
            cache = StaticCache(batch_size=2 * K, **kw)
        store["caches"][(K, bucket)] = cache
    else:
        cache.reset()

    embed_tokens = language_model.model.embed_tokens
    report(0.01, "Prefilling prompt")
    text_embeds = embed_tokens(text_ids)
    cache_position = torch.arange(prompt_len, device=device)
    output = language_model.model(
        inputs_embeds=text_embeds, past_key_values=cache,
        use_cache=True, cache_position=cache_position,
    )
    last_hidden = output.last_hidden_state[:, -1]  # [2K, H]
    if fresh_cache:
        turbo._mark_cache_static(cache)

    decode = store["decodes"].get((K, bucket))
    if decode is None:

        def _decode(feedback, position, _cache=cache):
            out = language_model.model(
                inputs_embeds=feedback, past_key_values=_cache,
                use_cache=True, cache_position=position,
            )
            lh = out.last_hidden_state[:, -1]
            return lh, head(lh)

        try:
            decode = torch.compile(_decode, mode="reduce-overhead", fullgraph=False)
        except Exception as exc:  # pragma: no cover
            log.warning("ensemble decode compile unavailable, eager: %s", exc)
            decode = _decode
        store["decodes"][(K, bucket)] = decode

    local_mask = torch.ones(width, dtype=torch.bool, device=device)
    local_mask[sem_local : sem_local + enc._SEMANTIC_VOCAB_SIZE] = False
    local_mask[eos_local] = False

    rvq_forward = type(rvq).forward
    projection = rvq.projection
    audio_heads = rvq.audio_heads
    audio_embeddings = rvq.audio_embeddings
    code_offsets = (torch.arange(num_codebooks - 1, device=device) * audio_vocab).unsqueeze(0)
    inv_sqrt = num_codebooks ** -0.5
    cfg = enc._AR_CFG_SCALE
    topk_cfg = enc._AR_CFG_TOP_K
    topk_samp = enc._AR_SAMPLING_TOP_K

    def sample_rows(logits_2k, g_k):
        """Vectorized per-song CFG + top-k + Gumbel argmax. logits_2k: [2K, V]."""
        logits = logits_2k.float()
        cond, unc = logits[0::2], logits[1::2]           # [K, V] each
        guided = unc + (cond - unc) * cfg
        thr = torch.topk(cond, topk_cfg, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(cond < thr, -float("inf"))
        vals = torch.nan_to_num(guided, nan=-1e9, posinf=1e9, neginf=-1e9)
        thr2 = torch.topk(vals, topk_samp, dim=-1).values[..., -1, None]
        vals = vals.masked_fill(vals < thr2, -float("inf"))
        return torch.argmax(vals + g_k, dim=-1)          # [K]

    def depth_chain(last_hidden_2k, sem2k, g_dep_k):
        """Batched port of _generate_depth_codes. Returns codes [2K, C],
        cond-row hidden cat [K, (C-1)*d]."""
        parts = [
            projection(last_hidden_2k).unsqueeze(1),
            projection(embed_tokens(sem2k + enc._AUDIO_CODE_OFFSET)).unsqueeze(1),
        ]
        codes = [sem2k]
        hidden_parts = []
        for index in range(1, num_codebooks):
            hidden = rvq_forward(rvq, torch.cat(parts, dim=1))[:, -1]  # [2K, d]
            hidden_parts.append(hidden[0::2])
            logits = audio_heads[index - 1](hidden)                    # [2K, 1024]
            code_k = sample_rows(logits, g_dep_k[:, index - 1])        # no vocab mask (reference)
            code = code_k.repeat_interleave(2)                         # [2K]
            codes.append(code)
            if index < num_codebooks - 1:
                embed = audio_embeddings(code + (index - 1) * audio_vocab)
                parts.append(projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    lm_logits = head(last_hidden)
    pos = prompt_len
    pos_t = torch.tensor([0], device=device)

    # (frame_index, local[K], cond_hidden[K,H], depth_hidden[K,d7]) — EOS checks
    # sync the GPU, so they are drained in batches like the turbo megastep.
    pending: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    active = [True] * K
    frames: list[list[torch.Tensor]] = [[] for _ in range(K)]
    g_sem_all = g_dep_all = None
    sync_every = 8

    def drain() -> bool:
        """Apply pending frames' bookkeeping; returns True when all songs done."""
        for fidx, loc, ch, dh in pending:
            for s, lid in enumerate(loc.tolist()):
                if active[s] and lid == eos_local:
                    active[s] = False
                if active[s] and fidx > 0 and len(frames[s]) < max_frames:
                    frames[s].append(torch.cat((ch[s : s + 1], dh[s : s + 1]), dim=-1))
        pending.clear()
        return all(
            (not active[s]) or len(frames[s]) >= max_frames for s in range(K)
        )

    report(0.02, f"Composing {K} variations")
    t_ar = time.time()
    for frame_index in range(max_frames + 1):
        slot = frame_index % turbo._GUMBEL_BLOCK
        if slot == 0:
            g_sem_all = torch.stack(
                [turbo._gumbel((turbo._GUMBEL_BLOCK, width), generators[s], device) for s in range(K)]
            )  # [K, B, V]
            g_dep_all = torch.stack(
                [turbo._gumbel((turbo._GUMBEL_BLOCK, num_codebooks - 1, audio_vocab), generators[s], device)
                 for s in range(K)]
            )  # [K, B, C-1, 1024]
        g_sem = g_sem_all[:, slot]
        g_dep = g_dep_all[:, slot]

        local = sample_rows(lm_logits.masked_fill(local_mask, -float("inf")), g_sem)  # [K]

        # Frozen songs keep decoding with a safe code (rows ignored at drain);
        # replacing EOS keeps every embedding lookup in range. No sync here.
        safe_local = torch.where(local == eos_local, torch.full_like(local, sem_local), local)
        sem2k = (safe_local + (slice_start - enc._AUDIO_CODE_OFFSET)).repeat_interleave(2)

        frame_codes, depth_hidden = depth_chain(last_hidden, sem2k, g_dep)
        pending.append((frame_index, local.clone(), last_hidden[0::2].clone(), depth_hidden.clone()))
        if len(pending) >= sync_every or frame_index == max_frames:
            if drain():
                break

        embeds = embed_tokens(frame_codes[:, :1] + enc._AUDIO_CODE_OFFSET)
        extra = audio_embeddings(frame_codes[:, 1:] + code_offsets).sum(dim=1, keepdim=True)
        feedback = (embeds + extra.to(embeds.dtype)) * inv_sqrt

        torch.compiler.cudagraph_mark_step_begin()
        pos_t.fill_(pos)
        last_hidden, lm_logits = decode(feedback, pos_t)
        last_hidden = last_hidden.clone()
        lm_logits = lm_logits.clone()
        pos += 1

        if frame_index % 50 == 0:
            done = max((len(f) for f in frames), default=0)
            report(0.02 + 0.55 * max(done, frame_index * 0.9) / max_frames,
                   f"Composing {K} variations")
    drain()

    log.info("ensemble AR: %d songs, %s frames in %.1fs",
             K, [len(f) for f in frames], time.time() - t_ar)
    for s in range(K):
        if not frames[s]:
            raise ValueError(f"variation {s} generated zero audio frames")

    # ---- DiT + vocoder, batched ACROSS songs with equal chunk counts ----
    report(0.60, "Rendering arrangements")
    n_steps = int(steps) if steps else 30
    results: list[np.ndarray | None] = [None] * K
    # Batch songs with IDENTICAL frame counts (usual case: nobody hit EOS
    # early, so all have max_frames). Equal counts mean equal tensor shapes in
    # every chunk, keeping the batched math exactly faithful — no padding.
    by_len: dict[int, list[int]] = {}
    for s in range(K):
        by_len.setdefault(len(frames[s]), []).append(s)

    done_groups = 0
    for _n, members in by_len.items():
        report(0.60 + 0.36 * done_groups / len(by_len),
               f"Rendering {len(members)} arrangement(s)")
        fh = [torch.stack(frames[s], dim=1) for s in members]     # each [1, n_s, D]
        gens = [generators[s] for s in members]
        audios = _denoise_and_vocode_group(components, fh, gens, n_steps)
        for s, audio in zip(members, audios):
            arr = np.squeeze(audio.astype(np.float32))
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            results[s] = arr
        done_groups += 1
    report(1.0, "Done")
    return results


@torch.no_grad()
def _denoise_and_vocode_group(components, frame_hiddens_list, generators, num_steps):
    """Faithful port of before_denoise + denoise + decoders, batched across G
    songs with IDENTICAL frame counts. Row layout per DiT step: rows [0..G) are
    each song's conditional branch, rows [G..2G) the zero-conditioned branch.
    Every op is row-independent; per-song noise comes from that song's own
    generator in the same draw order as the solo path.

    Returns a list of G numpy waveforms.
    """
    from diffusers.modular_pipelines.minimax_music3.before_denoise import (
        _CHUNK_FRAMES, _CHUNK_HOP,
    )
    from diffusers.modular_pipelines.minimax_music3.decoders import (
        _CROP_LEFT_LATENT, _CROP_RIGHT_LATENT,
    )
    from diffusers.modular_pipelines.minimax_music3.denoise import _OVERLAP_LATENT_LENGTH
    from diffusers.utils.torch_utils import randn_tensor

    device = components._execution_device
    transformer = components.transformer
    for m in (components.condition_encoder, transformer, components.vocoder):
        if getattr(m, "_hf_hook", None) is not None:
            m._hf_hook.pre_forward(m)

    fw = type(transformer).forward
    if turbo.V2_DIT:
        cache = getattr(_denoise_and_vocode_group, "_fw_cache", None)
        if cache is None:
            cache = _denoise_and_vocode_group._fw_cache = {}
        fw = cache.get(type(transformer))
        if fw is None:
            try:
                # No CUDA graphs for the DiT: capture pools for seq-689
                # batch-2G activations cost gigabytes of resident VRAM.
                fw = torch.compile(type(transformer).forward, dynamic=False, fullgraph=False)
            except Exception:
                fw = type(transformer).forward
            cache[type(transformer)] = fw

    guider = components.guider
    scale = float(
        getattr(guider, "guidance_scale", None)
        or getattr(getattr(guider, "config", None), "guidance_scale", None)
        or 1.7
    )

    G = len(frame_hiddens_list)
    hidden_stack = torch.cat(frame_hiddens_list, dim=0)  # [G, n, D] (equal n)
    num_frames = hidden_stack.shape[1]
    chunk_starts = [0] if num_frames <= _CHUNK_FRAMES else list(range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))

    previous_latent = None
    previous_condition = None
    latent_chunks = []
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + _CHUNK_FRAMES, num_frames)
        condition = components.condition_encoder(hidden_stack[:, chunk_start:chunk_end].to(device))
        condition = condition.to(transformer.dtype)  # [G, L, Dc]
        overlap = 0
        if previous_latent is not None:
            overlap = min(previous_latent.shape[-1], condition.shape[1])
            condition[:, :overlap] = previous_condition[:, :overlap]

        latents = torch.cat(
            [
                randn_tensor(
                    (1, components.num_channels_latents, condition.shape[1]),
                    generator=generators[g], device=device, dtype=condition.dtype,
                )
                for g in range(G)
            ],
            dim=0,
        )  # [G, C, L]
        noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None

        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        components.scheduler.set_timesteps(sigmas=sigmas, device=device)
        cond_pair = torch.cat([condition, torch.zeros_like(condition)], dim=0)  # [2G, L, Dc]

        for t in components.scheduler.timesteps:
            if overlap > 0:
                tv = t.to(latents.dtype)
                latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * tv) * noise_prompt + (
                    tv * previous_latent[..., :overlap]
                )
            timestep = t.expand(2 * G).to(latents.dtype)
            torch.compiler.cudagraph_mark_step_begin()
            noise_pred = fw(
                transformer,
                hidden_states=torch.cat([latents, latents], dim=0),
                timestep=timestep,
                encoder_hidden_states=cond_pair,
                return_dict=False,
            )[0].clone()
            velocity = noise_pred[G:] + (noise_pred[:G] - noise_pred[G:]) * scale
            latents = components.scheduler.step(velocity, t, latents, return_dict=False)[0]

        if overlap > 0:
            latents[..., :overlap] = previous_latent[..., :overlap]
        o_start = max(0, latents.shape[-1] - 2 * _OVERLAP_LATENT_LENGTH)
        o_end = max(o_start, latents.shape[-1] - _OVERLAP_LATENT_LENGTH)
        previous_latent = latents[..., o_start:o_end]
        previous_condition = condition[:, o_start:o_end]
        latent_chunks.append(latents)

    hop = components.latent_hop_length
    n_chunks = len(latent_chunks)
    waves: list[list[torch.Tensor]] = [[] for _ in range(G)]
    for i, latents in enumerate(latent_chunks):
        w = components.vocoder(latents.to(components.vocoder.dtype))  # [G, 2, S]
        left = 0 if i == 0 else _CROP_LEFT_LATENT * hop
        right = 0 if i == n_chunks - 1 else _CROP_RIGHT_LATENT * hop
        cropped = w[..., left : w.shape[-1] - right]
        for g in range(G):
            waves[g].append(cropped[g : g + 1])
    return [
        torch.cat(waves[g], dim=-1).float().clamp(-1.0, 1.0).cpu().numpy()
        for g in range(G)
    ]

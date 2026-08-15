# Portions of this file are derived from the diffusers MiniMax Music 3 modular
# pipeline (modular_pipelines/minimax_music3/, commit dafe3733):
# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Licensed under the Apache License, Version 2.0 — see LICENSES/Apache-2.0.txt.
# Modifications Copyright 2026 Xtraworks: rebuilt the autoregressive loop with
# a StaticCache and torch.compile CUDA graphs, sliced lm_head sampling,
# Gumbel-max fused sampling, and a batched-CFG denoise step.

"""Verified speed optimizations for the MiniMax Music 3 modular pipeline.

Measured on this machine (RTX 4090, 20s song, warm, seed-fixed):

    baseline                      50.5s   (AR 37.3, DiT 12.8)
    + compiled AR                 39.8s
    + batched-CFG DiT             31.0s   (AR ~22, DiT 8.8)   => 1.63x

Two independent changes:

1. Fast AR (`install_fast_ar`): the pipeline's autoregressive loop is
   CPU-dispatch bound (one 8B forward + seven 0.6B depth forwards per audio
   frame at 25 fps). We re-implement the semantic-generation block with a
   transformers StaticCache and torch.compile(mode="reduce-overhead") on the
   decode step (Qwen3Model + lm_head) and on the depth decoder — CUDA graphs
   replace per-token kernel-launch overhead. The math and the sampling order
   are identical to the original block.

2. Batched CFG (`install_batched_cfg`): the DiT step ran the conditional and
   unconditional branches as two sequential batch=1 transformer passes; one
   batch=2 pass computes the same guidance formula.

Quality note: compiled kernels change floating-point reduction order, so
logits differ from eager by ~ulps. In an autoregressive sampler this makes a
given seed take a different (equally valid) trajectory than the eager path —
comparable to using a different seed, not a degradation. The batched-CFG
change is formula-identical.

Both installs monkeypatch the pinned diffusers commit's block classes and
return undo callables. `install_all` wires everything and returns one undo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

log = logging.getLogger("music3.turbo")

# Persistent inductor cache: compiled artifacts survive server restarts, so
# only the first song after an install/driver change pays full compilation.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".inductor_cache"),
)


# v2 features (each independently disableable for bisection):
#   MUSIC3_TURBO_SLIM_HEAD=0  full-vocab lm_head instead of the sliced one
#   MUSIC3_TURBO_FUSED=0      eager depth-decoder chain instead of the fused graph
#   MUSIC3_TURBO_DIT=0        eager DiT forward instead of the compiled one
V2_SLIM_HEAD = os.environ.get("MUSIC3_TURBO_SLIM_HEAD", "1") != "0"
V2_FUSED = os.environ.get("MUSIC3_TURBO_FUSED", "1") != "0"
V2_DIT = os.environ.get("MUSIC3_TURBO_DIT", "1") != "0"
# A fully-fused per-frame "megastep" (sample+depth+embed+decode in one graph,
# EOS checks batched) was built and REMOVED: it measured no better than the
# split graphs below, because the AR stage sits on the memory-bandwidth floor
# (~40ms/frame of weight reads at realistic GEMV efficiency), not on Python.

# Pre-drawn Gumbel noise is allocated in blocks of this many frames.
_GUMBEL_BLOCK = 512

# CUDA graphs must be allowed to mutate the KV cache in-graph. transformers'
# own compiled-generate path marks StaticCache tensors as static addresses;
# building the cache manually skips that, and cudagraph trees then silently
# refuse to capture the decode graph (mutated non-static input).
try:  # pragma: no cover
    import torch._inductor.config as _inductor_config

    _inductor_config.triton.cudagraph_support_input_mutation = True
except Exception:
    pass


def _mark_cache_static(cache) -> None:
    tensors = []
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    tensors.append(t)
    for name in ("key_cache", "value_cache"):
        seq = getattr(cache, name, None)
        if isinstance(seq, (list, tuple)):
            tensors.extend(t for t in seq if isinstance(t, torch.Tensor))
    marked = 0
    for t in tensors:
        try:
            torch._dynamo.mark_static_address(t)
            marked += 1
        except Exception:
            pass
    log.info("static cache: marked %d tensors as static addresses", marked)


def _gumbel(shape, generator, device):
    """Standard Gumbel(0,1) noise. argmax(logits + G) samples the categorical
    softmax(logits) exactly (Gumbel-max trick) — the same distribution the
    reference's softmax+multinomial draws from, as pure tensor ops."""
    u = torch.rand(shape, generator=generator, device=device)
    neg_log_u = (-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20)  # positive
    return -torch.log(neg_log_u)


def install_fast_ar():
    from diffusers.modular_pipelines.minimax_music3 import encoders as enc
    from transformers import StaticCache

    # Sliced-head geometry. The EOS token (151670) and the semantic code range
    # [151675, 151675+16384) are the only rows that can ever be sampled — every
    # other row is masked to -inf by the reference code. Slicing the contiguous
    # row range [EOS, end-of-codes) therefore yields IDENTICAL logits for all
    # sampleable tokens while reading ~130MB instead of ~1.38GB per frame.
    if V2_SLIM_HEAD:
        _SLICE_START = enc._AUDIO_END_TOKEN_ID
        _SLICE_END = enc._AUDIO_CODE_OFFSET + enc._SEMANTIC_VOCAB_SIZE
    else:
        _SLICE_START = 0
        _SLICE_END = None  # resolved to vocab_size at call time

    def fast_call(self, components, state):
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        text_ids = block_state.text_ids
        max_frames = min(
            int(block_state.audio_duration * components.frame_rate), enc._MAX_AUDIO_FRAMES
        )
        if max_frames == 0:
            raise ValueError("audio_duration is shorter than one audio frame")
        generator = block_state.generator
        language_model = components.language_model
        rvq = components.rvq_depth_decoder
        device = text_ids.device

        # Same manual hook trigger as the original block: place both models
        # once, then run the loop hook-free on submodules.
        hooked = [
            m for m in (language_model, rvq) if getattr(m, "_hf_hook", None) is not None
        ]
        for m in hooked:
            m._hf_hook.pre_forward(m)

        vocab_size = int(language_model.config.vocab_size)
        slice_start = _SLICE_START
        slice_end = _SLICE_END if _SLICE_END is not None else vocab_size
        width = slice_end - slice_start
        eos_local = enc._AUDIO_END_TOKEN_ID - slice_start
        sem_local = enc._AUDIO_CODE_OFFSET - slice_start

        # Bucketed static KV cache: one compile per bucket size, reused across
        # songs. Bucketing avoids a recompile for every distinct duration.
        prompt_len = text_ids.shape[1]
        need = prompt_len + max_frames + 8
        bucket = ((need + 1023) // 1024) * 1024
        caches = getattr(self, "_turbo_caches", None)
        if caches is None:
            caches = self._turbo_caches = {}
        cache = caches.get(bucket)
        if cache is None:
            kw = dict(
                config=language_model.config,
                max_cache_len=bucket,
                device=device,
                dtype=language_model.dtype,
            )
            try:
                cache = StaticCache(max_batch_size=2, **kw)
            except TypeError:  # older transformers signature
                cache = StaticCache(batch_size=2, **kw)
            caches[bucket] = cache
            self._turbo_cache_unmarked = cache  # marked after prefill (lazy alloc)
        else:
            cache.reset()

        embed_tokens = language_model.model.embed_tokens
        head_weight = language_model.lm_head.weight
        head_bias = language_model.lm_head.bias

        def head(hidden):
            w = head_weight[slice_start:slice_end]
            b = head_bias[slice_start:slice_end] if head_bias is not None else None
            return torch.nn.functional.linear(hidden, w, b)

        text_embeds = embed_tokens(text_ids)
        cache_position = torch.arange(prompt_len, device=device)
        output = language_model.model(
            inputs_embeds=text_embeds,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        last_hidden = output.last_hidden_state[:, -1]

        # StaticCache allocates lazily; after prefill the tensors exist, so
        # mark them BEFORE the decode fns are traced — this is what licenses
        # CUDA graphs to capture the in-graph cache mutation at all.
        if getattr(self, "_turbo_cache_unmarked", None) is cache:
            _mark_cache_static(cache)
            self._turbo_cache_unmarked = None

        decode_fns = getattr(self, "_turbo_decode_fns", None)
        if decode_fns is None:
            decode_fns = self._turbo_decode_fns = {}
        decode = decode_fns.get(bucket)
        if decode is None:

            def _decode(feedback, position, _cache=cache):
                out = language_model.model(
                    inputs_embeds=feedback,
                    past_key_values=_cache,
                    use_cache=True,
                    cache_position=position,
                )
                lh = out.last_hidden_state[:, -1]
                return lh, head(lh)

            try:
                decode = torch.compile(_decode, mode="reduce-overhead", fullgraph=False)
            except Exception as exc:  # pragma: no cover
                log.warning("decode compile unavailable, eager: %s", exc)
                decode = _decode
            decode_fns[bucket] = decode

        # Disallowed = everything except EOS and the semantic-code range —
        # exactly the reference's vocab_mask, restricted to the slice.
        local_mask = torch.ones(width, dtype=torch.bool, device=device)
        local_mask[sem_local : sem_local + enc._SEMANTIC_VOCAB_SIZE] = False
        local_mask[eos_local] = False

        # Semantic sampling as one compiled graph. Mirrors the reference ops
        # (mask -> CFG -> CFG-top-k -> mask -> _sample_top_k) with the final
        # softmax+multinomial replaced by the equivalent Gumbel argmax.
        sample_fn = getattr(self, "_turbo_sample_fn", None)
        if sample_fn is None:

            def _sample(lm_logits, g):
                logits = lm_logits.float()
                logits = logits.masked_fill(local_mask, -float("inf"))
                conditional, unconditional = logits[0:1], logits[1:2]
                guided = unconditional + (conditional - unconditional) * enc._AR_CFG_SCALE
                threshold = torch.topk(conditional, enc._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
                guided = guided.masked_fill(conditional < threshold, -float("inf"))
                guided = guided.masked_fill(local_mask.unsqueeze(0), -float("inf"))
                vals = torch.nan_to_num(guided, nan=-1e9, posinf=1e9, neginf=-1e9)
                thr = torch.topk(vals, enc._AR_SAMPLING_TOP_K, dim=-1).values[..., -1, None]
                vals = vals.masked_fill(vals < thr, -float("inf"))
                return torch.argmax(vals + g.unsqueeze(0), dim=-1)

            if V2_FUSED:
                try:
                    sample_fn = torch.compile(_sample, mode="reduce-overhead", fullgraph=False)
                except Exception as exc:  # pragma: no cover
                    log.warning("sample compile unavailable, eager: %s", exc)
                    sample_fn = _sample
            else:
                sample_fn = _sample
            self._turbo_sample_fn = sample_fn

        # Depth-code chain as one compiled callable: seven 0.6B forwards plus
        # heads/embeddings/sampling with zero eager glue. Bypasses the per-call
        # component hook (placement already done above), calling the class
        # forward directly.
        num_codebooks = int(components.num_codebooks)
        audio_vocab = int(components.audio_vocab_size)
        depth_fn = getattr(self, "_turbo_depth_fn", None)
        if depth_fn is None:
            rvq_forward = type(rvq).forward
            projection = rvq.projection
            audio_heads = rvq.audio_heads
            audio_embeddings = rvq.audio_embeddings

            def _depth(last_hidden, sem_code2, g_depth):
                parts = [
                    projection(last_hidden).unsqueeze(1),
                    projection(embed_tokens(sem_code2 + enc._AUDIO_CODE_OFFSET)).unsqueeze(1),
                ]
                codes = [sem_code2]
                hidden_parts = []
                for index in range(1, num_codebooks):
                    hidden = rvq_forward(rvq, torch.cat(parts, dim=1))[:, -1]
                    hidden_parts.append(hidden[:1])
                    logits = audio_heads[index - 1](hidden)
                    conditional, unconditional = logits[:1].float(), logits[1:2].float()
                    guided = unconditional + (conditional - unconditional) * enc._AR_CFG_SCALE
                    vals = torch.nan_to_num(guided, nan=-1e9, posinf=1e9, neginf=-1e9)
                    thr = torch.topk(vals, enc._AR_SAMPLING_TOP_K, dim=-1).values[..., -1, None]
                    vals = vals.masked_fill(vals < thr, -float("inf"))
                    code = torch.argmax(vals + g_depth[index - 1].unsqueeze(0), dim=-1).repeat(2)
                    codes.append(code)
                    if index < num_codebooks - 1:
                        embed = audio_embeddings(code + (index - 1) * audio_vocab)
                        parts.append(projection(embed).unsqueeze(1))
                return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

            if V2_FUSED:
                try:
                    depth_fn = torch.compile(_depth, mode="reduce-overhead", fullgraph=False)
                except Exception as exc:  # pragma: no cover
                    log.warning("depth compile unavailable, eager: %s", exc)
                    depth_fn = _depth
            else:
                depth_fn = _depth
            self._turbo_depth_fn = depth_fn

        lm_logits = head(last_hidden)
        pos = prompt_len
        pos_t = torch.tensor([0], device=device)

        g_sem = g_dep = None
        frame_hiddens = []
        for frame_index in range(max_frames + 1):
            slot = frame_index % _GUMBEL_BLOCK
            if slot == 0:
                g_sem = _gumbel((_GUMBEL_BLOCK, width), generator, device)
                g_dep = _gumbel((_GUMBEL_BLOCK, num_codebooks - 1, audio_vocab), generator, device)

            torch.compiler.cudagraph_mark_step_begin()
            local = sample_fn(lm_logits, g_sem[slot]).clone()
            local_id = int(local.item())
            if local_id == eos_local:
                break

            sem_code2 = (local + (slice_start - enc._AUDIO_CODE_OFFSET)).repeat(2)
            torch.compiler.cudagraph_mark_step_begin()
            frame_codes, depth_hidden = depth_fn(last_hidden, sem_code2, g_dep[slot])
            frame_codes = frame_codes.clone()
            depth_hidden = depth_hidden.clone()
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = enc._embed_audio_frame(components, frame_codes)
            torch.compiler.cudagraph_mark_step_begin()
            pos_t.fill_(pos)
            last_hidden, lm_logits = decode(feedback, pos_t)
            last_hidden = last_hidden.clone()
            lm_logits = lm_logits.clone()
            pos += 1

        if not frame_hiddens:
            raise ValueError(
                "MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately"
            )
        block_state.frame_hiddens = torch.stack(frame_hiddens, dim=1)
        self.set_block_state(state, block_state)
        return components, state

    cls = enc.MiniMaxMusic3SemanticGenerationStep
    prev = cls.__call__
    cls.__call__ = fast_call

    def undo():
        cls.__call__ = prev

    return undo


def install_compiled_rvq(pipe):
    rvq = pipe.rvq_depth_decoder
    if getattr(rvq, "_turbo_compiled", False):
        return lambda: None
    orig_forward_fn = type(rvq).forward
    compiled = torch.compile(orig_forward_fn, mode="reduce-overhead", dynamic=False, fullgraph=False)

    def fw(self_, *a, **k):
        torch.compiler.cudagraph_mark_step_begin()
        return compiled(self_, *a, **k).clone()

    rvq.forward = fw.__get__(rvq)
    rvq._turbo_compiled = True

    def undo():
        if hasattr(rvq, "forward"):
            del rvq.forward  # fall back to the class attribute
        rvq._turbo_compiled = False

    return undo


def install_batched_cfg():
    from diffusers.modular_pipelines.minimax_music3 import denoise as dn

    compiled_fw = {}

    def fast_inner(self, components, block_state, k):
        latents = block_state.latents
        timesteps = block_state.timesteps
        overlap = block_state.overlap
        guider = components.guider
        scale = float(
            getattr(guider, "guidance_scale", None)
            or getattr(getattr(guider, "config", None), "guidance_scale", None)
            or 1.7
        )

        transformer = components.transformer
        if getattr(transformer, "_hf_hook", None) is not None:
            # Trigger placement once per chunk, then call the class forward
            # directly so the compiled graph is not re-wrapped per step.
            transformer._hf_hook.pre_forward(transformer)

        fw = type(transformer).forward
        if V2_DIT:
            fw = compiled_fw.get(type(transformer))
            if fw is None:
                try:
                    # Default mode (no CUDA graphs): DiT graphs capture
                    # seq-689 activations and their pools are the largest
                    # VRAM cost of compilation — kernel fusion keeps most of
                    # the win without pinning gigabytes.
                    fw = torch.compile(
                        type(transformer).forward, dynamic=False, fullgraph=False
                    )
                except Exception as exc:  # pragma: no cover
                    log.warning("DiT compile unavailable, eager: %s", exc)
                    fw = type(transformer).forward
                compiled_fw[type(transformer)] = fw

        cond = block_state.condition
        cond_pair = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        for i, t in enumerate(timesteps):
            if overlap > 0:
                time_value = t.to(latents.dtype)
                latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time_value) * block_state.noise_prompt + (
                    time_value * block_state.previous_latent[..., :overlap]
                )
            timestep = t.expand(2).to(latents.dtype)
            latent_pair = latents.expand(2, -1, -1)

            torch.compiler.cudagraph_mark_step_begin()
            noise_pred = fw(
                transformer,
                hidden_states=latent_pair,
                timestep=timestep,
                encoder_hidden_states=cond_pair,
                return_dict=False,
            )[0].clone()
            conditional, unconditional = noise_pred[0:1], noise_pred[1:2]
            velocity = unconditional + (conditional - unconditional) * scale
            latents = components.scheduler.step(velocity, t, latents, return_dict=False)[0]
            block_state.progress_bar.update()

        block_state.latents = latents
        return components, block_state

    cls = dn.MiniMaxMusic3ChunkDenoiseInner
    prev = cls.__call__
    cls.__call__ = fast_inner

    def undo():
        cls.__call__ = prev

    return undo


# FP8 weight-only was tried here and REMOVED: on Windows/torch 2.11 + torchao
# 0.18 it measured 2.1x SLOWER (fused dequant kernels never engage; every
# forward dequantizes to bf16 and breaks the CUDA-graph decode). See README.


def fold_vocoder_weight_norm(pipe):
    """Fold the vocoder's weight_norm parametrization into plain weights.

    The library wraps every vocoder conv in torch.nn.utils.weight_norm, which
    recomputes `g * v/||v||` on EVERY forward — a training-time construct.
    Folding computes the identical weight once. Mathematically exact.
    """
    from torch.nn.utils import remove_weight_norm

    folded = 0
    for module in pipe.vocoder.modules():
        try:
            remove_weight_norm(module)
            folded += 1
        except (ValueError, AttributeError, RuntimeError):
            pass
    log.info("vocoder: folded weight_norm on %d convs", folded)


def purge_runtime_state(pipe):
    """Free per-session KV caches and compiled-decode references after an OOM,
    so fallback retries start from clean memory instead of the poisoned state
    the failed attempt left behind (its caches otherwise stay referenced by
    the block instances and keep the GPU pinned near the cap)."""
    import gc

    blocks = getattr(getattr(pipe, "blocks", None), "sub_blocks", None)
    if blocks is not None:
        try:
            iterable = blocks.values()
        except AttributeError:
            iterable = list(blocks)
        for block in iterable:
            for attr in ("_turbo_caches", "_turbo_decode_fns", "_turbo_sample_fn",
                         "_turbo_depth_fn", "_turbo_cache_unmarked"):
                if hasattr(block, attr):
                    try:
                        delattr(block, attr)
                    except Exception:
                        pass
    if hasattr(pipe, "_ensemble_state"):
        try:
            del pipe._ensemble_state
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("turbo runtime state purged (caches/compiled decoders freed)")


def install_all(pipe):
    """Install every verified optimization; returns a single undo callable."""
    try:
        fold_vocoder_weight_norm(pipe)
    except Exception as exc:  # pragma: no cover
        log.warning("vocoder fold skipped: %s", exc)
    undos = [install_fast_ar(), install_batched_cfg()]
    if not V2_FUSED:
        # The fused depth chain subsumes the standalone rvq compile; only the
        # eager-chain fallback still benefits from it.
        undos.append(install_compiled_rvq(pipe))
    log.info(
        "turbo installed: compiled AR decode%s%s, %s depth chain, batched-CFG DiT%s",
        " + slim head" if V2_SLIM_HEAD else "",
        " + fused sampling" if V2_FUSED else "",
        "fused-compiled" if V2_FUSED else "eager",
        " (compiled)" if V2_DIT else "",
    )

    def undo_all():
        for u in reversed(undos):
            try:
                u()
            except Exception:  # pragma: no cover
                pass
        log.info("turbo reverted to reference implementation")

    return undo_all

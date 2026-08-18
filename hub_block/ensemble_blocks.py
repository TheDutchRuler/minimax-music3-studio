# Copyright 2026 Xtraworks.
# Portions derived from the diffusers MiniMax Music 3 modular pipeline
# (Copyright 2026 The MiniMax Team and The HuggingFace Team), licensed under
# the Apache License, Version 2.0. This file is likewise Apache-2.0.
"""Ensemble generation blocks for MiniMax Music 3: render K variations of one
prompt in a single batched pass.

Why: the autoregressive stage reads the full 8B language model plus seven
0.6B depth-decoder passes (~23GB of weights) for EVERY audio frame at 25fps —
it is memory-bandwidth bound at batch size 2 (one song's CFG pair). Decoding K
variations in lockstep (batch 2K rows) shares that weight read, so the
marginal variation is nearly free in the AR stage. Measured on an RTX 4090:
3 variations at 17.7s per 20s song vs ~31s solo (see
https://github.com/huggingface/diffusers/issues/14486).

Usage:
    from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentsManager
    import torch

    blocks = ModularPipelineBlocks.from_pretrained(
        "TheDutchRuler/minimax-music3-ensemble-blocks", trust_remote_code=True
    )
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = blocks.init_pipeline("MiniMaxAI/MiniMax-Music3", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)

    audios = pipe(
        prompt="Genre: acoustic pop. BPM: 96. Warm female vocals...",
        lyrics="[verse]\\n...\\n[chorus]\\n...",
        audio_duration=60.0,
        num_variations=3,
        seed=7,               # variation i uses seed + i; omit for random
        output="audios",
    )                          # -> list of (channels, samples) float32 arrays

Notes and honest caveats:
- Guidance is standard CFG (the checkpoint's recipe). For other guidance
  techniques use the default MiniMax Music 3 blocks with the guider.
- Each variation's math is row-independent and draws from its own seeded
  generator; quality is identical to solo generation. Batched kernels differ
  from solo kernels at the floating-point-ulp level, so a given seed can take
  a different (equally valid) trajectory than it would solo.
- VRAM: the AR stage needs the language model and depth decoder resident
  together plus a KV cache that grows with duration x variations. On 24GB,
  3 variations up to ~1 minute of audio is comfortable; reduce
  `num_variations` for longer songs.
- This block deliberately uses plain eager decoding (no torch.compile) so it
  runs on any diffusers install. Compiled variants live in
  https://github.com/TheDutchRuler/minimax-music3-studio.
"""

import numpy as np
import torch

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)
from diffusers.modular_pipelines import ModularPipelineBlocks, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from diffusers.modular_pipelines.minimax_music3 import encoders as _enc
from diffusers.modular_pipelines.minimax_music3.before_denoise import (
    _CHUNK_FRAMES,
    _CHUNK_HOP,
)
from diffusers.modular_pipelines.minimax_music3.decoders import (
    _CROP_LEFT_LATENT,
    _CROP_RIGHT_LATENT,
)
from diffusers.modular_pipelines.minimax_music3.denoise import _OVERLAP_LATENT_LENGTH
from diffusers.utils.torch_utils import randn_tensor


class MiniMaxMusic3EnsembleAutoregressiveStep(ModularPipelineBlocks):
    model_name = "minimax-music3"

    @property
    def description(self) -> str:
        return (
            "Batched autoregressive step: decodes `num_variations` takes of the same prompt in lockstep "
            "(batch 2K rows), sharing the per-frame weight reads that dominate autoregressive cost. Each "
            "variation samples from its own seeded generator with the reference sampling recipe."
        )

    @property
    def expected_components(self):
        return list(_enc.MiniMaxMusic3AutoregressiveStep().expected_components)

    @property
    def inputs(self):
        return [
            InputParam("text_ids", required=True, type_hint=torch.Tensor,
                       description="Tokenized conditional/unconditional prompt pair from the tokenize step."),
            InputParam("audio_duration", default=60.0, type_hint=float,
                       description="Upper bound in seconds per variation (capped at 9000 frames)."),
            InputParam("num_variations", default=3, type_hint=int,
                       description="How many takes to decode in one batched pass (1-8)."),
            InputParam("seed", default=None, type_hint=int,
                       description="Base seed; variation i uses seed + i. None draws random seeds."),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("frame_hiddens_list", type_hint=list,
                        description="Per-variation hidden-state tensors `[1, frames, D]` for the flow-matching stage."),
            OutputParam("variation_seeds", type_hint=list,
                        description="The seed actually used per variation."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        K = int(block_state.num_variations)
        if not 1 <= K <= 8:
            raise ValueError(f"`num_variations` must be in [1, 8], got {K}")
        max_frames = min(int(block_state.audio_duration * components.frame_rate), _enc._MAX_AUDIO_FRAMES)
        if max_frames == 0:
            raise ValueError("`audio_duration` is shorter than one audio frame")

        base = block_state.seed
        if base is None:
            seeds = [int(torch.randint(0, 2**31 - 1, (1,)).item()) for _ in range(K)]
        else:
            seeds = [int(base) + i for i in range(K)]

        language_model = components.language_model
        rvq = components.rvq_depth_decoder
        hooked = [m for m in (language_model, rvq) if getattr(m, "_hf_hook", None) is not None]
        for m in hooked:
            m._hf_hook.pre_forward(m)
        device = language_model.model.embed_tokens.weight.device

        text_ids = block_state.text_ids.to(device).repeat(K, 1)  # rows: s0c,s0u,s1c,s1u,...
        generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

        text_embeds = language_model.model.embed_tokens(text_ids)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]  # [2K, H]

        vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=device)
        vocab_mask[_enc._AUDIO_CODE_OFFSET:_enc._AUDIO_CODE_OFFSET + _enc._SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[_enc._AUDIO_END_TOKEN_ID] = False

        projection = rvq.projection
        audio_heads = rvq.audio_heads
        audio_embeddings = rvq.audio_embeddings
        embed_tokens = language_model.model.embed_tokens
        num_codebooks = int(rvq.config.num_codebooks)
        audio_vocab = int(rvq.config.audio_vocab_size)
        code_offsets = (torch.arange(num_codebooks - 1, device=device) * audio_vocab).unsqueeze(0)
        inv_sqrt = num_codebooks ** -0.5

        active = [True] * K
        frames = [[] for _ in range(K)]

        for frame_index in range(max_frames + 1):
            logits = language_model.lm_head(last_hidden).float()          # [2K, V]
            logits = logits.masked_fill(vocab_mask, -float("inf"))

            sem = torch.empty(K, dtype=torch.long, device=device)
            for s in range(K):
                cond, unc = logits[2 * s:2 * s + 1], logits[2 * s + 1:2 * s + 2]
                guided = unc + (cond - unc) * _enc._AR_CFG_SCALE
                thr = torch.topk(cond, _enc._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
                guided = guided.masked_fill(cond < thr, -float("inf"))
                guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
                sem[s] = _enc._sample_top_k(guided, generators[s])[0]

            for s in range(K):
                if active[s] and int(sem[s].item()) == _enc._AUDIO_END_TOKEN_ID:
                    active[s] = False
            if not any(active):
                break
            # Frozen variations keep decoding with a safe code; rows are ignored.
            safe = torch.where(sem == _enc._AUDIO_END_TOKEN_ID,
                               torch.full_like(sem, _enc._AUDIO_CODE_OFFSET), sem)
            sem_code2 = (safe - _enc._AUDIO_CODE_OFFSET).repeat_interleave(2)  # [2K]

            # Batched depth chain (reference math per row-pair, per-song sampling).
            parts = [projection(last_hidden).unsqueeze(1),
                     projection(embed_tokens(sem_code2 + _enc._AUDIO_CODE_OFFSET)).unsqueeze(1)]
            codes = [sem_code2]
            hidden_parts = []
            for index in range(1, num_codebooks):
                hidden = rvq(torch.cat(parts, dim=1))[:, -1]               # [2K, d]
                hidden_parts.append(hidden[0::2])
                logits_d = audio_heads[index - 1](hidden)
                code = torch.empty(K, dtype=torch.long, device=device)
                for s in range(K):
                    c, u = logits_d[2 * s:2 * s + 1].float(), logits_d[2 * s + 1:2 * s + 2].float()
                    g = u + (c - u) * _enc._AR_CFG_SCALE
                    code[s] = _enc._sample_top_k(g, generators[s])[0]
                code2 = code.repeat_interleave(2)
                codes.append(code2)
                if index < num_codebooks - 1:
                    emb = audio_embeddings(code2 + (index - 1) * audio_vocab)
                    parts.append(projection(emb).unsqueeze(1))
            frame_codes = torch.stack(codes, dim=1)                        # [2K, C]
            depth_hidden = torch.cat(hidden_parts, dim=-1)                 # [K, (C-1)d]

            if frame_index > 0:
                cond_hidden = last_hidden[0::2]
                for s in range(K):
                    if active[s] and len(frames[s]) < max_frames:
                        frames[s].append(torch.cat((cond_hidden[s:s + 1], depth_hidden[s:s + 1]), dim=-1))
                if all((not active[s]) or len(frames[s]) >= max_frames for s in range(K)):
                    break

            embeds = embed_tokens(frame_codes[:, :1] + _enc._AUDIO_CODE_OFFSET)
            extra = audio_embeddings(frame_codes[:, 1:] + code_offsets).sum(dim=1, keepdim=True)
            feedback = (embeds + extra.to(embeds.dtype)) * inv_sqrt
            output = language_model.model(inputs_embeds=feedback,
                                          past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        for s in range(K):
            if not frames[s]:
                raise ValueError(f"variation {s} generated zero audio frames")
        block_state.frame_hiddens_list = [torch.stack(f, dim=1) for f in frames]
        block_state.variation_seeds = seeds
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxMusic3EnsembleDenoiseDecodeStep(ModularPipelineBlocks):
    model_name = "minimax-music3"

    @property
    def description(self) -> str:
        return (
            "Flow-matches and vocodes every variation. Variations with identical frame counts are denoised "
            "together (batched CFG across variations, no padding); guidance is standard CFG with the "
            "checkpoint's guidance scale."
        )

    @property
    def expected_components(self):
        return [
            ComponentSpec("condition_encoder", MiniMaxMusic3ConditionEncoder),
            ComponentSpec("transformer", MiniMaxMusic3Transformer1DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            ComponentSpec("vocoder", MiniMaxMusic3Vocoder),
        ]

    @property
    def inputs(self):
        return [
            InputParam("frame_hiddens_list", required=True, type_hint=list),
            InputParam("variation_seeds", required=True, type_hint=list),
            InputParam("num_inference_steps", default=30, type_hint=int,
                       description="Flow-matching Euler steps per 200-frame chunk."),
            InputParam("output_type", default="np", type_hint=str),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("audios", type_hint=list,
                        description="One stereo waveform `(channels, samples)` per variation, in `[-1, 1]`."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        fh_list = block_state.frame_hiddens_list
        seeds = block_state.variation_seeds
        steps = int(block_state.num_inference_steps)

        for m in (components.condition_encoder, components.transformer, components.vocoder):
            if getattr(m, "_hf_hook", None) is not None:
                m._hf_hook.pre_forward(m)
        device = components._execution_device
        transformer = components.transformer
        scale = float(getattr(getattr(components, "guider", None), "guidance_scale", None) or 1.7)

        generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
        K = len(fh_list)
        results: list = [None] * K
        by_len: dict[int, list[int]] = {}
        for i, fh in enumerate(fh_list):
            by_len.setdefault(fh.shape[1], []).append(i)

        hop = components.latent_hop_length
        for _n, members in by_len.items():
            hidden = torch.cat([fh_list[i] for i in members], dim=0).to(device)  # [G, n, D]
            G = len(members)
            num_frames = hidden.shape[1]
            chunk_starts = [0] if num_frames <= _CHUNK_FRAMES else list(
                range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))

            previous_latent = previous_condition = None
            latent_chunks = []
            for chunk_start in chunk_starts:
                chunk_end = min(chunk_start + _CHUNK_FRAMES, num_frames)
                condition = components.condition_encoder(hidden[:, chunk_start:chunk_end])
                condition = condition.to(transformer.dtype)
                overlap = 0
                if previous_latent is not None:
                    overlap = min(previous_latent.shape[-1], condition.shape[1])
                    condition[:, :overlap] = previous_condition[:, :overlap]

                latents = torch.cat([
                    randn_tensor((1, components.num_channels_latents, condition.shape[1]),
                                 generator=generators[i], device=device, dtype=condition.dtype)
                    for i in members
                ], dim=0)
                noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None

                sigmas = np.linspace(1.0, 1.0 / steps, steps)
                components.scheduler.set_timesteps(sigmas=sigmas, device=device)
                cond_pair = torch.cat([condition, torch.zeros_like(condition)], dim=0)

                for t in components.scheduler.timesteps:
                    if overlap > 0:
                        tv = t.to(latents.dtype)
                        latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * tv) * noise_prompt + (
                            tv * previous_latent[..., :overlap])
                    timestep = t.expand(2 * G).to(latents.dtype)
                    noise_pred = transformer(
                        hidden_states=torch.cat([latents, latents], dim=0),
                        timestep=timestep, encoder_hidden_states=cond_pair, return_dict=False,
                    )[0]
                    velocity = noise_pred[G:] + (noise_pred[:G] - noise_pred[G:]) * scale
                    latents = components.scheduler.step(velocity, t, latents, return_dict=False)[0]

                if overlap > 0:
                    latents[..., :overlap] = previous_latent[..., :overlap]
                o_start = max(0, latents.shape[-1] - 2 * _OVERLAP_LATENT_LENGTH)
                o_end = max(o_start, latents.shape[-1] - _OVERLAP_LATENT_LENGTH)
                previous_latent = latents[..., o_start:o_end]
                previous_condition = condition[:, o_start:o_end]
                latent_chunks.append(latents)

            n_chunks = len(latent_chunks)
            waves = [[] for _ in range(G)]
            for ci, latents in enumerate(latent_chunks):
                w = components.vocoder(latents.to(components.vocoder.dtype))
                left = 0 if ci == 0 else _CROP_LEFT_LATENT * hop
                right = 0 if ci == n_chunks - 1 else _CROP_RIGHT_LATENT * hop
                cropped = w[..., left:w.shape[-1] - right]
                for g in range(G):
                    waves[g].append(cropped[g:g + 1])
            for g, i in enumerate(members):
                audio = torch.cat(waves[g], dim=-1).float().clamp(-1.0, 1.0)[0]
                results[i] = audio.cpu().numpy() if block_state.output_type == "np" else audio

        block_state.audios = results
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxMusic3EnsembleBlocks(SequentialPipelineBlocks):
    """Tokenize -> batched multi-variation AR -> grouped batched denoise+vocode."""

    block_classes = [
        _enc.MiniMaxMusic3TokenizeStep,
        MiniMaxMusic3EnsembleAutoregressiveStep,
        MiniMaxMusic3EnsembleDenoiseDecodeStep,
    ]
    block_names = ["tokenize", "ensemble_autoregressive", "ensemble_denoise_decode"]

    @property
    def description(self) -> str:
        return (
            "MiniMax Music 3 ensemble generation: K variations of one prompt in a single batched pass. "
            "The autoregressive stage is weight-read bound, so variations share its dominant cost — "
            "measured 3x20s variations in ~53s total vs ~31s per solo song on an RTX 4090."
        )

    @property
    def outputs(self):
        return [
            OutputParam("audios", type_hint=list,
                        description="One stereo waveform `(channels, samples)` per variation, in `[-1, 1]`."),
            OutputParam("variation_seeds", type_hint=list),
        ]

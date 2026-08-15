"""Turbo lab: measure and optimize MiniMax Music 3 stages on this machine.

Phases (one model load, several renders):
  1. Baseline warm render with per-block timings, recording the sampled
     semantic codes for a fixed seed.
  2. Fast AR: StaticCache + torch.compile'd decode step (LLM submodule +
     lm_head) and compiled RVQ depth decoder. Same seed; codes compared
     against baseline for agreement.
  3. Batched-CFG DiT: cond+uncond in one batch=2 transformer forward instead
     of two sequential batch=1 passes. Same seed; audio compared to baseline.

Key insight from reading the pipeline source: the AR loop calls
`language_model.model(...)` (a submodule), so compiling `pipe.language_model`
wraps a callable the loop never uses — compile must target the submodule call.
The ComponentsManager hooks fire once per song (triggered by hand in the block),
so the loop itself is hook-free and compile-friendly.

Run:  .venv\\Scripts\\python.exe turbo_lab.py [duration]
"""

import ctypes
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL = os.environ.get("MUSIC3_MODELS") or str(Path(__file__).parent / "models")
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
SEED = 7
OUT = Path(__file__).parent
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(OUT / ".inductor_cache"))

RESULTS: dict[str, dict] = {}


# ---------------------------------------------------------------- utilities

def vram(tag=""):
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    if tag:
        print(f"  [{tag}] VRAM {used:.1f}/{total/1024**3:.1f} GB", flush=True)
    return used


def watchdog():
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
        if st.ullAvailPhys / 1024**3 < 3.0:
            print("\n!! ABORT: <3GB RAM free", flush=True)
            os._exit(9)
        time.sleep(2)


# Per-block wall timers, attached by monkeypatching each block class __call__.
BLOCK_TIMES: dict[str, float] = {}


def install_block_timers():
    from diffusers.modular_pipelines.minimax_music3 import (
        before_denoise as bd,
        decoders as dec,
        denoise as dn,
        encoders as enc,
    )

    targets = {
        "text_encoder": enc.MiniMaxMusic3TextEncoderStep,
        "semantic_ar": enc.MiniMaxMusic3SemanticGenerationStep,
        "prepare_chunks": bd.MiniMaxMusic3PrepareChunksStep,
        "denoise_dit": dn.MiniMaxMusic3ChunkDenoiseStep,
        "vocoder": dec.MiniMaxMusic3VocoderDecodeStep,
    }
    for name, cls in targets.items():
        orig = cls.__call__

        def timed(self, components, state, _orig=orig, _name=name):
            torch.cuda.synchronize()
            t0 = time.time()
            out = _orig(self, components, state)
            torch.cuda.synchronize()
            BLOCK_TIMES[_name] = BLOCK_TIMES.get(_name, 0.0) + (time.time() - t0)
            return out

        cls.__call__ = timed


# Record sampled codes to prove the optimized path follows the same trajectory.
SAMPLED: list[int] = []


def install_sample_recorder():
    from diffusers.modular_pipelines.minimax_music3 import encoders as enc

    orig = enc._sample_top_k

    def recording(logits, generator):
        out = orig(logits, generator)
        SAMPLED.append(int(out.reshape(-1)[0].item()))
        return out

    enc._sample_top_k = recording
    return orig


def run_song(pipe, tag, steps=None):
    BLOCK_TIMES.clear()
    SAMPLED.clear()
    kwargs = dict(
        prompt=(
            "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. "
            "Vocals: soft female lead, close and breathy. Arrangement: "
            "fingerpicked guitar and soft piano."
        ),
        lyrics="[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe",
        audio_duration=DUR,
        generator=torch.Generator("cuda").manual_seed(SEED),
        output="audios",
    )
    if steps:
        kwargs["num_inference_steps"] = steps
    torch.cuda.synchronize()
    t0 = time.time()
    audio = pipe(**kwargs)[0]
    torch.cuda.synchronize()
    total = time.time() - t0

    arr = np.squeeze(np.asarray(audio, dtype=np.float32))
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    sf.write(str(OUT / f"lab_{tag}.wav"), arr, 44100, subtype="FLOAT")
    RESULTS[tag] = {
        "total": total,
        "blocks": dict(BLOCK_TIMES),
        "codes": list(SAMPLED),
        "audio": arr,
    }
    blocks = "  ".join(f"{k}={v:.1f}s" for k, v in BLOCK_TIMES.items())
    print(f"[{tag}] total={total:.1f}s  ({total/DUR:.2f}x RT)  {blocks}", flush=True)
    vram(tag)
    return arr


# ---------------------------------------------------------------- fast AR

def install_fast_ar():
    """Replace the semantic-generation block's __call__ with a StaticCache +
    compiled-decode implementation. Same math, same sampling order."""
    from diffusers.modular_pipelines.minimax_music3 import encoders as enc
    from transformers import StaticCache

    def fast_call(self, components, state):
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        text_ids = block_state.text_ids
        max_frames = min(int(block_state.audio_duration * components.frame_rate), enc._MAX_AUDIO_FRAMES)
        generator = block_state.generator
        language_model = components.language_model
        device = text_ids.device

        hooked = [
            m for m in (language_model, components.rvq_depth_decoder)
            if getattr(m, "_hf_hook", None) is not None
        ]
        for m in hooked:
            m._hf_hook.pre_forward(m)

        # Bucketed static cache: one compile per bucket, reused across songs.
        prompt_len = text_ids.shape[1]
        need = prompt_len + max_frames + 8
        bucket = ((need + 1023) // 1024) * 1024
        cache = getattr(self, "_static_caches", {}).get(bucket)
        if cache is None:
            if not hasattr(self, "_static_caches"):
                self._static_caches = {}
            kw = dict(config=language_model.config, max_cache_len=bucket,
                      device=device, dtype=language_model.dtype)
            try:
                cache = StaticCache(max_batch_size=2, **kw)
            except TypeError:  # older signature
                cache = StaticCache(batch_size=2, **kw)
            self._static_caches[bucket] = cache
        else:
            cache.reset()

        embed_tokens = language_model.model.embed_tokens
        text_embeds = embed_tokens(text_ids)
        cache_position = torch.arange(prompt_len, device=device)
        output = language_model.model(
            inputs_embeds=text_embeds, past_key_values=cache,
            use_cache=True, cache_position=cache_position,
        )
        last_hidden = output.last_hidden_state[:, -1]

        decode = getattr(self, "_decode_fn", None)
        if decode is None:
            def _decode(feedback, cache_position):
                out = language_model.model(
                    inputs_embeds=feedback, past_key_values=cache,
                    use_cache=True, cache_position=cache_position,
                )
                lh = out.last_hidden_state[:, -1]
                return lh, language_model.lm_head(lh)

            try:
                decode = torch.compile(_decode, mode="reduce-overhead", fullgraph=False)
            except Exception as exc:
                print(f"  (decode compile failed, eager: {exc})", flush=True)
                decode = _decode
            self._decode_fn = decode
        # NOTE: the closure binds `cache`; with bucket reuse the same cache
        # object persists, so recompiles happen only on a new bucket size.
        self._decode_cache_obj = cache

        vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=device)
        vocab_mask[enc._AUDIO_CODE_OFFSET : enc._AUDIO_CODE_OFFSET + enc._SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[enc._AUDIO_END_TOKEN_ID] = False

        lm_logits = language_model.lm_head(last_hidden)
        pos = prompt_len
        pos_t = torch.tensor([0], device=device)

        frame_hiddens = []
        for frame_index in range(max_frames + 1):
            logits = lm_logits.float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * enc._AR_CFG_SCALE
            threshold = torch.topk(conditional, enc._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = enc._sample_top_k(guided, generator)
            if int(sampled.item()) == enc._AUDIO_END_TOKEN_ID:
                break

            semantic_code = sampled - enc._AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = enc._generate_depth_codes(
                components, last_hidden, semantic_code.repeat(2), generator
            )
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
            raise ValueError("MiniMax Music 3 generated zero audio frames")
        block_state.frame_hiddens = torch.stack(frame_hiddens, dim=1)
        self.set_block_state(state, block_state)
        return components, state

    prev = enc.MiniMaxMusic3SemanticGenerationStep.__call__
    enc.MiniMaxMusic3SemanticGenerationStep.__call__ = fast_call
    return prev


def install_compiled_rvq(pipe):
    rvq = pipe.rvq_depth_decoder
    if getattr(rvq, "_lab_compiled", False):
        return
    orig_forward = type(rvq).forward
    compiled = torch.compile(orig_forward, mode="reduce-overhead", dynamic=False, fullgraph=False)

    def fw(self_, *a, **k):
        torch.compiler.cudagraph_mark_step_begin()
        out = compiled(self_, *a, **k)
        return out.clone()

    rvq.forward = fw.__get__(rvq)
    rvq._lab_compiled = True


# ---------------------------------------------------------------- batched CFG DiT

def install_batched_cfg():
    """Single batch=2 transformer forward per DiT step instead of two
    sequential batch=1 passes. Mathematically the same guidance formula."""
    from diffusers.modular_pipelines.minimax_music3 import denoise as dn

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

            noise_pred = components.transformer(
                hidden_states=latent_pair,
                timestep=timestep,
                encoder_hidden_states=cond_pair,
                return_dict=False,
            )[0]
            conditional, unconditional = noise_pred[0:1], noise_pred[1:2]
            velocity = unconditional + (conditional - unconditional) * scale
            latents = components.scheduler.step(velocity, t, latents, return_dict=False)[0]
            block_state.progress_bar.update()

        block_state.latents = latents
        return components, block_state

    prev = dn.MiniMaxMusic3ChunkDenoiseInner.__call__
    dn.MiniMaxMusic3ChunkDenoiseInner.__call__ = fast_inner
    return prev


# ---------------------------------------------------------------- main

def main():
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"duration={DUR}s seed={SEED} models={MODEL}", flush=True)
    print(f"torch {torch.__version__}", flush=True)

    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.time()
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(MODEL, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    print(f"LOAD {time.time()-t0:.1f}s  attn={pipe.language_model.config._attn_implementation}", flush=True)

    install_block_timers()
    install_sample_recorder()

    # Warm-up (pays one-off migration) then measured baseline.
    print("--- warm-up ---", flush=True)
    run_song(pipe, "warmup")
    print("--- baseline (warm) ---", flush=True)
    base = run_song(pipe, "baseline")

    # Fast AR.
    print("--- fast AR: first (compiles) ---", flush=True)
    prev_ar = install_fast_ar()
    install_compiled_rvq(pipe)
    try:
        run_song(pipe, "fast_ar_compile")
        print("--- fast AR: warm ---", flush=True)
        fast = run_song(pipe, "fast_ar")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"FAST AR FAILED: {type(exc).__name__}: {exc} — reverting", flush=True)
        from diffusers.modular_pipelines.minimax_music3 import encoders as enc
        enc.MiniMaxMusic3SemanticGenerationStep.__call__ = prev_ar
        fast = None

    # Code agreement.
    if fast is not None:
        a, b = RESULTS["baseline"]["codes"], RESULTS["fast_ar"]["codes"]
        n = min(len(a), len(b))
        agree = sum(1 for i in range(n) if a[i] == b[i])
        div = next((i for i in range(n) if a[i] != b[i]), n)
        print(f"CODE AGREEMENT: {agree}/{n} ({100*agree/max(n,1):.1f}%), first divergence at {div}", flush=True)

    # Batched CFG DiT (on top of whatever AR path is active).
    print("--- batched-CFG DiT ---", flush=True)
    prev_inner = install_batched_cfg()
    try:
        both = run_song(pipe, "fast_ar_plus_batched_dit")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"BATCHED DIT FAILED: {type(exc).__name__}: {exc} — reverting", flush=True)
        from diffusers.modular_pipelines.minimax_music3 import denoise as dn
        dn.MiniMaxMusic3ChunkDenoiseInner.__call__ = prev_inner
        both = None

    # Audio deltas vs baseline (same seed).
    ref = RESULTS["baseline"]["audio"]
    for tag in ("fast_ar", "fast_ar_plus_batched_dit"):
        if tag in RESULTS:
            x = RESULTS[tag]["audio"]
            n = min(len(ref), len(x))
            if n:
                d = ref[:n] - x[:n]
                rms_ref = float(np.sqrt((ref[:n] ** 2).mean()))
                rms_d = float(np.sqrt((d ** 2).mean()))
                print(f"AUDIO DELTA {tag}: rms(diff)/rms(ref) = {rms_d/max(rms_ref,1e-12):.4f} "
                      f"(len {len(x)} vs {len(ref)})", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for tag, r in RESULTS.items():
        blocks = "  ".join(f"{k}={v:.1f}" for k, v in r["blocks"].items())
        print(f"{tag:28s} {r['total']:7.1f}s   {blocks}", flush=True)


if __name__ == "__main__":
    main()

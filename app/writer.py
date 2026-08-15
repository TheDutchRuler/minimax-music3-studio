"""Songwriter: turn a short brief into a structured caption + tagged lyrics
using the ALREADY-LOADED Qwen3-8B language model.

The music checkpoint's Global LLM is initialized from Qwen3-8B and keeps the
full text vocabulary; a logits mask suppresses the music-token range (ids
>= 151650: audio markers, CFG token, and the 16k semantic codes) so generation
stays in plain text. Whether the music fine-tune preserved enough instruction-
following for good lyrics is an empirical question — the caller should treat
weak output as a signal to fall back to hand-written prompts.

Zero extra VRAM: the model is resident for rendering anyway.
"""

from __future__ import annotations

import logging
import re

import torch

log = logging.getLogger("music3.writer")

_TEXT_VOCAB_CUTOFF = 151650  # below: text + chat specials; at/above: music tokens

_SYSTEM = (
    "You are an expert songwriter and music producer writing input for a "
    "text-to-music model. Given a short brief, produce a complete song "
    "specification in EXACTLY this format:\n\n"
    "TITLE: <a short evocative song title>\n"
    "CAPTION: <one paragraph: genre and subgenre, BPM, key, emotional arc and "
    "production style. Then 'Vocals:' with gender, timbre, delivery, harmonies. "
    "Then 'Arrangement:' with instruments and how sections evolve.>\n"
    "LYRICS:\n"
    "<the full lyrics. Put each section tag on its own line, lowercase, from: "
    "[intro] [verse] [pre-chorus] [chorus] [bridge] [outro]. Write vivid, "
    "singable lines with a clear rhyme feel. 2 verses and 2 choruses minimum, "
    "and a bridge.>\n\n"
    "Output nothing else. No explanations."
)


class _TextOnly(torch.nn.Module):
    def __call__(self, input_ids, scores):
        scores[..., _TEXT_VOCAB_CUTOFF:] = -float("inf")
        return scores


@torch.no_grad()
def write_song(pipe, brief: str, instrumental: bool = False) -> dict:
    lm = pipe.language_model
    tokenizer = pipe.tokenizer
    if getattr(lm, "_hf_hook", None) is not None:
        lm._hf_hook.pre_forward(lm)
    device = lm.model.embed_tokens.weight.device

    ask = brief.strip()
    if instrumental:
        ask += "\n(This is an INSTRUMENTAL track: output 'LYRICS:' followed only by '[instrumental]'.)"
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": ask},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    out = lm.generate(
        **inputs,
        max_new_tokens=650,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.05,
        logits_processor=[_TextOnly()],
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log.info("writer raw output (%d chars)", len(raw))
    return _parse(raw, instrumental)


def _parse(raw: str, instrumental: bool) -> dict:
    title = ""
    caption = ""
    lyrics = ""

    m = re.search(r"TITLE:\s*(.+)", raw)
    if m:
        title = m.group(1).strip().strip('"').strip()[:60]
    m = re.search(r"CAPTION:\s*(.+?)(?=\nLYRICS:|\Z)", raw, re.DOTALL)
    if m:
        caption = " ".join(m.group(1).split())
    m = re.search(r"LYRICS:\s*\n?(.+)", raw, re.DOTALL)
    if m:
        lyrics = m.group(1).strip()
        # Keep only through the last plausible lyric content; strip trailing chatter.
        lyrics = re.split(r"\n(?:Note|Explanation|---)\b", lyrics)[0].strip()

    if instrumental:
        lyrics = "[instrumental]"
    if not caption:
        raise ValueError(f"writer produced no caption; raw output began: {raw[:200]!r}")
    if not instrumental and "[" not in lyrics:
        raise ValueError(f"writer produced no tagged lyrics; raw began: {raw[:200]!r}")
    return {"title": title or "Untitled", "caption": caption, "lyrics": lyrics, "raw": raw}

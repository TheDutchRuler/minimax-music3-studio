"""Songwriter: turn a short brief into a structured caption + tagged lyrics.

Uses a small companion instruct model (Qwen3-1.7B, ~3.4GB, downloaded on first
use). The obvious candidate — the music checkpoint's own 8B, which is resident
anyway — was tried first and CANNOT write text anymore: its embedding/output
layers were re-adapted to music tokens during the music fine-tune, and chat
prompting yields CJK gibberish. Documented so nobody retries it.

Device is chosen per call: GPU when it has room (music model not yet loaded —
seconds per song spec), CPU otherwise (a minute or two, still fine for a
button). The writer never touches the music pipeline, so briefs can be
expanded before the 4-minute model load has ever happened.
"""

from __future__ import annotations

import logging
import re
import threading

import torch

log = logging.getLogger("music3.writer")

WRITER_MODEL = "Qwen/Qwen3-1.7B"

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

_lock = threading.Lock()
_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading songwriter model %s (first use downloads ~3.4GB)", WRITER_MODEL)
    _tokenizer = AutoTokenizer.from_pretrained(WRITER_MODEL, token=False)
    _model = AutoModelForCausalLM.from_pretrained(
        WRITER_MODEL, torch_dtype=torch.bfloat16, token=False
    )
    _model.eval()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        if free / 1024**3 >= 6.0:
            return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def write_song(brief: str, instrumental: bool = False) -> dict:
    with _lock:
        _load()
        device = _pick_device()
        _model.to(device)
        if device.type == "cpu":
            _model.float()  # bf16 CPU inference is slow on many kernels
        log.info("songwriter running on %s", device)

        ask = brief.strip()
        if instrumental:
            ask += "\n(This is an INSTRUMENTAL track: output 'LYRICS:' followed only by '[instrumental]'.)"
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": ask},
        ]
        try:
            text = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # template without the thinking switch
            text = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = _tokenizer(text, return_tensors="pt").to(device)
        out = _model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=_tokenizer.eos_token_id,
        )
        raw = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Qwen3 may open with a reasoning block even when asked not to.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Free GPU room for renders; keep weights cached on CPU.
        if device.type == "cuda":
            _model.to("cpu")
            torch.cuda.empty_cache()
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
        lyrics = re.split(r"\n(?:Note|Explanation|---)\b", lyrics)[0].strip()

    if instrumental:
        lyrics = "[instrumental]"
    if not caption:
        raise ValueError(f"writer produced no caption; raw output began: {raw[:200]!r}")
    if not instrumental and "[" not in lyrics:
        raise ValueError(f"writer produced no tagged lyrics; raw began: {raw[:200]!r}")
    return {"title": title or "Untitled", "caption": caption, "lyrics": lyrics, "raw": raw}

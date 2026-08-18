"""Equivalence proof for the lm_head-slicing PR (diffusers #14486 follow-up).

Loads the real language model head weights and, for N random hidden states,
checks that the reference full-vocabulary path and the sliced-head path
produce (a) bitwise-identical logits on every sampleable token, and
(b) the identical sampled token sequence when driven by identical generators
through the exact reference sampling ops.

Loads ONLY the lm_head weight slice via safetensors (no 16GB model load).
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

MODELS = Path(__file__).parent.parent / "models_bf16" / "language_model"

_AUDIO_END = 151670
_OFFSET = 151675
_SEM_VOCAB = 16384
_CFG = 1.5
_TOPK_CFG = 50
_TOPK_SAMP = 50


def load_head(device):
    from safetensors import safe_open

    index_file = MODELS / "model.safetensors.index.json"
    if index_file.exists():
        shard = json.loads(index_file.read_text())["weight_map"]["lm_head.weight"]
    else:
        shard = "model.safetensors"
    with safe_open(str(MODELS / shard), framework="pt", device=device) as f:
        w = f.get_tensor("lm_head.weight")
    return w  # [vocab, hidden] bf16


def run(seed, w, device):
    """Reference vs sliced over 200 sampling steps on random hiddens."""
    hidden_dim = w.shape[1]
    vocab = w.shape[0]
    g_h = torch.Generator(device=device).manual_seed(999)
    mask_full = torch.ones(vocab, dtype=torch.bool, device=device)
    mask_full[_OFFSET:_OFFSET + _SEM_VOCAB] = False
    mask_full[_AUDIO_END] = False

    hs = _AUDIO_END
    he = _OFFSET + _SEM_VOCAB
    w_slim = w[hs:he]
    mask_slim = torch.ones(he - hs, dtype=torch.bool, device=device)
    mask_slim[_OFFSET - hs:_OFFSET - hs + _SEM_VOCAB] = False
    mask_slim[0] = False

    def sample(vals, gen):
        vals = torch.nan_to_num(vals.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
        thr = torch.topk(vals, _TOPK_SAMP, dim=-1).values[..., -1, None]
        vals = vals.masked_fill(vals < thr, -float("inf"))
        probs = torch.nan_to_num(F.softmax(vals, dim=-1), nan=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(probs, 1, generator=gen).squeeze(-1)

    g1 = torch.Generator(device=device).manual_seed(seed)
    g2 = torch.Generator(device=device).manual_seed(seed)
    max_logit_diff = 0.0
    mismatches = 0
    for step in range(200):
        h = torch.randn(2, hidden_dim, generator=g_h, device=device, dtype=torch.float32).to(w.dtype)

        # reference: full head
        lf = F.linear(h, w).float().masked_fill(mask_full, -float("inf"))
        c, u = lf[0:1], lf[1:2]
        gf = u + (c - u) * _CFG
        thr = torch.topk(c, _TOPK_CFG, dim=-1).values[..., -1, None]
        gf = gf.masked_fill(c < thr, -float("inf")).masked_fill(mask_full.unsqueeze(0), -float("inf"))
        s_ref = sample(gf, g1)

        # sliced head
        ls = F.linear(h, w_slim).float().masked_fill(mask_slim, -float("inf"))
        c2, u2 = ls[0:1], ls[1:2]
        gs = u2 + (c2 - u2) * _CFG
        thr2 = torch.topk(c2, _TOPK_CFG, dim=-1).values[..., -1, None]
        gs = gs.masked_fill(c2 < thr2, -float("inf")).masked_fill(mask_slim.unsqueeze(0), -float("inf"))
        s_slim = sample(gs, g2) + hs

        # logits on sampleable rows must be bitwise identical
        d = (lf[:, ~mask_full] - ls[:, ~mask_slim]).abs().max().item()
        max_logit_diff = max(max_logit_diff, d)
        if int(s_ref.item()) != int(s_slim.item()):
            mismatches += 1

    return max_logit_diff, mismatches


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = load_head(device)
    print(f"lm_head: {tuple(w.shape)} {w.dtype} on {device}", flush=True)
    for seed in (7, 8, 9):
        d, m = run(seed, w, device)
        print(f"seed {seed}: max |logit diff| on sampleable rows = {d}  sample mismatches = {m}/200", flush=True)
    print("=== SLIM HEAD EQUIVALENCE DONE ===", flush=True)


if __name__ == "__main__":
    main()

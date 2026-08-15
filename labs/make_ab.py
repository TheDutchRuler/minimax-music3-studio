"""Assemble the blind FP8 listening test.

Pairs bf16 ("AB Reference") and FP8 ("AB Candidate") renders of the same
prompt+seed, copies them to ab/pairN_A.wav / pairN_B.wav with the A/B
assignment randomized per pair, and seals the mapping in ab/key.json.
The mapping is not printed — judge first, then open the key.
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
LIB = ROOT / "library"
AB = ROOT / "ab"


def main():
    index = json.loads((LIB / "index.json").read_text(encoding="utf-8"))
    ref = {t["seed"]: t for t in index if t["title"] == "AB Reference"}
    cand = {t["seed"]: t for t in index if t["title"] == "AB Candidate"}
    seeds = sorted(set(ref) & set(cand))
    if not seeds:
        raise SystemExit("no matched pairs found")

    AB.mkdir(exist_ok=True)
    rng = random.SystemRandom()
    key = {}
    for i, seed in enumerate(seeds, 1):
        bf16_first = rng.random() < 0.5
        a, b = (ref[seed], cand[seed]) if bf16_first else (cand[seed], ref[seed])
        shutil.copy2(LIB / a["audio"], AB / f"pair{i}_A.wav")
        shutil.copy2(LIB / b["audio"], AB / f"pair{i}_B.wav")
        key[f"pair{i}"] = {"A": "bf16" if bf16_first else "fp8",
                           "B": "fp8" if bf16_first else "bf16", "seed": seed}
    (AB / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"prepared {len(seeds)} blind pairs in {AB}")
    print("listen to pairN_A.wav vs pairN_B.wav; the key stays sealed in ab/key.json")


if __name__ == "__main__":
    main()

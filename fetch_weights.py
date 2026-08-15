"""Download only the components the diffusers ModularPipeline actually loads.

Skips the SGLang-Omni-only artifacts (qwen_7B/, *.pth) which account for ~27GB
of the 53GB repo but are never referenced by modular_model_index.json.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "MiniMaxAI/MiniMax-Music3"
LOCAL_DIR = str(Path(__file__).resolve().parent / "models")

IGNORE = [
    "qwen_7B/*",  # SGLang path only (17.2GB)
    "*.pth",  # flowmatching_vae.pth + dav.pth, SGLang path only (9.9GB)
    "assets/*",  # sample audio (35MB)
    "figures/*",  # readme images
]

if __name__ == "__main__":
    path = snapshot_download(
        repo_id=REPO,
        local_dir=LOCAL_DIR,
        ignore_patterns=IGNORE,
        max_workers=8,
        # The repo is public; token=False forces anonymous access so a stale or
        # invalid token in the user's HF config can't turn this into a spurious
        # 401 "Repository Not Found".
        token=False,
    )
    print(f"DONE -> {path}")

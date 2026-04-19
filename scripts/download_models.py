"""Download all required HuggingFace models into the local models/ directory."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS = [
    ("BAAI/bge-m3",               MODELS_DIR / "bge-m3"),
    ("BAAI/bge-reranker-v2-m3",   MODELS_DIR / "bge-reranker-v2-m3"),
    ("minishlab/potion-base-32M",  MODELS_DIR / "potion-base-32M"),
]

for repo_id, local_dir in MODELS:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"  Done: {repo_id}\n")

print("All models downloaded.")

"""Reproducible development rebuild. Stops immediately if any stage fails."""
import subprocess
import sys

import _bootstrap  # noqa: F401
from app.artifacts import read_json
from app.config.settings import get_settings


if __name__ == "__main__":
    settings = get_settings()
    for script in ("prepare_multivent_dataset.py", "train_lda.py", "build_rag_index.py"):
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(settings.root / "scripts" / script)], cwd=settings.root, check=True)
    subprocess.run([sys.executable, "-m", "pytest"], cwd=settings.root, check=True)
    manifest = read_json(settings.corpus_dir / "manifest.json")
    print("Rebuild complete:", manifest["sizes"])
    print("Next: python scripts/evaluate.py; restart the local Uvicorn server.")

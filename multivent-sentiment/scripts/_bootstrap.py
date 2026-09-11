"""Allow documented `python scripts/name.py` commands on Windows."""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Keep newly downloaded caches inside this standalone workspace.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "models/embeddings/hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

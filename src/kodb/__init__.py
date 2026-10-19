import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"
# HF_HOME for every download (model weights, task sources, the xet chunk cache), so nothing lands outside the project.
# huggingface_hub reads it at import, and a kev.serve started by the kev-http adapter inherits it.
os.environ.setdefault("HF_HOME", str(WEIGHTS))

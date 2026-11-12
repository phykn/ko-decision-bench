import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"
# Keeps every Hugging Face download inside the project. Set here because huggingface_hub reads it at import;
# a kev.serve started by kodb inherits it.
os.environ.setdefault("HF_HOME", str(WEIGHTS))
# HF_HOME also moves the login token; keep reading the one `hf auth login` saved in the default place
os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))

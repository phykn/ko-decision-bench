from huggingface_hub import snapshot_download


class Adapter:
    def sync(self):
        """Wait for queued GPU work so the timing around decide() is real."""


class TorchAdapter(Adapter):
    runtime = "torch"

    def __init__(self, cfg):
        import torch
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.path = snapshot_download(cfg["repo"], revision=cfg["revision"],
                                      ignore_patterns=["assets/*", "figures/*", "onnx/*", "thinker/*", "mobile/*",
                                                       "*.gguf", "*.mp4", "*.gif", "*.png"])

    def sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device == "cuda":
            self.torch.cuda.synchronize()


def read_probs(q, answer):
    if q["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    if q["type"] == "choice":
        return [answer["probabilities"][k] for k in q["criteria"]]
    return list(answer["probabilities"].values())   # score levels, lowest first

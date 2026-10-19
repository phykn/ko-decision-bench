"""Model adapters. Each turns one typed question into probabilities over `options(q)`, in that order.

Question shapes (the Kev eval format):
  noul    {"instructions", "label": bool}                         options: ["false", "true"]
  choice  {"instructions", "criteria": {name: description}, "label": name}
  score   {"instructions", "criteria": [level, ...] low to high, "label": level index}
"""
import json
import sys

import httpx


def options(q):
    if q["type"] == "noul":
        return ["false", "true"]
    return list(q["criteria"])


def label_index(q):
    if q["type"] == "noul":
        return int(bool(q["label"]))
    if q["type"] == "choice":
        return list(q["criteria"]).index(q["label"])
    return int(q["label"])


class Adapter:
    """Load the model in __init__; decide() returns one probability per option of options(q)."""
    runtime = "?"
    device = "?"

    def __init__(self, cfg):
        self.cfg = cfg

    def decide(self, state, q):
        raise NotImplementedError

    def sync(self):
        """Wait for queued GPU work so the timing around decide() is real."""


class TorchAdapter(Adapter):
    runtime = "torch"

    def __init__(self, cfg):
        super().__init__(cfg)
        import torch
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.path = snapshot(cfg)

    def sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device == "cuda":
            self.torch.cuda.synchronize()


def snapshot(cfg):
    from huggingface_hub import snapshot_download
    return snapshot_download(cfg["repo"], revision=cfg["revision"],
                             ignore_patterns=["assets/*", "figures/*", "onnx/*", "*.gguf", "*.mp4", "*.gif", "*.png"])


class KevHttp(Adapter):
    """A running kev.serve (https://github.com/jaredpalmer/kev), through its System One API (POST /v1/systemone)."""
    runtime = "kev.serve (HTTP)"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.url = cfg.get("url", "http://127.0.0.1:8081")
        try:
            m = httpx.get(f"{self.url}/v1/models", timeout=5).json()["models"][0]
        except httpx.HTTPError as e:
            raise SystemExit(f"no kev.serve at {self.url} ({e}). Start one serving {cfg['repo']}@{cfg['revision']}")
        want = f"{cfg['repo']}@{cfg['revision']}"
        if not (m.get("run") or "").endswith(want):
            raise SystemExit(f"kev.serve at {self.url} serves {m.get('run')!r}, not {want!r}")
        self.device = f"{m.get('backend')} {m.get('device')} {m.get('dtype')}".strip()
        self.client = httpx.Client(timeout=300)

    def decide(self, state, q):
        spec = {k: v for k, v in q.items() if k != "label"}
        r = self.client.post(f"{self.url}/v1/systemone", json={"state": state, "model": "kev-latest", "questions": {"q": spec}})
        r.raise_for_status()
        a = r.json()["answers"]["q"]
        if q["type"] == "noul":
            return [1 - a["noul"], a["noul"]]
        if q["type"] == "choice":
            return [a["probabilities"][k] for k in q["criteria"]]
        by_level = {a["legend"][k]: p for k, p in a["probabilities"].items()}
        return [by_level[lvl] for lvl in q["criteria"]]


class Laya(TorchAdapter):
    """convaiinnovations/laya-* and fine-tunes of it, through the `laya` package."""

    def __init__(self, cfg):
        super().__init__(cfg)
        import laya
        self.agent = laya.load(self.path, device=self.device)

    def decide(self, state, q):
        spec = {k: v for k, v in q.items() if k != "label"}
        a = self.agent.predict(state, {"q": spec})["answers"]["q"]
        if q["type"] == "noul":
            return [1 - a["noul"], a["noul"]]
        return list(a["probabilities"].values()) if q["type"] == "score" else [a["probabilities"][k] for k in q["criteria"]]


class JevStyle(TorchAdapter):
    """chaoliangUNSW/Jev-Style-*: the PyTorch runtime shipped in the repo (jev_style_decision.py)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        sys.path.insert(0, self.path)
        from jev_style_decision import JevStyleDecision
        self.model = JevStyleDecision(self.path, device=self.device)

    def decide(self, state, q):
        if q["type"] == "noul":
            r = self.model.decide(state, q["instructions"], qtype="noul")
            p = r["probabilities"]
            return [p["false"], p["true"]]
        r = self.model.decide(state, q["instructions"], options=q["criteria"], qtype=q["type"])
        return list(r["probabilities"].values())


class Nari(TorchAdapter):
    """blackmallangcow/nari-decision-*: trust_remote_code model with a decide() method."""

    def __init__(self, cfg):
        super().__init__(cfg)
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True,
                                               dtype=self.torch.bfloat16).to(self.device).eval()

    def decide(self, state, q):
        if q["type"] == "noul":
            spec = {"type": "noul", "instructions": q["instructions"]}
        elif q["type"] == "choice":
            spec = {"type": "choice", "instructions": q["instructions"], "options": list(q["criteria"]),
                    "criteria": "; ".join(f"{k}: {v}" for k, v in q["criteria"].items())}
        else:
            spec = {"type": "score", "instructions": q["instructions"], "options": list(q["criteria"])}
        a = self.model.decide(self.tok, state=state, questions={"q": spec})["answers"]["q"]
        if q["type"] == "noul":
            return [1 - a["p_true"], a["p_true"]]
        return [a["probs"][k] for k in options(q)]


class CoCo(TorchAdapter):
    """corners-ai/CoCo-Decision-2.4-4B: the readout recipe from its model card (letter-token readout head)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch, dev = self.torch, self.device
        self.tok = AutoTokenizer.from_pretrained(self.path)
        model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16).to(dev).eval()
        self.table = json.load(open(f"{self.path}/decision_config.json"))["readout_token_ids"]
        self.readout = load_file(f"{self.path}/readout.safetensors")["proj.weight"].to(dev, torch.float32)
        self.inner = model.model
        st = {}
        self.inner.register_forward_pre_hook(lambda m, a, kw: st.update(mask=kw.get("attention_mask")), with_kwargs=True)

        # the card's recipe: full-attention layers see the whole prompt (bidirectional over non-padding tokens)
        def open_mask(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            pad = st["mask"]
            if pad is None:
                pad = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            kwargs["attention_mask"] = pad[:, None, None, :].bool()
            return args, kwargs

        for kind, layer in zip(self.inner.config.layer_types, self.inner.layers):
            if kind == "full_attention":
                layer.register_forward_pre_hook(open_mask, with_kwargs=True)

    def decide(self, state, q):
        if q["type"] == "noul":
            opts = {"no": "the statement is false", "yes": "the statement is true"}
        elif q["type"] == "choice":
            opts = q["criteria"]
        else:
            opts = {str(i): lvl for i, lvl in enumerate(q["criteria"])}
        letters = [chr(ord("A") + i) for i in range(len(opts))]
        lines = ["State:", state, "", f"Question: {q['instructions']}", "Options:"]
        lines += [f"{l}. {k}: {t}" for l, (k, t) in zip(letters, opts.items())]
        lines.append("Answer with the letter only.")
        prompt = self.tok.apply_chat_template([{"role": "user", "content": "\n".join(lines)}], tokenize=False,
                                              add_generation_prompt=True) + "Answer:"
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            hidden = self.inner(**ids, use_cache=False).last_hidden_state[0, -1].float()
        rows = [self.table.index(self.tok.encode(" " + l, add_special_tokens=False)[0]) for l in letters]
        return self.torch.softmax((self.readout @ hidden)[rows], dim=-1).tolist()


ADAPTERS = {"kev-http": KevHttp, "laya": Laya, "jev-style": JevStyle, "nari": Nari, "coco": CoCo}

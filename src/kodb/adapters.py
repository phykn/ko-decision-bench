import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from . import ROOT, WEIGHTS
from .tasks import options, unlabeled


class Adapter:
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
        from huggingface_hub import snapshot_download
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


KEV = "kev[serve] @ git+https://github.com/jaredpalmer/kev@6b719c3c3f367295f6ef336f4f751cf5ff970abc"   # the commit the results were measured with


def kev_info(url):
    try:
        return httpx.get(f"{url}/v1/models", timeout=5).json()["models"][0]
    except httpx.HTTPError:
        return None


def port_of(url):
    u = httpx.URL(url)
    if u.host not in ("127.0.0.1", "localhost"):
        raise SystemExit(f"no server at {url}, and kodb starts one only on this machine")
    return u.port or 80


def start_server(name, cmd, url, ready, timeout=900, **popen):
    """Start a decision server in the background, stop it on exit, and return ready() once it is truthy."""
    log = ROOT / "logs" / f"{name}-{port_of(url)}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"starting {name} on {url} (log: {log})", flush=True)
    with log.open("w") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, **popen)
    atexit.register(lambda: (p.terminate(), p.wait()))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise SystemExit(f"{name} exited with {p.returncode}; see {log}")
        if r := ready():
            return r
        time.sleep(2)
    raise SystemExit(f"{name} not ready after {timeout}s; see {log}")


def start_kev(run, url):
    # its own uv environment: Kev pins torch<2.9, which the models extra does not
    cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with", KEV,
           "python", "-m", "kev.serve", "--run", run, "--port", str(port_of(url))]
    return start_server("kev-serve", cmd, url, lambda: kev_info(url))


def health(url):
    try:
        r = httpx.get(f"{url}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def system_one_probs(q, a):
    if q["type"] == "noul":
        return [1 - a["noul"], a["noul"]]
    if q["type"] == "choice":
        return [a["probabilities"][k] for k in q["criteria"]]
    return list(a["probabilities"].values())   # score levels, lowest first


class SystemOneHttp(Adapter):
    def __init__(self, cfg, url):
        super().__init__(cfg)
        self.url = url
        self.client = httpx.Client(timeout=300)

    def decide(self, state, q):
        r = self.client.post(f"{self.url}/v1/systemone", json={"state": state, "questions": {"q": unlabeled(q)}})
        r.raise_for_status()
        return system_one_probs(q, r.json()["answers"]["q"])


STARTLUX = ("startlux-decide[transformers] @ git+https://github.com/StartLuxLabs/StartLux-Decision"
            "@a99977d9332decc9fb27a2488b4a83b533b6cd5f")


# the package's reference engine in its own uv environment: its bidirectional read-out requires transformers 5.8.x
class StartLux(SystemOneHttp):
    runtime = "startlux-decide transformers (HTTP)"

    def __init__(self, cfg):
        from huggingface_hub import snapshot_download
        super().__init__(cfg, cfg.get("url", "http://127.0.0.1:8090"))
        path = snapshot_download(cfg["repo"], revision=cfg["revision"])
        self.device = "mps" if sys.platform == "darwin" else "cuda"
        cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with", STARTLUX, "startlux-decide", "serve",
               "--model", path, "--backend", "transformers", "--device", self.device,
               "--port", str(port_of(self.url))]
        start_server("startlux-decide", cmd, self.url, lambda: health(self.url))
        self.device += " bfloat16"


H2O_MLX = ("https://github.com/internet-projects/h2o-lightning-mlx", "v1.4.2")


# the card's Apple Silicon route: the h2o-lightning-mlx facade (a vLLM stand-in) under H2O's official shim, two-way
class H2O(SystemOneHttp):
    runtime = "h2o shim + MLX facade (HTTP)"

    def __init__(self, cfg):
        super().__init__(cfg, cfg.get("url", "http://127.0.0.1:8741"))
        engine = WEIGHTS / "h2o-lightning-mlx"
        if not (engine / "model" / "h2o_lightning_shim.py").exists():
            if not engine.exists():
                subprocess.run(["git", "clone", "--depth", "1", "--branch", H2O_MLX[1], H2O_MLX[0], str(engine)],
                               check=True)
            # setup.sh downloads the official repo into engine/model at this revision and runs the shim's tests
            subprocess.run(["bash", "setup.sh"], cwd=engine, check=True,
                           env={**os.environ, "H2O_REVISION": cfg["revision"]})
        have = (engine / "model" / ".h2o_revision").read_text().strip()
        if have != cfg["revision"]:
            raise SystemExit(f"{engine}/model holds {have}, not {cfg['revision']}; remove it to download again")
        port = port_of(self.url)
        env = {**os.environ, "MODEL": "model", "PORT": str(port), "FACADE_PORT": str(port + 2)}
        start_server("h2o-lightning-mlx", ["bash", "serve.sh"], self.url, lambda: health(self.url),
                     cwd=engine, env=env)
        self.device = "mlx bfloat16"


class KevHttp(Adapter):
    runtime = "kev.serve (HTTP)"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.url = cfg.get("url", "http://127.0.0.1:8081")
        want = f"{cfg['repo']}@{cfg['revision']}"
        m = kev_info(self.url) or start_kev(want, self.url)
        if not (m.get("run") or "").endswith(want):
            raise SystemExit(f"kev.serve at {self.url} serves {m.get('run')!r}, not {want!r}")
        self.device = f"{m.get('backend')} {m.get('device')} {m.get('dtype')}".strip()
        self.client = httpx.Client(timeout=300)

    def decide(self, state, q):
        r = self.client.post(f"{self.url}/v1/systemone",
                             json={"state": state, "model": "kev-latest", "questions": {"q": unlabeled(q)}})
        r.raise_for_status()
        a = r.json()["answers"]["q"]
        if q["type"] == "noul":
            return [1 - a["noul"], a["noul"]]
        if q["type"] == "choice":
            return [a["probabilities"][k] for k in q["criteria"]]
        by_level = {a["legend"][k]: p for k, p in a["probabilities"].items()}
        return [by_level[lvl] for lvl in q["criteria"]]


class Laya(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        import laya
        self.agent = laya.load(self.path, device=self.device)

    def decide(self, state, q):
        a = self.agent.predict(state, {"q": unlabeled(q)})["answers"]["q"]
        if q["type"] == "noul":
            return [1 - a["noul"], a["noul"]]
        return list(a["probabilities"].values()) if q["type"] == "score" else [a["probabilities"][k] for k in q["criteria"]]


# the PyTorch runtime shipped in the model repo, so results follow the authors' code
class JevStyle(TorchAdapter):
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


# the readout recipe from the model card; the repo ships no inference code
class CoCo(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch, dev = self.torch, self.device
        self.tok = AutoTokenizer.from_pretrained(self.path)
        model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16).to(dev).eval()
        self.table = json.loads(Path(self.path, "decision_config.json").read_text())["readout_token_ids"]
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


class Kanana(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True,
                                               dtype=self.torch.bfloat16).to(self.device).eval()

    def decide(self, state, q):
        a = self.model.system_one(state=state, questions={"q": unlabeled(q)})["answers"]["q"]
        return system_one_probs(q, a)


# one question per request, the mode the card recommends for this model
class PplxDecider(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        sys.path.insert(0, f"{self.path}/source/src")
        from pplx_decider import Decider
        self.decider = Decider.from_pretrained(self.path, device=self.device)

    def decide(self, state, q):
        a = self.decider.decide({"state": state, "questions": {"q": unlabeled(q)}})["answers"]["q"]
        return system_one_probs(q, a)


# the transformers + peft recipe from the model card; the repo ships no inference code
class AutoTrust(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch, p = self.torch, Path(self.path)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        base = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16).to(self.device)
        self.model = PeftModel.from_pretrained(base, self.path, subfolder="adapter").merge_and_unload().eval()
        head = load_file(p / "head.safetensors")
        self.w, self.b = head["proj.weight"].to(self.device).float(), head["proj.bias"].to(self.device).float()
        self.slots = json.loads((p / "judge_config.json").read_text())["slots"]["ranges"]
        self.temp = json.loads((p / "calibration.json").read_text())["per_kind"]

    def decide(self, state, q):
        kind, question = q["type"], q["instructions"]
        if kind == "noul":
            lines = ["false", "true"]
        elif kind == "choice":
            lines = [f"{'ABCDEFGHIJKLMNOP'[i]}) {k}: {v}" for i, (k, v) in enumerate(q["criteria"].items())]
        else:
            # the head's score scale is fixed to the bare levels 0-5, so the level meanings go in the question
            lines = [str(i) for i in range(len(q["criteria"]))]
            question = f"{question} ({'; '.join(q['criteria'])})"
        text = f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            h = self.model.model(**ids).last_hidden_state[0, -1].float()
        z = (self.w @ h + self.b) / self.temp[kind]
        s = self.slots[kind][0]
        return self.torch.softmax(z[s:s + len(lines)], 0).tolist()


# the repo's thx02 runtime without a thinker server, so every answer is a direct read (the card's 63.83)
class Thx(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        sys.path.insert(0, self.path)
        from thx02 import THX02
        self.model = THX02.load(self.path, device=self.device)

    def decide(self, state, q):
        return system_one_probs(q, self.model.decide(state, {"q": unlabeled(q)})["answers"]["q"])


ADAPTERS = {"kev-http": KevHttp, "laya": Laya, "jev-style": JevStyle, "nari": Nari, "coco": CoCo,
            "kanana": Kanana, "pplx-decider": PplxDecider, "autotrust": AutoTrust, "thx02": Thx,
            "startlux": StartLux, "h2o": H2O}

import json
import sys
from pathlib import Path

from ..tasks import list_options, strip_label
from .base import TorchAdapter, read_probs


class Laya(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        import laya
        self.agent = laya.load(self.path, device=self.device)

    def decide(self, state, q):
        return read_probs(q, self.agent.predict(state, {"q": strip_label(q)})["answers"]["q"])


class Kanana(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True,
                                               dtype=self.torch.bfloat16).to(self.device).eval()

    def decide(self, state, q):
        return read_probs(q, self.model.system_one(state=state, questions={"q": strip_label(q)})["answers"]["q"])


# one question per request, the mode the card recommends for this model
class PplxDecider(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        sys.path.insert(0, f"{self.path}/source/src")
        from pplx_decider import Decider
        self.decider = Decider.from_pretrained(self.path, device=self.device)

    def decide(self, state, q):
        return read_probs(q, self.decider.decide({"state": state, "questions": {"q": strip_label(q)}})["answers"]["q"])


# the repo's thx02 runtime without a thinker server, so every answer is a direct read (the card's 63.83)
class Thx(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        sys.path.insert(0, self.path)
        from thx02 import THX02
        self.model = THX02.load(self.path, device=self.device)

    def decide(self, state, q):
        return read_probs(q, self.model.decide(state, {"q": strip_label(q)})["answers"]["q"])


# the PyTorch runtime shipped in the model repo, so results follow the authors' code
class JevStyle(TorchAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        sys.path.insert(0, self.path)
        from jev_style_decision import JevStyleDecision
        self.model = JevStyleDecision(self.path, device=self.device)

    def decide(self, state, q):
        if q["type"] == "noul":
            p = self.model.decide(state, q["instructions"], qtype="noul")["probabilities"]
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
        return [a["probs"][k] for k in list_options(q)]


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

import atexit
import os
import socket
import subprocess
import sys
import time

import httpx
from huggingface_hub import snapshot_download

from .. import ROOT, WEIGHTS
from ..tasks import strip_label
from .base import Adapter, read_probs


def get_local_port(url):
    u = httpx.URL(url)
    if u.host not in ("127.0.0.1", "localhost"):
        raise SystemExit(f"no server at {url}, and kodb starts one only on this machine")
    return u.port or 80


def start_server(name, cmd, url, ready, timeout=900, **popen):
    port = get_local_port(url)
    # a server already on the port would answer ready() and be measured in place of this model
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"{url} is already in use; stop that server or pass another --url")
    log = ROOT / "logs" / f"{name}-{port}.log"
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


def get_health(url):
    try:
        r = httpx.get(f"{url}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


class SystemOneHttp(Adapter):
    request = {}

    def __init__(self, url):
        self.url = url
        self.client = httpx.Client(timeout=300)

    def ask(self, state, q):
        r = self.client.post(f"{self.url}/v1/systemone",
                             json={"state": state, **self.request, "questions": {"q": strip_label(q)}})
        r.raise_for_status()
        return r.json()["answers"]["q"]

    def decide(self, state, q):
        return read_probs(q, self.ask(state, q))


def get_kev_model(url):
    try:
        return httpx.get(f"{url}/v1/models", timeout=5).json()["models"][0]
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None


class KevHttp(SystemOneHttp):
    runtime = "kev.serve (HTTP)"
    request = {"model": "kev-latest"}

    def __init__(self, cfg):
        super().__init__(cfg.get("url", "http://127.0.0.1:8081"))
        want = f"{cfg['repo']}@{cfg['revision']}"
        m = get_kev_model(self.url)
        if m is None:
            # its own uv environment: Kev pins torch<2.9, which the models extra does not; the commit the results used
            cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with",
                   "kev[serve] @ git+https://github.com/jaredpalmer/kev@6b719c3c3f367295f6ef336f4f751cf5ff970abc",
                   "python", "-m", "kev.serve", "--run", want, "--port", str(get_local_port(self.url))]
            m = start_server("kev-serve", cmd, self.url, lambda: get_kev_model(self.url))
        if not (m.get("run") or "").endswith(want):
            raise SystemExit(f"kev.serve at {self.url} serves {m.get('run')!r}, not {want!r}")
        self.device = f"{m.get('backend')} {m.get('device')} {m.get('dtype')}".strip()

    def decide(self, state, q):
        a = self.ask(state, q)
        if q["type"] != "score":
            return read_probs(q, a)
        by_level = {a["legend"][k]: p for k, p in a["probabilities"].items()}
        return [by_level[lvl] for lvl in q["criteria"]]


# the package's reference engine in its own uv environment: its bidirectional read-out requires transformers 5.8.x
class StartLux(SystemOneHttp):
    runtime = "startlux-decide transformers (HTTP)"

    def __init__(self, cfg):
        super().__init__(cfg.get("url", "http://127.0.0.1:8090"))
        path = snapshot_download(cfg["repo"], revision=cfg["revision"])
        device = "mps" if sys.platform == "darwin" else "cuda"
        cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with",
               "startlux-decide[transformers] @ git+https://github.com/StartLuxLabs/StartLux-Decision"
               "@a99977d9332decc9fb27a2488b4a83b533b6cd5f",
               "startlux-decide", "serve", "--model", path, "--backend", "transformers", "--device", device,
               "--port", str(get_local_port(self.url))]
        start_server("startlux-decide", cmd, self.url, lambda: get_health(self.url))
        self.device = f"{device} bfloat16"


# the card's Apple Silicon route: the h2o-lightning-mlx facade (a vLLM stand-in) under H2O's official shim, two-way
class H2O(SystemOneHttp):
    runtime = "h2o shim + MLX facade (HTTP)"
    device = "mlx bfloat16"

    def __init__(self, cfg):
        super().__init__(cfg.get("url", "http://127.0.0.1:8741"))
        engine = WEIGHTS / "h2o-lightning-mlx"
        if not (engine / "model" / "h2o_lightning_shim.py").exists():
            if not engine.exists():
                subprocess.run(["git", "clone", "--depth", "1", "--branch", "v1.4.2",
                                "https://github.com/internet-projects/h2o-lightning-mlx", str(engine)], check=True)
            # setup.sh downloads the official repo into engine/model at this revision and runs the shim's tests
            subprocess.run(["bash", "setup.sh"], cwd=engine, check=True,
                           env={**os.environ, "H2O_REVISION": cfg["revision"]})
        have = (engine / "model" / ".h2o_revision").read_text().strip()
        if have != cfg["revision"]:
            raise SystemExit(f"{engine}/model holds {have}, not {cfg['revision']}; remove it to download again")
        port = get_local_port(self.url)
        env = {**os.environ, "MODEL": "model", "PORT": str(port), "FACADE_PORT": str(port + 2)}
        start_server("h2o-lightning-mlx", ["bash", "serve.sh"], self.url, lambda: get_health(self.url),
                     cwd=engine, env=env)

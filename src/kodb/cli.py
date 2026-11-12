import argparse
import tomllib
from pathlib import Path

from . import tasks as T
from .run import run_model
from .score import print_tables


def load_models(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))["models"]


def main():
    ap = argparse.ArgumentParser(prog="kodb")
    ap.add_argument("--data", default="data", help="folder with the built <task>.jsonl files")
    ap.add_argument("--results", default="results", help="folder for <model>/<task>.json")
    ap.add_argument("--models", default="models.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download the sources and write data/<task>.jsonl")
    b.add_argument("tasks", nargs="*", help=f"default: all ({', '.join(T.TASKS)})")
    b.add_argument("--n", type=int, default=T.N)
    b.add_argument("--seed", type=int, default=T.SEED)
    r = sub.add_parser("run", help="run one model on tasks and write results/<model>/<task>.json")
    r.add_argument("model")
    r.add_argument("tasks", nargs="+", help="task names, or 'all'")
    r.add_argument("--url", help="HTTP adapters: server address (defaults: kev-http :8081, startlux :8090, h2o :8741)")
    s = sub.add_parser("score", help="print the README tables")
    s.add_argument("tasks", nargs="*", help="default: every task with results")
    s.add_argument("--lang", choices=("en", "ko"), default="en", help="language of the table headers")
    sub.add_parser("list", help="list tasks and models")
    a = ap.parse_args()

    if a.cmd == "build":
        T.build(a.tasks or list(T.TASKS), a.data, a.n, a.seed)
    elif a.cmd == "run":
        models = load_models(a.models)
        if a.model not in models:
            raise SystemExit(f"unknown model {a.model!r}; see {a.models}")
        names = list(T.TASKS) if a.tasks == ["all"] else a.tasks
        unknown = [t for t in names if t not in T.TASKS]
        if unknown:
            raise SystemExit(f"unknown task(s) {unknown}; known: {list(T.TASKS)}")
        run_model(a.model, names, dict(models[a.model], **({"url": a.url} if a.url else {})), a.data, a.results)
    elif a.cmd == "score":
        print_tables(a.results, load_models(a.models), a.tasks or None, a.lang)
    else:
        for t, d in T.TASKS.items():
            print(f"task  {t:<14} {d['type']:<7} {d['repo']} ({d['split']}), {d['license']}")
        for m, d in load_models(a.models).items():
            print(f"model {m:<27} {d['adapter']:<13} {d['repo']}@{d['revision'][:12]}")

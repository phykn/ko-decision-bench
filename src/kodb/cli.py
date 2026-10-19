import argparse
import hashlib
import json
import time
import tomllib
from datetime import date
from pathlib import Path

from . import tasks as T
from .adapters import ADAPTERS
from .score import brier, report


def load_models(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))["models"]


def run(model, task_names, cfg, data_dir, results_dir):
    t0 = time.time()
    adapter = ADAPTERS[cfg["adapter"]](cfg)
    load_s = time.time() - t0
    for task in task_names:
        recs = T.load(task, data_dir)
        first = next(iter(recs[0]["questions"].values()))
        adapter.decide(recs[0]["state"], first)   # warm-up, not timed
        rows = []
        for rec in recs:
            for qid, q in rec["questions"].items():
                t1 = time.perf_counter()
                p = adapter.decide(rec["state"], q)
                adapter.sync()
                ms = (time.perf_counter() - t1) * 1000
                total = sum(p)
                p = [x / total for x in p]
                y = T.label_index(q)
                top = max(range(len(p)), key=p.__getitem__)
                rows.append({"id": rec["id"], "question": qid, "type": q["type"], "label": T.options(q)[y],
                             "answer": T.options(q)[top], "probs": [round(x, 5) for x in p], "correct": top == y,
                             "brier": round(brier(p, y), 5), "ms": round(ms, 1)})
        data = (Path(data_dir) / f"{task}.jsonl").read_bytes()
        acc = sum(r["correct"] for r in rows) / len(rows)
        out = Path(results_dir) / model / f"{task}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "model": model, "repo": cfg["repo"], "revision": cfg["revision"], "task": task,
            "data_sha256": hashlib.sha256(data).hexdigest(), "date": date.today().isoformat(),
            "runtime": adapter.runtime, "device": adapter.device, "load_s": round(load_s, 1),
            "summary": {"n": len(rows), "acc": acc, "ms": sum(r["ms"] for r in rows) / len(rows)},
            "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{model} {task}: acc {acc:.3f} on {len(rows)} -> {out}", flush=True)


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
    r.add_argument("--url", help="kev-http adapter: kev.serve address (default http://127.0.0.1:8081)")
    s = sub.add_parser("score", help="print the comparison tables")
    s.add_argument("tasks", nargs="*", help="default: every task with results")
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
        cfg = dict(models[a.model], **({"url": a.url} if a.url else {}))
        run(a.model, names, cfg, a.data, a.results)
    elif a.cmd == "score":
        report(a.results, load_models(a.models), a.tasks or None)
    else:
        for t, d in T.TASKS.items():
            print(f"task  {t:<14} {d['type']:<7} {d['repo']} ({d['split']}), {d['license']}")
        for m, d in load_models(a.models).items():
            print(f"model {m:<18} {d['adapter']:<10} {d['repo']}@{d['revision'][:12]}")


if __name__ == "__main__":
    main()

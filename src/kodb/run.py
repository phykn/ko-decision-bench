import hashlib
import json
import time
from datetime import date
from pathlib import Path

from . import tasks as T
from .adapters import ADAPTERS
from .score import compute_brier


def run_model(model, task_names, cfg, data_dir, results_dir):
    t0 = time.time()
    adapter = ADAPTERS[cfg["adapter"]](cfg)
    load_s = time.time() - t0
    for task in task_names:
        recs = T.load(task, data_dir)
        adapter.decide(recs[0]["state"], next(iter(recs[0]["questions"].values())))   # warm-up, not timed
        rows = []
        for rec in recs:
            for qid, q in rec["questions"].items():
                t1 = time.perf_counter()
                p = adapter.decide(rec["state"], q)
                adapter.sync()
                ms = (time.perf_counter() - t1) * 1000
                total = sum(p)
                p = [x / total for x in p]
                y = T.find_label(q)
                top = max(range(len(p)), key=p.__getitem__)
                opts = T.list_options(q)
                rows.append({"id": rec["id"], "question": qid, "type": q["type"], "label": opts[y],
                             "answer": opts[top], "probs": [round(x, 5) for x in p], "correct": top == y,
                             "brier": round(compute_brier(p, y), 5), "ms": round(ms, 1)})
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

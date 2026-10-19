"""Uniform metrics over results/<model>/<task>.json, whatever runtime produced them."""
import json
import math
from pathlib import Path

MARK = {"split": "†", "related": "‡", "unknown": "?"}


def ece(tops, correct, bins=10):
    total, n = 0.0, len(tops)
    for b in range(bins):
        idx = [i for i, t in enumerate(tops) if b / bins < t <= (b + 1) / bins or (b == 0 and t == 0)]
        if idx:
            gap = sum(tops[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(gap)
    return total


def metrics(rows):
    n = len(rows)
    tops = [max(r["probs"]) for r in rows]
    corr = [bool(r["correct"]) for r in rows]
    acc = sum(corr) / n
    return {"n": n, "acc": acc, "ci": 1.96 * math.sqrt(acc * (1 - acc) / n),
            "brier": sum(r["brier"] for r in rows) / n, "ece": ece(tops, corr),
            "wrong90": sum(1 for t, c in zip(tops, corr) if t >= 0.9 and not c) / n,
            "ms": sum(r["ms"] for r in rows) / n}


def load_results(results_dir):
    out = {}
    for f in sorted(Path(results_dir).glob("*/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.setdefault(d["model"], {})[d["task"]] = d
    return out


def report(results_dir, models_cfg, tasks=None):
    res = load_results(results_dir)
    if not res:
        raise SystemExit(f"no results in {results_dir}. Run: kodb run <model> <task>")
    tasks = tasks or sorted({t for m in res.values() for t in m})
    name_w = max(len(m) for m in res) + 2
    col_w = max(max(len(t) for t in tasks), 6) + 1

    print("Accuracy per task.  † trained on another split of the source  ‡ related Korean data  ? card unclear")
    print(f"{'model':<{name_w}}" + "".join(f"{t:>{col_w}}" for t in tasks) + f"{'mean':>7}{'clean':>7}")
    order = sorted(res, key=lambda m: -sum(res[m][t]["summary"]["acc"] for t in tasks if t in res[m]))
    for m in order:
        trained = models_cfg.get(m, {}).get("trained_on", {})
        cells, accs, clean = [], [], []
        for t in tasks:
            if t not in res[m]:
                cells.append(f"{'-':>{col_w}}")
                continue
            a = res[m][t]["summary"]["acc"]
            accs.append(a)
            if t not in trained:
                clean.append(a)
            cells.append(f"{a:.3f}{MARK.get(trained.get(t), ' ')}".rjust(col_w))
        mean = f"{sum(accs) / len(accs):.3f}" if len(accs) == len(tasks) else "-"
        cmean = f"{sum(clean) / len(clean):.3f}" if clean and len(accs) == len(tasks) else "-"
        print(f"{m:<{name_w}}" + "".join(cells) + f"{mean:>7}{cmean:>7}")
    print("mean: over all tasks. clean: over tasks with no mark for that model (a different set per model).")

    print()
    print("All questions pooled, per model (top probability for ECE and the >=0.9-wrong share)")
    print(f"{'model':<{name_w}}{'n':>6}{'acc':>7}{'±95%':>6}{'brier':>7}{'ECE':>7}{'p>=.9 wrong':>12}{'ms/q':>7}  runtime")
    for m in order:
        rows = [r for t in tasks if t in res[m] for r in res[m][t]["rows"]]
        s = metrics(rows)
        rt = next(iter(res[m].values()))
        print(f"{m:<{name_w}}{s['n']:>6}{s['acc']:>7.3f}{s['ci']:>6.3f}{s['brier']:>7.3f}{s['ece']:>7.3f}"
              f"{s['wrong90']:>12.1%}{s['ms']:>7.0f}  {rt['runtime']} [{rt['device']}]")

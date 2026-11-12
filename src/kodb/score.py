import json
import math
from pathlib import Path

from .tasks import TASKS

MARK = {"split": "†", "related": "‡", "unknown": "?"}
TEXT = {
    "en": {"acc": "**Accuracy per task**", "model": "Model", "mean": "Mean",
           "legend": "`†` `‡` `?` mark possible training-data overlap ([Method](#method)). "
                     "**Clean** is the mean over unmarked tasks, a different set per model.",
           "cal": "**Calibration, all questions pooled**",
           "cal_cols": ["Accuracy ±95%", "Brier", "ECE", "p ≥ .9 wrong", "ms/q", "Runtime"]},
    "ko": {"acc": "**과제별 정확도**", "model": "모델", "mean": "평균",
           "legend": "`†` `‡` `?`는 학습 데이터가 겹칠 수 있음을 나타낸다([방법](#방법)). "
                     "**Clean**은 표시가 없는 과제만의 평균이다. 모델마다 들어가는 과제가 다르다.",
           "cal": "**보정(calibration), 전체 문항 합산**",
           "cal_cols": ["정확도 ±95%", "Brier", "ECE", "p ≥ .9 오답", "ms/문항", "런타임"]},
}


def compute_brier(probs, y):
    return sum((p - (k == y)) ** 2 for k, p in enumerate(probs))


def compute_ece(tops, correct, bins=10):
    total, n = 0.0, len(tops)
    for b in range(bins):
        idx = [i for i, t in enumerate(tops) if b / bins < t <= (b + 1) / bins or (b == 0 and t == 0)]
        if idx:
            gap = sum(tops[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(gap)
    return total


def compute_metrics(rows):
    n = len(rows)
    tops = [max(r["probs"]) for r in rows]
    corr = [bool(r["correct"]) for r in rows]
    acc = sum(corr) / n
    return {"acc": acc, "ci": 1.96 * math.sqrt(acc * (1 - acc) / n),
            "brier": sum(r["brier"] for r in rows) / n, "ece": compute_ece(tops, corr),
            "wrong90": sum(1 for t, c in zip(tops, corr) if t >= 0.9 and not c) / n,
            "ms": sum(r["ms"] for r in rows) / n}


def load_results(results_dir):
    out = {}
    for f in sorted(Path(results_dir).glob("*/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out.setdefault(d["model"], {})[d["task"]] = d
    return out


def summarize(res, trained, tasks):
    accs = {t: res[t]["summary"]["acc"] for t in tasks if t in res}
    clean = [a for t, a in accs.items() if t not in trained]
    full = len(accs) == len(tasks)
    return {"accs": accs, "mean": sum(accs.values()) / len(accs) if full else None,
            "clean": sum(clean) / len(clean) if clean and full else None,
            # ranked by the clean mean, so in-domain training does not buy a higher place; no clean task goes last
            "rank": (-(sum(clean) / len(clean)) if clean else 1, -sum(accs.values()) / len(accs))}


def format_best(x, best):
    if x is None:
        return "–"
    return f"**{x:.3f}**" if x == best else f"{x:.3f}"


def print_tables(results_dir, models_cfg, tasks, lang):
    res = load_results(results_dir)
    if not res:
        raise SystemExit(f"no results in {results_dir}. Run: kodb run <model> <task>")
    tasks = tasks or sorted({t for m in res.values() for t in m})
    trained = {m: models_cfg.get(m, {}).get("trained_on", {}) for m in res}
    sums = {m: summarize(res[m], trained[m], tasks) for m in res}
    order = sorted(res, key=lambda m: sums[m]["rank"])
    best_mean = max((s["mean"] for s in sums.values() if s["mean"] is not None), default=None)
    best_clean = max((s["clean"] for s in sums.values() if s["clean"] is not None), default=None)
    text = TEXT[lang]

    print(text["acc"], "", sep="\n")
    cols = ["#", text["model"], *(TASKS[t]["name"] for t in tasks), text["mean"], "Clean"]
    print("| " + " | ".join(cols) + " |", "|---" * len(cols) + "|", sep="\n")
    for i, m in enumerate(order, 1):
        s, first = sums[m], next(iter(res[m].values()))
        cells = [f"{s['accs'][t]:.3f}{MARK.get(trained[m].get(t), '')}" if t in s["accs"] else "–" for t in tasks]
        print(f"| {i} | [{m}](https://huggingface.co/{first['repo']}) | " + " | ".join(cells)
              + f" | {format_best(s['mean'], best_mean)} | {format_best(s['clean'], best_clean)} |")
    print("", text["legend"], "", text["cal"], "", sep="\n")
    cols = ["#", text["model"], *text["cal_cols"]]
    print("| " + " | ".join(cols) + " |", "|---" * len(cols) + "|", sep="\n")
    for i, m in enumerate(order, 1):
        k = compute_metrics([r for t in tasks if t in res[m] for r in res[m][t]["rows"]])
        first = next(iter(res[m].values()))
        print(f"| {i} | {m} | {k['acc']:.3f} ± {k['ci']:.3f} | {k['brier']:.3f} | {k['ece']:.3f} | "
              f"{k['wrong90']:.1%} | {k['ms']:.0f} | {first['runtime']}, {first['device']} |")

import json

from kodb.score import compute_brier, compute_ece, print_tables


def write_result(root, model, task, correct, probs):
    rows = [{"probs": p, "correct": c, "brier": compute_brier(p, 0 if c else 1), "ms": 10.0}
            for c, p in zip(correct, probs)]
    path = root / model / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "task": task, "repo": f"org/{model}", "runtime": "torch",
                                "device": "cpu", "summary": {"acc": sum(correct) / len(correct)}, "rows": rows}))


def test_metrics():
    assert compute_brier([0.75, 0.25], 0) == 0.125
    assert abs(compute_ece([0.9, 0.6], [True, False]) - 0.35) < 1e-12


def test_tables_rank_by_clean_mean_and_put_models_without_one_last(tmp_path, capsys):
    p = [[0.95, 0.05], [0.6, 0.4]]
    write_result(tmp_path, "leaky", "klue_nli", [True, True], p)        # higher mean, but in-domain
    write_result(tmp_path, "leaky", "klue_ynat", [True, True], p)
    write_result(tmp_path, "honest", "klue_nli", [True, False], p)
    write_result(tmp_path, "honest", "klue_ynat", [True, True], p)
    write_result(tmp_path, "mixed", "klue_nli", [True, True], p)
    write_result(tmp_path, "mixed", "klue_ynat", [False, False], p)
    cfg = {"leaky": {"trained_on": {"klue_nli": "split", "klue_ynat": "split"}},
           "mixed": {"trained_on": {"klue_nli": "split"}}}
    print_tables(tmp_path, cfg, None, "en")
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ") and "](" in line]
    assert [r.split("[")[1].split("]")[0] for r in rows] == ["honest", "mixed", "leaky"]
    assert rows[0].endswith("| 0.750 | **0.750** |")
    assert rows[2].endswith("| **1.000** | – |")
    assert "1.000†" in rows[2]

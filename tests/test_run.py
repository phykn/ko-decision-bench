import json

from kodb import run
from kodb.adapters.base import Adapter


class Fixed(Adapter):
    runtime, device = "fake", "cpu"

    def __init__(self, cfg):
        pass

    def decide(self, state, q):
        return [3.0, 1.0]   # unnormalised on purpose


def test_run_writes_the_result_schema_the_scores_read(tmp_path, monkeypatch):
    monkeypatch.setitem(run.ADAPTERS, "fixed", Fixed)
    q = {"type": "noul", "instructions": "?", "label": False}
    rec = {"id": "x-0", "state": "s", "questions": {"kobest_boolq": q}}
    (tmp_path / "kobest_boolq.jsonl").write_text(json.dumps(rec))
    run.run_model("m", ["kobest_boolq"], {"adapter": "fixed", "repo": "org/m", "revision": "r"}, tmp_path, tmp_path)
    d = json.loads((tmp_path / "m" / "kobest_boolq.json").read_text())
    assert list(d) == ["model", "repo", "revision", "task", "data_sha256", "date", "runtime", "device", "load_s",
                       "summary", "rows"]
    assert list(d["rows"][0]) == ["id", "question", "type", "label", "answer", "probs", "correct", "brier", "ms"]
    assert d["rows"][0]["probs"] == [0.75, 0.25] and d["rows"][0]["correct"] and d["summary"]["acc"] == 1.0

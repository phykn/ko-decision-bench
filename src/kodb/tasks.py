import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

N = 200
SEED = 0
NLI = {"함의": "가설이 반드시 참이다", "중립": "참인지 거짓인지 알 수 없다", "모순": "가설이 반드시 거짓이다"}
NLI_NAMES = ["함의", "중립", "모순"]   # label 0 entailment, 1 neutral, 2 contradiction in both KorNLI and KLUE-NLI
YNAT = {"IT과학": "IT, 과학, 기술", "경제": "경제, 금융, 산업", "사회": "사회, 사건, 교육, 노동", "생활문화": "생활, 문화, 건강, 여행",
        "세계": "국제, 외국 소식", "스포츠": "스포츠, 경기", "정치": "정치, 국회, 정부, 외교"}
STS = ["0: 의미가 전혀 다르다", "1: 거의 다르다", "2: 일부만 같다", "3: 대체로 같지만 중요한 차이가 있다",
       "4: 거의 같고 사소한 차이만 있다", "5: 의미가 같다"]

TASKS = {
    "unsmile_hate": {"repo": "smilegate-ai/kor_unsmile", "revision": "6ef17c27881aa44b0d74e9fe29e8d031b40ba07f",
                     "file": "data/valid-00000-of-00001.parquet", "split": "valid", "type": "noul",
                     "license": "CC BY-NC-ND 4.0", "redistribute": False},
    "kornli_xnli": {"repo": "kakaobrain/kor_nli", "revision": "3e0e4626f66911b344c490c26e3cc07e6c3bb0f9",
                    "file": "xnli/test-00000-of-00001.parquet", "split": "xnli test", "type": "choice",
                    "license": "CC BY-SA 4.0 (translation of XNLI, CC BY-NC 4.0)", "redistribute": False},
    "klue_nli": {"repo": "klue/klue", "revision": "349481ec73fff722f88e0453ca05c77a447d967c",
                 "file": "nli/validation-00000-of-00001.parquet", "split": "validation", "type": "choice",
                 "license": "CC BY-SA 4.0", "redistribute": True},
    "klue_ynat": {"repo": "klue/klue", "revision": "349481ec73fff722f88e0453ca05c77a447d967c",
                  "file": "ynat/validation-00000-of-00001.parquet", "split": "validation", "type": "choice",
                  "license": "CC BY-SA 4.0", "redistribute": True},
    "klue_sts": {"repo": "klue/klue", "revision": "349481ec73fff722f88e0453ca05c77a447d967c",
                 "file": "sts/validation-00000-of-00001.parquet", "split": "validation", "type": "score",
                 "license": "CC BY-SA 4.0", "redistribute": True},
    "kobest_boolq": {"repo": "skt/kobest_v1", "revision": "a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
                     "file": "boolq/test.jsonl", "split": "test", "type": "noul",
                     "license": "CC BY-SA 4.0", "redistribute": True},
    "kobest_copa": {"repo": "skt/kobest_v1", "revision": "a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
                    "file": "copa/test.jsonl", "split": "test", "type": "choice",
                    "license": "CC BY-SA 4.0", "redistribute": True},
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def load_rows(task):
    t = TASKS[task]
    path = hf_hub_download(t["repo"], t["file"], repo_type="dataset", revision=t["revision"])
    if path.endswith(".parquet"):
        return pq.read_table(path).to_pylist()
    return read_jsonl(path)


def sample(rows, n, rnd):
    return [rows[i] for i in rnd.sample(range(len(rows)), min(n, len(rows)))]


def nli_record(r):
    q = {"type": "choice", "instructions": "전제가 참일 때 가설은 어떤가요?", "criteria": NLI, "label": NLI_NAMES[r["label"]]}
    return f"전제: {r['premise']}\n가설: {r['hypothesis']}", q


def build_records(task, n=N, seed=SEED):
    rows, rnd = load_rows(task), random.Random(seed)
    out = []
    if task == "unsmile_hate":
        hate = [r for r in rows if r["clean"] == 0]
        clean = [r for r in rows if r["clean"] == 1]
        picked = sample(hate, n // 2, rnd) + sample(clean, n - n // 2, rnd)
        rnd.shuffle(picked)
        for r in picked:
            out.append((r["문장"], {"type": "noul", "instructions": "이 댓글에 혐오 표현이나 욕설, 악플이 있나요?",
                                   "label": r["clean"] == 0}))
    elif task in ("kornli_xnli", "klue_nli"):
        out = [nli_record(r) for r in sample([r for r in rows if r["label"] in (0, 1, 2)], n, rnd)]
    elif task == "klue_ynat":
        names = list(YNAT)
        for r in sample(rows, n, rnd):
            out.append((r["title"], {"type": "choice", "instructions": "이 뉴스 제목은 어느 분야의 기사인가요?",
                                     "criteria": YNAT, "label": names[r["label"]]}))
    elif task == "klue_sts":
        for r in sample(rows, n, rnd):
            level = min(5, int(r["labels"]["label"] + 0.5))   # the source label is real-valued
            out.append((f"문장 1: {r['sentence1']}\n문장 2: {r['sentence2']}",
                        {"type": "score", "instructions": "두 문장의 의미가 얼마나 비슷한가요?", "criteria": STS,
                         "label": level}))
    elif task == "kobest_boolq":
        for r in sample(rows, n, rnd):
            out.append((r["paragraph"], {"type": "noul",
                                         "instructions": f"지문에 따르면 다음 질문의 답은 '예'인가요? 질문: {r['question']}",
                                         "label": r["label"] == 1}))
    elif task == "kobest_copa":
        for r in sample(rows, n, rnd):
            out.append((r["premise"], {"type": "choice",
                                       "instructions": f"다음 중 이 상황의 {r['question']}{'으로' if r['question'] == '원인' else '로'} 더 그럴듯한 것은 무엇인가요?",
                                       "criteria": {"1": r["alternative_1"], "2": r["alternative_2"]},
                                       "label": "12"[r["label"]]}))
    return [{"id": f"{task}-{i}", "state": s, "questions": {task: q}} for i, (s, q) in enumerate(out)]


def build(tasks, out_dir, n=N, seed=SEED):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        recs = build_records(task, n, seed)
        path = out_dir / f"{task}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        print(f"{task}: {len(recs)} questions -> {path}")


def load(task, data_dir):
    path = Path(data_dir) / f"{task}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Build it first: kodb build {task}")
    return read_jsonl(path)


def options(q):
    if q["type"] == "noul":
        return ["false", "true"]
    return list(q["criteria"])


def label_index(q):
    if q["type"] == "noul":
        return int(bool(q["label"]))
    if q["type"] == "choice":
        return list(q["criteria"]).index(q["label"])
    return int(q["label"])


def unlabeled(q):
    return {k: v for k, v in q.items() if k != "label"}

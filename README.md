# ko-decision-bench

A reproducible Korean benchmark for **decision models**. These are Jev / System One-style models that read a text and a typed question. In one forward pass, they return a probability for each option instead of generating text.

Few Korean scores have been published for these models, and model cards each use their own prompts and splits. This benchmark asks every model the same questions and scores every model the same way.

> 텍스트를 생성하지 않고 확률로 답하는 판정 모델들을 같은 한국어 문항과 같은 지표로 비교합니다.

## Results

Measured on 2026-10-19: 8 models, 7 tasks, and 1,400 questions per model. Models are sorted by mean accuracy.

**Accuracy per task**

| Model | KLUE NLI | KLUE STS | KLUE YNAT | KoBEST BoolQ | KoBEST COPA | KorNLI XNLI | UnSmile | Mean | Clean |
|---|---|---|---|---|---|---|---|---|---|
| [nari-4b](https://huggingface.co/blackmallangcow/nari-decision-4b) | 0.880† | 0.470† | 0.865† | 0.945† | 0.940† | 0.790† | 0.750‡ | **0.806** | – |
| [coco-2.4-4b](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B) | 0.900? | 0.160 | 0.855 | 0.940? | 0.940? | 0.745? | 0.690 | 0.747 | 0.568 |
| [kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | 0.815 | 0.355 | 0.730 | 0.920 | 0.955 | 0.725 | 0.685 | 0.741 | **0.741** |
| [malkuth-2b](https://huggingface.co/dhtocks/malkuth-2b) | 0.785† | 0.175 | 0.835† | 0.905† | 0.860† | 0.730 | 0.765 | 0.722 | 0.557 |
| [jev-style-0.8b-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) | 0.690? | 0.260 | 0.565 | 0.760 | 0.620 | 0.710? | 0.620 | 0.604 | 0.565 |
| [kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 0.680 | 0.150 | 0.645 | 0.700 | 0.685 | 0.685 | 0.665 | 0.601 | 0.601 |
| [laya-ko](https://huggingface.co/2nugu/laya-ko) | 0.590† | 0.240† | 0.820† | 0.600 | 0.530 | 0.545 | 0.570 | 0.556 | 0.561 |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 0.535 | 0.175 | 0.450 | 0.550 | 0.515 | 0.540 | 0.650 | 0.488 | 0.488 |

`†` marks a model that trained on another split of the same source. `‡` marks related Korean training data. `?` marks a card that is unclear. **Clean** is the mean over the tasks with no mark. See [Method](#method) for details.

**Calibration, all questions pooled**

| Model | Accuracy ±95% | Brier | ECE | p ≥ .9 wrong | ms/q | Runtime |
|---|---|---|---|---|---|---|
| nari-4b | 0.806 ± 0.021 | 0.276 | 0.054 | 3.4% | 781 | torch MPS |
| coco-2.4-4b | 0.747 ± 0.023 | 0.351 | 0.118 | 5.6% | 519 | torch MPS |
| kev-4b | 0.741 ± 0.023 | 0.345 | 0.054 | 3.2% | 148 | kev.serve MLX |
| malkuth-2b | 0.722 ± 0.023 | 0.369 | 0.060 | 2.4% | 59 | kev.serve MLX |
| jev-style-0.8b-v3 | 0.604 ± 0.026 | 0.520 | 0.120 | 5.3% | 155 | torch MPS |
| kev-0.8b | 0.601 ± 0.026 | 0.510 | 0.055 | 2.5% | 28 | kev.serve MLX |
| laya-ko | 0.556 ± 0.026 | 0.621 | 0.232 | 10.0% | 17 | torch MPS |
| laya-multilingual | 0.488 ± 0.026 | 0.676 | 0.207 | 8.4% | 17 | torch MPS |

**Summary**

- On tasks with no mark, kev-4b is the best model, and it is also among the best calibrated.
- nari-4b has the highest accuracy, but it trained on other splits of all seven sources, so its score is in-domain.
- malkuth-2b has the fewest high-confidence errors (2.4%), and its latency is 59 ms on MLX.
- In the 0.8B class, kev-0.8b and jev-style-0.8b-v3 have similar accuracy, and kev-0.8b is better calibrated.

## Tasks

Each task has 200 questions, sampled with `random.Random(0)` from a pinned commit of its source. Each task also has one fixed Korean instruction, which every model receives ([`tasks.py`](src/kodb/tasks.py)).

| Task | Type | Source (split) | Options | License |
|---|---|---|---|---|
| `unsmile_hate` | yes/no | smilegate-ai/kor_unsmile (valid), 100 per class | Hate or abuse present | CC BY-NC-ND 4.0 |
| `kornli_xnli` | choice | kakaobrain/kor_nli (xnli test) | 함의 / 중립 / 모순 | CC BY-SA 4.0; translated from XNLI (CC BY-NC 4.0) |
| `klue_nli` | choice | klue/klue nli (validation) | 함의 / 중립 / 모순 | CC BY-SA 4.0 |
| `klue_ynat` | choice | klue/klue ynat (validation) | 7 news sections | CC BY-SA 4.0 |
| `klue_sts` | score | klue/klue sts (validation) | Similarity, 6 levels | CC BY-SA 4.0 |
| `kobest_boolq` | yes/no | skt/kobest_v1 boolq (test) | yes / no | CC BY-SA 4.0 |
| `kobest_copa` | choice | skt/kobest_v1 copa (test) | 2 alternatives | CC BY-SA 4.0 |

The repository contains no data and no per-question results. Two sources do not allow redistribution, and result rows include gold labels. You build the data locally, for research use only.

## Method

- Each model returns one probability per option, normalized to sum to 1. Each model uses its own shipped temperatures. Nothing is refitted.
- **Accuracy**: the answer is the option with the highest probability. The 95% CI is ±1.96·√(p(1−p)/n).
- **Brier**: Σₖ (pₖ − yₖ)² over the options. The best score is 0 and the worst is 2.
- **ECE**: computed on the top probability, with 10 equal-width bins.
- **p ≥ .9 wrong**: the share of questions answered wrongly with a top probability of at least 0.9. A confidence threshold cannot catch these errors.
- **ms/q**: the time for one question per call, measured after a warm-up.
- `klue_sts` is scored by exact level. Chance is 0.167.

Scores depend on these prompts. Model cards report results with each model's own input format, so they can be much higher. For example, laya-ko reports 0.815 on KLUE-NLI and gets 0.590 here.

**Contamination marks.** [`models.toml`](models.toml) records what each model card says about its training data. Cards were read on 2026-10-19.

- `†` The model trained on another split of the same dataset. Its score is in-domain, not zero-shot.
- `‡` The model trained on Korean data for the same task from another source.
- `?` The card describes data that might cover the task.

`clean` is the mean over the tasks that have no mark for that model, so each model is averaged over a different set of tasks.

**Hardware.** Every model ran on an Apple Silicon Mac with 32 GB of memory, one model at a time. laya, Jev-Style, nari, and CoCo ran in PyTorch on MPS. Kev-format models (kev-4b, kev-0.8b, malkuth-2b) ran in [kev.serve](https://github.com/jaredpalmer/kev) on MLX bf16 and were called over HTTP. Compare ms/q only between models on the same runtime.

## Usage

```bash
uv sync --extra models          # torch, transformers, laya (kev-http models need only the core)
uv run kodb build               # download the pinned sources and write data/<task>.jsonl
uv run kodb run nari-4b all     # write results/nari-4b/<task>.json
uv run kodb score               # print the tables
```

Kev-format models need a running kev.serve with the same `repo@revision` as in `models.toml`:

```bash
python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8081
uv run kodb run kev-4b all --url http://127.0.0.1:8081
```

**Add a model**: subclass `Adapter` in [`adapters.py`](src/kodb/adapters.py) so that `decide(state, q)` returns probabilities in `options(q)` order, and register it in `ADAPTERS`. Then add `repo`, `revision`, and `trained_on` to `models.toml`.

## License

The code is Apache-2.0. Datasets and models keep their own licenses: see the table above and each model card. The malkuth-2b weights are CC BY-NC 4.0.

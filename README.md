# ko-decision-bench

[Korean](README.ko.md)

A reproducible Korean benchmark for **decision models**: Jev / System One-style models that read a text and a typed question and return a probability per option in one forward pass, without generating text. Model cards each use their own prompts and splits; this benchmark asks every model the same questions and scores them the same way.

## Results

Measured on an Apple M6 with 32 GB: 14 models, 7 tasks, 1,400 questions per model. Ranked by **Clean**, the mean over tasks with no possible training-data overlap, so training on the evaluated distribution does not buy a higher place. Models with no clean task go last.

**Accuracy per task**

| # | Model | KLUE NLI | KLUE STS | KLUE YNAT | KoBEST BoolQ | KoBEST COPA | KorNLI XNLI | UnSmile | Mean | Clean |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [kanana-2-v-decide-3b](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611) | 0.885 | 0.465 | 0.895 | 0.915 | 0.960 | 0.780 | 0.820 | **0.817** | **0.817** |
| 2 | [pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | 0.905 | 0.470 | 0.810 | 0.960 | 0.950 | 0.780 | 0.750 | 0.804 | 0.804 |
| 3 | [startlux-4b-instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) | 0.910 | 0.415 | 0.770 | 0.960 | 0.970 | 0.810 | 0.700 | 0.791 | 0.791 |
| 4 | [thx-02](https://huggingface.co/HAL-X-AI/THX-02) | 0.910 | 0.365 | 0.825 | 0.960 | 0.950 | 0.770 | 0.750 | 0.790 | 0.790 |
| 5 | [h2o-lightning-4b](https://huggingface.co/h2oai/h2o-lightning-4b) | 0.870 | 0.405 | 0.800 | 0.935 | 0.955 | 0.780 | 0.730 | 0.782 | 0.782 |
| 6 | [autotrust-decider-mini-9b](https://huggingface.co/autotrust/AutoTrust-Decider-Mini-9B) | 0.865 | 0.265 | 0.720 | 0.955 | 0.950 | 0.750 | 0.735 | 0.749 | 0.749 |
| 7 | [kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | 0.815 | 0.355 | 0.730 | 0.920 | 0.955 | 0.725 | 0.685 | 0.741 | 0.741 |
| 8 | [kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 0.680 | 0.150 | 0.645 | 0.700 | 0.685 | 0.685 | 0.665 | 0.601 | 0.601 |
| 9 | [coco-2.4-4b](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B) | 0.900? | 0.160 | 0.855 | 0.940? | 0.940? | 0.745? | 0.690 | 0.747 | 0.568 |
| 10 | [jev-style-0.8b-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) | 0.690? | 0.260 | 0.565 | 0.760 | 0.620 | 0.710? | 0.620 | 0.604 | 0.565 |
| 11 | [laya-ko](https://huggingface.co/2nugu/laya-ko) | 0.590† | 0.240† | 0.820† | 0.600 | 0.530 | 0.545 | 0.570 | 0.556 | 0.561 |
| 12 | [malkuth-2b](https://huggingface.co/dhtocks/malkuth-2b) | 0.785† | 0.175 | 0.835† | 0.905† | 0.860† | 0.730 | 0.765 | 0.722 | 0.557 |
| 13 | [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 0.535 | 0.175 | 0.450 | 0.550 | 0.515 | 0.540 | 0.650 | 0.488 | 0.488 |
| 14 | [nari-4b](https://huggingface.co/blackmallangcow/nari-decision-4b) | 0.880† | 0.470† | 0.865† | 0.945† | 0.940† | 0.790† | 0.750‡ | 0.806 | – |

`†` `‡` `?` mark possible training-data overlap ([Method](#method)). **Clean** is the mean over unmarked tasks, a different set per model.

**Calibration, all questions pooled**

| # | Model | Accuracy ±95% | Brier | ECE | p ≥ .9 wrong | ms/q | Runtime |
|---|---|---|---|---|---|---|---|
| 1 | kanana-2-v-decide-3b | 0.817 ± 0.020 | 0.254 | 0.046 | 1.5% | 135 | torch MPS |
| 2 | pplx-decider-flash-v1.2 | 0.804 ± 0.021 | 0.256 | 0.015 | 1.0% | 624 | torch MPS fp32 |
| 3 | startlux-4b-instant | 0.791 ± 0.021 | 0.286 | 0.020 | 1.5% | 235 | startlux-decide MPS |
| 4 | thx-02 | 0.790 ± 0.021 | 0.259 | 0.041 | 1.2% | 590 | torch MPS fp32 |
| 5 | h2o-lightning-4b | 0.782 ± 0.022 | 0.302 | 0.059 | 2.5% | 164 | h2o shim MLX |
| 6 | autotrust-decider-mini-9b | 0.749 ± 0.023 | 0.323 | 0.036 | 1.5% | 368 | torch MPS |
| 7 | kev-4b | 0.741 ± 0.023 | 0.345 | 0.054 | 3.2% | 148 | kev.serve MLX |
| 8 | kev-0.8b | 0.601 ± 0.026 | 0.510 | 0.055 | 2.5% | 28 | kev.serve MLX |
| 9 | coco-2.4-4b | 0.747 ± 0.023 | 0.351 | 0.118 | 5.6% | 519 | torch MPS |
| 10 | jev-style-0.8b-v3 | 0.604 ± 0.026 | 0.520 | 0.120 | 5.3% | 155 | torch MPS |
| 11 | laya-ko | 0.556 ± 0.026 | 0.621 | 0.232 | 10.0% | 17 | torch MPS |
| 12 | malkuth-2b | 0.722 ± 0.023 | 0.369 | 0.060 | 2.4% | 59 | kev.serve MLX |
| 13 | laya-multilingual | 0.488 ± 0.026 | 0.676 | 0.207 | 8.4% | 17 | torch MPS |
| 14 | nari-4b | 0.806 ± 0.021 | 0.276 | 0.054 | 3.4% | 781 | torch MPS |

- kanana-2-v-decide-3b ranks first: 0.817 on seven unmarked tasks, above even the mean of nari-4b (0.806), which trained on other splits of all seven sources.
- pplx-decider-flash-v1.2 is second at 0.804 and the best calibrated (ECE 0.015, 1.0% high-confidence errors). Off CUDA it runs in fp32, so it is slow on MPS.
- startlux-4b-instant (0.791) and thx-02 (0.790) are tied within the ±0.021 interval. THX-02 ships pplx-decider-flash's code and saved temperature, and scores within 0.015 of it on six of seven tasks (all but KLUE STS).
- autotrust-decider-mini-9b is the largest model here (9B) and ranks sixth. Its score scale is a fixed 0–5, so on KLUE STS the level meanings go in the question rather than the options.
- nari-4b has a mark on every task, so it has no Clean score and ranks last.
- Among 0.8B models, kev-0.8b matches jev-style-0.8b-v3 in accuracy and is better calibrated.

## Tasks

200 questions per task, sampled with `random.Random(0)` from a pinned commit of the source, each with one fixed Korean instruction ([`tasks.py`](src/kodb/tasks.py)).

| Task | Type | Source (split) | Options | License |
|---|---|---|---|---|
| `unsmile_hate` | yes/no | smilegate-ai/kor_unsmile (valid), 100 per class | Hate or abuse present | CC BY-NC-ND 4.0 |
| `kornli_xnli` | choice | kakaobrain/kor_nli (xnli test) | entailment / neutral / contradiction | CC BY-SA 4.0; translated from XNLI (CC BY-NC 4.0) |
| `klue_nli` | choice | klue/klue nli (validation) | entailment / neutral / contradiction | CC BY-SA 4.0 |
| `klue_ynat` | choice | klue/klue ynat (validation) | 7 news sections | CC BY-SA 4.0 |
| `klue_sts` | score | klue/klue sts (validation) | Similarity, 6 levels | CC BY-SA 4.0 |
| `kobest_boolq` | yes/no | skt/kobest_v1 boolq (test) | yes / no | CC BY-SA 4.0 |
| `kobest_copa` | choice | skt/kobest_v1 copa (test) | 2 alternatives | CC BY-SA 4.0 |

The repository ships no data or per-question results: two sources forbid redistribution, and result rows contain gold labels. Build the data locally, for research use only.

## Method

- Each model returns one probability per option, normalized to sum to 1, with its own shipped temperatures. Nothing is refitted.
- **Accuracy**: the top-probability option. 95% CI is ±1.96·√(p(1−p)/n).
- **Brier**: Σₖ (pₖ − yₖ)², from 0 (best) to 2.
- **ECE**: on the top probability, 10 equal-width bins.
- **p ≥ .9 wrong**: share of questions answered wrongly with top probability ≥ 0.9, which no confidence threshold catches.
- **ms/q**: one question per call, after a warm-up. Compare only within the same runtime.
- `klue_sts` is scored by exact level (chance 0.167).

Scores depend on these prompts. Model cards use each model's own format and can report much higher; laya-ko reports 0.815 on KLUE-NLI and gets 0.590 here.

**Contamination marks**, from the model cards at the pinned revisions ([`models.toml`](models.toml)):

- `†` trained on another split of the same dataset (in-domain, not zero-shot)
- `‡` trained on Korean data for the same task from another source
- `?` the card describes data that might cover the task

**Hardware.** Apple M6, 32 GB, one model at a time. laya, Jev-Style, nari, CoCo, and Kanana ran in PyTorch on MPS; pplx-decider and THX-02 ran in PyTorch on MPS in fp32 (their code uses fp32 off CUDA); AutoTrust ran in PyTorch on MPS in bf16 with the card's transformers + peft recipe; StartLux ran in its own [startlux-decide](https://github.com/StartLuxLabs/StartLux-Decision) server (transformers backend, MPS bf16); h2o-lightning-4b ran its official bf16 weights under H2O's shim on the [h2o-lightning-mlx](https://github.com/internet-projects/h2o-lightning-mlx) MLX engine, two-way, default profile; kev-4b, kev-0.8b, and malkuth-2b ran in [kev.serve](https://github.com/jaredpalmer/kev) on MLX bf16 over HTTP.

## Usage

```bash
uv sync --extra models          # torch, transformers, laya, ...
uv run kodb build               # write data/<task>.jsonl
uv run kodb run nari-4b all     # write results/nari-4b/<task>.json
uv run kodb score               # print the tables
```

All Hugging Face downloads (weights and sources) go to `weights/`, which kodb sets as `HF_HOME` unless you set one.

Kev-format models run in kev.serve. `kodb run` uses the server at `--url` (default `http://127.0.0.1:8081`) if it serves the `repo@revision` in `models.toml`; otherwise it starts one there in a separate uv environment (Python 3.12, the Kev commit the results used), logs to `logs/`, and stops it on exit. StartLux and h2o start their servers the same way: startlux-decide in a uv environment with transformers 5.8 (`--url` default `:8090`), and h2o from h2o-lightning-mlx v1.4.2, which kodb clones into `weights/` and sets up on first use (`:8741`).

**Add a model**: subclass `Adapter` in [`adapters.py`](src/kodb/adapters.py) so `decide(state, q)` returns probabilities in `options(q)` order, register it in `ADAPTERS`, and add `repo`, `revision`, and `trained_on` to `models.toml`.

## License

Code: Apache-2.0. Datasets and models keep their own licenses (table above and each model card); the malkuth-2b and StartLux weights are CC BY-NC 4.0, and the Kanana weights use the Kanana Open License.

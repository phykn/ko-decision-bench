<div align="center">

# ko-decision-bench

A reproducible Korean benchmark for decision models: the same questions and the same scoring for every model.

![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Models](https://img.shields.io/badge/models-14-green) ![Tasks](https://img.shields.io/badge/tasks-7-green)

[한국어](README.ko.md)

</div>

Decision models (Jev / System One style) read a text and a typed question and return a probability for every option in one forward pass, without generating text. Model cards each use their own prompts and splits; this benchmark does not.

- 7 Korean tasks, 200 questions each, from pinned source commits
- Ranked by **Clean** accuracy, so training on the evaluated distribution does not buy a place
- Calibration next to accuracy: Brier, ECE, and confident wrong answers
- One adapter per model family, each following its model card

## Results

Apple M6, 32 GB, one model at a time.

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
| 1 | kanana-2-v-decide-3b | 0.817 ± 0.020 | 0.254 | 0.046 | 1.5% | 135 | torch, mps |
| 2 | pplx-decider-flash-v1.2 | 0.804 ± 0.021 | 0.256 | 0.015 | 1.0% | 624 | torch, mps |
| 3 | startlux-4b-instant | 0.791 ± 0.021 | 0.286 | 0.020 | 1.5% | 235 | startlux-decide transformers (HTTP), mps bfloat16 |
| 4 | thx-02 | 0.790 ± 0.021 | 0.259 | 0.041 | 1.2% | 590 | torch, mps |
| 5 | h2o-lightning-4b | 0.782 ± 0.022 | 0.302 | 0.059 | 2.5% | 164 | h2o shim + MLX facade (HTTP), mlx bfloat16 |
| 6 | autotrust-decider-mini-9b | 0.749 ± 0.023 | 0.323 | 0.036 | 1.5% | 368 | torch, mps |
| 7 | kev-4b | 0.741 ± 0.023 | 0.345 | 0.054 | 3.2% | 148 | kev.serve (HTTP), mlx mps bfloat16 |
| 8 | kev-0.8b | 0.601 ± 0.026 | 0.510 | 0.055 | 2.5% | 28 | kev.serve (HTTP), mlx mps bfloat16 |
| 9 | coco-2.4-4b | 0.747 ± 0.023 | 0.351 | 0.118 | 5.6% | 519 | torch, mps |
| 10 | jev-style-0.8b-v3 | 0.604 ± 0.026 | 0.520 | 0.120 | 5.3% | 155 | torch, mps |
| 11 | laya-ko | 0.556 ± 0.026 | 0.621 | 0.232 | 10.0% | 17 | torch, mps |
| 12 | malkuth-2b | 0.722 ± 0.023 | 0.369 | 0.060 | 2.4% | 59 | kev.serve (HTTP), mlx mps bfloat16 |
| 13 | laya-multilingual | 0.488 ± 0.026 | 0.676 | 0.207 | 8.4% | 17 | torch, mps |
| 14 | nari-4b | 0.806 ± 0.021 | 0.276 | 0.054 | 3.4% | 781 | torch, mps |

- kanana-2-v-decide-3b ranks first: 0.817 on seven unmarked tasks, above even the mean of nari-4b (0.806), which trained on other splits of all seven sources.
- pplx-decider-flash-v1.2 is second and the best calibrated (ECE 0.015, 1.0% confident wrong answers).
- startlux-4b-instant and thx-02 are tied within the ±0.021 interval. THX-02 ships pplx-decider-flash's code and saved temperature and scores within 0.015 of it on every task but KLUE STS.
- autotrust-decider-mini-9b is the largest model (9B) and ranks sixth. Its score head has fixed 0–5 levels, so on KLUE STS the level meanings go in the question instead of the options.
- nari-4b has a mark on every task, so it has no Clean score and ranks last.

## Usage

```bash
uv sync --extra models                     # torch, transformers, laya, peft, ...
uv run kodb build                          # download the sources, write data/<task>.jsonl
uv run kodb list                           # tasks and models
uv run kodb run laya-multilingual klue_nli # write results/<model>/<task>.json; 'all' runs every task
uv run kodb score                          # the tables above; --lang ko for Korean headers
```

Downloads go to `weights/`. kev.serve, StartLux, and h2o models start their own server on first use (logs in `logs/`); pass `--url` to use another address.

## Tasks

| Task | Type | Source (split) | Options | License |
|---|---|---|---|---|
| `unsmile_hate` | yes/no | smilegate-ai/kor_unsmile (valid), 100 per class [4] | hate or abuse present | CC BY-NC-ND 4.0 |
| `kornli_xnli` | choice | kakaobrain/kor_nli (xnli test) [2, 5] | entailment / neutral / contradiction | CC BY-SA 4.0; from XNLI (CC BY-NC 4.0) |
| `klue_nli` | choice | klue/klue nli (validation) [1] | entailment / neutral / contradiction | CC BY-SA 4.0 |
| `klue_ynat` | choice | klue/klue ynat (validation) [1] | 7 news sections | CC BY-SA 4.0 |
| `klue_sts` | score | klue/klue sts (validation) [1] | similarity, 6 levels | CC BY-SA 4.0 |
| `kobest_boolq` | yes/no | skt/kobest_v1 boolq (test) [3] | yes / no | CC BY-SA 4.0 |
| `kobest_copa` | choice | skt/kobest_v1 copa (test) [3] | 2 alternatives | CC BY-SA 4.0 |

Questions are sampled with `random.Random(0)`, each task with one fixed Korean instruction. The repository ships no data or per-question results: two sources forbid redistribution, and result rows hold gold labels. Build the data locally, for research use only.

## Method

- Each model returns one probability per option with its own shipped temperatures, normalized to sum to 1. Nothing is refitted.
- **Accuracy** takes the top option; the 95% interval is ±1.96·√(p(1−p)/n). `klue_sts` counts the exact level (chance 0.167).
- **Brier** is Σₖ (pₖ − yₖ)², from 0 to 2. **ECE** uses the top probability in 10 equal bins. **p ≥ .9 wrong** is the share of wrong answers given with at least 0.9.
- **ms/q** is one question per call after a warm-up; compare it only within one runtime. pplx-decider and THX-02 run in fp32 off CUDA.
- Marks come from the model cards at the pinned revisions in [`models.toml`](models.toml): `†` trained on another split of the same dataset, `‡` trained on Korean data for the same task from another source, `?` the card describes data that might cover the task.

Scores depend on these prompts: laya-ko reports 0.815 on KLUE NLI in its own format and gets 0.590 here.

To add a model, write an adapter in [`src/kodb/adapters/`](src/kodb/adapters) whose `decide(state, q)` returns probabilities in `list_options(q)` order, register it in `ADAPTERS`, and add `repo`, `revision`, and `trained_on` to `models.toml`.

## License

Code: Apache-2.0. Datasets and models keep their own licenses; the malkuth-2b and StartLux weights are CC BY-NC 4.0, and the Kanana weights use the Kanana Open License.

## References

1. Sungjoon Park et al. KLUE: Korean Language Understanding Evaluation. NeurIPS Datasets and Benchmarks, 2021. https://github.com/KLUE-benchmark/KLUE
2. Jiyeon Ham et al. KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding. Findings of EMNLP, 2020. https://github.com/kakaobrain/kor-nlu-datasets
3. Dohyeong Kim et al. KoBEST: Korean Balanced Evaluation of Significant Tasks. COLING, 2022. https://huggingface.co/datasets/skt/kobest_v1
4. Smilegate AI. Korean UnSmile Dataset. 2022. https://github.com/smilegate-ai/korean_unsmile_dataset
5. Alexis Conneau et al. XNLI: Evaluating Cross-lingual Sentence Representations. EMNLP, 2018. https://github.com/facebookresearch/XNLI

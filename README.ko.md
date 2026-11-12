<div align="center">

# ko-decision-bench

판정 모델을 위한 재현 가능한 한국어 벤치마크다. 모든 모델에 같은 문항을 묻고 같은 방식으로 채점한다.

![License](https://img.shields.io/badge/license-Apache--2.0-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Models](https://img.shields.io/badge/models-14-green) ![Tasks](https://img.shields.io/badge/tasks-7-green)

[English](README.md)

</div>

판정 모델(Jev / System One 계열)은 텍스트와 유형이 정해진 질문을 읽고, 텍스트를 생성하지 않은 채 한 번의 forward pass로 선택지마다 확률을 낸다. 모델 카드마다 프롬프트와 분할이 다르지만, 이 벤치마크는 그렇지 않다.

- 한국어 과제 7개, 과제당 200문항, 출처 커밋 고정
- **Clean** 정확도로 순위를 매겨, 평가 분포로 학습한 모델이 그만큼 앞서지 않는다
- 정확도와 함께 보정을 본다: Brier, ECE, 확신 높은 오답
- 모델 계열마다 adapter 하나, 각 모델 카드의 방식을 따른다

## 결과

Apple M6 32GB에서 한 번에 한 모델씩 측정했다.

**과제별 정확도**

| # | 모델 | KLUE NLI | KLUE STS | KLUE YNAT | KoBEST BoolQ | KoBEST COPA | KorNLI XNLI | UnSmile | 평균 | Clean |
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

`†` `‡` `?`는 학습 데이터가 겹칠 수 있음을 나타낸다([방법](#방법)). **Clean**은 표시가 없는 과제만의 평균이다. 모델마다 들어가는 과제가 다르다.

**보정(calibration), 전체 문항 합산**

| # | 모델 | 정확도 ±95% | Brier | ECE | p ≥ .9 오답 | ms/문항 | 런타임 |
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

- kanana-2-v-decide-3b가 1위다. 표시 없는 7개 과제 평균이 0.817로, 7개 출처 모두의 다른 분할로 학습한 nari-4b의 평균(0.806)보다도 높다.
- pplx-decider-flash-v1.2는 2위이고 보정이 가장 좋다(ECE 0.015, 확신 높은 오답 1.0%).
- startlux-4b-instant와 thx-02는 ±0.021 구간 안에서 동률이다. THX-02는 pplx-decider-flash의 코드와 저장된 temperature를 그대로 쓰며, KLUE STS를 뺀 모든 과제에서 0.015 이내로 같은 점수를 낸다.
- autotrust-decider-mini-9b는 가장 큰 모델(9B)이고 6위다. score 헤드의 단계가 0~5로 고정이라 KLUE STS에서는 단계의 뜻을 선택지가 아니라 질문에 넣었다.
- nari-4b는 모든 과제에 표시가 있어 Clean 점수가 없고 맨 아래다.

## 사용법

```bash
uv sync --extra models                     # torch, transformers, laya, peft 등
uv run kodb build                          # 출처를 받아 data/<task>.jsonl 생성
uv run kodb list                           # 과제와 모델 목록
uv run kodb run laya-multilingual klue_nli # results/<model>/<task>.json 생성, 'all'은 모든 과제
uv run kodb score --lang ko                # 위의 표, 기본은 영어 머리글
```

다운로드는 `weights/`에 저장된다. kev.serve, StartLux, h2o 모델은 처음 실행할 때 자기 서버를 띄우고(로그는 `logs/`), 다른 주소를 쓰려면 `--url`을 준다.

## 과제

| 과제 | 유형 | 출처 (분할) | 선택지 | 라이선스 |
|---|---|---|---|---|
| `unsmile_hate` | 예/아니오 | smilegate-ai/kor_unsmile (valid), 클래스당 100 [4] | 혐오·욕설 여부 | CC BY-NC-ND 4.0 |
| `kornli_xnli` | 선택 | kakaobrain/kor_nli (xnli test) [2, 5] | 함의 / 중립 / 모순 | CC BY-SA 4.0, XNLI(CC BY-NC 4.0) 번역 |
| `klue_nli` | 선택 | klue/klue nli (validation) [1] | 함의 / 중립 / 모순 | CC BY-SA 4.0 |
| `klue_ynat` | 선택 | klue/klue ynat (validation) [1] | 뉴스 분야 7개 | CC BY-SA 4.0 |
| `klue_sts` | 점수 | klue/klue sts (validation) [1] | 유사도 6단계 | CC BY-SA 4.0 |
| `kobest_boolq` | 예/아니오 | skt/kobest_v1 boolq (test) [3] | 예 / 아니오 | CC BY-SA 4.0 |
| `kobest_copa` | 선택 | skt/kobest_v1 copa (test) [3] | 대안 2개 | CC BY-SA 4.0 |

문항은 `random.Random(0)`으로 뽑고, 과제마다 고정된 한국어 지시문 하나를 쓴다. 저장소에는 데이터와 문항별 결과가 없다. 두 출처가 재배포를 금지하고, 결과 행에 정답 레이블이 들어 있기 때문이다. 데이터는 로컬에서 생성하며 연구용으로만 쓴다.

## 방법

- 모델마다 자체 temperature로 선택지별 확률을 내고, 합이 1이 되게 정규화한다. 다시 맞추는 값은 없다.
- **정확도**는 확률이 가장 높은 선택지로 채점하고, 95% 구간은 ±1.96·√(p(1−p)/n)이다. `klue_sts`는 정확한 단계만 정답이다(우연 0.167).
- **Brier**는 Σₖ (pₖ − yₖ)²로 0에서 2 사이다. **ECE**는 최고 확률을 10개 구간으로 나눠 계산한다. **p ≥ .9 오답**은 0.9 이상 확신하고 틀린 문항의 비율이다.
- **ms/문항**은 warm-up 뒤 한 번에 한 문항씩 잰 값이라 같은 런타임 안에서만 비교한다. pplx-decider와 THX-02는 CUDA가 아니면 fp32로 돈다.
- 표시는 [`models.toml`](models.toml)에 고정한 revision의 모델 카드를 따른다. `†`는 같은 데이터셋의 다른 분할로 학습, `‡`는 다른 출처의 같은 과제 한국어 데이터로 학습, `?`는 카드에 이 과제를 포함할 수 있는 데이터가 적혀 있다는 뜻이다.

점수는 이 프롬프트에 따라 달라진다. laya-ko는 자체 형식으로 KLUE NLI 0.815를 보고하지만 여기서는 0.590이다.

모델을 추가하려면 [`src/kodb/adapters/`](src/kodb/adapters)에 `decide(state, q)`가 `list_options(q)` 순서로 확률을 돌려주는 adapter를 쓰고, `ADAPTERS`에 등록한 뒤 `models.toml`에 `repo`, `revision`, `trained_on`을 추가한다.

## 라이선스

코드는 Apache-2.0이다. 데이터셋과 모델은 각자의 라이선스를 따르며, malkuth-2b와 StartLux 웨이트는 CC BY-NC 4.0, Kanana 웨이트는 Kanana Open License다.

## 참고 문헌

1. Sungjoon Park et al. KLUE: Korean Language Understanding Evaluation. NeurIPS Datasets and Benchmarks, 2021. https://github.com/KLUE-benchmark/KLUE
2. Jiyeon Ham et al. KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding. Findings of EMNLP, 2020. https://github.com/kakaobrain/kor-nlu-datasets
3. Dohyeong Kim et al. KoBEST: Korean Balanced Evaluation of Significant Tasks. COLING, 2022. https://huggingface.co/datasets/skt/kobest_v1
4. Smilegate AI. Korean UnSmile Dataset. 2022. https://github.com/smilegate-ai/korean_unsmile_dataset
5. Alexis Conneau et al. XNLI: Evaluating Cross-lingual Sentence Representations. EMNLP, 2018. https://github.com/facebookresearch/XNLI

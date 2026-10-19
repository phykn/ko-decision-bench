# ko-decision-bench

[English](README.md)

**판정 모델**을 위한 재현 가능한 한국어 벤치마크예요. 판정 모델은 Jev / System One 계열 모델로, 텍스트와 유형이 정해진 질문을 읽고 텍스트 생성 없이 한 번의 forward pass로 선택지마다 확률을 내요. 모델 카드마다 프롬프트와 분할이 달라서, 이 벤치마크는 모든 모델에 같은 문항을 묻고 같은 방식으로 채점해요.

## 결과

2026-10-19, Apple M6 32GB에서 측정했어요. 모델 8개, 과제 7개, 모델당 1,400문항이고 평균 정확도 순으로 정렬했어요.

**과제별 정확도**

| 모델 | KLUE NLI | KLUE STS | KLUE YNAT | KoBEST BoolQ | KoBEST COPA | KorNLI XNLI | UnSmile | 평균 | Clean |
|---|---|---|---|---|---|---|---|---|---|
| [nari-4b](https://huggingface.co/blackmallangcow/nari-decision-4b) | 0.880† | 0.470† | 0.865† | 0.945† | 0.940† | 0.790† | 0.750‡ | **0.806** | – |
| [coco-2.4-4b](https://huggingface.co/corners-ai/CoCo-Decision-2.4-4B) | 0.900? | 0.160 | 0.855 | 0.940? | 0.940? | 0.745? | 0.690 | 0.747 | 0.568 |
| [kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | 0.815 | 0.355 | 0.730 | 0.920 | 0.955 | 0.725 | 0.685 | 0.741 | **0.741** |
| [malkuth-2b](https://huggingface.co/dhtocks/malkuth-2b) | 0.785† | 0.175 | 0.835† | 0.905† | 0.860† | 0.730 | 0.765 | 0.722 | 0.557 |
| [jev-style-0.8b-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) | 0.690? | 0.260 | 0.565 | 0.760 | 0.620 | 0.710? | 0.620 | 0.604 | 0.565 |
| [kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 0.680 | 0.150 | 0.645 | 0.700 | 0.685 | 0.685 | 0.665 | 0.601 | 0.601 |
| [laya-ko](https://huggingface.co/2nugu/laya-ko) | 0.590† | 0.240† | 0.820† | 0.600 | 0.530 | 0.545 | 0.570 | 0.556 | 0.561 |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 0.535 | 0.175 | 0.450 | 0.550 | 0.515 | 0.540 | 0.650 | 0.488 | 0.488 |

`†` `‡` `?`는 학습 데이터가 겹칠 수 있다는 표시예요([방법](#방법)). **Clean**은 표시가 없는 과제만의 평균이에요.

**보정(calibration), 전체 문항 합산**

| 모델 | 정확도 ±95% | Brier | ECE | p ≥ .9 오답 | ms/문항 | 런타임 |
|---|---|---|---|---|---|---|
| nari-4b | 0.806 ± 0.021 | 0.276 | 0.054 | 3.4% | 781 | torch MPS |
| coco-2.4-4b | 0.747 ± 0.023 | 0.351 | 0.118 | 5.6% | 519 | torch MPS |
| kev-4b | 0.741 ± 0.023 | 0.345 | 0.054 | 3.2% | 148 | kev.serve MLX |
| malkuth-2b | 0.722 ± 0.023 | 0.369 | 0.060 | 2.4% | 59 | kev.serve MLX |
| jev-style-0.8b-v3 | 0.604 ± 0.026 | 0.520 | 0.120 | 5.3% | 155 | torch MPS |
| kev-0.8b | 0.601 ± 0.026 | 0.510 | 0.055 | 2.5% | 28 | kev.serve MLX |
| laya-ko | 0.556 ± 0.026 | 0.621 | 0.232 | 10.0% | 17 | torch MPS |
| laya-multilingual | 0.488 ± 0.026 | 0.676 | 0.207 | 8.4% | 17 | torch MPS |

- 표시 없는 과제에서는 kev-4b가 가장 좋고, 보정도 가장 좋은 축에 들어요.
- nari-4b가 정확도는 가장 높지만, 7개 출처 모두의 다른 분할로 학습했어요.
- malkuth-2b는 확신 높은 오답이 가장 적고(2.4%), 59ms예요.
- 0.8B급에서는 kev-0.8b와 jev-style-0.8b-v3의 정확도가 비슷하고, kev-0.8b의 보정이 더 좋아요.

## 과제

과제마다 200문항이에요. 출처의 고정 커밋에서 `random.Random(0)`으로 뽑았고, 과제마다 고정된 한국어 지시문이 하나씩 있어요([`tasks.py`](src/kodb/tasks.py)).

| 과제 | 유형 | 출처 (분할) | 선택지 | 라이선스 |
|---|---|---|---|---|
| `unsmile_hate` | yes/no | smilegate-ai/kor_unsmile (valid), 클래스당 100 | 혐오·욕설 여부 | CC BY-NC-ND 4.0 |
| `kornli_xnli` | choice | kakaobrain/kor_nli (xnli test) | 함의 / 중립 / 모순 | CC BY-SA 4.0; XNLI(CC BY-NC 4.0) 번역 |
| `klue_nli` | choice | klue/klue nli (validation) | 함의 / 중립 / 모순 | CC BY-SA 4.0 |
| `klue_ynat` | choice | klue/klue ynat (validation) | 뉴스 분야 7개 | CC BY-SA 4.0 |
| `klue_sts` | score | klue/klue sts (validation) | 유사도 6단계 | CC BY-SA 4.0 |
| `kobest_boolq` | yes/no | skt/kobest_v1 boolq (test) | 예 / 아니오 | CC BY-SA 4.0 |
| `kobest_copa` | choice | skt/kobest_v1 copa (test) | 대안 2개 | CC BY-SA 4.0 |

저장소에는 데이터와 문항별 결과가 없어요. 두 출처가 재배포를 금지하고, 결과 행에 정답 레이블이 들어 있기 때문이에요. 데이터는 로컬에서 만들어 연구용으로만 쓰세요.

## 방법

- 모델마다 선택지별 확률을 하나씩 내고, 합이 1이 되도록 정규화해요. 온도는 각 모델이 배포한 값을 그대로 쓰고, 다시 맞추지 않아요.
- **정확도**: 확률이 가장 높은 선택지가 답이에요. 95% CI는 ±1.96·√(p(1−p)/n)이에요.
- **Brier**: Σₖ (pₖ − yₖ)². 0이 최고, 2가 최악이에요.
- **ECE**: 최고 확률 기준, 같은 폭 10개 구간이에요.
- **p ≥ .9 오답**: 최고 확률이 0.9 이상인데 틀린 문항의 비율이에요. 확신도 임계값으로는 거를 수 없는 오류예요.
- **ms/문항**: 워밍업 후 호출당 한 문항의 시간이에요. 같은 런타임끼리만 비교하세요.
- `klue_sts`는 단계가 정확히 맞아야 정답이에요(우연 수준 0.167).

점수는 이 프롬프트에 따라 달라져요. 모델 카드는 각 모델의 입력 형식을 써서 훨씬 높게 나올 수 있어요. 예를 들어 laya-ko는 KLUE-NLI 0.815를 보고하지만 여기서는 0.590이에요.

**오염 표시**는 2026-10-19에 읽은 모델 카드 기준이에요([`models.toml`](models.toml)).

- `†` 같은 데이터셋의 다른 분할로 학습 (zero-shot이 아니라 in-domain)
- `‡` 다른 출처의 같은 과제 한국어 데이터로 학습
- `?` 카드에 적힌 데이터가 이 과제를 포함할 수 있음

**하드웨어.** Apple M6 32GB에서 한 번에 한 모델씩 돌렸어요. laya, Jev-Style, nari, CoCo는 PyTorch MPS로, kev-4b, kev-0.8b, malkuth-2b는 [kev.serve](https://github.com/jaredpalmer/kev)(MLX bf16)에서 HTTP로 돌렸어요.

## 사용법

```bash
uv sync --extra models          # torch, transformers, laya
uv run kodb build               # data/<task>.jsonl 생성
uv run kodb run nari-4b all     # results/nari-4b/<task>.json 생성
uv run kodb score               # 표 출력
```

Hugging Face에서 받는 것(웨이트와 출처 데이터)은 모두 `weights/`에 저장돼요. kodb가 `HF_HOME`을 여기로 잡아요. 직접 설정한 `HF_HOME`이 있으면 그걸 따라요.

Kev 형식 모델은 kev.serve에서 돌아가요. `kodb run`은 `--url`(기본 `http://127.0.0.1:8081`)의 서버가 `models.toml`의 `repo@revision`을 서빙하면 그 서버를 써요. 서버가 없으면 그 주소에 직접 띄워요. 별도 uv 환경(Python 3.12, 결과를 낼 때 쓴 Kev 커밋)에서 실행하고, 로그는 `logs/`에 남기고, 끝나면 종료해요.

**모델 추가**: [`adapters.py`](src/kodb/adapters.py)에서 `Adapter`를 상속해 `decide(state, q)`가 `options(q)` 순서로 확률을 돌려주게 하고, `ADAPTERS`에 등록한 뒤 `models.toml`에 `repo`, `revision`, `trained_on`을 추가하세요.

## 라이선스

코드는 Apache-2.0이에요. 데이터셋과 모델은 각자의 라이선스를 따르고(위 표와 각 모델 카드), malkuth-2b 웨이트는 CC BY-NC 4.0이에요.

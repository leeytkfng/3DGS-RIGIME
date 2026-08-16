---
title: "ReSplat 논문 정리"
subtitle: "Learning Recurrent Gaussian Splatting (arXiv 2510.08575) — 우리 연구 관점의 독해"
date: "Sparse-view 3DGS Regime 연구 | 2026-08"
reader: "이용수 (Tier 1 정독)"
status: "완료 — 결정 사항은 experiments/docs/checklist/experiment_checklist.md 및 daily_report_2026-08-16.md 참고"
---

# 0. 한 줄 요약과 우리에게 갖는 의미

**논문 한 줄** — Feed-forward로 3D Gaussian을 한 번 예측한 뒤, **렌더링 오차를 피드백 신호로 삼아 학습된 네트워크가 Gaussian을 반복 갱신**한다. 경사하강 없이(gradient-free) 갱신하며, 4회 반복 후 포화한다.

**저자** — Haofei Xu 외 (ETH Zurich / Tübingen). **DepthSplat과 같은 저자**이며 DepthSplat 아키텍처를 초기화 모델로 사용한다. 즉 DepthSplat의 후속작이다.

**우리에게 갖는 의미 — 세 줄 요약**

1. **위협**: 그들의 실험 환경에서 ReSplat은 **모든 계산 예산에서 3DGS를 압도**한다(0.8초 결과가 3DGS 70초 결과보다 4.2\,dB 높음). 우리 논문이 "예산을 늘리면 optimization이 이긴다"만 주장한다면 반박 대상이 된다.
2. **기회**: 그러나 **그들 자신의 부록 Tab.~S1이 view 수가 늘수록 격차가 단조 축소됨을 보인다**(8-view 2.76\,dB → 32-view 0.44\,dB, LPIPS는 32-view에서 3DGS가 역전). 이는 경쟁 논문의 데이터가 우리 가설(역전 경계 존재)을 지지하는 사례다.
3. **결정적 공백**: 그들은 view 수를 바꿀 때 **촬영 범위와 test view를 함께 바꿨다.** 즉 view 수와 overlap이 교란되어 있으며, 본인들이 이를 각주로 인정한다. 우리 논문의 핵심 문제의식이 최신 SOTA에서도 해소되지 않았다는 직접 증거다.

---

# 1. 사전 지식 — 이것만 알면 읽힌다

## 1.1 이 논문이 서 있는 세 갈래

| 계열 | 방식 | 대표 |
|---|---|---|
| Per-scene optimization | 장면마다 수천 번 경사하강 | 3DGS |
| Single-pass feed-forward | 한 번의 순전파로 예측 | MVSplat, DepthSplat |
| **Recurrent refinement** | 예측 후 **학습된 갱신**을 반복 | **ReSplat**, Diff3R, ForeSplat |

세 번째가 첫 두 개의 중간 지대다. ReSplat은 이 계열에서 현재 가장 강한 결과를 낸다.

## 1.2 "Learning to Optimize"

최적화 문제를 경사하강으로 푸는 대신, **갱신 규칙 자체를 신경망으로 학습**하는 접근이다. 광학 흐름(RAFT), 스테레오 매칭(RAFT-Stereo), SLAM(DROID-SLAM) 계열에서 이미 성공한 패러다임이며, ReSplat은 이를 3DGS로 가져왔다.

핵심 차이를 수식으로 보면:

$$\text{경사하강:}\quad \theta_{t+1} = \theta_t - \eta\,\nabla_\theta L$$
$$\text{ReSplat:}\quad \theta_{t+1} = \theta_t + f_\phi(\theta_t,\ z_t,\ e_t)$$

여기서 $f_\phi$는 학습된 네트워크, $z_t$는 은닉 상태, $e_t$는 렌더링 오차다. **경사를 계산하지 않으므로 iteration당 비용이 표준 최적화보다 싸다.**

## 1.3 왜 "렌더링 오차"가 신호가 되는가

테스트 시점에도 **입력 view의 정답 이미지는 가지고 있다.** 따라서 현재 Gaussian으로 입력 view를 렌더링해 원본과 비교하면, 별도의 감독 없이 "지금 예측이 어디서 틀렸는지"를 알 수 있다. 이것이 이 논문의 핵심 착상이다.

$$\hat{E}^t = \underbrace{\{\hat F_i^t - F_i\}}_{\text{특징 공간 오차}} + \underbrace{\mathrm{proj}(\{\hat I_i^t - I_i\})}_{\text{픽셀 공간 오차}}$$

특징 공간 오차는 ImageNet 사전학습 ResNet-18의 초기 3단계 특징 차이이고, 픽셀 공간 오차는 단순 차분이다. Ablation에서 **렌더링 오차를 제거하면 1.9\,dB 하락**하여 이 신호가 성능의 핵심임을 보였다.

## 1.4 $16\times$ subsampling

기존 feed-forward는 **픽셀당 Gaussian 하나**를 만든다(per-pixel Gaussian). View가 많거나 해상도가 높으면 Gaussian이 수백만 개로 폭증한다. ReSplat은 depth를 1/4 해상도로 예측해 역투영하므로 Gaussian 수가 $N \times \frac{HW}{16}$, 즉 **$16\times$ 적다.**

이것이 recurrent 갱신을 가능하게 하는 전제다. 3D 공간에서 반복 갱신하려면 Gaussian 수가 적어야 계산이 감당된다. 손실 보상은 kNN attention과 global attention으로 3D 문맥을 집계해 메운다(ablation: kNN attention 제거 시 1.5\,dB 하락).

## 1.5 우리 실험과 직결되는 용어

- **Iteration** — ReSplat에서는 recurrent 갱신 횟수(0\,--\,4)를 뜻한다. 3DGS의 optimization step과 혼동하면 안 된다.
- **Recon. Time** — 초기 예측 + recurrent 갱신까지의 전체 재구성 시간. 우리 예산 축과 직접 비교 가능한 값이다.

---

# 2. 방법 구조

## 2.1 전체 흐름

```
입력 N장 (pose 있음)
   ↓ depth 예측 (1/4 해상도) → 역투영
3D 점군 + 이미지 특징  M = N × HW/16 개
   ↓ kNN attention + global attention × 6 블록
초기 Gaussian G⁰ + 은닉 상태 z⁰
   ↓ ─────── 반복 (T = 4) ───────┐
   │  현재 Gaussian으로 입력 view 렌더링
   │  렌더링 오차 계산 (픽셀 + 특징)
   │  global attention으로 오차를 3D에 전파
   │  kNN attention 블록 4개가 Δg, Δz 예측
   │  g ← g + Δg,  z ← z + Δz
   └───────────────────────────┘
최종 Gaussian
```

## 2.2 학습

**2단계 학습**이다. 1단계에서 초기 재구성 모델만 학습하고, 2단계에서 초기 모델을 **동결**한 뒤 recurrent 모델만 학습한다. 손실은 모든 iteration의 렌더링 손실에 지수 가중($\gamma = 0.9$)을 준다.

$$L_{2\text{nd}} = \sum_{t=0}^{T-1}\gamma^{T-1-t}\sum_{v=1}^{V}\ell_{\text{render}}(\hat I_v^t,\ I_v)$$

**학습 시 $T$를 1\,--\,4에서 무작위 추출**하므로, 추론 시 iteration 수를 자유롭게 바꿀 수 있다.

## 2.3 우리가 주목할 설계 결정 두 가지

**① 좌표계 선택이 성능을 크게 좌우한다.** COLMAP 기본 전역 좌표계 대신 **입력 view 중 공간적으로 중앙에 있는 view**를 기준으로 삼으면 0.9\,dB 개선된다(Tab.~6b: COLMAP 28.14 → middle view 29.07). 우리가 C1-b 렌더 등가성 gate에서 좌표계 매핑을 명시적으로 다룬 것과 같은 문제의식이다.

**② Densification이 없다.** 저자들이 한계로 직접 밝힌다 — 정제 과정에서 **Gaussian 개수를 고정**하며, adaptive pruning/densification 통합은 향후 과제로 남긴다.

---

# 3. 우리가 확인하려던 다섯 가지 — 답

## ① Recurrent iteration의 실제 wall-clock

**답: 있다. 그리고 매우 짧다.** DL3DV 8-view, $512\times960$ 기준(Tab.~1):

| Iteration | PSNR | 재구성 시간 |
|---|---|---|
| 0 (초기 예측만) | 26.21 | 0.311\,s |
| 1 | 27.15 | 0.437\,s |
| 2 | 27.51 | 0.563\,s |
| 3 | 27.65 | 0.689\,s |
| 4 | 27.70 | 0.816\,s |

**iteration당 약 0.126초이고, 전 과정이 1초 안에 끝난다.** 즉 우리 예산 축의 **가장 짧은 칸(1초) 안에 ReSplat의 모든 것이 들어간다.**

## ② "100$\times$ faster"의 실체

**답: 완전 수렴 대비 속도이며, 동일 예산 비교가 아니다.** 3DGS 4000 iteration(70초)과 ReSplat 4 iteration(0.816초)의 비교로, 실제 배율은 약 $86\times$다.

**다만 이 사실이 우리에게 유리하지 않다.** Tab.~1의 원 데이터로 동일 예산 비교를 직접 구성하면 다음과 같다.

| 예산 | 3DGS | ReSplat |
|---|---|---|
| 약 1초 | 결과 없음(1000 iter = 15초) | **27.70** |
| 15초 | 20.36 | 27.70 (0.8초에 도달) |
| 70초 | 23.46 | 27.70 |

**모든 예산에서 ReSplat이 우세하다.** 우리가 기대했던 "동일 예산에서는 다를 것"이라는 여지는, 최소한 그들의 실험 환경에서는 없다.

## ③ view 수 실험에서 overlap을 통제했는가

**답: 하지 않았다. 그리고 이것이 우리에게 가장 중요한 발견이다.**

부록 Tab.~S1의 각주가 이렇게 밝힌다 — view 수가 늘어남에 따라 장면 커버리지를 넓히기 위해 **표본 추출 영역을 확대했고, 그 결과 test view도 설정마다 달라진다.**

이 진술의 함의는 두 가지다.

1. **view 수와 overlap이 교란되어 있다.** 촬영 범위를 넓히면 view 간 겹침이 줄어들므로, "view 수 효과"와 "overlap 효과"가 분리되지 않는다.
2. **설정 간 수치를 직접 비교할 수 없다.** test view 자체가 다르므로 8-view와 32-view의 PSNR은 엄밀히는 다른 문제에 대한 점수다.

**이것이 본 연구의 문제의식이 최신 SOTA에서도 해소되지 않았다는 직접적 증거다.** Related Work에 그대로 인용할 수 있다.

## ④ 예산에 맞춰 중간에 끊을 수 있는가

**답: 가능하다.** 학습 시 $T$를 1\,--\,4에서 무작위 추출하므로 추론 시 iteration 수를 자유롭게 지정할 수 있으며, 저자들이 이를 "속도-정확도 절충"으로 명시한다.

**단, 4회 이후 포화한다.** 저자들이 한계로 밝힌 부분이다. 따라서 **0.8초를 넘는 예산은 ReSplat에게 무의미하다.** 이는 우리 예산 축에서 ReSplat이 갖는 고유한 특성이며, 그 자체로 보고할 만한 성질이다.

## ⑤ 학습 view 분포

**답: 체크포인트마다 다르다.**

| 체크포인트 | 학습 view 수 | 학습 해상도 |
|---|---|---|
| DL3DV | 8-view → 16-view (점진적) | $256\times448$ → $512\times960$ |
| RealEstate10K | **2-view** | $256\times256$ |

우리가 RE10K에서 8/12-view로 평가한다면 **MVSplat과 동일한 학습 분포 이탈 문제**가 발생한다. 이는 §3.2에서 다루는 교란요인의 또 하나의 사례다.

한편 Fig.~5b에서 저자들은 **recurrent 모델이 초기 모델보다 추가 view로부터 더 많은 이득을 얻는다**고 보고한다. 렌더링 오차 피드백이 분포 이탈을 일부 완화한다는 주장이다.

---

# 4. 위협 분석 — 정직하게

## 4.1 위협이 되는 지점

**ReSplat이 우리 논문의 결론을 약화시킬 수 있는 시나리오는 하나다.** 우리 결론이 "예산을 충분히 주면 per-scene optimization이 feed-forward를 이긴다"로 요약될 경우, "ReSplat은 0.8초 만에 그보다 잘한다"는 반박이 가능하다.

특히 다음 세 수치가 강력하다.

- DL3DV 8-view: ReSplat 27.70 (0.8초) vs 3DGS 23.46 (70초) — **4.2\,dB 차이**
- 깊이 정규화를 추가한 강화 3DGS와 비교해도 24.54 vs 27.70 — **3.2\,dB 차이**
- RE10K 2-view: ReSplat 29.75 vs MVSplat 26.39, DepthSplat 27.47

## 4.2 그러나 조건이 다르다

**해상도와 데이터셋이 우리와 다르다.** 그들의 주 실험은 DL3DV $512\times960$ 또는 $256\times448$이고, 우리 주 실험은 RE10K $256\times256$이다. 낮은 해상도와 실내 위주 장면에서는 per-scene 3DGS가 훨씬 빨리 수렴하고 더 높은 PSNR에 도달한다(우리 파일럿: 12-view/60초에서 Vanilla 3DGS 20.6\,dB, MVSplat 17.1\,dB). **동일한 결론이 나온다고 가정할 수 없다.**

**그들의 3DGS 설정도 우리와 다르다.** 그들은 3DGS를 4000 iteration에서 끊고 그 결과를 보고했는데, 근거는 "sparse 입력에서는 더 오래 최적화하면 과적합이 발생하므로 최선의 결과를 보고한다"이다.

> **이 문장은 우리 H2를 경쟁 논문이 독립적으로 확인해 준 사례다.** 인용 가치가 높다.

동시에 이는 **그들이 3DGS 쪽에는 사후에 최적 지점을 골라 준 것**이기도 하다. 우리 프로토콜은 예산 종료 시점을 쓰므로 규칙이 다르다는 점을 명시해야 한다.

## 4.3 그들의 데이터가 우리 가설을 지지한다

**부록 Tab.~S1이 이 논문에서 우리에게 가장 값진 부분이다.** DL3DV $256\times448$에서 view 수를 늘려가며 3DGS(4000 iter)와 ReSplat(4 iter)을 비교한 결과:

| View 수 | 3DGS PSNR | ReSplat PSNR | 격차 | LPIPS (3DGS / ReSplat) |
|---|---|---|---|---|
| 8 | 26.44 | 29.20 | $+2.76$ | 0.134 / **0.104** |
| 16 | 27.38 | 29.01 | $+1.63$ | 0.119 / **0.105** |
| 32 | 27.86 | 28.30 | $+0.44$ | **0.113** / 0.114 |

**격차가 단조 축소된다.** 그리고 32-view에서는 **LPIPS 지표가 이미 역전**되었다(3DGS 0.113 vs ReSplat 0.114). 저자들도 "32 view가 주어지면 optimization 기반 3DGS와의 품질 격차가 줄어든다"고 인정한다.

이는 우리 논문의 핵심 명제 — **우위는 조건의 함수이며 역전 경계가 존재한다** — 를 경쟁 논문의 데이터가 지지하는 것이다. 다만 그 경계는 우리가 예상한 8\,--\,12-view보다 훨씬 바깥에 있을 수 있다.

**또한 지표 간 불일치가 실제로 나타났다는 점**도 주목할 만하다. PSNR로는 ReSplat이 앞서지만 LPIPS로는 3DGS가 앞서는 조건이 존재한다. 우리가 프로토콜에 "지표 간 불일치를 별도 분석"을 넣어 둔 것이 정당화된다.

## 4.4 ReSplat에는 densification이 없다

저자들이 한계로 명시한다 — 정제 중 Gaussian 개수가 고정된다.

**함의**: 우리 H1(초기 geometry 오차가 densification을 통해 증폭된다)과 §5.2(관측 부족 Gaussian)는 **ReSplat에 적용되지 않는다.** 즉 ReSplat은 우리 실패 분석의 대상이 아니라, "그 실패 메커니즘을 구조적으로 회피한 설계"로 위치 지을 수 있다.

이는 오히려 우리 서사를 강화한다. densification이 sparse 조건에서 문제라면, densification을 없앤 방법이 강한 것은 우리 가설과 정합적이다.

---

# 5. 결론 — 어떻게 다룰 것인가

## 5.1 권장: Related Work 인용 + 조건부 소규모 비교

**① Related Work에 반드시 인용한다(필수).**

인용 근거가 세 개나 된다.

- view 수와 overlap을 분리하지 않았다는 자기 진술(Tab.~S1 각주)
- sparse 입력에서 오래 최적화하면 과적합한다는 관찰(우리 H2 지지)
- view 수가 늘수록 격차가 축소된다는 데이터(우리 V1/V2 지지)

**② 메인 Regime Map에는 넣지 않는다.**

사전 등록 비교군이 아니며, 패러다임 축이 다르고(recurrent refinement), 무엇보다 **넣으면 C1-a 전체를 재실행해야 한다.** 현재 실험 일정상 불가능하다.

**③ 별도 절로 소규모 확장 비교(조건부).**

코드와 사전학습 모델이 공개되어 있으므로($\texttt{github.com/cvg/resplat}$) 기술적으로는 가능하다. 다만 아래 조건을 먼저 확인해야 한다.

- 우리 seed 기반 view split을 강제 주입할 수 있는가 (FSGS에서 겪은 문제)
- RE10K 체크포인트가 2-view 학습이므로 8/12-view는 분포 밖 — MVSplat과 동일한 교란
- 우리 test view·해상도·pose 규칙과 정렬 가능한가

조건이 맞으면 **역전 경계 근처(4/8/12-view $\times$ 10/60초)만 소규모로** 돌린다. 전 격자는 불필요하다.

## 5.2 서술 수위 조정

이 논문을 읽은 뒤 우리 논문에서 피해야 할 표현이 하나 있다.

> **(피해야 할 표현)** "예산을 늘리면 per-scene optimization이 feed-forward를 역전한다"

이는 recurrent refinement 계열을 고려하지 않은 진술이다. 대신:

> "**단일 pass feed-forward**와 per-scene optimization 사이에서, 예산과 view 수에 따른 우위 역전이 관찰된다. 최근의 recurrent refinement 계열은 이 두 축의 중간에 위치하며, 별도의 검토가 필요하다."

## 5.3 새로 생긴 연구 질문

ReSplat의 **4회 포화**는 흥미로운 성질이다. 우리 예산 축 관점에서 보면:

- 1초 예산: recurrent refinement가 지배적
- 10초 이상: recurrent는 이미 포화, per-scene만 계속 개선
- 따라서 **"학습된 갱신은 빠르지만 천장이 있고, 경사 기반 최적화는 느리지만 천장이 높다"**는 가설이 성립한다

이 천장이 어디서 만나는지가 곧 recurrent 계열과 optimization 계열의 역전 경계이며, **본 연구의 자연스러운 확장**이다. 후속 연구 절에 명시할 가치가 있다.

---

# 6. 인용 가능한 문장 정리

논문 작성 시 바로 쓸 수 있도록 정리한다.

**Related Work — 경계 지대 연구**

> ReSplat은 렌더링 오차를 피드백 신호로 삼아 학습된 recurrent 갱신을 반복함으로써, 단일 pass feed-forward와 per-scene 최적화의 중간 지점을 차지한다. 그러나 이 계열의 연구들도 view 수와 overlap을 분리된 축으로 통제하지 않는다. 실제로 ReSplat은 입력 view 수를 늘릴 때 표본 추출 영역을 함께 확대하여 평가 view 자체가 설정마다 달라진다고 밝히고 있으며, 따라서 view 수 효과와 겹침 효과가 분리되지 않는다.

**서론 — 통념의 조건부성**

> 최근 연구는 sparse 입력에서 per-scene 최적화를 오래 수행하면 오히려 과적합이 발생함을 보고하고, 이 때문에 최적화를 조기에 중단한 결과를 보고한다. 이는 "시간을 더 주면 최적화가 낫다"는 통념이 무조건 성립하지 않음을 시사한다.

**결과 해석 — 격차 축소**

> 선행 연구는 8, 16, 32 입력 view에서 recurrent feed-forward와 per-scene 최적화를 비교하여, view 수가 증가함에 따라 두 방법의 PSNR 격차가 단조 감소하고(2.76 → 1.63 → 0.44\,dB) 32-view에서는 LPIPS 지표가 역전됨을 보고했다. 다만 해당 실험은 view 수와 촬영 범위를 함께 변화시켜 두 요인이 교란되어 있다.

**한계 — 비교 범위**

> 본 연구의 주 비교는 단일 pass feed-forward와 per-scene 최적화의 두 축이다. 학습된 recurrent 갱신을 사용하는 계열은 별도의 패러다임으로 간주하여 본 비교에서 제외하였으며, 이에 대한 통제된 비교는 후속 과제로 남긴다.

---

# 부록. 핵심 수치 요약

**Tab.~1 (DL3DV, 8-view, $512\times960$)**

| 방법 | 계열 | Iter | PSNR | Gaussian 수 | 재구성 시간 |
|---|---|---|---|---|---|
| 3DGS | Optimization | 4000 | 23.46 | 359K | 70\,s |
| MVSplat | Feed-forward | 0 | 22.49 | 3932K | 0.129\,s |
| DepthSplat | Feed-forward | 0 | 24.17 | 3932K | 0.190\,s |
| ReSplat | Recurrent FF | 4 | **27.70** | **246K** | 0.816\,s |

**Tab.~4 (RE10K, 2-view, $256\times256$)** — ReSplat 29.75 / DepthSplat 27.47 / MVSplat 26.39 / pixelSplat 25.89

**Tab.~S4 (깊이 정규화 3DGS 비교)** — 3DGS 23.46(70\,s) → 깊이 손실 추가 24.54(75\,s) vs ReSplat 27.70(0.8\,s)

**모델 규모** — ReSplat-Base 223M(초기화 209M + recurrent 14M). recurrent 부분은 15M에 불과하나, 단일 pass 559M 모델보다 좋은 결과를 낸다.

**학습 자원** — 16$\times$ GH200 GPU, 단계당 80K step. **재현 불가 규모이므로 공개 체크포인트 사용이 유일한 선택이다.**

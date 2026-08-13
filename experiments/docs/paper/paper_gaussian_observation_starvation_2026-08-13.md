# Per-Gaussian Observation Starvation — view 수와 densification gradient의 관계, 2026-08-13

## 이 분석이 논문 어디에 들어가는가

C1-a Regime Map에서 sparse-view일수록 Vanilla3DGS(optimization)가 불리해지는 이유를 지금까지는 "관측 데이터가 적다"는 추상적 수준으로만 설명했다. 이 분석은 그 이유를 **gsplat의 실제 densification 코드 경로까지 추적해 정량화**한다 — "데이터가 적다"가 구체적으로 "Gaussian 예산 중 몇 %가 어떤 학습 window에서도 유의미한 gradient 신호를 못 받는가"로 번역된다.

**직접 연결되는 것**: H2/H3(초기화·refinement 관련 가설), §6(설계 과정에서 교정한 오류 — v1 가설이 기각되고 v2로 재설계된 과정 포함), DTU floater/free-space-opacity 지표(같은 현상의 다른 측정치일 가능성), C1-a의 view-count 축 자체.

---

## 1. 배경 — gsplat 코드에서 확인한 densification 판단 경로

`gsplat/strategy/default.py::DefaultStrategy`(원본, 우리가 수정하지 않음). 체인:

$$\frac{\partial L}{\partial \mu_{2D}} \;\to\; \|\nabla_{\mu_{2D}} L\| \;\to\; \text{accumulation/average} \;\to\; \tau_{\text{pos}} \;\to\; \text{densification candidate}$$

핵심 두 함수:

```python
# _update_state() — 매 step(=1개 view render)마다 호출
grads = info["means2d"].grad.clone()          # dL/dmu_2D, 이번 view에서 렌더된 모든 Gaussian
sel = (info["radii"] > 0.0).all(dim=-1)        # 이번 view에서 실제로 "보인" Gaussian만
state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))   # norm을 SUM으로 누적
state["count"].index_add_(0, gs_ids, ones)     # "보인 횟수" 누적

# _grow_gs() — refine_every(=100) step마다 한 번
grads = state["grad2d"] / count.clamp_min(1)   # 여기서 처음 평균이 나옴
is_grad_high = grads > self.grow_grad2d        # tau_pos = 0.0002
...
state["grad2d"].zero_(); state["count"].zero_()  # 그리고 즉시 리셋(다음 100-step은 새로 누적)
```

`count[g]`는 "그 Gaussian이 현재 100-step window에서 몇 번 관측됐는가"라는 effective observation 수다. 원 가설: sparse-view일수록 `count[g]`가 작아지고, 적은 표본으로 낸 평균(`grad2d[g]/count[g]`)이 노이즈에 취약해져 τ 판정을 왜곡할 것이다.

---

## 2. v1 — null 결과 (기록해둠, 설계 자체가 잘못됐던 사례)

densification(grow/prune)을 끄고 `_update_state()`만 500 step 동안 리셋 없이 누적해 count 분포를 봤다. 결과: 살아남은 모든 Gaussian이 `count≈500`(=n_steps)에 거의 포화 — view 수(4/8/12)와 무관하게 `count≤2` 비율이 정확히 0%.

**원인**: COLMAP triangulation으로 만든 초기 점은 정의상 ≥2개 view에서 correspondence가 맞아야 존재한다(survivorship bias) — "가끔만 보이는 점"은 애초에 초기 집합에 들어올 자격이 없다. 게다가 densification을 꺼서 낮은 count가 나올 진짜 후보군(densification이 새로 만드는, 아직 검증 안 된 점)을 실험에서 배제해버렸다. **가설이 아니라 측정 위치가 틀렸다** — v2로 재설계.

---

## 3. v2 — densification을 켜고 실제 판단 시점에 스냅샷

`step_post_backward()`를 그대로 쓰지 않고, 그 내부 로직(`_update_state`→(refine 조건 만족 시) **스냅샷**→`_grow_gs`/`_prune_gs`→reset)을 직접 오케스트레이션해 gsplat이 실제로 τ와 비교하는 바로 그 순간의 `count`/`grad2d` 배열을 매 100-step window마다 기록했다. `_update_state`/`_grow_gs`/`_prune_gs` 함수 자체는 gsplat 원본 그대로 재사용.

**데이터**: RE10K main subset 3개 scene(`0588138dfec165a1`, `0c52996355b23d76`, `1214f2a11a9fc1ed`) × view_count{2,4,8,12}, 2000 step(=`refine_start_iter=500` 이후 15개 window), seed=0. 초기화는 다른 러너와 동일한 규칙(COLMAP sparse triangulation, 실패 시 `MIN_SFM_POINTS=200` 기준 random-sphere fallback).

### 3.1 결과 — 마지막 window(step 2000) 기준

| scene | 2-view | 4-view | 8-view | 12-view | init 경로 |
|---|---:|---:|---:|---:|---|
| 0588138... | 69.4% | 49.7% | 0.8% | 0.5% | fallback → fallback → colmap → colmap |
| 0c52996... | 29.6% | 10.3% | 0.5% | 0.4% | fallback → fallback → colmap → colmap |
| **1214f2a...** | **55.3%** | **32.3%** | **8.3%** | **1.3%** | **fallback (4개 조건 전부 동일)** |
| pooled 평균 | **51.5%** | **30.8%** | **3.2%** | **0.7%** | |

(수치는 `count[g] ≤ 2`인 Gaussian의 비율 — 해당 100-step window 동안 2번 이하로만 관측된 Gaussian.)

**초기화 방식이라는 교란요인을 제거한 가장 깨끗한 증거는 `1214f2a11a9fc1ed`행이다** — 이 scene은 2/4/8/12-view 전부 triangulation이 `MIN_SFM_POINTS`(200)에 못 미쳐 동일하게 random-sphere fallback을 썼다. 즉 초기화 메커니즘을 완전히 고정한 채로도 55%→32%→8%→1%의 단조 감소가 나온다 — view 수 자체의 효과다.

### 3.2 원래 가설의 운명 — 부분 기각, 다른 발견으로 대체

`count≤2` 집단이 τ를 초과하는 비율은 **모든 조건에서 사실상 0%**(0%~0.003%)였다. 즉 "관측이 적어 평균이 노이즈로 부풀어 densification을 잘못 유발한다"는 원래 예측은 **틀렸다** — 관측이 적으면 분자(누적 gradient)도 같이 작으므로 비율 자체가 부풀지 않는다. 반면 `count>10` 집단은 조건에 따라 2.5%~19.7%가 τ를 넘어 densification 후보가 됐다.

**대신 나온 발견**: 원래 예측한 "노이즈로 인한 오탐"이 아니라, sparse-view에서는 **Gaussian 예산의 상당 부분이 어떤 window에서도 거의 관측되지 않는 채로 방치된다**는 것. 이런 Gaussian은 gradient 신호가 거의 없어 densification 후보에도 못 들고(τ를 넘지 않으므로), 그렇다고 딱히 문제를 일으키지도 않는 채 그냥 "죽은 예산"으로 남는다. 2-view에서 random-sphere fallback으로 흩뿌린 100,000개 중 절반 이상이 소수의 카메라 절두체(frustum) 안에 충분히 자주 들어오지 않는다는 뜻 — 무작위 초기화 자체가 sparse-view에서 손해를 구조적으로 키운다.

---

## 4. 논문에 쓸 수 있는 문장 (초안)

> We trace the per-Gaussian densification criterion in gsplat's `DefaultStrategy` — a running mean of 2D positional gradient norm compared against a fixed threshold ($\tau=2\times10^{-4}$) every 100 iterations — and find that the originally hypothesized failure mode (few observations inflating the gradient-average estimate into false-positive densification) does not occur: Gaussians observed in $\le 2$ of the last 100 training steps essentially never exceed $\tau$ (0.00–0.003% across all conditions), since both the numerator and the observation count shrink together. Instead we find a different, larger effect: the *fraction* of such severely under-observed Gaussians is strongly and monotonically dependent on the number of input views, from 51.5% at 2 views down to 0.7% at 12 views (pooled over 3 RE10K scenes), with the cleanest evidence coming from a scene where all four view-count conditions shared the same (COLMAP-triangulation-failed, random-sphere-fallback) initialization mechanism, isolating view count as the driver (55.3% → 32.3% → 8.3% → 1.3%). We interpret this as observation starvation: under sparse views, a substantial fraction of the Gaussian budget receives too little gradient signal in any refinement window to ever become a densification candidate, and — because random-sphere initialization is a direct consequence of triangulation failure at low view counts — a large share of this starved population never had a principled reason to be visible from the available cameras in the first place. This offers a code-traceable, quantitative mechanism for why per-scene optimization underperforms feed-forward methods at low view counts beyond the general notion of "less data": part of the optimization's Gaussian capacity is effectively wasted rather than merely under-trained.

**주의**: (1) v1(카운트 리셋 없이 관측)은 null 결과였고, 이건 측정 설계(triangulation의 survivorship bias + densification 비활성화)가 틀렸기 때문이지 원래 질문이 무의미해서가 아니다 — v2로 재설계해서 나온 결과다. (2) pilot 규모(scene 3개)라 통계적으로 확정된 결론은 아니다 — scene을 늘려 재현되는지 확인 필요. (3) `count>10` 집단의 τ 초과 비율(2.5%~19.7%)은 scene별 변동이 커서 view_count에 대한 명확한 단조 패턴이 없다 — 이 부분은 별도 조사가 필요하면 추가.

## 5. 재현 방법

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate ps3
python3 experiments/scripts/analysis/gaussian_gradient_accumulation_probe.py \
  --scenes 0588138dfec165a1 0c52996355b23d76 1214f2a11a9fc1ed \
  --view-counts 2 4 8 12 --n-steps 2000
```

원본 데이터: `experiments/outputs/gaussian_grad_probe/summary_v2.json`(scene×view_count별 15개 window 전체 snapshot 포함, 위 표는 마지막 window만 발췌).

## 6. 남은 일

- [ ] scene 수를 늘려(pilot 3개 → main subset 규모) 51.5%/30.8%/3.2%/0.7% 패턴이 재현되는지 확인
- [ ] overlap 축(고/저)에서도 같은 패턴이 나오는지 — view 수뿐 아니라 co-visibility 자체가 독립적으로 영향을 주는지
- [ ] "죽은 예산" 비율과 최종 test PSNR/floater 비율의 상관관계 — 이 메커니즘이 실제 렌더링 품질 손실과 얼마나 직결되는지
- [ ] §6(설계 과정에서 교정한 오류)에 v1→v2 재설계 과정 자체도 기록할지 결정(다른 confound 분석 문서와 같은 패턴)

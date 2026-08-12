# Geometry Uncertainty ↔ Overlap 교란(confounding) 분석 — 2026-08-12

## 이 분석이 논문 어디에 들어가는가

`overall.md` §5.3(overlap 계산식 정의)의 근거 자료다. §5.3은 co-visibility overlap을 sparse-view 실험의 주 지표로 쓰기로 결정했는데, 그 결정이 타당한지 — "overlap이 실제로 기하학적 불확실성과 관련 있는 지표인가"를 검증하는 게 이 분석의 원래 목적이었다(`geometry_uncertainty_figure.py` 최초 작성 의도). 그 과정에서 예상 밖의 결과(부호가 반대로 나옴)가 나왔고, 오늘 A-1(가우스-뉴턴) 학습이 끝난 뒤 그 원인을 끝까지 추적했다.

**직접 연결되는 것**: H1(초기 geometry 오차가 클수록, 특히 low-overlap 조건에서 품질이 나빠진다), §5.3의 overlap 정의, C2(기하 불확실성 개입 실험)의 이론적 배경.

---

## 1. 배경 수식 — 어디서 왔고 우리가 뭘 더했는가

**기존 것(우리가 유도하지 않음)**: 두 view로 3D 점 $X$를 삼각측량하는 문제는 재투영 오차를 최소화하는 비선형최소제곱이다.

$$r(X) = \begin{pmatrix} u_i - \pi_i(X) \\ v_i - \pi_i(X) \\ u_j - \pi_j(X) \\ v_j - \pi_j(X) \end{pmatrix}, \quad J = \frac{\partial r}{\partial X} \in \mathbb{R}^{4\times 3}$$

가우스-뉴턴에서 이 $J$는 원래 "다음 스텝을 어디로 갈까"를 위한 것이지만, $r$을 실제 관측 노이즈 $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$로 재해석하면 MLE의 점근 공분산(=Cramér-Rao bound)이 된다:

$$\text{Cov}(\hat{X}) \approx \sigma^2 (J^\top J)^{-1}$$

이건 통계학의 표준 결과이고, multi-view geometry에서 삼각측량 공분산에 이렇게 적용하는 것도 고전적이다(Hartley & Zisserman류 교과서 내용). **우리가 새로 만든 부분은 이 표준 공식을 실제 COLMAP 결과에서 직접 계산해 sparse-view 3DGS 연구의 overlap 지표를 검증하는 진단 도구로 쓴 것**이다.

`geometry_uncertainty_figure.py::two_view_depth_uncertainty()` 구현:

$$J_{\text{view}} = \underbrace{\begin{pmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{pmatrix}}_{\partial(u,v)/\partial X_{\text{cam}}} \cdot R, \qquad J = \begin{pmatrix} J_{\text{view }i} \\ J_{\text{view }j} \end{pmatrix}$$

3×3 공분산 $\sigma^2(J^\top J)^{-1}$을 view $i$의 시선 방향 $\hat{d}$로 투영해 depth 방향 불확실성 하나로 압축한다: $\sqrt{\hat{d}^\top \text{Cov}(\hat X) \hat{d}}$.

---

## 2. 실측 — DTU scan1, 861개 view pair

`experiments/outputs_dense_sanity`의 42-view dense reconstruction에서 모든 view pair(861쌍)에 대해 baseline(카메라 중심 간 거리), overlap($O_{ij}=2|P_i\cap P_j|/(|P_i|+|P_j|)$), 위 공식의 depth uncertainty를 계산했다(`pairwise_geometry.csv`).

### 2.1 처음 나온 "이상한" 결과

$$\text{corr}(O_{ij}, \log \hat\sigma_{\text{depth}}) = +0.952$$

**쉽게**: overlap이 높을수록(사진 두 장이 많이 겹칠수록) 오히려 depth 불확실성이 **더 크게** 나왔다. H1이 기대하는 "overlap 높으면 불확실성 낮다"(음의 상관)와 정반대.

### 2.2 첫 가설 — baseline 교란, 선형으로 통제해봄

당시(A-1 완료 전) 세운 가설: baseline이 둘 다를 같은 방향으로 끌고 간다.
- $\text{corr}(\text{baseline}, O_{ij}) = -0.874$ (baseline↑ → overlap↓, 예상대로 — 두 힘이 반대 방향)
- $\text{corr}(\text{baseline}, \log\hat\sigma) = -0.951$ (baseline↑ → uncertainty↓, 예상대로 — 삼각측량 각도가 넓어지니까)

baseline을 **선형으로** 통제한 partial correlation:

$$r_{O,\log\hat\sigma \,|\, b} = \frac{r_{O,\log\hat\sigma} - r_{O,b}\, r_{\log\hat\sigma,b}}{\sqrt{(1-r_{O,b}^2)(1-r_{\log\hat\sigma,b}^2)}} = +0.801$$

**문제**: 0.952에서 0.801로 조금만 줄고 여전히 강한 양의 상관이 남았다 — "순수한 baseline 교란"이라는 가설이 맞다면 0 근처로 꺼져야 하는데 안 꺼짐.

### 2.3 두 번째 가설 — "평균 내는 점 집합이 다르다"(shared_points selection effect), 기각됨

overlap이 높은 pair는 공유점(`shared_points`) 수가 많고(≈overlap 공식 자체가 이 수에서 나옴), 그 안에 약한 점(먼 배경, 비스듬한 각도)까지 다 섞여 평균을 끌어올릴 거라는 가설을 세웠다.

- $\text{corr}(\log(\text{shared\_points}), \log\hat\sigma \,|\, \text{baseline}) = +0.071$ — 거의 0, shared_points는 무죄.
- baseline과 shared_points를 **같이** 통제해도 $r_{O,\log\hat\sigma} = +0.818$ — 오히려 안 줄어듦.

**결론: 기각.** 좋은 시도였지만 데이터가 지지하지 않았다 — 정직하게 기록.

### 2.4 진짜 원인 — 선형 통제가 틀렸다(함수형 오지정)

$$\text{corr}(\log(\text{baseline}), \log\hat\sigma) = -0.989 \quad (\text{선형 버전: } -0.951)$$

로그-로그 공간에서 거의 완벽한 직선이 됐다 — 즉 $\hat\sigma_{\text{depth}} \propto \text{baseline}^{-k}$ 꼴의 **거듭제곱(power-law) 관계**이지 선형 관계가 아니었다. 처음에 baseline을 원래 스케일로(선형으로) 통제한 게 잘못이었다: 곡선 관계를 직선으로 통제하면 baseline의 진짜 영향을 다 못 지우고 일부가 "설명 안 된 잔차"로 새어나가 마치 새로운 현상이 있는 것처럼 보인다.

**log(baseline)으로 다시 통제**:

$$r_{O,\log\hat\sigma \,|\, \log b} = +0.301$$

$$0.952 \;\longrightarrow\; 0.801\ (\text{선형 통제, 불완전}) \;\longrightarrow\; 0.301\ (\text{log 통제, 정답에 근접})$$

**결론**: 애초 가설(baseline 교란)이 **맞았다**. 통계 도구를 잘못 써서(선형 vs log) 가짜 미스터리를 만든 것이었다. 남은 +0.301도 완전히 0은 아니라 추가 요인의 여지는 있지만, raw +0.952의 대부분은 baseline 교란으로 설명된다.

---

## 3. 논문에 쓸 수 있는 문장 (초안)

> The raw correlation between pairwise co-visibility overlap and Gauss-Newton-derived depth uncertainty is strongly *positive* ($r=0.95$), which appears to contradict the expectation that higher overlap implies lower geometric uncertainty. We find this is a confounding effect of camera baseline: baseline drives overlap and depth uncertainty in the same direction (wider baseline reduces both), and because depth uncertainty follows a power-law relationship with baseline rather than a linear one, naively partialling out baseline on its raw scale leaves substantial residual confounding ($r=0.80$). Controlling for $\log(\text{baseline})$ — matching the underlying power-law geometry of stereo triangulation — reduces the partial correlation to $r=0.30$, confirming that baseline, not overlap per se, is the primary geometric driver of depth uncertainty in this dataset. We therefore treat baseline and overlap as related but non-interchangeable axes: overlap remains the appropriate primary metric for the sparse-view regime study (it measures actual co-visibility, which baseline alone does not capture on non-orbital or irregular camera configurations), but claims about *why* low-overlap conditions are harder should be attributed to the underlying baseline/parallax geometry, not to overlap as an independent causal factor.

**주의**: 이 문장은 DTU의 orbital rig(카메라가 중심 물체를 둘러싼 배치) 데이터로 나온 결과다. RE10K/DL3DV처럼 카메라가 일직선 경로를 따라가는 데이터셋에서는 baseline-overlap 관계가 다를 수 있어 일반화 전에 확인이 필요하다(아래 "남은 일" 참고).

## 4. 재현 방법

```bash
PATH="/opt/conda/envs/ps3/bin:$PATH" /opt/conda/envs/ps3/bin/python3 \
  experiments/scripts/analysis/geometry_uncertainty_figure.py
```

`print_confound_analysis()`가 raw / 선형 통제 / log 통제 세 값을 매번 출력한다(2026-08-12부로 스크립트에 통합, 이 문서의 §2.2~2.4 재현).

## 5. 남은 일

- [ ] RE10K/DL3DV(경로형 카메라 배치)에서도 같은 baseline-overlap-uncertainty 관계가 성립하는지 확인 — DTU(orbital)와 다를 수 있음
- [ ] 남은 잔차(+0.301)의 정체 — 완전한 log-log도 아직 근사이므로 더 나은 함수형(예: 실제 parallax angle을 baseline 대신 직접 계산)이 있는지 검토
- [ ] 이 결과를 §5.3/§6(설계 과정에서 교정한 오류) 본문에 반영

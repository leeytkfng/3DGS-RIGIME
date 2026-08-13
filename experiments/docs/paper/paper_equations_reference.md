# 논문에서 쓰는 수식 전체 정리 — 2026-08-13

지금까지 코드/문서 전체에 흩어져 있던 수식을 한 곳에 모았다. 각 수식은 **목적 → 어디서/어떻게 쓰이는가 → 수식 → 쉬운 해석** 순서로 적는다. 실제 구현 위치(파일:함수)도 같이 적어서, 수식과 코드가 항상 대응되게 했다. 신뢰구간(confidence interval)은 6번에 따로 모았다.

---

## 1. Co-visibility Overlap — sparse-view 정도를 재는 자

**목적**: "view 몇 장이 서로 얼마나 겹치는가"를 하나의 숫자로 만든다. 이 연구 전체의 x축(가로축)이 되는 핵심 지표.

**어디서 쓰이나**: `protocol_utils.py::compute_pairwise_overlaps()`, §5.3. RE10K/DTU/DL3DV 전부에서 view pair마다 이 값을 계산해서 low/high overlap bucket을 나눈다.

$$
O_{ij} = \frac{2\,|P_i \cap P_j|}{|P_i| + |P_j|}
$$

- $P_i$ = view $i$가 관측한 SfM(Structure-from-Motion) 3D 점들의 집합
- 분자에 2를 곱하는 이유: 두 view가 완전히 같은 점을 본다면($P_i=P_j$) 값이 1이 되게 하려고(Dice 계수와 같은 형태)

**쉬운 해석**: 두 사진이 찍은 공통 배경(=서로 매칭된 3D 점)이 많을수록 1에 가까워지고, 아예 겹치는 게 없으면 0. **중요한 규칙**: COLMAP 매칭이 아예 실패한 pair도 $O_{ij}=0$으로 그냥 포함시킨다(제외하지 않음) — 실패 자체가 "여기 겹침이 거의 없다"는 강한 증거이기 때문에, 빼버리면 오히려 어려운 장면의 overlap이 과대평가된다.

**부속 지표** (`aggregate_overlap()`): 전체 pair의 평균(`mean_overlap`, 주 지표), 하위 25% 분위수(`q25_overlap`, 꼬리 확인용). Median은 안 씀 — 0이 많은 sparse 조건에서 median이 0값들을 다 삼켜버려 신호를 지우기 때문.

---

## 2. Gauss-Newton 삼각측량 & 깊이 불확실도 — "이 3D 점, 얼마나 믿을 만한가"

**목적**: 카메라 기하학만으로 "이 3D 점의 위치가 얼마나 불확실한가"를 계산한다. overlap이 왜 중요한 지표인지를 이론적으로 뒷받침하는 근거.

**어디서 쓰이나**: `geometry_uncertainty_figure.py::two_view_depth_uncertainty()`, `paper_gauss_newton_notation.md`.

### 2.1 문제 설정과 업데이트 식

두 개 이상의 view에서 같은 3D 점 $\beta=(X,Y,Z)$를 봤을 때, 각 view가 "이 점이 화면의 여기 있어야 한다"고 예측한 위치와 실제 관측된 위치의 차이(재투영 오차) $r(\beta)$를 최소화한다.

$$
S(\beta) = \|r(\beta)\|^2, \qquad \Delta\beta = -(J^\top J)^{-1} J^\top r
$$

- $J = \partial r/\partial\beta$: 점 위치가 조금 바뀌면 화면상 위치가 얼마나 바뀌는지(야코비안)
- 이건 COLMAP이 3D 점 하나하나를 삼각측량할 때 실제로 반복하는 계산

### 2.2 그 점의 신뢰도(공분산) — 이 문서의 핵심 수식

$r$을 "최적화가 아직 덜 된 오차"가 아니라 "카메라/센서가 원래 갖고 있는 관측 노이즈 $\varepsilon\sim\mathcal N(0,\sigma^2 I)$"로 다시 해석하면, 위와 똑같은 식에서 다른 것이 나온다:

$$
\boxed{\operatorname{Cov}(\hat\beta) \approx \sigma^2 (J^\top J)^{-1}}
$$

**쉬운 해석**: 2.1의 "다음 스텝 어디로 갈까" 공식과 겉보기엔 똑같은데, $r$을 무엇으로 보느냐만 바꾸면 "이 추정치를 얼마나 믿어도 되나"(공분산=불확실도)가 나온다. 통계학의 표준 결과(MLE의 점근 공분산, Cramér-Rao bound)를 그대로 가져온 것 — 우리가 새로 유도한 게 아니라 기존 정리다.

### 2.3 실제 구현 (multi-view, depth 방향만)

$N$개 view의 카메라 파라미터가 주어지면:

$$
J_k = \underbrace{\begin{bmatrix} f_x/z_k & 0 & -f_x x_k/z_k^2 \\ 0 & f_y/z_k & -f_y y_k/z_k^2 \end{bmatrix}}_{\partial(u,v)/\partial X_{cam}} R_k, \qquad J=\begin{bmatrix}J_1\\\vdots\\J_N\end{bmatrix} \in \mathbb R^{2N\times 3}
$$

3×3 공분산 $\sigma^2(J^\top J)^{-1}$을 view의 시선 방향 단위벡터 $\hat d$로 투영해서 "깊이 방향 표준편차" 하나로 압축: $\sqrt{\hat d^\top \operatorname{Cov}(\hat\beta)\,\hat d}$.

**실측 근거(σ)**: 이론값이 아니라 COLMAP의 `mean_reprojection_error`를 그대로 씀(실측 0.58~0.61px, DTU scan1 기준) — "우리가 측정한 σ"라고 논문에 인용 가능.

---

## 3. Baseline-Overlap-Uncertainty 삼각관계 — partial correlation

**목적**: "overlap이 높을수록 depth 불확실도가 낮다"는 raw 상관관계가 처음에 반대 부호로 나왔던 걸 baseline(카메라 간 거리)의 교란효과로 설명하기 위해 썼다.

**어디서 쓰이나**: `paper_geometry_confound_analysis_2026-08-12.md`, `geometry_uncertainty_figure.py::print_confound_analysis()`.

$$
r_{XY\cdot Z} = \frac{r_{XY} - r_{XZ}\,r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}
$$

**쉬운 해석**: $X$(overlap)와 $Y$(불확실도) 사이의 "진짜" 관계를 보고 싶은데, 둘 다 $Z$(baseline)의 영향을 받는다면 순수한 상관을 알 수 없다. $Z$를 통해 설명되는 부분을 수학적으로 빼고 남은 상관이 $r_{XY\cdot Z}$. **함정 주의(우리가 직접 겪음)**: $Z$와 $X$/$Y$의 관계가 곡선(비선형)인데 이 공식은 직선 관계만 제거한다 — baseline을 원래 스케일로 쓰면 곡률이 안 지워져서 가짜 잔차가 남는다. $\log(\text{baseline})$으로 바꾸니(진짜 관계가 거듭제곱 형태라서) 훨씬 깨끗하게 지워졌다.

---

## 4. 승패 판정 — Feed-forward vs Optimization, 언제 누가 이기나

**목적**: 두 패러다임을 비교할 때 "미세한 차이도 다 승패로 셀 것인가"를 막기 위한 규칙. C1-a Regime Map의 핵심 판정 로직.

**어디서 쓰이나**: `protocol_utils.py::classify_delta()`, `compute_tau()`.

$$
\Delta = \mathrm{PSNR}_{FF} - \mathrm{PSNR}_{OPT}, \qquad
\text{label} = \begin{cases} \text{feedforward\_win} & \Delta > \tau \\ \text{optimization\_win} & \Delta < -\tau \\ \text{tie} & |\Delta| \le \tau \end{cases}
$$

$$
\tau = \max(\text{seed 변동성},\ \text{실용적 최소 차이}=0.5\text{dB})
$$

**쉬운 해석**: PSNR이 0.01dB만 높아도 "이겼다"고 하면 의미 없는 승부가 남발된다. 그래서 "이 정도는 이겼다고 부를 만하다"는 문턱값 $\tau$를 두는데, 이걸 seed 변동성만으로 정하면 변동성이 큰 조건일수록 tie 구간이 저절로 넓어지는 자기참조 문제가 생긴다 — 그래서 "seed 변동성"과 "실용적으로 의미 있는 최소 차이(문헌 관행 0.3~0.5dB)" 둘 중 **큰 쪽**을 쓴다.

---

## 5. 계산 예산 시점 결과 선택 — 테스트 누출 방지 규칙

**목적**: "여러 체크포인트 중 결과가 제일 좋은 걸 골라 쓰는" 사후 선택(test leakage)을 원천 차단.

**어디서 쓰이나**: `protocol_utils.py::budget_checkpoint()`, `oracle_checkpoint()`, §5.7.

$$
t_B = \max\{t : t \le B\}, \qquad Q_m(B) = Q_m(t_B)
$$

**쉬운 해석**: 예산 $B$초가 주어지면, 그 예산을 넘지 않는 마지막 체크포인트($t_B$) 딱 하나만 메인 결과로 쓴다. "봤더니 이 체크포인트가 제일 좋더라"는 식으로 나중에 고르는 건 절대 안 되고(그건 `oracle_checkpoint`로 완전히 분리해서 진단용으로만 씀), Feed-forward의 추론 시간이 $B$보다 크면 그 예산 칸은 그냥 "결과 없음"으로 비워둔다.

---

## 6. 신뢰구간(Confidence Interval) — Scene Cluster Bootstrap

**목적**: 같은 장면(scene)에서 seed를 3번 돌린 걸 "독립적인 표본 3개"로 잘못 세면 신뢰구간이 실제보다 훨씬 좁게(과신) 나온다. 이걸 막고 진짜 불확실도를 반영한 신뢰구간을 만드는 게 목적.

**어디서 쓰이나**: `protocol_utils.py::scene_cluster_bootstrap_ci()`, §5.12(통계 분석 계획).

**절차** (부트스트랩 리샘플링):
1. scene별로 seed 3회 결과를 먼저 평균 낸다 → scene마다 값 하나(scene을 "독립 단위"로 취급, seed는 그 안의 반복측정일 뿐)
2. scene 목록에서 **복원추출(같은 scene이 여러 번 뽑힐 수 있음)**로 scene 개수만큼 다시 뽑고, 그 표본의 평균을 계산 — 이걸 2000번 반복해서 2000개의 "가상의 평균"을 만든다
3. 이 2000개 값을 정렬해서 하위 2.5%, 상위 97.5% 지점을 잘라내면 95% 신뢰구간

$$
\hat\mu_{\text{scene}} = \frac{1}{S}\sum_{s=1}^{S}\bar y_s, \qquad
\text{CI}_{95\%} = \big[\,\text{percentile}_{2.5}(\{\hat\mu^{(b)}\}_{b=1}^{2000}),\ \ \text{percentile}_{97.5}(\{\hat\mu^{(b)}\}_{b=1}^{2000})\,\big]
$$

- $S$ = scene 개수(독립 단위), $\bar y_s$ = scene $s$ 내 seed 평균, $\hat\mu^{(b)}$ = $b$번째 부트스트랩 재표본의 평균

**쉬운 해석**: "이 실험을 scene 구성만 살짝 다르게 해서(같은 성질의 scene들 중에서 다시 뽑아서) 여러 번 반복했다면 평균이 어느 범위 안에서 움직였을까"를 컴퓨터로 흉내 낸 것. 진짜 반복 실험은 불가능하니(scene을 또 모을 수 없으니) 갖고 있는 scene들에서 복원추출로 그걸 시뮬레이션한다. **표본 수는 run 개수(scene×seed)가 아니라 scene 개수** — 오늘(8/13) C1-a 파일럿의 경우 5 scene이 표본 수, 15(5scene×3seed)가 아니다.

**오늘 실측 사례(참고, 정식 CI 계산 전 단계)**: seed 3회의 scene별 평균 표준편차를 봤을 때(엄밀한 부트스트랩 CI는 아니고 단순 std), 2~4-view는 seed 표준편차 0.03~0.11dB(매우 안정), 8~12-view는 0.1~1.13dB로 커짐 — scene 5개로는 아직 case-by-case 변동이 꽤 있다는 뜻. scene 수를 20개로 늘려야 이 문서의 정식 부트스트랩 CI가 의미 있게 좁아진다.

### 6.1 다중 비교 보정 — Holm–Bonferroni

**목적**: view 수 × overlap × budget 조합마다 승패 검정을 반복하면, 그중 일부는 우연히 유의하게 나올 확률(false positive)이 누적된다. 이걸 통제.

**어디서 쓰이나**: `protocol_utils.py::holm_adjust()`.

$$
p_{(i)}^{\text{adj}} = \max_{j\le i} \big\{(n-j+1)\cdot p_{(j)}\big\}, \quad p_{(1)}\le p_{(2)}\le\cdots\le p_{(n)}
$$

**쉬운 해석**: p-value를 작은 것부터 정렬해서, 각각에 "몇 번째로 작은가"에 비례하는 배수를 곱해 보정한다(가장 작은 p-value에 가장 큰 배수를 곱함). Bonferroni(모든 p-value에 $n$을 곱하는 방식)보다 덜 보수적이면서도 여전히 false positive를 통제하는 표준 방법.

---

## 7. C2 — Depth 불확실도 개입 (sensitivity analysis)

**목적**: 초기 depth 추정이 나쁠수록 재구성이 어떻게 나빠지는지, 통제된 크기의 노이즈를 인위적으로 주입해서 확인.

**어디서 쓰이나**: §5.9, `experiment_config.yaml`의 `c2` 섹션.

$$
\text{(a) 개별 노이즈: } d' = d\,(1+\varepsilon),\ \varepsilon\sim\mathcal N(0,\sigma^2),\ \sigma\in\{0,0.01,0.03,0.05,0.10\}
$$

$$
\text{(b) 전역 스케일 편향: } d' = s\cdot d,\ s\in\{0.9, 0.95, 1.0, 1.05, 1.1\}
$$

**쉬운 해석**: (a)는 depth 추정이 점마다 조금씩 들쭉날쭉 틀리는 상황(monocular depth 모델의 전형적 실패), (b)는 전체가 일괄적으로 몇 % 크거나 작게 틀리는 상황(monocular depth의 대표적 실패 모드인 scale ambiguity)을 흉내 낸 것. §2.2의 $\operatorname{Cov}(\hat\beta)$를 인위적으로 통제된 크기로 키우는 실험이라고 볼 수 있다 — 즉 이 문서 2절과 개념적으로 직결된다.

---

## 8. 3D Gaussian Splatting 렌더링 핵심 수식

**목적**: FF 모델(MVSplat/DepthSplat) 출력을 Vanilla3DGS가 이어받게 변환할 때(C1-b warm-start) 정확히 어떤 파라미터를 어떻게 바꿔야 하는지.

**어디서 쓰이나**: `core/ff_gaussian_convert.py`, `runners/vanilla_3dgs_runner.py::init_gaussians()`.

### 8.1 Scale/Opacity 재매개변수화

$$
\text{scale} = \exp(s_{\text{raw}}), \qquad \text{opacity} = \sigma(o_{\text{raw}}) = \frac{1}{1+e^{-o_{\text{raw}}}}
$$

**쉬운 해석**: 학습 파라미터 자체는 아무 실수 값이나 가능한 $s_{\text{raw}}, o_{\text{raw}}$로 두고, 실제 렌더링에 쓸 때만 지수함수/시그모이드를 씌워 "크기는 항상 양수", "불투명도는 항상 0~1"이 되도록 강제한다. FF 모델의 opacity는 이미 0~1 확률값으로 나오므로, 우리 파라미터화에 넣으려면 역시그모이드 $o_{\text{raw}}=\log\frac{p}{1-p}$로 되돌려야 한다(`ff_gaussian_convert.py::inverse_sigmoid()`).

### 8.2 Covariance ↔ Scale/Quaternion 상호변환

$$
\Sigma = R\,\mathrm{diag}(\text{scale}^2)\,R^\top
$$

**쉬운 해석**: 하나의 Gaussian이 3D 공간에서 어떤 모양(길쭉한 정도와 방향)인지는 공분산 행렬 $\Sigma$ 하나로 표현되는데, 이걸 최적화하기 편한 형태(회전 $R$=quaternion, 크기 3개)로 분해하거나 반대로 합칠 때 쓴다. FF 모델은 $\Sigma$를 직접 주고, 우리 쪽은 quaternion+scale로 따로 갖고 있어서 이 변환(고유값 분해, `covariance_to_scale_quat()`)이 C1-b 변환기의 핵심이었다.

### 8.3 색상(Spherical Harmonics) 변환

$$
c_0 = \frac{\text{RGB}/255 - 0.5}{Y_0^0}, \qquad Y_0^0 = \frac{1}{2\sqrt\pi} \approx 0.28209479177387814
$$

**쉬운 해석**: Gaussian의 기본 색은 구면조화함수(spherical harmonics)의 0차항 계수로 저장된다. $Y_0^0$은 "보는 각도와 무관한 균일한 밝기" 성분의 정규화 상수 — RGB 값을 이 상수로 나누면 SH 계수로, 곱하면 다시 RGB로 변환된다.

### 8.4 평가 지표: PSNR / SSIM

$$
\mathrm{PSNR} = -10\log_{10}\big(\mathrm{MSE}(\hat I, I)\big), \qquad \mathrm{MSE} = \frac{1}{HWC}\sum (\hat I - I)^2
$$

$$
\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}
$$

**쉬운 해석**: PSNR은 "픽셀 값이 평균적으로 얼마나 다른가"를 로그 스케일 dB로 표현한 것(값이 클수록 원본과 가까움, 통상 8~35dB대에서 논의됨). SSIM은 단순 픽셀 차이가 아니라 밝기($\mu$)·대비($\sigma$)·구조(공분산 $\sigma_{xy}$)를 함께 비교하는 지표로, 사람이 느끼는 "비슷해 보임"에 더 가깝다고 알려져 있다. 우리 코드는 11×11 가우시안 윈도우로 지역적으로 계산해 평균 낸다(`vanilla_3dgs_runner.py::ssim()`).

---

## 부록: 이 수식들이 논문 어디로 흘러가는가 (한눈에)

```
overlap(§1) ──┬── H1 검증(§4의 τ 판정과 결합) → C1-a Regime Map
              └── §5.3 x축 정의
JᵀJ 공분산(§2) ── H1의 수학적 정의, C2(§7) 설계 근거, §5.3 이론적 뒷받침
partial correlation(§3) ── baseline confound 재해석 (paper_geometry_confound_analysis)
τ/승패판정(§4) ── C1-a Regime Map의 셀마다 색칠(feedforward_win/optimization_win/tie)
budget_checkpoint(§5) ── 모든 메인 결과 표의 leakage 방지 규칙
bootstrap CI(§6) ── C1-a/C1-b/C2 결과표 전부의 신뢰구간
Holm 보정(§6.1) ── 다중 view×overlap×budget 비교의 유의성 검정
C2 depth 모델(§7) ── C2 실험 자체의 정의
3DGS 렌더링(§8) ── C1-b warm-start 변환기, 모든 러너의 렌더링 코어
```

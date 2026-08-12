# Gauss-Newton / $J^\top J$ — 논문용 수식·표기 정리

이 문서는 A-1(공부방향.md)에서 유도한 내용 중 **논문에 그대로 쓸 수식과 표기**만 추린다.
학습 과정 전체(왜/어떻게)는 `공부방향.md`와 `experiments/test.md`를 참고.

## 0. 이 문서가 답하는 질문

H1이 주장하는 "overlap이 낮으면 초기 geometry 오차가 커진다"를 수식으로 무엇이라 부를지,
그리고 그 수식이 §5.3(overlap 정의)·C2(depth 개입)와 어떻게 연결되는지.

## 1. 표기 (Notation)

일반 nonlinear least squares 표기와, 우리 triangulation 문제로의 대응.

| 기호 | 일반적 의미 | 우리 문제에서의 의미 |
|---|---|---|
| $\beta$ | 추정할 파라미터 | 3D point 위치 $P = (X,Y,Z)$ |
| $f(x,\beta)$ | 파라미터가 만드는 예측값 | 카메라 투영 $\pi(P) = f\frac{(X,Y)}{Z} + c$ (pixel 좌표) |
| $y_i$ | 관측값 | SIFT/COLMAP feature의 관측 pixel 좌표 $u_{obs}$ |
| $r_i(\beta) = y_i - f(x_i,\beta)$ | residual | reprojection error (한 view당 2차원: $u,v$) |
| $J = \partial r/\partial\beta$ | Jacobian | $\partial(u,v)/\partial P$, view마다 2×3, $N$개 view를 쌓으면 $2N\times 3$ |
| $\sigma$ | 관측 노이즈 표준편차 (픽셀) | COLMAP의 `mean_reprojection_error`로 실측 (오늘 scan1: 0.58~0.61px) |
| $\hat\beta$ | 추정치 | COLMAP이 triangulation으로 낸 3D point |

## 2. 핵심 수식 세 개

### 2.1 목적함수와 Gauss-Newton update

$$
S(\beta) = \sum_i r_i(\beta)^2 = \|r(\beta)\|^2, \qquad
\Delta\beta = -(J^\top J)^{-1} J^\top r
$$

COLMAP이 매 3D point 삼각측량 시 내부적으로 반복하는 것이 이것이다 (pose-given이므로 pose는
고정, point 위치 $P$에 대해서만 반복).

### 2.2 추정치의 공분산 (H1의 수학적 정의)

$\beta^*$ 근방에서 관측 노이즈 $\varepsilon \sim \mathcal N(0,\sigma^2 I)$를 가정하고 위와 동일한
선형화를 $\beta^*$에서 하면:

$$
\hat\beta - \beta^* \approx -(J^\top J)^{-1} J^\top \varepsilon
\quad\Longrightarrow\quad
\boxed{\operatorname{Cov}(\hat\beta) \approx \sigma^2 (J^\top J)^{-1}}
$$

**이 식이 논문에서 인용할 핵심 수식이다.** 2.1의 update 식과 대수적으로 동일한 형태이고,
$r$을 "잔차"가 아니라 "관측 노이즈"로 재해석하면 바로 얻어진다.

### 2.3 Multi-view 확장 (우리가 실제로 구현한 것)

view $k=1,\dots,N$에 대해 카메라 파라미터 $R_k, t_k, K_k$가 주어졌을 때:

$$
J_k = \underbrace{\begin{bmatrix} f_x/z_k & 0 & -f_x x_k/z_k^2 \\ 0 & f_y/z_k & -f_y y_k/z_k^2 \end{bmatrix}}_{\partial(u,v)/\partial X_{cam}} R_k,
\qquad (x_k,y_k,z_k) = R_k P + t_k
$$

$$
J = \begin{bmatrix} J_1 \\ \vdots \\ J_N \end{bmatrix} \in \mathbb R^{2N\times 3}
$$

구현: `experiments/scripts/analysis/geometry_uncertainty_figure.py`의
`two_view_depth_uncertainty()` (현재 $N=2$ pairwise 버전). 깊이 방향 성분만 뽑을 때는
view의 시선 방향 단위벡터 $\hat d$로 $\hat d^\top \operatorname{Cov}(\hat\beta)\, \hat d$를 사용.

## 3. §5.3 / H1과의 연결, 그리고 아직 안 끝난 것

- $\sigma$(관측 노이즈)는 COLMAP `mean_reprojection_error`로 **실측 가능** — 이론값이 아니라
  데이터에서 바로 나오는 수라 논문에 "우리가 측정한 $\sigma$"로 인용 가능.
- $(J^\top J)^{-1}$은 순수하게 **카메라 기하(포즈+intrinsics)의 함수**이지 overlap이나 SfM point
  개수와 직접 관련 없다 — overlap은 "$(J^\top J)^{-1}$이 작은 view 쌍을 몇 개나 확보했는가"를
  간접적으로만 반영한다. 이 간접성이 2026-08-10 실측(§13, audit log)에서 나온 반직관적 결과
  (overlap↑인데 depth uncertainty도↑, baseline이 confound)의 근본 원인.
- **아직 안 끝난 것**: baseline이 좁을 때 $J^\top J$가 어느 방향으로 왜 ill-conditioned가
  되는지의 기하학적 증명(두 시선 벡터가 평행에 가까울 때 $J$의 행들이 선형종속에 가까워짐,
  §C-1 관련). 이게 끝나야 §5.3의 "overlap이 $(J^\top J)^{-1}$과 정확히 어떤 관계인가" 문장을
  쓸 수 있음.

## 4. 이 프로젝트의 다른 곳에서 참조하는 수식과의 관계

- Diff3R(공부방향.md 읽기 목록)의 Hessian 근사 $H \approx J^\top J$는 §2.1과 같은 근사
  (2차항 무시). C1-b/C4-a 서술에서 이 대응 명시할 것.
- C2의 depth 개입(`d' = d(1+\varepsilon)`, `d'=s·d`)은 본질적으로 $\varepsilon$을 인위적으로
  주입해 §2.2의 $\operatorname{Cov}(\hat\beta)$를 통제된 크기로 키우는 실험이다 — C2 방법론
  문단에서 이 문서의 2.2절을 직접 인용 가능.

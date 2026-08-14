# 학습지 v2: Gauss-Newton과 Levenberg-Marquardt

**도달 목표** — 다음 두 문장을 스스로 유도하고 설명할 수 있으면 통과.

> ① 두 시점의 시차비 $B/Z$ 가 작아지면 $J^\top J$ 가 ill-conditioned가 되고, 깊이 불확실성이 $Z^2/(fB)$ 에 비례해 폭발한다.
>
> ② damping $\lambda I$ 는 그 폭발을 $1/\lambda$ 로 막아주지만, **정보를 더해주지는 않는다.**

---

## 0. 왜 우리가 이걸 배우는가

우리는 Gauss-Newton으로 **문제를 풀 일이 없다.** COLMAP이 대신 풀어준다. 그럼에도 배우는 이유는 세 가지다.

| 일반적인 용도 | 우리의 용도 |
|---|---|
| $J^\top J$ = $\Delta\beta$ 를 구하려고 역행렬 취하는 대상 | $J^\top J$ = **관측이 파라미터를 얼마나 제약하는가의 척도** |
| 알고리즘 | **H1의 "초기 geometry 오차"를 정의하는 언어** |
| 삼각측량 풀이 | **sparse-view 과적합의 정체를 서술하는 틀** |

한 문장으로:

> **우리에게 Gauss-Newton은 알고리즘이 아니라 "관측이 파라미터를 얼마나 제약하는가"를 재는 언어다.**

---

## 1. 선형 최소제곱 — 복습

$A\mathbf{x} = \mathbf{b}$ 에서 $\mathbf{b}$ 가 $A$ 의 column space $C(A)$ 밖에 있으면 정확한 해가 없다. 대신 $\mathbf{b}$ 를 $C(A)$ 에 **직교투영**한 점에 도달하는 $\hat{\mathbf{x}}$ 를 찾는다.

residual $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$ 가 $C(A)$ 에 수직이어야 하므로 $A^\top \mathbf{r} = 0$, 전개하면

$$A^\top A\,\hat{\mathbf{x}} = A^\top \mathbf{b} \qquad \text{(Normal Equation)}$$

**연습 1.1** — Normal Equation은 어떤 기하학적 조건에서 나오는가?

> **풀이** — residual이 $A$ 의 column space에 직교해야 한다는 조건. 외우는 공식이 아니다.

---

## 2. 비선형 문제와 Jacobian

### 2-1. 문제 설정

일반형으로 쓰면, 입력 $x_i$, 파라미터 $\beta$, 관측 $y_i$ 에 대해

$$S(\beta) = \sum_{i=1}^{n}\bigl(y_i - f(x_i,\beta)\bigr)^2 = \|\mathbf{r}(\beta)\|^2,
\qquad r_i(\beta) = y_i - f(x_i,\beta)$$

삼각측량이면 $\beta = P = (X,Y,Z)$, 그리고

$$\hat u = f\frac{X}{Z},\qquad \hat v = f\frac{Y}{Z}$$

$$\mathbf{e}(P) = \begin{bmatrix} f X/Z - u_{\text{obs}} \\[2pt] f Y/Z - v_{\text{obs}}\end{bmatrix}$$

$X/Z$ 때문에 $P$ 에 대해 **비선형**이다. 따라서 $A^\top A x = A^\top b$ 를 한 번에 적용할 수 없다.

### 2-2. Jacobian 유도

**연습 2.1** — $e_u = fX/Z - u_{\text{obs}}$ 를 $X, Y, Z$ 로 편미분하시오.

> **풀이**
>
> $$\frac{\partial e_u}{\partial X} = \frac{f}{Z},\quad \frac{\partial e_u}{\partial Y} = 0,\quad \frac{\partial e_u}{\partial Z} = -\frac{fX}{Z^2}$$
>
> $$\frac{\partial e_v}{\partial X} = 0,\quad \frac{\partial e_v}{\partial Y} = \frac{f}{Z},\quad \frac{\partial e_v}{\partial Z} = -\frac{fY}{Z^2}$$
>
> $$J = \begin{bmatrix} f/Z & 0 & -fX/Z^2 \\[2pt] 0 & f/Z & -fY/Z^2 \end{bmatrix}$$

**연습 2.2** — $J$ 의 $Z$ 열이 $1/Z^2$ 에 비례한다. 물리적으로 무슨 뜻인가?

> **풀이** — 멀리 있는 점일수록 깊이를 크게 움직여도 이미지에서는 거의 변하지 않는다. 즉 **이미지 관측으로 깊이를 구분하는 능력이 거리가 멀수록 급격히 떨어진다.** 뒤에서 $\sigma_Z \propto Z^2$ 가 나오는 이유의 씨앗.

---

## 3. Gauss-Newton — 선형화 + 최소제곱의 반복

현재 추정치 $\beta_k$ 에서 $\Delta\beta$ 만큼 움직인다고 놓고 1차 Taylor 전개:

$$\mathbf{r}(\beta_k + \Delta\beta) \;\approx\; \mathbf{r}(\beta_k) + J\,\Delta\beta,
\qquad J = \frac{\partial \mathbf{r}}{\partial \beta}\bigg|_{\beta_k}$$

$\mathbf{r}(\beta_k)$ 와 $J$ 는 현재 위치에서 계산된 **숫자**이므로 미지수는 $\Delta\beta$ 뿐이다.

$$\min_{\Delta\beta}\;\bigl\|\mathbf{r}(\beta_k) + J\Delta\beta\bigr\|^2
\quad\Longleftrightarrow\quad J\,\Delta\beta \approx -\mathbf{r}$$

$A \to J,\; x \to \Delta\beta,\; b \to -\mathbf{r}$ 로 대입하면 Normal Equation이 그대로:

$$\boxed{\;J^\top J\,\Delta\beta = -J^\top \mathbf{r}
\quad\Longrightarrow\quad
\Delta\beta = -(J^\top J)^{-1}J^\top \mathbf{r}\;}$$

그리고 갱신 후 **다시 선형화**한다.

$$\beta_{k+1} = \beta_k + \Delta\beta \;\longrightarrow\; \mathbf{r}(\beta_{k+1}),\, J(\beta_{k+1}) \;\text{재계산}$$

**연습 3.1** — 왜 매 iteration마다 $J$ 를 다시 계산하는가?

> **풀이** — 1차 Taylor 근사는 현재 위치 **근방에서만** 유효하다. 위치가 바뀌면 그 근방에서 다시 선형화해야 한다.

**연습 3.2** — Gauss-Newton이라는 이름의 유래는? (Newton method와의 차이)

> **풀이** — 목적함수 $F(\beta) = \tfrac12\|\mathbf{r}\|^2$ 의 정확한 Hessian은
> $$\nabla^2 F = J^\top J + \sum_i r_i \nabla^2 r_i$$
> Gauss-Newton은 둘째 항(2차 도함수)을 **무시**하여 $\nabla^2 F \approx J^\top J$ 로 근사한다. 잔차가 작으면 타당한 근사이며, 계산이 훨씬 싸다.
>
> ※ Diff3R이 3DGS 최적화의 Hessian을 $J^\top J$ 로 근사하는 것이 정확히 이것.

---

## 4. 손계산 — 한 iteration 직접 돌리기

**설정**

- 초점거리 $f = 100$
- 카메라 1: 원점 / 카메라 2: $x$ 축으로 $B = 10$ 이동 (카메라 2 좌표계에서 점은 $(X-10,\,Y,\,Z)$)
- 참값 $P^\ast = (0,0,50)$ → 관측 $p_1 = (0,0)$, $p_2 = (-20, 0)$
- 초기 추정 $P_0 = (0,0,40)$

**연습 4.1** — residual $\mathbf{e}$ (4×1)를 계산하시오.

> **풀이**
> 카메라 1: $\hat u = 100\cdot 0/40 = 0$, $\hat v = 0$ → $\mathbf{e}_1 = (0,0)$
> 카메라 2: $\hat u = 100\cdot(0-10)/40 = -25$, 관측은 $-20$ → $\mathbf{e}_2 = (-5, 0)$
> $$\mathbf{e} = (0,\;0,\;-5,\;0)^\top$$

**연습 4.2** — $J$ (4×3)를 계산하시오.

> **풀이** — $Z = 40$ 이므로 $f/Z = 2.5$.
> 카메라 2의 $Z$ 편미분: $-f(X-B)/Z^2 = -100\cdot(-10)/1600 = 0.625$
>
> $$J = \begin{bmatrix} 2.5 & 0 & 0 \\ 0 & 2.5 & 0 \\ 2.5 & 0 & 0.625 \\ 0 & 2.5 & 0 \end{bmatrix}$$

**연습 4.3** — $J^\top J$ 와 $J^\top \mathbf{e}$ 를 계산하시오.

> **풀이**
> $$J^\top J = \begin{bmatrix} 12.5 & 0 & 1.5625 \\ 0 & 12.5 & 0 \\ 1.5625 & 0 & 0.390625\end{bmatrix},
> \qquad J^\top\mathbf{e} = \begin{bmatrix} -12.5 \\ 0 \\ -3.125\end{bmatrix}$$

**연습 4.4** — $\Delta P$ 를 구하고 $P_1$ 을 계산하시오.

> **풀이** — $J^\top J\,\Delta P = -J^\top\mathbf{e} = (12.5,\,0,\,3.125)^\top$. $X$–$Z$ 블록만 풀면
> $$12.5\,\delta X + 1.5625\,\delta Z = 12.5,\qquad 1.5625\,\delta X + 0.390625\,\delta Z = 3.125$$
> 첫 식에서 $\delta X = 1 - 0.125\,\delta Z$, 대입하면 $0.1953125\,\delta Z = 1.5625$
>
> $$\delta Z = 8,\quad \delta X = 0,\quad \delta Y = 0
> \;\Longrightarrow\; P_1 = (0,0,48)$$
>
> $40 \to 48$. 참값 $50$ 에 접근했으나 비선형이므로 한 번에 도달하지 않는다. 다음 iteration에서 재선형화.

---

## 5. ★ $J^\top J$ 는 불확실성이다

여기서부터가 우리 연구의 핵심이다. 강의노트에는 없다.

### 5-1. 공분산으로서의 $(J^\top J)^{-1}$

관측 노이즈가 $\sigma_{px}$ 일 때 최소제곱 추정치의 공분산은

$$\operatorname{Cov}(\hat P) \;\approx\; \sigma_{px}^2\,(J^\top J)^{-1}$$

즉 $(J^\top J)^{-1}$ 의 대각 성분이 각 방향의 불확실성이다. $J^\top J$ 에 **작은 고유값**이 있으면 그 방향으로 추정이 극도로 불안정해진다.

### 5-2. 스테레오 표준식을 $J^\top J$ 에서 복원하기

참값 위치 $P = (0,0,Z)$, baseline $B$ 로 일반화.

**연습 5.1** — $J$ 를 구성하고 $J^\top J$ 를 계산하시오.

> **풀이**
> $$J_1 = \begin{bmatrix} f/Z & 0 & 0 \\ 0 & f/Z & 0\end{bmatrix},\qquad
> J_2 = \begin{bmatrix} f/Z & 0 & fB/Z^2 \\ 0 & f/Z & 0\end{bmatrix}$$
>
> $$J^\top J = \begin{bmatrix}
> 2f^2/Z^2 & 0 & f^2B/Z^3 \\
> 0 & 2f^2/Z^2 & 0 \\
> f^2B/Z^3 & 0 & f^2B^2/Z^4
> \end{bmatrix}$$

**연습 5.2** — $X$–$Z$ 블록의 역행렬에서 $ZZ$ 성분을 구하시오.

> **풀이**
> $$M = \begin{bmatrix} 2f^2/Z^2 & f^2B/Z^3 \\ f^2B/Z^3 & f^2B^2/Z^4\end{bmatrix},
> \qquad \det M = \frac{2f^4B^2}{Z^6} - \frac{f^4B^2}{Z^6} = \frac{f^4B^2}{Z^6}$$
>
> $$(M^{-1})_{ZZ} = \frac{2f^2/Z^2}{f^4B^2/Z^6} = \frac{2Z^4}{f^2B^2}$$
>
> $$\boxed{\;\sigma_Z \;\approx\; \sqrt{2}\,\sigma_{px}\,\frac{Z^2}{fB}\;}$$

**★ 도달점** — 스테레오 표준식 $\partial Z/\partial d = Z^2/(fB)$ 와 **정확히 같은 형태**다. 차이는:

| | 스테레오 근사식 | $J^\top J$ 유도 |
|---|---|---|
| 적용 범위 | rectified 2-view | **임의의 $N$-view, 임의 배치** |
| 얻는 것 | 깊이 오차 크기 | **전 방향 공분산 행렬** |
| 확장 | 어려움 | $J$ 에 행을 추가하면 끝 |

우리 실험은 2/4/8/12-view이므로 **일반형이 반드시 필요**하다.

### 5-3. 가로 방향과의 비교 — 지배 변수는 시차비

**연습 5.3** — $(M^{-1})_{XX}$ 를 구하고 $\sigma_Z/\sigma_X$ 를 계산하시오.

> **풀이**
> $$(M^{-1})_{XX} = \frac{f^2B^2/Z^4}{f^4B^2/Z^6} = \frac{Z^2}{f^2}
> \;\Longrightarrow\; \sigma_X \approx \sigma_{px}\frac{Z}{f}$$
>
> $$\boxed{\;\frac{\sigma_Z}{\sigma_X} = \sqrt{2}\,\frac{Z}{B}\;}$$

**이 한 줄이 전부다.** 지배 변수는 baseline 자체가 아니라 **시차비 $B/Z$** 다.

**수치 확인** — $f=100,\ Z=50,\ B=10,\ \sigma_{px}=1$ 이면 $\sigma_Z \approx 3.54$, $\sigma_X \approx 0.5$ → 깊이가 가로보다 **7배** 부정확. $B$ 가 절반($B=5$)이면 이 비는 **14배**.

**연습 5.4** — 로봇이 직선 복도를 지날 때와 코너를 돌 때, 어느 쪽이 더 나쁜가?

> **풀이** — 단순하지 않다. 코너에서는 시점 방향이 크게 바뀌어 baseline은 오히려 클 수 있으나 **공통으로 보이는 영역이 급감**해 삼각측량할 점 자체가 사라진다. 직선 구간에서는 overlap은 크지만 전방 이동이라 시차가 작아 $B/Z$ 가 나쁠 수 있다.
>
> → **baseline과 overlap은 다른 개념이며, 이것이 §5.3에서 overlap을 co-visibility 수치로 정의한 근거다.**

---

## 6. ★ Levenberg-Marquardt — damping

### 6-1. 형태

Gauss-Newton에서 한 항만 추가된다.

$$\text{GN:}\quad (J^\top J)\,\Delta\beta = -J^\top\mathbf{r}$$
$$\text{LM:}\quad (J^\top J + \lambda I)\,\Delta\beta = -J^\top\mathbf{r}$$

**극단의 직관**

| $\lambda$ | 거동 |
|---|---|
| $\lambda \to 0$ | GN과 동일. 빠르지만 $J^\top J$ 가 특이하면 폭주 |
| $\lambda \to \infty$ | $\Delta\beta \approx -\tfrac{1}{\lambda}J^\top\mathbf{r}$ → 단순 gradient descent. 느리지만 안전 |

실제로는 iteration마다 조절한다 — 오차가 줄면 $\lambda$ 를 낮춰 GN에 가깝게, 늘면 $\lambda$ 를 높여 보수적으로.

**기하학적 의미**: $+\lambda I$ 는 $J^\top J$ 의 **모든 고유값에 $\lambda$ 를 더해** rank-deficient 행렬을 강제로 가역으로 만든다. 즉 관측이 제약하지 못하는 방향(null space)에 **인공적인 제약**을 넣는 것.

### 6-2. damping이 불확실성을 어떻게 막는가 (핵심 연습)

**연습 6.1** — 5-2의 $M$ 에 $\lambda I$ 를 더한 $M_\lambda$ 의 $ZZ$ 성분을 구하고, $B \to 0$ 극한을 계산하시오.

> **풀이**
> $$M_\lambda = \begin{bmatrix} 2f^2/Z^2 + \lambda & f^2B/Z^3 \\ f^2B/Z^3 & f^2B^2/Z^4 + \lambda\end{bmatrix}$$
>
> $$\det M_\lambda = \frac{f^4B^2}{Z^6} + \lambda\left(\frac{2f^2}{Z^2} + \frac{f^2B^2}{Z^4}\right) + \lambda^2$$
>
> $$(M_\lambda^{-1})_{ZZ} = \frac{2f^2/Z^2 + \lambda}{\det M_\lambda}$$
>
> $B \to 0$ 일 때 $\det M_\lambda \to \lambda\left(\tfrac{2f^2}{Z^2} + \lambda\right)$ 이므로
>
> $$\boxed{\;(M_\lambda^{-1})_{ZZ} \;\longrightarrow\; \frac{1}{\lambda}\;}$$

**★ 이 결과의 의미 — 논문에 그대로 들어갈 문장**

damping이 없으면 $B \to 0$ 에서 $\sigma_Z \to \infty$ 로 발산한다. damping이 있으면 $1/\lambda$ 로 **유계**가 된다.

그러나 그 상한은 **데이터에서 온 것이 아니라 우리가 넣은 것**이다. 즉:

> **정규화는 정보를 더하지 않는다. 발산을 막을 뿐이다.**

이것이 sparse-view 정규화 기법들(depth regularization, opacity 제약, proximal term)이 **안정성은 주지만 진짜 기하를 복원하지는 못하는** 이유다.

### 6-3. Diff3R = 학습된 방향별 damping

Diff3R의 inner loss는

$$\mathcal{L}_{\text{inner}}(\Theta) = \tfrac12\|\mathbf{r}(\Theta)\|^2 + \tfrac12(\Theta - \Theta_0)^\top \operatorname{diag}(\Lambda)(\Theta - \Theta_0)$$

이고, 최적성 조건을 미분하면 시스템이 $\bigl(J^\top J + \operatorname{diag}(\Lambda)\bigr)v = \nabla\mathcal{L}_{\text{outer}}$ 가 된다. **LM의 normal equation과 구조가 동일**하며, 논문 스스로 proximal weight가 LM damping factor로 번역된다고 서술한다.

즉 Diff3R이 하는 일은:

> **Gaussian별·속성별로 서로 다른 damping $\Lambda$ 를 학습하는 것.** 확신 있는 파라미터는 큰 $\lambda$ 로 묶고, 불확실한 파라미터는 작은 $\lambda$ 로 풀어준다.

**여기서 C4-a가 나온다** — 그 방향별 $\Lambda$ 를 반드시 학습으로 얻어야 하는가? 기성 feed-forward 모델이 이미 출력하는 confidence가 그 역할을 대신할 수 있는가?

---

## 7. GN / LM / Adam — 왜 3DGS는 Adam인가

| | GN / LM | Adam (3DGS) |
|---|---|---|
| 사용 정보 | 2차 근사 $J^\top J$ | 1차 gradient만 |
| 파라미터 규모 | $10^1$ (점 하나) ~ $10^5$ (BA) | $10^7\!\sim\!10^8$ |
| 수렴 | 빠름 (near-quadratic) | 느림 |
| 비용 | $J^\top J$ 구성·역행렬 | 저렴 |
| null space 방어 | LM의 $\lambda I$ 가 막아줌 | **없음** |

3DGS가 Adam을 쓰는 이유는 파라미터가 수천만 개라 $J^\top J$ 를 만들 수도, 뒤집을 수도 없기 때문이다. (Diff3R이 matrix-free PCG solver를 따로 만든 것도 같은 이유 — 행렬을 명시하지 않고 행렬-벡터 곱만으로 푼다.)

**중요한 함의** — Adam으로 푼다고 $J^\top J$ 가 사라지지 않는다. 손실의 곡률은 여전히 $J^\top J$ 가 결정하고 rank-deficiency도 그대로다. **다만 Adam은 그 방향에 브레이크를 걸지 않는다.** 이것이 sparse-view 과적합이 3DGS에서 특히 잘 일어나는 이유의 한 축이다.

### 덤 — robust loss

$\sum r_i^2$ 대신 $\sum \rho(r_i)$ 를 쓰면(Huber, Cauchy 등) 큰 잔차의 영향이 억제된다. COLMAP BA가 이를 사용하고, 3DGS는 $L_1 + \text{D-SSIM}$ 조합이라 이미 순수 제곱이 아니다.

우리에게 필요한 인식은 하나 — **재투영 오차 0.61px는 순수 제곱 최소화 결과가 아니라 robust loss를 거친 값일 수 있다.** 수치를 해석할 때 알고 있어야 한다.

---

## 8. ★ 우리 연구로의 번역 — 3DGS를 같은 틀로 보기

| | 삼각측량 | 3DGS 최적화 |
|---|---|---|
| 파라미터 | 점 1개 = 3 DOF | Gaussian $N$ 개 × 약 59 파라미터 |
| Residual | $2 \times$ (관측 view 수) | 픽셀 수 × 학습 view 수 |
| 필요 조건 | view 2장 이상 | residual > 파라미터 |

이 틀에서 sparse-view 과적합이 이렇게 서술된다:

> 입력 view가 적으면 residual 개수가 파라미터 개수에 비해 부족해져 $J^\top J$ 가 rank-deficient가 된다. 그 **null space 방향으로는 파라미터를 아무리 움직여도 학습 view의 손실이 변하지 않는다.** 그러나 held-out view에서는 그 움직임이 그대로 렌더링 오차로 나타난다.

그리고 **densification의 해로움이 자동으로 따라 나온다:**

- clone/split → **파라미터 개수 증가** (Gaussian당 약 59개씩)
- 입력 view는 그대로 → **residual 개수 불변**
- 따라서 $\dfrac{\#\text{파라미터}}{\#\text{residual}}$ 이 단조 증가 → **null space 확대**

$$\boxed{\text{densification} \;\Longrightarrow\; \text{null space 확대} \;\Longrightarrow\; \text{held-out 성능 저하}}$$

이것이 **H1의 메커니즘 그 자체**다. "Gaussian 수는 늘지만 품질은 떨어진다"가 상관관계 관찰이 아니라 설명되는 명제가 된다.

**1일차·2일차 관측과의 연결**

| 관측 | 이 틀에서의 해석 |
|---|---|
| sparse 8-view: 100k→221k Gaussian, 10.69→9.85 dB (densification 활발 구간) | null space가 **커지는 과정**. H1 |
| dense 42-view: refine_stop 이후 Gaussian 고정인데 24.11→24.06 미세 하강 | null space가 이미 작은 상태의 **미세 드리프트**. 다른 메커니즘 |

→ 두 하강은 **같은 현상이 아니다.** D 산출물에서 iteration 축에 refine_stop = 15,000 을 표시하면 이 구분이 그림에서 바로 보인다.

---

## 9. 학습 항목 → 설계 결정 매핑

| 학습 항목 | 연구에서의 역할 |
|---|---|
| 재투영 오차 최소화 | COLMAP `point_triangulator`가 하는 일. 1일차 reproj error 0.61px가 이 목적함수 값 |
| $(J^\top J)^{-1}$ = 공분산 | **H1의 "초기 geometry 오차"를 정량화하는 정의** |
| $\sigma_Z \propto Z^2/(fB)$ | C2 개입에서 noise를 곱셈형 $d(1+\varepsilon)$ 으로 준 이유 |
| $\sigma_Z/\sigma_X = \sqrt2\,Z/B$ | §5.3에서 overlap을 baseline으로 통제하지 않은 근거 |
| $J^\top J$ rank-deficiency | **sparse-view 과적합의 정의**, densification 해로움의 메커니즘 |
| LM damping $\lambda I$ | 정규화 기법들의 통일적 해석. "정보가 아니라 발산 방지" |
| $H \approx J^\top J$ | Diff3R 부록 B의 절반 |
| 학습된 $\Lambda$ | **C4-a의 출발점** — 학습 없이 기성 confidence로 대체 가능한가 |

---

## 10. 실측 과제 (반나절) — 학습을 논문 figure로

이미 있는 COLMAP 출력을 재활용. 새 실험 불필요.

**데이터**: `outputs_dense_sanity/colmap_work/dtu_scan1_dense_sanity/49view_seed0/sparse_triangulated/` 의 `points3D.bin`, `images.bin` (pycolmap)

**절차**

1. view 쌍별 baseline $B$ 계산 (카메라 중심 간 거리)
2. 각 3D point의 깊이 $Z$ 와 관측 view 목록 추출
3. 점마다 $J$ 를 구성해 $(J^\top J)^{-1}$ 의 깊이 성분 계산 (또는 근사로 $\sqrt2\,\sigma_{px}Z^2/(fB)$)
4. 가로축 **시차비 $B/Z$ 또는 co-visibility**, 세로축 **깊이 불확실성** 산점도

**확인할 것**

- 이론값과 실측 분포가 일치하는가
- co-visibility가 낮은 쌍에서 불확실성이 실제로 큰가
- view 수가 늘면 불확실성이 어떻게 줄어드는가 ($J$ 에 행이 추가되는 효과)

**산출물**: §5.3 정당화 figure 1장 + H1의 실데이터 근거

---

## 11. 자가 점검

1. Normal Equation은 어떤 기하학적 조건에서 나오는가?
2. GN이 매 iteration마다 $J$ 를 다시 계산하는 이유는?
3. $J$ 의 $Z$ 열이 $1/Z^2$ 에 비례하는 것의 물리적 의미는?
4. $(J^\top J)^{-1}$ 의 대각 성분은 무엇을 뜻하는가?
5. $\sigma_Z \approx \sqrt2\,\sigma_{px}Z^2/(fB)$ 를 $J^\top J$ 에서 유도할 수 있는가?
6. baseline이 크면 항상 좋은가? 트레이드오프는 무엇인가?
7. view를 추가하면 $J$, $J^\top J$ 는 어떻게 변하며 불확실성은 왜 줄어드는가?
8. GN과 Newton method의 차이는? (Hessian 근사)
9. LM의 $\lambda I$ 는 기하학적으로 무엇을 하는가? $B\to0$ 에서 $\sigma_Z$ 는 어떻게 되는가?
10. **정규화가 정보를 더하지 않는다는 말의 의미는?**
11. 3DGS는 왜 GN/LM이 아니라 Adam을 쓰는가? 그 대가는?
12. densification이 왜 held-out 성능을 떨어뜨리는가? (파라미터/residual 비로 설명)

---

## 다음 단계

- **CS231A 노트 4 §3.3** (15분) — similarity ambiguity. 절대 스케일이 복원 불가한 이유 → C2의 $d' = s\cdot d$ 개입 근거
- **§5.1 Bundle adjustment** (10분, 개념만) — 같은 GN/LM을 다중 카메라·다중 점으로 확장
- 그 다음 **노트 5 (volumetric / plane sweep)** → MVSNet → MVSplat
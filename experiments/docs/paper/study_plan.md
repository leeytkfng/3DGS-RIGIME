# 학습 계획 — 논문 이해를 위한 개인 스터디 로드맵

이 문서는 "지금 뭘 아는지"와 "다음에 뭘 봐야 하는지"를 계속 갱신하는 살아있는 문서다(체크리스트와 같은 패턴). 각 단계는 (읽을 자료) + (대조할 코드) + (이해 목표) + (스스로 확인할 질문)으로 구성해서, 읽고 끝이 아니라 코드로 검증까지 하도록 짰다.

목표는 논문을 그냥 읽는 게 아니라 **이 프로젝트가 실제로 만든 코드/실험과 항상 대조하면서** 보는 것 — 그래야 "어디에 쓰이는 개념인지" 바로 연결된다.

---

## 완료됨 (2026-08-13 기준)

### 1. 야코비안 / Gauss-Newton 삼각측량 → 공분산

- 다룬 내용: 재투영 오차 최소화 문제, $J=\partial r/\partial\beta$, $\mathrm{Cov}(\hat\beta)\approx\sigma^2(J^\top J)^{-1}$
- 실제 어디 쓰였나: `experiments/docs/paper/paper_geometry_confound_analysis_2026-08-12.md`의 baseline-overlap-uncertainty 분석. Overleaf 초안 §5.1(`sec:confound`)에 이미 들어감.
- 파일: `experiments/scripts/analysis/geometry_uncertainty_figure.py::two_view_depth_uncertainty()`

### 2. Vanilla 3D Gaussian Splatting 기초

- 다룬 내용: Gaussian 파라미터(위치·크기·회전·불투명도·색), rasterization 렌더링, 대략적인 학습 루프
- 실제 어디 쓰였나: `experiments/scripts/runners/vanilla_3dgs_runner.py`가 이 개념들을 gsplat 라이브러리로 구현한 것

---

## 다음 단계 — 순서대로

### Step 1. Densification 메커니즘 심화 ⭐ 최우선 추천

**왜 지금인가**: 야코비안(관측이 적으면 불확실도가 커진다는 감각)과 Gaussian(파라미터가 뭔지)이 만나는 지점이라, 방금 배운 두 개를 바로 이어붙일 수 있다. 오늘 이 프로젝트가 실제로 발견한 것과 정확히 겹친다.

- **읽을 자료**: `experiments/docs/paper/paper_gaussian_observation_starvation_2026-08-13.md` (전체 — 특히 §1 코드 체인 부분과 §3.2 "가설의 운명")
- **대조할 코드**:
  - `/opt/conda/envs/ps3/lib/python3.9/site-packages/gsplat/strategy/default.py`의 `_update_state()`, `_grow_gs()` — 실제 densification 판단 로직
  - `experiments/scripts/analysis/gaussian_gradient_accumulation_probe.py` — 우리가 이걸 어떻게 계측했는지(v1이 왜 null 결과였는지도 같이 보면 좋음 — "가설 검증 설계를 잘못하면 뭐가 틀리는가"의 좋은 사례)
- **이해 목표**:
  - `index_add_()`로 gradient norm이 어떻게 누적되는지, 왜 100-step마다 리셋되는지
  - `count[g]`가 왜 "표본 크기"처럼 작동하는지
- **스스로 확인할 질문**: "`count[g]`가 0인 Gaussian은 왜 생기는가?" / "관측이 적은 Gaussian은 왜 절대 τ를 못 넘는가(분자·분모가 같이 작아진다는 게 무슨 뜻인가)?"

### Step 2. 3DGS 원 논문의 Adaptive Density Control 절

**왜 이 순서인가**: Step 1에서 gsplat "구현"을 봤으니, 이제 그게 원래 어떤 논문 아이디어를 재현한 건지 원문으로 확인한다.

- **읽을 자료**: Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023) — Adaptive Density Control 절(clone/split, opacity reset, pruning 부분)
- **대조할 코드**: `vanilla_3dgs_runner.py`의 `--densification {on,off}` 옵션과 `strategy.step_post_backward()` 호출부
- **이해 목표**: 논문이 말하는 "under-reconstruction"(clone 대상)과 "over-reconstruction"(split 대상)의 구분 기준이 코드의 `is_small`/`is_large` 판정과 어떻게 대응하는지

### Step 3. FSGS 논문 + 코드 — Vanilla 3DGS에서 "최소한의 추가"

**왜 이 순서인가**: Step 1~2로 Vanilla 3DGS를 확실히 알았으니, 거기서 뭘 더했는지만 보면 되는 가장 작은 다음 스텝이다.

- **읽을 자료**: Zhu et al., *FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting* (ECCV 2024, arXiv:2312.00451) — Proximity-guided Gaussian Unpooling, Pseudo-view 정규화 절 중심
- **대조할 코드**:
  - `/data/Re-feem/code/fsgs/train.py`의 학습 루프 — 특히 `pseudo_stack`(가상 카메라 샘플링), `depth_loss`/`depth_loss_pseudo`(Pearson correlation 기반 깊이 정합) 부분
  - `experiments/scripts/runners/fsgs_runner.py` — 우리가 이 루프를 wall-clock budget 체계로 어떻게 감쌌는지
- **이해 목표**: "실제로 안 찍은 카메라 위치"(pseudo-view)를 어떻게 만들어내는지(`generate_random_poses_llff`), MiDaS로 추정한 깊이와 렌더된 깊이를 어떻게 비교하는지(상관계수를 손실로 쓰는 이유)
- **스스로 확인할 질문**: "pseudo-view loss는 실제 정답(GT) 없이 어떻게 학습 신호가 되는가?"

### Step 4. Feed-forward 아키텍처 전환 — MVSplat

**왜 지금인가**: 여기서부터 패러다임이 완전히 바뀐다("장면 하나를 반복 학습"이 아니라 "사전학습된 신경망이 한 번에 예측"). Step 1~3과 이어지는 게 적어서 새로 시작하는 느낌일 수 있으니, 앞의 세 스텝으로 자신감이 붙은 뒤에 보는 게 낫다.

- **선행 배경지식(생소하면 먼저 검색)**: epipolar geometry(두 카메라 사이 대응점이 만족하는 기하학적 제약), plane-sweep cost volume(MVSNet 계열 — "이 픽셀이 후보 깊이 $d_1, d_2, \dots$ 중 어디 있을 때 다른 view들과 가장 잘 맞는가"를 전부 계산해보는 방식)
- **읽을 자료**: Chen et al., *MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images* (ECCV 2024, arXiv:2403.14627)
- **대조할 코드**: `/data/Re-feem/code/mvsplat/src/model/encoder/` — cost volume 구성과 depth 예측 head
- **이해 목표**: 여러 사진의 feature를 어떻게 하나의 depth map으로 합치는지, Gaussian 파라미터를 왜 이 depth에서 직접 회귀할 수 있는지

### Step 5. DepthSplat — MVSplat 대비 추가된 것

- **읽을 자료**: Xu et al., *DepthSplat: Connecting Gaussian Splatting and Depth* (CVPR 2025, arXiv:2410.13862)
- **이해 목표**: 사전학습된 monocular depth feature가 cost volume에 어떻게 융합되는지, 이게 왜 더 넓은 view 수 범위에서 강인한지

---

## 전체 로드맵 한눈에

```
[완료] 야코비안/공분산 ──┐
[완료] Vanilla 3DGS 기초 ─┤
                          ├─→ Step1: densification 메커니즘(오늘 발견과 직결)
                          │        ↓
                          │   Step2: 3DGS 원논문 Adaptive Density Control
                          │        ↓
                          │   Step3: FSGS(Vanilla 3DGS + 최소 추가)
                          │
                          └─────────────→ Step4: MVSplat(패러다임 전환) → Step5: DepthSplat
```

Step 1~3은 지금까지 배운 것과 직접 이어지는 "낮은 언덕"이고, Step 4~5는 완전히 다른 아이디어라 "새 언덕"이다 — 순서를 바꾸지 않는 게 심리적으로도 유리하다.

## 진행 기록

| 일자 | 완료한 것 | 비고 |
|---|---|---|
| ~2026-08-13 | 야코비안/공분산, Vanilla 3DGS 기초 | 이 문서 작성 시점 기준 |

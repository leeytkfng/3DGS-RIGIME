# Sparse-view 3DGS의 패러다임 선택 Regime 분석

**입력 희소성·겹침·계산 예산에 따른 Feed-forward와 장면별 최적화의 품질–효율 우위 조건 규명**

KCI 논문 연구계획서 | 2차 최종본 | 2026. 08. 09 | 목표 제출: 2026년 10~11월 | 2인 공동 | H200 ×1

---

## 0. 한 문단 요약

사진 몇 장으로 3D 장면을 만드는 방법에는 크게 두 갈래가 있다. 하나는 미리 학습한 모델에 사진을 넣으면 즉시 3D가 나오는 feed-forward 방식이고, 다른 하나는 새 장면마다 그 장면 전용으로 반복 최적화하는 per-scene optimization 방식이다. 본 연구는 새 알고리즘을 만드는 대신 **입력 사진 수·사진끼리 겹치는 정도·사용할 수 있는 계산 시간**을 축으로 대표 시스템들의 실용적 승패 지도를 만들고, **동일한 초기값에서 standard 3DGS refinement를 켜고 끄는 통제 실험**으로 최적화의 순효과를 분리하며, 승패가 갈리는 원인을 기하 불확실성과 최적화 동역학에서 설명하고, 마지막에 **조건별 선택 가이드라인**으로 마무리한다.

---

## 1. 문제 제기

### 1.1 두 패러다임

| 구분 | Feed-forward 3DGS | Per-scene Optimization (3DGS) |
|---|---|---|
| 동작 | 사전학습된 네트워크가 입력 이미지를 받아 단일 forward pass로 Gaussian 출력 | 새 장면마다 초기 point cloud에서 시작해 렌더링 오차를 줄이는 방향으로 Gaussian 반복 갱신 |
| 속도 | 초 단위 이하 또는 수 초 | 수십 초 ~ 수십 분 |
| 강점 | 학습된 prior 덕분에 정보가 부족해도 그럴듯한 형상 보완 | 입력이 충분하면 해당 장면에 맞춰 높은 품질 달성 |
| 약점 | 학습 분포를 벗어난 장면에서 성능 저하 | 입력이 희소하면 잘못된 초기 형상에 과적합 |
| 대표 | pixelSplat, MVSplat, DepthSplat, NoPoSplat, ReSplat | 3DGS 원논문, SparseGS, FSGS |

### 1.2 흔들리는 통념

일반적으로 "느리지만 정확한 optimization 대 빠르지만 부정확한 feed-forward"라는 구도가 통용된다. 그러나 최근 보고는 이 구도가 조건부임을 시사한다. ReSplat은 8-view sparse 입력에서 per-scene 3DGS가 과적합으로 악화되어 feed-forward가 앞서는 결과를 보고했고, Diff3R은 sparse 세팅의 test-time optimization이 context view에 과적합하며 geometry를 훼손하는 현상을 핵심 문제로 다룬다. 즉 **어느 쪽이 이기는지는 입력 조건과 계산 예산에 따라 달라진다.**

### 1.3 남은 공백

ForeSplat·Diff3R 등 최근 연구는 이 경계 지대를 hybrid 방식으로 활용하는 데 집중한다. 그러나 입력 희소성·view 겹침·계산 예산을 동일 프로토콜 아래 공동 분석하여 **경계가 어디에 있고 무엇이 결정하는지** 정리한 연구는 제한적이다.

---

## 2. 연구 질문과 포지셔닝

> **"Sparse-view 3D 재구성에서 입력 view 수, view overlap, 계산 시간 예산에 따라 feed-forward와 per-scene optimization의 품질–효율 우위는 어떻게 변화하며, 그 역전 경계는 무엇이 결정하는가?"**

본 논문은 method paper가 아니라 **Empirical Study + Controlled Analysis + Failure Analysis**를 결합한 분석 논문이다. 핵심 메시지는 "보편적으로 우월한 패러다임은 없으며, 실용적 선택은 조건의 함수"라는 것이다.

### 2.1 논문의 서사 사슬

기여들은 병렬 나열이 아니라 하나의 사슬로 읽혀야 한다. 각 절은 앞 절이 남긴 질문에 답한다.

```
Regime Map (현상: 어디서 갈리는가)
        ↓  "왜 갈리는가?"
기하 프록시 + 최적화 동역학 (증거)
        ↓  "정말 그 원인인가?"
Depth noise 개입 + Refinement On/Off (통제)
        ↓  "그래서 어떻게 하라는 것인가?"
Practical Guideline (처방)
```

제목 후보 — 영문: *"When Does Feed-Forward Reconstruction Beat Per-Scene Optimization? A Controlled Regime Study of Sparse-View 3D Gaussian Splatting"* / 국문: *"Sparse-view 3D Gaussian Splatting에서 Feed-forward와 장면별 최적화의 우위 조건 분석"*

---

## 3. 기여

| 기여 | 유형 | 내용 |
|---|---|---|
| **C1-a. 품질–시간 Regime Map** | Benchmark | view 수 × overlap × 계산 예산에서 대표 시스템들의 **실용적** 승패 영역을 매핑하고 품질–시간 Pareto frontier를 함께 제시 |
| **C1-b. 동일 초기값 Refinement On/Off** | Controlled Analysis | 동일한 feed-forward Gaussian 출력을 고정한 뒤 standard 3DGS refinement를 off(0초)/on(10·60·300초)으로 비교. 모든 optimization의 보편적 효과가 아니라 **동일 초기값에 적용한 해당 refinement 절차의 순효과**를 측정 |
| **C2. 기하 불확실성–렌더링 실패 연결** | Knowledge | 기하 프록시·최적화 동역학·통제된 depth noise 개입으로 초기 geometry 오차가 과적합과 렌더링 실패로 이어지는 경로 분석. **논문 비중은 C1보다 C2에 더 둔다** |
| **C4. Practical Guideline** | Application | 분석 결과를 조건별 선택 규칙(의사결정 표)으로 요약. 실험 결과에서 파생되므로 추가 실험 비용이 없으며, C3가 빠져도 실용 기여가 남는다 |
| **C3. 경량 Paradigm Selector** | Method (여유 시) | co-visibility·texture 통계·geometry confidence로 승자 예측. **scene 단위 group split** 필수. 일정 압박 시 완전히 분리 |

### 3.1 C2 사전 가설 — 파일럿 전 동결

C2가 사후적 failure case 설명이 아니라 **확증적 분석**이 되도록, 방향성 있는 가설을 미리 고정한다. 각 가설에는 지지·기각 판정 기준을 함께 명시한다.

| 가설 | 내용 | 지지되는 관찰 |
|---|---|---|
| **H1** | 초기 geometry 오차(σ)가 커질수록 densification 이후 Gaussian 수는 증가하지만 held-out 품질은 감소하며, 이 관계는 **low-overlap 조건에서 강화**된다 | σ↑에 따라 Gaussian count 단조 증가 + test PSNR 단조 감소, 그리고 σ×overlap 상호작용이 유의(low-overlap에서 기울기가 더 가파름) |
| **H2** | Optimization의 품질 정점 도달 시점은 view 수가 줄수록 **앞당겨지고**, 정점 이후 하강 기울기가 가팔라진다 | view 수와 정점 iteration 사이 양의 관계, view 수와 정점 후 하강 기울기 사이 음의 관계 |
| **H3** | 초기 geometry 품질이 일정 수준을 넘으면 refinement의 한계 이득이 **소멸**한다 | C1-b의 on/off 품질 격차가 σ 증가에 따라 단조 감소하며, 낮은 σ 구간에서 신뢰구간이 0을 포함 |

세 가설 중 어느 것이 기각되더라도 그 자체가 보고할 결과다. 기각된 가설을 사후에 삭제하지 않고 결과와 함께 기술한다.

### 3.2 C4 Practical Guideline — 출력 형태 (예시)

실제 값은 실험 결과로 채우며, 아래는 표의 형식만 보여주는 예시다.

| 조건 | 권장 |
|---|---|
| view ≤ 4 AND overlap 낮음 | Feed-forward |
| view ≥ 8 AND overlap 높음 | Per-scene optimization |
| 예산 < 10초 | Feed-forward |
| 예산 > 300초 AND geometry 신뢰도 높음 | Per-scene optimization |
| 그 외 | Tie 영역 — 품질–시간 Pareto frontier 참조 |

---

## 4. 실험 설계

### 공정성 원칙 ① — Pose-given track 통일

모든 방법에 동일한 카메라 pose를 제공한다. 한쪽에 정답 pose, 다른 쪽에 COLMAP 추정 pose를 주면 패러다임 비교가 아니라 **pose 정확도 비교**가 된다. Pose-free 세팅은 후속 연구로 분리한다.

### 공정성 원칙 ② — 주장 수위와 교란요인

서로 다른 시스템 간 성능 차이는 학습 데이터·모델 크기·학습 view 수·입력 해상도·depth prior·구현 최적화의 영향을 함께 받는다. C1-a는 "대표 시스템들의 실용적 우위 영역"으로 서술하고, refinement의 효과에 관한 진술은 모델 요인을 고정한 **C1-b 범위 안에서만** 제기한다.

### 4.1 통제 축

| 축 | 수준 |
|---|---|
| View 수 | 2 / 4 / 8 / 12 (스모크 테스트에서 모델 지원 범위 미달 시 2 / 4 / 8로 축소) |
| View overlap | 고 / 저 — co-visibility 수치 구간으로 정의(§5.3) |
| 계산 예산 | 1초 / 10초 / 60초 / 300초 — **end-to-end 기준이 메인 비교**, post-initialization 기준은 동역학 분석으로 병기 |

Domain은 완전교차 축에서 제외하고 외부 검증으로 사용한다. Geometry 불확실성은 view 수와 overlap에 종속되는 설명 변수이므로 독립 축이 아니라 C2의 측정 변수로 취급한다.

### 4.2 비교군

| 구분 | 구성 |
|---|---|
| Feed-forward (2종) | DepthSplat + MVSplat. 공개 체크포인트 zero-shot 추론. ReSplat은 코드 공개·재현 상태 확인 후 대체 후보 |
| Optimization (2종) | Vanilla 3DGS + sparse-view 특화 1종(SparseGS 또는 FSGS 중 코드 상태가 안정적인 1개). 강한 기준선을 포함해 편향 비판 차단 |
| C1-b 트랙 | Feed-forward 출력(모델별 각각)을 고정하고 standard 3DGS refinement off/on 비교 |
| 초기화 ablation | COLMAP vs VGGT/DA3. 별도 패러다임이 아니라 optimization 내부 부가 실험 |

---

## 5. 측정 프로토콜 — 본 실험 전 사전 확정 12항목

아래 항목은 1~3주 차 스모크 테스트와 파일럿의 산출물이며, 본 실험 착수 전에 문서와 설정 파일로 동결한다. **결과를 본 뒤 기준을 바꾸지 않는다.**

### 5.1 시간 측정 범위 — 초기화 경계 포함

- **공통 사전 제공(측정 제외):** 입력 이미지, 동일 카메라 pose, train/test view 분할
- **End-to-end time(메인):** 공통 사전 제공 시점 이후 방법별 초기화 시간을 모두 포함하여 최종 Gaussian 생성 완료까지. COLMAP sparse point, VGGT/DA3 초기 geometry 생성 시간도 해당 방법의 비용으로 계상
- **Optimization-only time(부록·동역학):** 초기 Gaussian 생성 완료부터 gradient update 종료까지. 초기화 산출물을 사전 제공하면 **post-initialization reconstruction time**으로 명칭을 좁힌다

CUDA 최초 컴파일은 제외하되 모든 방법을 같은 조건으로 warm-up한다. 렌더링·평가 시간은 별도 보고한다.

### 5.2 모델별 지원 view 수 — 스모크 테스트로 채울 표

2026-08-12 갱신: 공식 repo(config/README)와 DTU smoke test(2/4/8/12-view forward pass) 결과로 채움. "학습 분포"와 "forward pass가 죽지 않는 범위"는 다른 질문이므로 열을 분리한다.

| 모델 | 학습 시 입력 view (체크포인트 기준) | 공식 문서상 지원 범위 | Forward pass 생존 확인(실측) | Pose 필요 | 입력 해상도 |
|---|---|---|---|---|---|
| MVSplat | RE10K: 고정 2-view (`num_context_views: 2`, `config/dataset/view_sampler/bounded.yaml`) | DTU eval index는 N=2,3만 공식 제공(`assets/evaluation_index_dtu_nctx{2,3}.json`). README가 직접 "12-view까지 필요하면 DepthSplat 쓰라"고 안내 — 저자 스스로 4-view 이상은 지원 범위 밖으로 간주 | 2/4/8/12 전부 crash 없이 통과(DTU scan1, 2026-08-11 smoke) — 단, "안 죽는다"≠"학습 분포 안" | 필요 | RE10K/DTU 256×256(`config/experiment/re10k.yaml`, `dtu.yaml`) |
| DepthSplat | 체크포인트별로 다름. 우리가 쓰는 기본 체크포인트 `depthsplat-gs-base-dl3dv-256x448-randview2-6`는 **2~6-view 랜덤 샘플링으로 학습**(`view_sampler/boundedv2_360.yaml` 기본값 num_context_views=4, 체크포인트명의 randview2-6이 실제 학습 범위) | 공식 README: 별도 체크포인트(`randview4-10`, 448×768)로 4~10-view, 최대 12-view(512×960, A100 0.6초)까지 문서화·검증됨. 우리는 이 상위 체크포인트를 아직 받지 않음 | 2-view만 실측(DL3DV in-domain probe, mean PSNR 20.0dB). 4/8/12-view는 우리 쪽에서 아직 미실행 | 필요 | DL3DV 256×448(우리 체크포인트), RE10K 전용 체크포인트는 256×256 |
| Vanilla 3DGS | 해당 없음(optimization) | 해당 없음 — view 수 제약 자체가 없는 방법론 | 2/4/8/12-view(DTU smoke) + 49-view(dense sanity, 42 train) 전부 정상 | 필요 | 가변 |
| Sparse-view opt (SparseGS/FSGS, 택1) | 미착수 | 확인 필요 | 확인 필요 | 필요 | 확인 필요 |

**결론:** main 실험 축의 view_counts=[2,4,8,12] 중, MVSplat은 사실상 2-view만 학습 분포 내이고 4/8/12는 저자 기준으로도 공식 범위 밖이다. DepthSplat은 우리 체크포인트 기준 2~6-view가 학습 분포이므로 8/12-view는 분포 밖, 4-view까지는 분포 내로 볼 수 있다(단 실측은 아직 2-view뿐). `model_registry.py`의 `supports_views`가 지금까지 네 모델 모두 `[2, 4, 8, 12]`로 동일한 placeholder였던 것을 이 표 기준으로 갱신했다(아래 참고). 학습 분포를 벗어난 view 수의 결과는 정상 성능 비교가 아니라 분포 밖 사용 결과이므로, 4/8/12-view MVSplat과 8/12-view DepthSplat 결과는 regime map 본문에서 "OOD 사용"으로 별도 표기하고 §5.2 경계 밖 취급한다.

### 5.3 Overlap 계산식 — 수식·집계·선택 편향 차단

- **Pairwise 주 지표:** `O_ij = 2|P_i ∩ P_j| / (|P_i| + |P_j|)`, `P_i`는 view `i`가 관측한 SfM point 집합
- **Non-edge 처리(중요):** SfM 매칭이 실패하거나 공통 point가 임계 미만인 쌍은 **제외하지 않고 `O_ij = 0`으로 취급**한다. 매칭 실패 자체가 낮은 overlap의 증거이며, 이런 쌍을 빼면 low-texture·low-overlap 장면에서 overlap이 **체계적으로 과대평가**되어 regime map의 x축이 왜곡된다
- **집계:** 유효 edge의 중앙값이 아니라 **전체 쌍(0 포함)의 평균과 하위 25% 분위수를 함께** 산출한다. 중앙값은 0값을 흡수해 low-overlap 신호를 지운다. 주 지표는 평균, 보조 지표로 하위 분위수를 병기
- **층화:** overlap 고/저 구간은 전체 분포가 아니라 **view 수 수준 내에서 층화**해 정의한다(view 수가 늘면 pairwise overlap 분포 자체가 이동하므로 공통 절대 기준은 부적절)
- **Graph connectivity 최소 기준:** 입력 view 집합이 연결 그래프를 이루지 못하는 경우(고립 view 존재)는 별도 범주로 기록하고 메인 집계에서 분리 보고
- **Texture 교란 분리:** feature co-visibility는 텍스처 양에 영향을 받으므로 장면별 texture 통계(gradient energy 등)를 **공변량으로 로깅**해 overlap 효과와 분리 분석한다. DTU에서는 geometry overlap을 함께 계산해 두 지표의 상관을 확인

### 5.4 장면 수·Sampling seed

| 단계 | 규모 |
|---|---|
| 파일럿 | 장면 5개, 조건 전 조합 1회전, 변동성 측정 |
| 본 실험 | 장면 20~30개, 입력 view sampling seed 3회 |
| 외부 검증(DTU) | 장면 8~15개 |

**2026-08-12 재계산** — 최초 추정("약 1,200 run, 200~300 GPU-hour")은 감이었고, 8/9 audit에서 "budget을 별도 run으로 잘못 세면 실제로는 2.4배"라는 우려가 나왔다(`paper_scaffold_audit_log.md` §7). 이번엔 실제 manifest(`experiment_manifest.json`, 12,480 row)를 budget-checkpoint를 하나의 trajectory로 접어서(= budget은 재학습이 아니라 한 trajectory 안의 체크포인트) 실행 단위 수를 다시 세고, DTU smoke(2026-08-11)의 실측 wall-clock을 대입했다.

| 구간 | Trajectory 수 | 근거 | per-trajectory 추정 | 소계 |
|---|---:|---|---:|---:|
| main, optimization(Vanilla3DGS+SparseGS) | 960 | 20 scene × 3 seed × 4 view × 2 overlap × 2 method | COLMAP 초기화 오버헤드(실측 평균 8.15s, DTU 10s-budget smoke 4건: 17.2~19.2s − 10s) + max budget 300s ≈ **308s** | 82.1 GPU-hour |
| main, feed-forward(MVSplat+DepthSplat) | 960 | 20 scene × 3 seed × 4 view × 2 overlap × 2 method | MVSplat 실측 6.36~7.98s(DTU, 동일 256×256) 평균 ≈7s를 두 모델에 임시 적용(DepthSplat은 monodepth 백본이 있어 과소추정 가능성 있음, 미실측) | 1.9 GPU-hour |
| C1-b, refinement=on | 960 | FF 초기값 고정 후 재최적화, main-optimization과 같은 trajectory 구조로 가정 | ≈308s(위와 동일 가정) | 82.1 GPU-hour |
| C1-b, refinement=off | 960 | FF 출력을 그대로 렌더 등가성 gate만 통과시키는 평가 전용 | ≈7.5s(추정, 실측 없음) | 2.0 GPU-hour |
| C2 (depth noise + scale bias) | 960 | 8 external scene × 3 seed × 4 조건 × (5 noise + 5 scale) | §5.9에서 main phase와 동일한 300s trajectory로 확정(2026-08-12) | 82.1 GPU-hour |

**합계: 약 250 GPU-hour** (초기화·평가 오버헤드는 위 per-trajectory 추정에 이미 포함, 실패 재실행 여유는 별도 가산 필요). 2026-08-12 이전에는 C2 budget 미정으로 186~250h 범위였으나, §5.9에서 main phase와 동일한 300s trajectory로 확정하며 상한(250h)으로 고정했다.

결론: 최초 추정(200~300 GPU-hour)이 걱정했던 것과 달리 실측 기반 재계산도 같은 범위(250h)에 들어온다. 8/9 audit의 "2.4배" 우려는 budget을 trajectory로 접어 세면 해소된다(사실 audit 자신도 이미 이렇게 셌었다 — 2,880 "실제 optimization 실행"이라는 숫자 자체가 이 접힌 카운트였다). 그 외 가정: (a) SparseGS는 미구현이라 Vanilla3DGS와 같은 시간 프로파일로 가정, (b) DepthSplat 단독 실측 wall-clock 없음(probe 스크립트가 elapsed를 로깅하지 않음 — 정식 러너 승격 시 같이 고칠 것), (c) H200 병렬 실행으로 처리량이 늘지 않는다는 것은 이미 실측 확인됨(연산 병목)이므로 시간 단축 요인으로 넣지 않았다. 평균, 표준편차 또는 95% CI, **장면별 win rate**를 함께 보고한다.

### 5.5 Crossover 승패 판정 기준

`ΔPSNR = PSNR_FF − PSNR_OPT`. `Δ > +τ` FF 우세 / `Δ < −τ` OPT 우세 / 그 사이 Tie.

τ는 **두 근거의 최댓값**으로 정의한다: (a) 파일럿의 seed 간 변동성, (b) 실용적으로 의미 있는 최소 차이(문헌 관행상 PSNR 0.3~0.5dB 수준). seed 변동성만으로 정하면 변동성이 클수록 tie 구간이 넓어지는 자기참조가 생기므로, 두 근거를 병기해 고정한다. LPIPS·SSIM이 반대 결론을 내는 조건은 별도 분석하며, 품질–시간 Pareto frontier를 보조 결과로 제시한다.

### 5.6 데이터셋별 지표 분리

| 데이터셋 | 적용 지표 |
|---|---|
| 주 데이터셋(RE10K 또는 DL3DV) | PSNR·SSIM·LPIPS, reprojection consistency, 재구성 시간, Gaussian 수, peak VRAM |
| DTU(GT geometry 보유) | 위 지표 + depth AbsRel/RMSE, Chamfer Distance, floater 비율, free-space opacity |

floater와 free-space opacity는 GT geometry가 필요한 DTU 전용 지표다. 주 데이터셋은 **모델 공개 평가 프로토콜의 재현 가능성**을 기준으로 파일럿에서 확정한다.

### 5.7 계산 예산 시점의 결과 선택 규칙 — 테스트 누출 방지

**메인 비교는 예산 종료 시점 결과를 사용한다.** 방법 `m`과 예산 `B`에 대해 `t_B = max{t : t ≤ B}`, `Q_m(B) = Q_m(t_B)`로 정의한다. Optimization은 예산 `B` 직전의 마지막 체크포인트를 사용하며, **test PSNR이 가장 높은 체크포인트를 사후 선택하지 않는다.**

Feed-forward의 실측 추론 시간이 `B`를 초과하면 해당 예산 칸은 **No result**로 둔다. 추론 완료 이후의 더 큰 예산에서는 동일 출력이 사용 가능하므로 같은 품질을 표시하되 실제 완료 시간을 함께 표기한다. 테스트 PSNR 기준 최고점은 메인 결과가 아니라 **Oracle peak performance**로만 별도 분석한다. C1-b의 시간은 feed-forward 초기값 생성부터 refinement 종료까지의 누적 end-to-end로 계산한다.

### 5.8 C1-b 변환 호환성·렌더 등가성 검증 (refinement 전 필수 gate)

Feed-forward Gaussian을 gsplat/standard 3DGS 표현으로 변환할 때 **중심 좌표계, scene scale, scale 파라미터화, quaternion convention, opacity의 pre/post-sigmoid 상태, RGB/SH 표현**을 명시적으로 매핑한다. 변환 전 원래 렌더러와 변환 후 refinement 렌더러가 동일 입력 view에서 사실상 같은 영상을 출력하는지 확인하고, 허용 오차는 파일럿 전에 수치로 동결한다.

**렌더 등가성이 확보되지 않으면 C1-b를 진행하지 않는다.** 변환 자체의 오차가 refinement 효과로 오인될 수 있기 때문이다.

**2026-08-12 구현·실측**: `core/ff_gaussian_convert.py`(covariance 고유분해 → scale/quaternion, opacity inverse-sigmoid, harmonics 재배열)와 `analysis/check_renderer_equivalence.py`(gsplat 재렌더링 vs MVSplat 자체 decoder render 비교, cross-env이므로 `mvsplat_runner.py`가 저장한 `gaussians.pt`/`render_reference.pt`를 통해 hand-off)로 구현했다. 검증 절차:

- 변환 함수 자체는 합성 데이터 round-trip으로 먼저 검증(covariance 재구성 오차 최대 2.6e-6, quaternion 항상 unit-norm, 회전행렬 항상 det=+1) — 수학적으로 정확함을 확인.
- DTU scan1 2-view 실제 MVSplat 출력(13만 Gaussian)으로 held-out 7-view 전부 재렌더링해 비교: **mean MSE 0.00006~0.00028, mean abs diff 0.005(픽셀값 0~1 기준 약 0.5%)**. 오차는 background(alpha≈0) 영역에서는 거의 0(5.5e-5 vs 5.58e-5)이고 alpha coverage와의 상관은 0.55 — 변환 버그가 아니라 서로 다른 두 CUDA rasterizer(MVSplat 자체 fork vs gsplat) 간 흔한 수치적 차이(경계 픽셀의 blending 순서/정밀도)로 해석된다.

**config의 `renderer_equivalence_tolerance: 0.0001`은 파일럿 전에 채워야 할 추정치였는데, 실측해보니 이 값이 너무 타이트하다** — 두 개의 "정확히 같은" Gaussian을 서로 다른 정상 rasterizer로 렌더링해도 view별 MSE가 0.0001을 넘는 경우가 있었다(7-view 중 4개가 근소하게 초과). PSNR로 환산하면 실측 범위는 약 35.6~42.0dB — 우리가 실제로 비교하는 재구성 품질(대체로 8~25dB대)과는 충분히 구분되는 값이라, 이 정도 cross-renderer noise를 gate 통과 기준으로 삼아도 refinement 효과와 혼동될 위험은 낮다. 잠정 gate 기준을 MSE 0.0001(고정 픽셀 오차)에서 PSNR ≥ 33dB(view별)로 바꿔 C1-b 스케일업(§표 아래 참고)에 계속 적용해왔다.

**2026-08-13 최종 동결**: DTU/RE10K(MVSplat)·DL3DV(DepthSplat) 스케일업에서 쌓인 gate 로그 130건(개별 view-PSNR 샘플 390개)을 전부 모아 재검토했다.

| 구간 | n | min | p5 | median | max |
|---|---:|---:|---:|---:|---:|
| 전체 | 390 | 26.76dB | 35.55dB | 45.21dB | 60.11dB |
| MVSplat/RE10K | 240 | 28.98dB | 33.99dB | 43.63dB | 50.67dB |
| DepthSplat/DL3DV | 150 | 26.76dB | 42.27dB | 50.05dB | 60.11dB |

**PSNR ≥ 33dB 기준을 그대로 최종 확정한다.** 근거: 이 기준으로 390개 샘플 중 미달은 9개(2.3%)뿐이고 대부분 33dB 바로 아래(28.98~33dB 근방)에 몰려 있어 정상적인 cross-renderer noise 범위 안이다. DepthSplat은 개별 최솟값(26.76dB)이 MVSplat 최솟값(28.98dB)보다 낮아 "DepthSplat이 항상 더 정밀하다"고 단정할 순 없지만, 두 모델 모두 median이 43~50dB대로 33dB 기준선과 충분히 떨어져 있어 하나의 공통 기준으로 다뤄도 무리가 없다. 더 낮춰서(예: 30dB) 관대하게 잡을 수도 있었지만, 재구성 품질 범위(8~25dB대)와의 거리를 넉넉히 유지하는 쪽을 택해 33dB를 유지한다. 지금까지 스케일업에서 이미 이 값으로 실행해왔으므로 소급 변경 없음.

### 5.9 C2 개입 실험 — 대상과 범위

예측 depth를 back-projection해 초기 3D point를 만들 때 두 종류의 교란을 적용한다. 그 외 초기화 요소는 고정.

- **(a) iid 오차:** `d' = d(1 + ε), ε ~ N(0, σ²)`, **σ = 0 / 0.01 / 0.03 / 0.05 / 0.10**
- **(b) Global scale bias:** `d' = s·d`, **s = 0.9 / 0.95 / 1.0 / 1.05 / 1.1**. Monocular depth의 대표적 실패 모드인 scale ambiguity를 모사하며 구현 비용이 거의 없다

**주장 수위 제한:** 본 개입은 현실의 모든 depth 오류를 재현하는 것이 아니라, **초기 depth uncertainty 증가가 refinement 동역학에 미치는 민감도를 통제된 조건에서 측정**하는 것이다. 실제 depth 오류는 물체 경계·textureless 영역에서 구조적으로 발생하므로, spatially correlated noise는 여유가 있을 때 추가한다. Vanilla 3DGS의 COLMAP 초기화는 dense depth를 직접 입력받지 않으므로 개입 트랙은 VGGT/DA3 계열 depth back-projection 초기화에서 수행한다.

C2는 전체 실험 격자를 반복하지 않는다. 결과를 본 뒤 고르지 않도록 **파일럿 전에 대표 조건을 고정**한다: 최저 view–low overlap / 중간 view–low overlap / 중간 view–high overlap / 최고 view–high overlap. 12-view 미지원 시 최고 지원 view로 대체. DTU 5~8개 장면에서 깊게 분석하며, C1-b와 C2의 완성도를 selector보다 우선한다.

**2026-08-12 budget 결정(§5.4 GPU-hour 재계산에서 발견된 미결 항목)**: C2 row에는 원래 `budget_seconds`가 없어서(매니페스트 생성 코드에 빠져 있었음) 별도 예산으로 새로 정의할지 고민이 있었으나, **main phase/C1-b와 동일하게 `budgets_seconds=[1,10,60,300]` 체크포인트를 가진 단일 300s trajectory로 통일**하기로 한다. 근거:

1. 이미 존재하는 budget_snapshot 메커니즘을 그대로 재사용하므로 새 코드가 필요 없다.
2. Sensitivity analysis의 목적 자체가 "depth noise 효과가 학습 진행에 따라 어떻게 변하는가"이므로, 한 시점만 보는 것보다 전체 궤적(1/10/60/300s)을 남기는 쪽이 정보량이 많다.
3. 이 선택은 §5.4의 GPU-hour 추정에서 "C2 60s 가정 시 18.1h / 300s 가정 시 82.1h"로 4배 차이 나던 범위 중 **상한(300s, 82.1h)으로 확정**한다는 뜻이다 — 총 GPU-hour는 약 186h가 아니라 **약 250h**로 잡는다.
4. C1-b warm-start에서 발견한 "opacity reset이 짧은 예산에서 비정상 초기값(우리 경우는 FF warm-start, C2는 perturbed-depth init)을 파괴할 수 있다"(§5.8 인접 발견)는 위험이 C2에도 그대로 적용될 수 있다 — C2 파일럿 때 같은 현상이 재현되는지 반드시 확인하고, 필요하면 C1-b와 동일하게 `reset_every`를 조정한다.

### 5.10 로깅 항목

체크포인트는 예산 지점만 남기지 않고 **궤적 전체**를 기록한다. 최소 스키마:

```
experiment_id / scene / seed / method / iteration / wall-clock /
train loss / validation metric(사용 시) / test PSNR·SSIM·LPIPS /
Gaussian count / peak VRAM / checkpoint path
```

- **Optimization dynamics:** 품질 정점과 하강 시점으로 과적합 시작 지점 분석
- **Gaussian density:** Gaussian 수는 증가하나 품질이 정체·하락하는 구간을 densification 실패 증거로 분석
- **Selection audit:** 메인 체크포인트가 고정 예산 시점 규칙으로 선택되었는지 자동 검증하고 oracle 결과와 분리 저장

### 5.11 조기 종료 규칙 — 파일럿 전 결정 필요 (미결 항목)

Sparse 조건에서 optimization이 **언제 멈추는가**가 승패를 직접 좌우하지만, 현재 설계에는 정당한 정지 규칙이 없다. 예산 종료 컷은 공정하나 "더 일찍 멈췄어야 한다"는 반론에 약하고, oracle peak는 test leakage라 메인이 될 수 없다(ReSplat도 이 문제로 iteration 상한을 두었다).

| 선택지 | 평가 |
|---|---|
| ① 예산 컷만 (현행) | 단순·공정. 정지 규칙 부재 비판 감수 |
| ② Training-view loss 기반 조기 종료 | held-out 불필요하나 train loss는 계속 하강하므로 과적합을 못 잡음 |
| ③ Held-out validation view 기반 | 과적합 탐지 가능. 단 2·4-view에서 한 장을 빼면 입력의 25~50%가 소실되어 조건 자체가 바뀜 |

**권고:** ①을 메인으로 유지하되, view 수가 많은 조건(8·12-view)에 한해 ③을 **보조 실험**으로 1회 수행하여 "조기 종료가 있었다면 경계가 얼마나 이동했을까"를 보고한다. 메인의 일관성을 지키면서 정지 규칙 비판에 답할 수 있다. 최종 결정과 규칙은 파일럿 착수 전 동결한다.


### 5.12 통계 분석 계획 — 독립 단위와 추론

**실질적 독립 단위는 run이 아니라 scene이다.** 같은 장면의 seed 3회는 상관된 표본이므로 1,200 run을 표본 수처럼 해석하면 신뢰구간이 과소추정된다.

- **주 방법:** **scene 단위 cluster bootstrap** — 장면을 단위로 복원추출해 CI와 win rate를 산출. 구현이 단순하고 상관 구조를 올바르게 반영
- **보조(여유 시):** scene을 random effect로 둔 mixed-effects model로 `method × view 수 × overlap × budget` 상호작용 검정. 부록에 배치
- **보고:** 평균 + scene 단위 cluster bootstrap CI + 장면별 win rate를 함께 제시. seed는 반복 측정으로 처리하고 표본 수로 계상하지 않는다
- **다중 비교:** 조건 격자에서 다수의 비교가 발생하므로, 주 결론에 사용하는 검정에는 다중 비교 보정(Holm 등)을 적용하고 그 범위를 명시

---

## 6. 설계 과정에서 교정한 오류

내부 인수인계와 심사 대응을 위한 기록이다. 외부 제출용 요약본에서는 생략할 수 있다.

1. **협업 구도와 주제의 불일치** — 로보틱스와 3DGS를 동시에 다루는 초기안은 3개월 일정에 과도했다. 비전 단일 주제로 확정해 시뮬레이터·정책학습 스택 제거
2. **불공정한 pose 조건** — 두 진영에 서로 다른 품질의 pose를 제공하던 문제를 Pose-given track 통일로 교정
3. **초기화 3종의 잘못된 병렬 배치** — COLMAP/VGGT/DA3는 별도 패러다임이 아니라 optimization 내부 초기값 차이이므로 ablation으로 강등
4. **약한 기준선만 배치** — 최신 FF 대 vanilla 3DGS만 비교하는 편향을 막기 위해 sparse-view 특화 optimization을 강한 기준선으로 추가
5. **시간 예산 축의 부재** — 최종 PSNR만 비교하면 속도 차이가 지워지므로 계산 예산을 독립 축으로 승격하고 domain은 외부 검증으로 이동
6. **측정 불가능한 Gaussian 파라미터 오차** — 정답 Gaussian이 존재하지 않으므로(같은 표면을 100개로도 1,000개로도 표현 가능) depth·floater·free-space opacity·reprojection consistency 등 측정 가능한 프록시로 교정
7. **과도한 인과 주장** — 관찰만으로 인과를 주장하지 않고 통제된 depth noise 개입을 추가했으며, 나머지는 메커니즘 분석으로 표현
8. **모호한 측정 정의** — 시간 시작점, overlap 분모와 다중 view 집계, 예산 시점 규칙을 수식과 문장으로 고정
9. **교란요인과 패러다임 효과의 혼재** — C1-a는 대표 시스템의 실용적 우위로 주장 수위를 제한하고, C1-b에서 동일 초기값 standard refinement의 효과만 분석
10. **규모 산정과 누출 위험** — 장면 수·seed·GPU-hour를 명시하고 selector는 scene 단위 group split으로 누출 방지
11. **테스트 PSNR 기반 oracle 체크포인트 선택** — 예산 이내 최고 test PSNR을 고르면 test leakage가 발생하며, 특히 **과적합이 주제인 본 연구에서 과적합 현상 자체가 측정에서 지워진다.** 메인 결과를 예산 종료 시점 체크포인트로 고정하고 oracle peak는 부가 분석으로 분리
12. **FF → refinement 표현 변환 오차** — Gaussian 표현·렌더러 차이가 refinement 효과로 섞일 수 있으므로 파라미터 매핑과 렌더 등가성 검증을 C1-b의 사전 조건(gate)으로 추가
13. **기여의 병렬 나열** — C1-a/C1-b/C2가 따로 노는 구조였다. "현상 → 메커니즘 → 통제 → 처방"의 서사 사슬(§2.1)로 묶고 마지막에 C4 Practical Guideline을 배치해 "그래서?"에 답하도록 교정
14. **Overlap 측정의 선택 편향** — 유효 edge만으로 집계하면 SfM 매칭이 실패한 쌍(주로 low-texture·low-overlap)이 빠져 overlap이 과대평가되고, regime map의 x축 자체가 왜곡된다. non-edge를 0으로 포함하고 집계를 평균·하위 분위수로 바꾸며 view 수 내 층화를 도입해 교정
15. **통계 단위의 오인** — run을 독립 표본으로 취급하면 CI가 과소추정된다. 독립 단위를 scene으로 두고 cluster bootstrap으로 교정하며, τ도 seed 변동성 단독이 아니라 실용적 최소 차이와 병기
16. **개입 실험의 단순성** — iid multiplicative noise만으로는 실제 depth 오류(경계·textureless·scale bias)를 대표하지 못한다. global scale bias를 추가하고 주장 수위를 '민감도 측정'으로 제한

---

## 7. 일정 (12주)

| 주차 | 내용 |
|---|---|
| 1~2주 | 스모크 테스트: FF 2종 구동, 지원 view 표 확정, optimization 파이프라인·전체 궤적 로깅 구현, 시간·overlap·평가 코드 통일, **C1-b 파라미터 변환 및 렌더 등가성 검증** |
| 3주 | 파일럿(장면 5개): 주 데이터셋, overlap 경계, τ, 예산 지점, 렌더 등가성 허용 오차, C2 대표 조건, **조기 종료 규칙(§5.11)·C2 사전 가설(§3.1)·통계 계획(§5.12)** 동결 및 GPU-hour 재계산 |
| 4~6주 | 본 벤치마크(C1-a): 전 조합 × seed 3회, end-to-end regime map + post-initialization dynamics + Pareto frontier. FF 출력 전량 보존 |
| 7주 | C1-b: 동일 FF 초기값에서 standard 3DGS refinement off/on. 변환 호환성 실패 시 C1-b 축소 또는 한 모델만 수행 |
| 8~9주 | C2 핵심 분석: 대표 조건의 기하 프록시, depth noise 개입, DTU 5~8개 장면 외부 검증. **이 구간을 논문의 심장으로 우선 보호** |
| 10주 | 초기화 ablation, 시각화, failure case, 지표 불일치 분석, **C4 Guideline 도출**. C1+C2가 완성된 경우에만 selector 착수 |
| 11~12주 | 집필·수정. Intro/Related Work 초안은 4~6주와 병행. 외부 공유용 2~3쪽 요약본 별도 작성 |

---

## 8. 역할 분담

| 담당 | 범위 |
|---|---|
| 연구자 A | Optimization 파이프라인, standard refinement 변환·등가성 검증, 시간 프로토콜, C2 기하·개입 실험, regime map 설계 |
| 연구자 B | Feed-forward 셋업·추론, 지원 view 검증, overlap 측정기, 평가 지표 통합, 여유 시 selector |
| 공동 | 파일럿 설계, 사전 확정 12항목 동결, 결과 해석, 집필, 실험 기준 변경 승인 기록 |

---

## 9. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| "모델 차이이지 패러다임 차이가 아니다" | C1-a는 실용적 시스템 비교로 제한하고, C1-b는 동일 초기값에서 특정 standard refinement의 효과만 주장 |
| FF Gaussian 변환 비호환 | 1~2주 차 렌더 등가성 검증을 gate로 사용. 실패 시 한 모델만 C1-b에 사용하거나 제한적으로 보고 |
| 정지 규칙 부재 비판 | §5.11 결정을 파일럿 전 동결하고, 8·12-view 보조 실험으로 경계 이동 폭을 함께 보고 |
| **결과가 밋밋함**(예: 한 모델이 전 구간 우세) | 승패 자체보다 원인 분석(C2)과 Pareto·Guideline이 기여가 되도록 설계. "view 수보다 overlap이 승패를 더 크게 결정한다", "일정 geometry 품질 이상에서는 refinement 효과가 소멸한다" 같은 관찰을 적극 탐색 |
| Crossover가 선명하지 않음 | 역전이 특정 조건에만 발생한다는 결과와 품질–시간 Pareto 지도 자체를 knowledge 기여로 정리 |
| 코드베이스 통합 난항 | 좌표계·전처리·평가 이미지 동일성 체크리스트, 공용 로그 스키마, 실패 모델 즉시 교체·제외 |
| GPU 시간 초과 | 축소 순서: seed 3→2, 외부 검증 장면 축소, 초기화 ablation 축소, view 단계 축소. C3를 가장 먼저 포기 |
| C2 실험 폭발 | 전체 격자가 아니라 사전 동결한 대표 조건과 DTU 5~8개 장면에 집중 |

---

## 10. 논문 구성과 분량 배분 (7~8쪽 기준)

| 절 | 분량 | 내용 |
|---|---|---|
| 1. Introduction | 1쪽 | Sparse-view 문제 → feed-forward 등장 → 언제 무엇을 쓸지 모른다 → 본 논문의 목적 |
| 2. Related Work | 1쪽 | Per-scene optimization / Feed-forward / Hybrid(ForeSplat·Diff3R) / 기존 benchmark |
| 3. Experimental Protocol | 1~1.5쪽 | 통제 축, overlap 정의, 예산·시간 측정, 지표, 공정성 원칙 (§4~5 축약) |
| 4. Results: Regime Map | 2쪽 | 승패 지도, Pareto frontier, 장면별 win rate, 지표 불일치 |
| 5. Failure Analysis | 2쪽 | 사전 가설 H1~H3 검증, 기하 프록시, optimization dynamics, Gaussian density, depth noise·scale bias 개입, C1-b on/off |
| 6. Practical Guideline | 0.5쪽 | 조건별 선택 규칙 표 |
| 7. Conclusion & Limitations | 0.5쪽 | 요약, 한계, 후속 regime-aware hybrid |

Figure 우선순위: ① Regime Map(대표 그림) ② 품질–시간 Pareto ③ optimization dynamics 곡선 ④ Gaussian count vs 품질 ⑤ failure case 정성 비교.

---

## 11. 후속 확장 (석사 연구 연결)

후속 연구는 **regime-aware hybrid**다. 입력 조건과 계산 예산을 보고 refinement를 수행할지, 어느 영역을 얼마나 고칠지를 결정하는 선택적 최적화로 확장한다. 본 논문의 경계 분석과 failure analysis가 후속 method 연구에서 "언제·어디를 고칠지"를 정하는 근거가 된다.

---

## 12. 지금부터 할 일 — 실행 순서

원칙: **논문 읽기와 코드를 병행하되, 읽기가 코드를 이끈다.** 각 논문에서 아래 5개 항목을 채우는 것을 읽기 완료 기준으로 삼는다.

| ① 입력 가정 | ② 학습 데이터 | ③ 평가 프로토콜 | ④ 저자 한계 | ⑤ 우리가 상속할 것 |
|---|---|---|---|---|
| pose·view 범위 | 데이터셋·해상도 | split·context/target 선정 | limitation 절 | 체크포인트·코드·설정 |

### STEP 0. 착수 전 (이번 주)

1. 공동연구자에게 본 문서를 공유하고 역할 분담과 **사전 확정 12항목의 책임자**를 문장으로 남긴다
2. 공용 repo, experiment ID, 로그 스키마, 설정 파일 버전, 주 1회 동기화 시간을 정한다
3. **메인 결과와 oracle 결과의 디렉터리·파일명을 처음부터 분리**해 test leakage를 구조적으로 방지한다

### STEP 1. 1~2주 차 — 읽기와 셋업 병행

- **읽기(A):** ReSplat의 per-scene 세팅, iteration 상한, 8-view 구성, 조기 종료 근거 정리 → §5.1·§5.5·§5.11의 직접 근거
- **읽기(A):** 3DGS 원논문 재확인 — densification·opacity reset·pruning이 왜 필요한지, gradient가 무엇을 의미하는지, 언제 최적화가 실패하는지. 이 질문에 답할 수 있으면 feed-forward 논문의 절반은 "이 한계를 어떻게 우회하는가"로 읽힌다
- **읽기(B):** MVSplat → DepthSplat 순으로 평가 프로토콜, depth feature 주입 위치, 지원 view 범위 정리
- **코드(A):** gsplat 기반 vanilla optimization + 전체 궤적 로깅으로 1개 장면 학습·평가 성공
- **코드(B):** DepthSplat·MVSplat을 동일 장면에서 구동하고 2/4/8/12 view 지원 여부 확인 → §5.2 표 채우기
- **코드(공동):** 동일 평가 이미지에서 PSNR·SSIM·LPIPS가 일치하는지 교차 검증 — **이 단계를 건너뛰면 이후 모든 비교가 무효**
- **C1-b gate:** FF Gaussian → standard 표현 변환 후 원래 렌더러와 변환 렌더러의 출력 등가성 확인

*완료 기준: 지원 view 표, 시간 경계, 평가 일치, C1-b 렌더 등가성 결과가 채워진다.*

### STEP 2. 3주 차 — 파일럿

장면 5개로 전 조합을 1회 실행한다. overlap 분포와 seed 변동성으로 고/저 경계와 τ를 확정하고, RE10K·DL3DV 후보를 시험해 주 데이터셋을 정한다. C2 대표 조건·사전 가설(§3.1), 렌더 등가성 허용 오차, 조기 종료 규칙(§5.11), 통계 분석 계획(§5.12)도 이 시점에 동결하며 실측 GPU-hour를 재계산한다.

*완료 기준: 사전 확정 12항목을 설정 파일·표·회의 기록으로 동결한다. 이후 기준 변경은 **버전과 사유를 공개 기록**한다.*

### STEP 3. 4~6주 차 — 본 벤치마크

전 조합 × seed 3회를 자동 실행하고 중단·재개가 가능하도록 구성한다. Feed-forward 출력은 C1-b 재사용을 위해 전량 보존한다. 메인 결과는 고정 예산 종료 시점 체크포인트로 작성하고 oracle peak는 별도 파일로 관리한다. 실험이 도는 동안 Intro·Related Work 초안을 병행한다.

### STEP 4. 7~9주 차 — 원인 분석

C1-b 동일 초기값 off/on → C2 기하 프록시 → depth noise 5수준 → DTU 외부 검증 순으로 진행한다. 일정이 밀리면 selector와 ablation을 줄이고 **이 구간을 지킨다.**

### STEP 5. 10~12주 차 — 마무리

초기화 ablation, 시각화, failure case, 지표 불일치 분석, C4 Guideline 도출, 집필을 진행한다. 10주 차에 C1+C2가 충분히 완성된 경우에만 selector에 착수한다.

---

### 흔들릴 때 돌아올 기준 세 가지

1. **기준은 결과보다 먼저:** 측정·선택 규칙은 파일럿에서 동결하고 test 결과를 본 뒤 바꾸지 않는다
2. **"왜"가 "누가"보다 중요하다:** 표 하나를 늘리는 것보다 실패 원인을 설명하는 증거를 우선한다
3. **줄일 때는 정해진 순서로:** seed → 외부 검증 장면 → ablation → view 단계. C3는 언제든 분리한다
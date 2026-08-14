# Implementation Summary — updated 2026-08-13

이 문서는 지금까지 작성한 코드와 문서의 역할을 한 곳에 정리한다. 프로젝트는 scaffold 단계를 완전히 지나 **세 방법론(Vanilla3DGS/MVSplat/FSGS)이 RE10K에서 실제로 맞붙는 본 실험(C1-a)이 30-scene 규모로 진행 중**이다. 다음 병목은 코드가 도는지가 아니라 GPU 시간(현재 실행 중인 job이 약 40시간 소요)과 남은 축(overlap_level)의 확장이다.

## 핵심 방향

본 프로젝트는 새 3DGS 방법을 만드는 코드가 아니라, sparse-view 조건에서 feed-forward와 per-scene optimization의 우위가 언제·왜 바뀌는지 측정하는 실험 scaffold다. 코드의 역할은 overlap 축, 시간 예산, checkpoint 선택, 입력 해상도, 통계 단위를 실험 전에 고정하고, 세 종류의 서로 다른 모델을 같은 조건에서 공정하게 비교할 수 있게 만드는 것이다.

---

## 우리가 비교하는 네 가지 방법론 — 목적과 원리

같은 문제("사진 몇 장으로 3D 장면을 재구성하기")를 푸는 두 가지 완전히 다른 접근을 비교한다. 앞의 둘(MVSplat, DepthSplat)은 **feed-forward**(추론만), 뒤의 둘(Vanilla3DGS, FSGS)은 **per-scene optimization**(그 장면 전용으로 학습) 계열이다.

### MVSplat — feed-forward, 우리의 "빠른 쪽" 대표

**쉽게**: 사진 몇 장을 모델에 넣으면, 미리 학습해둔 신경망이 "이 픽셀은 카메라에서 대략 이 정도 거리에 있겠다"를 한 번에 계산해서 3D Gaussian들을 즉석에서 만들어낸다. 학습은 이미 끝나 있으니 실제 사용할 땐 몇 초도 안 걸린다. 대신 학습 때 본 상황(RE10K에서는 정확히 2장짜리 입력)과 많이 다른 조건(예: 사진을 12장 준다거나)에서는 원래 실력이 잘 안 나온다 — 배운 적 없는 시험을 보는 셈이다.

**전문 용어**: cost-volume 기반 depth 추정 + Gaussian parameter regression을 단일 forward pass로 수행. 우리가 쓰는 RE10K 체크포인트는 고정 2-view로 학습됨(`config/dataset/view_sampler/bounded.yaml`) — 4/8/12-view 사용은 명시적으로 분포 밖(OOD, §5.2에서 실측 확인) 취급한다.

**우리 코드에서**: `mvsplat_runner.py`(DTU), `mvsplat_re10k_runner.py`(RE10K, C1-a/C1-b 겸용).

### DepthSplat — feed-forward, MVSplat의 "더 넓은 view 범위" 버전

**쉽게**: MVSplat과 같은 계열(사진→3D를 한 번에)인데, "사진 한 장만 봐도 대충 깊이를 추측하는" 보조 모델(monocular depth)의 힘을 같이 빌려서, 2장이 아니라 2~6장처럼 좀 더 다양한 개수의 입력에도 버틸 수 있게 만든 버전이다.

**전문 용어**: monodepth backbone을 cost-volume에 융합. 우리가 쓰는 체크포인트(`depthsplat-gs-base-dl3dv-256x448-randview2-6`)는 DL3DV에서 2~6-view 랜덤 샘플링으로 학습됨 — 8/12-view는 이 체크포인트 기준 분포 밖.

**우리 코드에서**: `depthsplat_dl3dv_runner.py`(C1-b 파이프라인에 연결, full trajectory 로깅은 아직 미완성).

### Vanilla 3DGS — per-scene optimization, 우리의 "느린 쪽" 기준선

**쉽게**: 원조 3D Gaussian Splatting 방식. feed-forward처럼 "한 번에 뚝딱" 만드는 게 아니라, 그 장면 전용으로 처음부터 학습시킨다 — 사진들을 반복해서 보면서 "이 부분은 이렇게, 저 부분은 저렇게" 조금씩 고쳐나간다(경사하강법). 시간이 오래 걸리는 대신, 시간을 충분히 주면 feed-forward보다 훨씬 정교한 결과가 나올 수 있다 — "빨리 대충" vs "천천히 정교하게"의 대비가 이 연구의 핵심 질문(언제 어느 쪽이 이기는가)이다.

**전문 용어**: COLMAP sparse triangulation(또는 실패 시 random-sphere fallback, `MIN_SFM_POINTS=200` 기준)으로 초기화한 Gaussian을, gradient 기반 densification(grow/split/prune, `gsplat.strategy.DefaultStrategy`)으로 반복 최적화. 오늘 이 densification 메커니즘 자체를 코드로 추적해 "sparse-view에서 Gaussian 예산의 절반 이상이 관측 부족으로 죽은 채 방치된다"는 걸 실측으로 확인했다(§`paper_gaussian_observation_starvation_2026-08-13.md`).

**우리 코드에서**: `vanilla_3dgs_runner.py` — DTU/RE10K/DL3DV 세 데이터셋 다 지원, warm-start(C1-b) 경로와 일반 COLMAP-init(C1-a) 경로 둘 다 있음.

### FSGS — per-scene optimization, "sparse-view 특화" 강한 기준선

**쉽게**: Vanilla3DGS랑 같은 "장면 전용 학습" 계열인데, "사진이 몇 장 안 되는 상황"에 맞춰 특화된 보조 기법들이 추가로 들어있다. 예를 들어 (1) 실제로 찍지 않은 가상의 카메라 위치(pseudo-view)를 상상해서 "이쪽에서 보면 대충 이런 깊이여야 한다"는 힌트를 스스로에게 추가로 주고, (2) monocular depth 모델(MiDaS)의 깊이 추정을 학습 신호로 같이 활용한다. 이런 기법 없이도 사진이 많으면 Vanilla3DGS가 알아서 잘 배우지만, 사진이 적을 때는 이런 "추가 힌트"가 도움이 될 거라는 가설로 만들어진 방법이다. 이 방법이 있어야 "Vanilla3DGS가 진 건 sparse-view 특화 기법을 안 써서가 아니냐"는 반박을 막을 수 있다 — 그래서 강한 기준선으로 포함했다.

**전문 용어**: proximity-guided pseudo-view sampling + depth correlation loss(MiDaS 기반, Pearson correlation) + 표준 L1/D-SSIM. 우리는 FSGS 원 논문의 dense-MVS 초기화 대신 Vanilla3DGS와 동일한 sparse COLMAP 초기화를 쓴다(공정성을 위한 의도적 편차, `overall.md` §4.2에 명시).

**우리 코드에서**: `fsgs_runner.py`(2026-08-13 신규) — FSGS 원본 repo의 Scene/GaussianModel/render()/loss를 그대로 재사용하되 바깥 학습 루프만 우리 wall-clock budget 체계로 감쌌다. DTU/RE10K/DL3DV 세 데이터셋 지원.

---

## 현재 디렉토리 구조

- `experiments/configs/experiment_config.yaml`: 전체 실험 축과 protocol guard 설정. view 수, overlap level, seed(현재 `[0,1]`), scenes_primary(현재 30), budget, C1-b, C2, 통계 설정.
- `experiments/scripts/core/`: 여러 runner가 공유하는 핵심 모듈.
  - `protocol_utils.py`: overlap 집계, budget checkpoint 선택, tau 계산, scene cluster bootstrap.
  - `model_registry.py`: method별 conda python, runner script, external repo, default checkpoint 관리 (DepthSplat/MVSplat/Vanilla3DGS/**FSGS**).
  - `dtu_dataset.py`, `re10k_dataset.py`, `dl3dv_dataset.py`: 데이터셋별 camera/image loader.
  - `colmap_init.py`: pose-given COLMAP triangulation(데이터셋 무관 공용 코어)과 random fallback.
  - `ff_gaussian_convert.py`: feed-forward Gaussian → gsplat 파라미터 변환기(C1-b용).
  - **`view_selector.py`**(신규): co-visibility 기반 view selector — farthest-point-sampling을 좁은 window(high overlap)/전체 범위(low overlap) 두 pool에 적용.
- `experiments/scripts/runners/`: protocol_utils 스키마를 따르는 정식 모델 러너.
  - `vanilla_3dgs_runner.py`, `mvsplat_runner.py`, `mvsplat_re10k_runner.py`, `depthsplat_dl3dv_runner.py`, **`fsgs_runner.py`**(신규).
- `experiments/scripts/batch/`: manifest 생성과 batch 실행 driver.
  - `run_experiment.py`: protocol manifest 생성(scene×seed×view×overlap×budget×method 전체 격자).
  - `run_re10k_c1a_pilot.py`: C1-a 5-scene 파일럿(완료).
  - **`run_re10k_c1a_main.py`**(신규): C1-a 30-scene 본 실험 — 지금 백그라운드에서 실행 중.
  - `run_re10k_c1b_scaleup.py`, `run_dl3dv_c1b_scaleup.py`: C1-b 20-scene 스케일업.
- `experiments/scripts/analysis/`: overlap report, view selector 검증, geometry uncertainty, gradient accumulation 등.
  - `generate_overlap.py`, `generate_re10k_view_overlap.py`, `generate_dl3dv_view_overlap.py`(v2): overlap 측정.
  - `generate_re10k_overlap_candidates.py`, `generate_dl3dv_overlap_candidates.py`(신규): selector 검증.
  - `prep_dtu_for_fsgs.py`(신규): FSGS용 데이터 prep(DTU 전용 + dataset-agnostic 버전 둘 다).
  - `gaussian_gradient_accumulation_probe.py`(신규): per-Gaussian densification gradient 관측 실험.
  - `geometry_uncertainty_figure.py`: baseline/overlap/depth uncertainty 관계 실측 + confound 분석.
- `experiments/docs/paper/`: 논문용 분석 문서 — 수식 정리(`paper_equations_reference.md`), geometry confound 분석, observation starvation 분석.
- `experiments/docs/reference/`: 이 문서, 데이터셋 인용(`dataset_citations.md`), checkpoint-domain 표.

## 현재 검증 상태 (2026-08-13 기준)

- Protocol unit tests: 9개 통과(변동 없음).
- 세 방법론(Vanilla3DGS/MVSplat/FSGS) 모두 DTU + RE10K + DL3DV에서 실제 학습·평가 완료.
- **C1-a 첫 파일럿**(RE10K 5 scene, seed×3): view가 적으면(2/4-view) MVSplat이 압승, 12-view·budget≥60s에서 Vanilla3DGS가 확실히 역전(τ 적용 후에도 robust). 8-view는 pooled 평균상 "역전"처럼 보였지만 τ 적용·scene별 분해 결과 Tie로 정정.
- **C1-a 본 실험**: RE10K 30 scene × view×budget×method×seed 격자로 착수, 현재 진행 중(~40시간 예상).
- Co-visibility selector: RE10K 30 scene 전체 98.3%, DL3DV 25 scene 전체 94% 조건에서 의도한 대로 동작 확인.
- Per-Gaussian gradient observation 실험: sparse-view에서 densification 신호가 왜 약해지는지에 대한 정량적 메커니즘 확인(§`paper_gaussian_observation_starvation_2026-08-13.md`).

## 중요한 프로토콜 규칙 (변경/추가분만)

기존 규칙(non-edge=0, budget/oracle 분리, scene cluster bootstrap, 해상도 상속, renderer equivalence gate)은 유지. 오늘 추가/확정된 것들:

### 7. τ(승패 판정 임계값)는 view_count에 따라 다르다

seed 노이즈 실측 결과 2/4-view(0.04~0.17dB)와 8/12-view(0.13~1.41dB)가 한 자릿수 이상 차이 나서, 단일 τ 대신 구간별 τ(0.5dB / 1.4dB)를 쓴다. 이 결정 과정에서 기존에 "8-view 역전"이라 서술했던 것이 pooled 평균의 착시였음을 발견·정정했다(`overall.md` §5.12).

### 8. FSGS는 sparse COLMAP init을 쓴다(dense-MVS 아님)

원 논문 프로토콜과의 의도적 편차 — Vanilla3DGS와 같은 초기화를 써야 두 optimization 방법의 알고리즘 차이만 격리된다. 논문 methods/limitations 양쪽에 명시 예정(`overall.md` §4.2).

### 9. Co-visibility selector는 "선택"이지 "측정 후 분류"가 아니다

overlap_level(고/저) 축은 같은 scene에서 두 가지 다른 view 후보를 **직접 선택**해서 만든다(좁은 window FPS = high, 전체 범위 FPS = low) — 사후에 threshold로 scene을 분류하는 방식이 아니다. `run_experiment.py`의 manifest가 scene×view×overlap의 완전 교차를 전제하고 있어서 이 방식이 필요했다.

## 현재 로컬 데이터

- RE10K: `/data/Re-feem/datasets/re10k`, test 114 scene 확보, **main subset 30 scene**(`re10k_main_subset.json`, 20→30 추가 확장).
- DL3DV: `/data/Re-feem/datasets/dl3dv`, 25 scene(pilot 규모, 공식 eval split과 중복 0 확인).
- DTU: `/data/Re-feem/datasets/dtu`, 공식 split 16개 포함 총 29 scan.
- External repos/checkpoints: `/data/Re-feem/code/mvsplat`, `/data/Re-feem/code/depthsplat`, `/data/Re-feem/code/fsgs`.

## 다음 작업 (2026-08-13 갱신 — 실제 checklist와 동기화)

1. ~~Main benchmark를 RE10K-first로 확정~~ ✅
2. ~~DepthSplat 정식 runner 승격~~ 🔶 부분 완료(full trajectory 로깅은 아직)
3. ~~RE10K chunk loader와 256x256 protocol을 Vanilla3DGS runner에 연결~~ ✅
4. ~~co-visibility 기반 view selector~~ ✅ (전체 scene 규모 검증까지 완료)
5. ~~MVSplat/DepthSplat의 지원 view 수·confidence 출력 유무 표 완성~~ ✅
6. ~~densification on/off CLI~~ ✅
7. ~~§5.4 GPU-hour 예산 갱신~~ ✅ (실측 기반 250h로 확정)
8. **[진행 중]** C1-a 본 실험(30 scene) 완료 대기
9. **[다음]** overlap_level 축을 C1-a 본 실험에 추가(selector는 검증 끝났으니 실행만 남음)
10. **[다음]** DL3DV 밀린 것들 — DepthSplat C1-b를 v2 view 선택으로 재실행, Vanilla3DGS/MVSplat DL3DV 통제 스모크
11. **[다음]** C1-b/C2 본격 착수(C1-a 완료 후)

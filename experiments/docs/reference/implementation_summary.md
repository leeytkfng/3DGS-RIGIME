# Implementation Summary — updated 2026-08-11

이 문서는 지금까지 작성한 코드와 문서의 역할을 한 곳에 정리한다. 현재 프로젝트는 단순 scaffold 단계를 지나, **Vanilla 3DGS / MVSplat / DepthSplat이 모두 실데이터에서 최소 검증된 상태**다. 다음 병목은 코드가 도는지 여부가 아니라 main dataset 결정, view selection 통제, runner 정식화다.

## 핵심 방향

본 프로젝트는 새 3DGS 방법을 만드는 코드가 아니라, sparse-view 조건에서 feed-forward와 per-scene optimization의 우위가 언제 바뀌는지 측정하는 실험 scaffold다. 따라서 코드의 가장 중요한 역할은 overlap 축, 시간 예산, checkpoint 선택, 입력 해상도, 통계 단위를 실험 전에 고정하는 것이다.

## 현재 디렉토리 구조

- `experiments/configs/experiment_config.yaml`: 전체 실험 축과 protocol guard 설정. view 수, overlap level, seed, budget, C1-b, C2, 통계 설정이 들어 있다.
- `experiments/scripts/core/`: 여러 runner가 공유하는 핵심 모듈.
  - `protocol_utils.py`: overlap 집계, budget checkpoint 선택, tau 계산, scene cluster bootstrap.
  - `model_registry.py`: method별 conda python, runner script, external repo, default checkpoint 관리.
  - `dtu_dataset.py`: DTU camera/projection parsing과 loader.
  - `colmap_init.py`: pose-given COLMAP triangulation과 random fallback.
- `experiments/scripts/runners/`: protocol_utils 스키마를 따르는 정식 모델 러너.
  - `vanilla_3dgs_runner.py`: gsplat 기반 per-scene optimization runner.
  - `mvsplat_runner.py`: MVSplat inference runner.
- `experiments/scripts/probes/`: 정식 runner 전 1회성 검증 스크립트.
  - `mvsplat_re10k_probe.py`: RE10K in-domain probe.
  - `depthsplat_dl3dv_probe.py`: DepthSplat DL3DV probe.
- `experiments/scripts/batch/`: manifest 생성과 batch 실행 driver.
  - `run_experiment.py`, `run_experiment.sh`: protocol manifest 생성.
  - `run_experiment_batch.py`: registry 기반 일반 batch driver.
  - `run_dtu_batch.py`, `run_dtu_dense_sanity.sh`: DTU 검증용 batch.
- `experiments/scripts/analysis/`: overlap report와 geometry uncertainty figure 생성.
  - `generate_overlap.py`: SfM visibility에서 pairwise overlap 생성.
  - `geometry_uncertainty_figure.py`: baseline/overlap/depth uncertainty 관계 실측.
- `tests/test_protocol_utils.py`: 중요한 프로토콜 규칙이 깨지지 않는지 확인하는 단위 테스트.
- `experiments/docs/`: 일일 보고서, 데이터 확보 현황, checkpoint-domain 표, 수식 정리, 논문 읽기 로그.

## 현재 검증 상태

- Protocol unit tests: 9개 통과.
- Manifest 생성: `bash experiments/scripts/batch/run_experiment.sh` 정상 동작.
- DTU official split batch: 16 scan x 2-view x seed0에서 Vanilla3DGS와 MVSplat 로그 생성 완료.
- DTU dense-view sanity: scan1, 42 train view, 30k iteration에서 PSNR 24.0~24.1dB, SSIM 0.843, LPIPS 0.218. sparse pilot의 10dB대 결과가 단순 배관 고장 때문일 가능성은 낮아졌다.
- MVSplat RE10K in-domain probe: mean PSNR 25.6dB, 추가 mirror chunk 22.4dB. DTU zero-shot 저하가 OOD penalty였음을 뒷받침한다.
- DepthSplat DL3DV in-domain probe: near-context mean PSNR 20.0dB. view 선택이 성능을 크게 좌우한다는 신호를 확인했다.
- Geometry uncertainty 1차 figure: baseline과 uncertainty 관계는 이론 방향이지만 overlap과 uncertainty 부호가 예상과 달라 A-1 재독 후 재해석 필요.
- DTU scan1 2/4/8/12 control smoke: `experiments/scripts/analysis/generate_dtu_view_overlap_smoke.py`로 view set과 overlap report를 만들고, `run_dtu_batch.py`로 Vanilla3DGS(10초)와 MVSplat을 모두 통과시켰다.

## 중요한 프로토콜 규칙

### 1. Non-edge pair를 0으로 포함

Overlap 계산에서 매칭 실패 pair를 제외하면 low-texture 또는 low-overlap 장면의 overlap이 실제보다 높게 계산된다. 그래서 `protocol_utils.compute_pairwise_overlaps()`는 공통 point가 부족하거나 매칭이 실패한 pair를 버리지 않고 `overlap = 0`으로 기록한다.

### 2. Budget checkpoint와 oracle 분리

Optimization run에서 test PSNR이 가장 높은 checkpoint를 고르면 test leakage가 생긴다. `budget_checkpoint()`는 예산 직전 마지막 checkpoint만 고르고, `oracle_checkpoint()`는 diagnostic helper로만 둔다.

### 3. Scene 단위 bootstrap

같은 scene의 seed 3회는 독립 표본이 아니다. `scene_cluster_bootstrap_ci()`는 seed run을 scene 안에서 평균내고 scene을 cluster로 resampling한다.

### 4. 입력 해상도 상속

Feed-forward checkpoint가 학습한 입력 해상도가 protocol이다. RE10K main은 256x256, DepthSplat DL3DV track은 256x448을 따른다. Vanilla3DGS/SparseGS도 같은 이미지로 학습·평가해야 공정하다.

### 5. View selection 통제

DepthSplat probe에서 context 간격만 바꿔도 PSNR이 크게 달라졌다. 정식 실험은 임의 프레임 간격이 아니라 co-visibility 기반 selector와 overlap bucket을 써야 한다.

### 6. C1-b renderer equivalence gate + densification on/off

Feed-forward Gaussian을 standard 3DGS 표현으로 변환할 때 renderer equivalence gate를 통과해야 refinement 효과를 해석할 수 있다. 추가로 densification on/off 조건이 H1 해석에 중요해졌으므로 C1-b에 포함한다.

## 현재 로컬 데이터

- RE10K: `/data/Re-feem/datasets/re10k`, test 114 scene, 약 1.2GB.
- DL3DV: `/data/Re-feem/datasets/dl3dv`, 25 scene, 약 1.9GB.
- DTU: `/data/Re-feem/datasets/dtu`, 공식 split 16개 포함 총 29 scan.
- External repos/checkpoints: `/data/Re-feem/code/mvsplat`, `/data/Re-feem/code/depthsplat`.

## 다음 작업

1. Main benchmark를 RE10K-first로 확정할지 결정하고 20~30 scene subset index를 만든다.
2. DepthSplat 정식 runner를 작성해 probe에서 protocol_utils 스키마로 승격한다.
3. RE10K chunk loader와 256x256 protocol을 Vanilla3DGS runner에 붙인다.
4. RE10K 본 실험용 co-visibility 기반 view selector를 공통 모듈로 만들고 overlap bucket 생성과 연결한다. DTU scan1 smoke에서는 도구 검증 완료.
5. MVSplat/DepthSplat의 지원 view 수와 confidence 출력 유무를 채워 §5.2 표를 완성한다.
6. densification on/off CLI를 Vanilla3DGS runner에 추가하고 C1-b manifest에 반영한다.
7. §5.4 GPU-hour 예산을 256x256 파일럿 속도와 병렬화 무효 실측으로 갱신한다.

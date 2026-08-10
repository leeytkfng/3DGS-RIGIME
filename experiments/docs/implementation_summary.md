# Implementation Summary

이 문서는 지금까지 작성한 코드와 문서의 역할을 한 곳에 정리한다. 목적은 실험 코드를 붙이기 전에 프로토콜의 중요한 규칙을 추적 가능하게 만드는 것이다.

## 핵심 방향

본 프로젝트는 새 3DGS 방법을 만드는 코드가 아니라, sparse-view 조건에서 feed-forward와 per-scene optimization의 우위가 언제 바뀌는지 측정하는 실험 scaffold다. 따라서 코드의 가장 중요한 역할은 성능 숫자를 빨리 뽑는 것이 아니라, overlap 축, 시간 예산, checkpoint 선택, 통계 단위를 실험 전에 고정하는 것이다.

## 주요 파일

- `experiments/configs/experiment_config.yaml`: 전체 실험 축과 protocol guard 설정. view 수, overlap level, seed, budget, C1-b, C2, 통계 설정이 들어 있다.
- `experiments/scripts/run_experiment.py`: config를 읽어 전체 실험 manifest를 만든다. 실제 모델 실행기는 아직 붙어 있지 않지만, 어떤 run을 해야 하는지와 어떤 규칙을 적용해야 하는지 먼저 고정한다.
- `experiments/scripts/protocol_utils.py`: 논문 프로토콜에서 반복 사용되는 계산 규칙 모음. overlap 집계, budget checkpoint 선택, tau 계산, scene cluster bootstrap을 담당한다.
- `experiments/scripts/generate_overlap.py`: SfM visibility에서 overlap report를 생성하는 CLI. COLMAP `images.txt` 또는 단순 JSON 입력을 받아 `summary.json`과 `pairwise_overlap.csv`를 쓴다.
- `tests/test_protocol_utils.py`: 중요한 프로토콜 규칙이 깨지지 않는지 확인하는 단위 테스트.
- `experiments/docs/early_experiment/README.md`: 본 실험 전에 동결해야 하는 STEP 0~2 절차 문서.

## 어려운 부분과 주의점

### 1. Non-edge pair를 0으로 포함

Overlap 계산에서 매칭 실패 pair를 제외하면 low-texture 또는 low-overlap 장면의 overlap이 실제보다 높게 계산된다. 그래서 `protocol_utils.compute_pairwise_overlaps()`는 공통 point가 부족하거나 매칭이 실패한 pair를 버리지 않고 `overlap = 0`으로 기록한다. 이 규칙이 regime map의 x축을 결정한다.

### 2. COLMAP `images.txt` 파싱

COLMAP text export의 `images.txt`는 이미지 하나당 두 줄로 구성된다. 첫 줄은 이미지 metadata이고, 둘째 줄은 `(x, y, POINT3D_ID)` triplet이다. `POINT3D_ID = -1`은 SfM track이 아니므로 visibility에서 제외한다.

### 3. Budget checkpoint와 oracle 분리

Optimization run에서 test PSNR이 가장 높은 checkpoint를 고르면 test leakage가 생긴다. 특히 본 연구는 sparse 조건의 과적합을 분석하므로 oracle 선택을 메인 결과에 쓰면 핵심 현상이 지워진다. `budget_checkpoint()`는 예산 직전 마지막 checkpoint만 고르고, `oracle_checkpoint()`는 diagnostic helper로만 둔다.

### 4. Scene 단위 bootstrap

같은 scene의 seed 3회는 독립 표본이 아니다. `scene_cluster_bootstrap_ci()`는 seed run을 scene 안에서 평균내고 scene을 cluster로 resampling한다. 이 규칙은 confidence interval이 지나치게 좁아지는 것을 막는다.

### 5. C1-b renderer equivalence gate

Feed-forward Gaussian을 standard 3DGS 표현으로 변환할 때 좌표계, scale, quaternion, opacity, SH/RGB convention이 틀리면 refinement 효과가 아니라 변환 오류를 측정하게 된다. 그래서 manifest에는 C1-b run마다 `renderer_equivalence_gate = true`를 기록한다.

## 현재 검증 상태

- 단위 테스트: `python3 -m unittest discover -s tests`
- manifest 생성: `bash experiments/scripts/run_experiment.sh`
- overlap CLI 샘플 실행 확인 완료

## 다음 작업

1. 주 데이터셋 결정: RE10K vs DL3DV. feed-forward checkpoint 학습 도메인과 맞춰 결정한다.
2. DTU dense-view sanity check: 49-view + 충분한 iteration으로 Vanilla 3DGS가 정상 PSNR 범위에 도달하는지 확인한다.
3. §5.2 모델별 지원 view 수 표 갱신: MVSplat 4/8/12-view, DepthSplat checkpoint별 2/4/6/12-view 지원을 확인한다.
4. DTU는 공식 split 기준으로 external/C2 track에 둔다.
5. sanity check 이후에만 overlap threshold, tau, bootstrap CI 산출용 batch를 스케일업한다.

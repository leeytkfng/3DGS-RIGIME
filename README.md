# Sparse-view 3DGS Regime Study

Repository alias: `3DGS-RIGIME`

이 폴더는 Sparse-view 3DGS regime analysis 실험을 위한 기본 문서, 설정, 실행 스크립트를 담는다.

## 폴더 구성

- [Sparse-view_3DGS_Regime_연구계획서_정리.md](Sparse-view_3DGS_Regime_연구계획서_정리.md): 연구 목표와 실험 설계 요약
- [experiments/configs/experiment_config.yaml](experiments/configs/experiment_config.yaml): 실험 설정
- [experiments/scripts/run_experiment.sh](experiments/scripts/run_experiment.sh): 실행 진입점 스텁
- [experiments/scripts/generate_overlap.py](experiments/scripts/generate_overlap.py): SfM visibility 기반 overlap report 생성기
- [experiments/docs/early_experiment](experiments/docs/early_experiment): 파일럿 전 동결 절차 문서

## 빠른 시작

```bash
cd /root/task\ 5
bash experiments/scripts/run_experiment.sh
```

이제 스크립트는 설정 파일을 읽고, 수정본 프로토콜에 맞춘 실험 계획을 자동 생성해 결과 폴더에 manifest로 저장합니다. manifest에는 pose-given track, budget-end checkpoint 규칙, oracle 결과 분리, C1-b 렌더 등가성 gate, C2 depth perturbation, scene 단위 bootstrap 분석 설정이 함께 기록됩니다.

## Overlap report 생성

```bash
python3 experiments/scripts/generate_overlap.py \
  --colmap-images /data/Re-feem/datasets/re10k/scene_000/sparse/0/images.txt \
  --output-dir experiments/outputs/overlap/re10k_scene_000/4view_seed0 \
  --scene re10k_scene_000 \
  --view-count-label 4view_seed0
```

입력이 JSON이면 `{"view_id": [point_id, ...]}` 형태로 저장한 뒤 `--visibility-json`을 쓰면 됩니다.

## 현재 반영된 프로토콜

- Overlap: 매칭 실패/non-edge view pair를 0으로 포함하고, zero-included mean을 주 지표로 사용
- 승패 판정: `tau = max(파일럿 seed 변동성, 실용 최소 PSNR 차이)`로 고정
- 체크포인트: 메인 결과는 예산 종료 시점만 사용하고 oracle peak는 별도 경로에 저장
- 통계: scene 단위 cluster bootstrap CI를 기본으로 사용하고 seed는 scene 내 반복 측정으로 처리
- C1-b: feed-forward Gaussian 변환 후 렌더 등가성 gate를 통과한 경우에만 standard 3DGS refinement off(0초)/on(10, 60, 300초) 비교
- C2: iid depth noise와 global scale bias를 sensitivity analysis 범위로 실행

## 권장 다음 작업

1. 실제 DepthSplat/MVSplat/3DGS 실행 코드 연결
2. RE10K/DTU 데이터셋과 COLMAP export 경로 반영
3. 파일럿 5장면으로 view 수별 overlap threshold 산출
4. 파일럿 로그로 bootstrap CI와 tau 값 동결

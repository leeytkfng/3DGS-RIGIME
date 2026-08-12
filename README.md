# Sparse-view 3DGS Regime Study

Repository alias: `3DGS-RIGIME`

이 폴더는 Sparse-view 3DGS regime analysis 실험을 위한 기본 문서, 설정, 실행 스크립트를 담는다. 현재는 초기 scaffold를 넘어, Vanilla 3DGS / MVSplat / DepthSplat이 모두 실데이터에서 최소 검증된 상태다.

## 폴더 구성

- [Sparse-view_3DGS_Regime_연구계획서_정리.md](Sparse-view_3DGS_Regime_연구계획서_정리.md): 연구 목표와 실험 설계 요약
- [experiments/configs/experiment_config.yaml](experiments/configs/experiment_config.yaml): 실험 설정
- [experiments/docs](experiments/docs): 일일 보고서, 데이터 확보 현황, checkpoint-domain 표, 구현 요약, 논문 읽기 로그
- [experiments/scripts/core](experiments/scripts/core): protocol_utils, model_registry, DTU loader, COLMAP init 등 공유 모듈
- [experiments/scripts/runners](experiments/scripts/runners): 정식 모델 runner
- [experiments/scripts/probes](experiments/scripts/probes): MVSplat/DepthSplat 1회성 검증 스크립트
- [experiments/scripts/batch](experiments/scripts/batch): manifest 생성과 batch driver
- [experiments/scripts/analysis](experiments/scripts/analysis): overlap report와 geometry uncertainty figure 생성
- [tests/test_protocol_utils.py](tests/test_protocol_utils.py): 프로토콜 핵심 규칙 단위 테스트

## 빠른 시작

```bash
cd /root/task\ 5
bash experiments/scripts/batch/run_experiment.sh
python3 -m unittest discover -s tests
```

`run_experiment.sh`는 설정 파일을 읽고, 수정본 프로토콜에 맞춘 실험 계획을 `experiments/outputs/experiment_manifest.json`으로 생성한다. manifest에는 pose-given track, budget-end checkpoint 규칙, oracle 결과 분리, C1-b 렌더 등가성 gate, C2 depth perturbation, scene 단위 bootstrap 분석 설정이 함께 기록된다.

## Overlap report 생성

```bash
python3 experiments/scripts/analysis/generate_overlap.py \
  --colmap-images /data/Re-feem/datasets/re10k/scene_000/sparse/0/images.txt \
  --output-dir experiments/outputs/overlap/re10k_scene_000/4view_seed0 \
  --scene re10k_scene_000 \
  --view-count-label 4view_seed0
```

입력이 JSON이면 `{"view_id": [point_id, ...]}` 형태로 저장한 뒤 `--visibility-json`을 쓰면 된다.

## 현재 반영된 프로토콜

- Overlap: 매칭 실패/non-edge view pair를 0으로 포함하고, zero-included mean을 주 지표로 사용
- 승패 판정: `tau = max(파일럿 seed 변동성, 실용 최소 PSNR 차이)`로 고정
- 체크포인트: 메인 결과는 예산 종료 시점만 사용하고 oracle peak는 별도 경로에 저장
- 통계: scene 단위 cluster bootstrap CI를 기본으로 사용하고 seed는 scene 내 반복 측정으로 처리
- 입력 해상도: feed-forward checkpoint의 학습 해상도를 상속. RE10K는 256x256, DepthSplat DL3DV track은 256x448
- C1-b: renderer equivalence gate 통과 후 refinement on/off와 densification on/off를 분리 비교
- C2: iid depth noise와 global scale bias를 sensitivity analysis 범위로 실행

## Local Data Status

- RE10K: `/data/Re-feem/datasets/re10k`, test 8 chunk, 114 scene, 약 1.2GB. MVSplat in-domain probe 정상.
- DL3DV: `/data/Re-feem/datasets/dl3dv`, 25 scene, 약 1.9GB. DepthSplat DL3DV probe 정상, 공식 benchmark split과 중복 0.
- DTU: `/data/Re-feem/datasets/dtu`, 공식 sparse-view split 16개 포함 총 29 scan. external validation / C2 / dense sanity 용도.
- External code/checkpoints: `/data/Re-feem/code/mvsplat`, `/data/Re-feem/code/depthsplat`.

## Current Status

- DTU dense-view sanity check 완료: Vanilla 3DGS scan1, 42 train view, 30k iteration에서 PSNR 24.0~24.1dB, SSIM 0.843, LPIPS 0.218.
- MVSplat RE10K in-domain 확인: mean PSNR 25.6dB, 추가 mirror chunk 22.4dB.
- DepthSplat DL3DV in-domain 확인: near-context mean PSNR 20.0dB.
- DTU 공식 split 16 scan x 2-view x seed0에서 Vanilla3DGS와 MVSplat batch 로그 생성 완료.
- Geometry uncertainty 1차 figure 생성 완료. overlap과 uncertainty 부호 해석은 A-1 재독 후 재검토 필요.

## 권장 다음 작업

1. Main benchmark를 RE10K-first로 확정할지 결정하고 20~30 scene subset index를 만든다.
2. RE10K 256x256 protocol을 Vanilla3DGS runner와 FF runner에 공통 적용한다.
3. DepthSplat probe를 `depthsplat_runner.py` 정식 runner로 승격한다.
4. co-visibility 기반 view selector를 만들고 overlap bucket 생성과 연결한다.
5. MVSplat/DepthSplat의 지원 view 수와 confidence 출력 유무를 채워 §5.2 표를 완성한다.
6. densification on/off CLI를 Vanilla3DGS runner에 추가하고 C1-b 설계에 반영한다.

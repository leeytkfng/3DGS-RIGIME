#!/usr/bin/env bash
# 실험 manifest 생성 진입점.
# 목적:
#   - 사용자가 Python 파일 경로를 외우지 않아도 기본 config로 manifest를 만들 수 있게 한다.
# 예상 결과:
#   - 기본 실행 시 experiments/outputs/experiment_manifest.json 생성.
#   - 인자 1개를 주면 config 경로, 인자 2개를 주면 output directory를 바꿀 수 있다.
set -euo pipefail

# SCRIPT_DIR은 이 shell script가 위치한 디렉터리다.
# 어느 위치에서 실행해도 repo root를 안정적으로 찾기 위해 사용한다.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# 기본 config와 output dir.
# 예상 결과: 별도 인자가 없으면 repository 내부 outputs에 manifest와 빈 결과 폴더가 생긴다.
CONFIG=${1:-$REPO_ROOT/experiments/configs/experiment_config.yaml}
OUTDIR=${2:-$REPO_ROOT/experiments/outputs}

python3 "$SCRIPT_DIR/run_experiment.py" "$CONFIG" "$OUTDIR"

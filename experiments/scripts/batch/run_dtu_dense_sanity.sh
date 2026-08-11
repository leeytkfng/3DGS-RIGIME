#!/usr/bin/env bash
# DTU dense-view sanity check.
# 목적:
#   sparse-view 결과를 해석하기 전에, pose/rasterization/evaluation 파이프라인이
#   dense-view 조건에서 정상 PSNR 범위로 올라가는지 확인한다.
#
# 주의:
#   현재 vanilla_3dgs_runner.py는 held-out test split(1,8,15,...)을 고정하므로
#   --view-count 49를 줘도 실제 학습 view는 test를 제외한 42장이다.
#   즉 이 스크립트의 의미는 "42 train / 7 held-out dense-ish sanity"다.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PS3_PYTHON=${PS3_PYTHON:-/opt/conda/envs/ps3/bin/python3}
SCAN_ID=${1:-1}
SCAN_DIR=/data/Re-feem/datasets/dtu/scan${SCAN_ID}
SCENE=dtu_scan${SCAN_ID}_dense_sanity
OUTDIR=${2:-$REPO_ROOT/experiments/outputs_dense_sanity}

PATH="/opt/conda/envs/ps3/bin:$PATH" "$PS3_PYTHON" "$SCRIPT_DIR/../runners/vanilla_3dgs_runner.py"   --scan-dir "$SCAN_DIR"   --scene "$SCENE"   --view-count 49   --seed 0   --max-budget-seconds 7200   --budget-snapshots 60 300 1800 3600 7200   --max-iterations 30000   --iteration-snapshots 30000   --output-dir "$OUTDIR"

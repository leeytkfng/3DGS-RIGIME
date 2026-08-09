#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CONFIG=${1:-$REPO_ROOT/experiments/configs/experiment_config.yaml}
OUTDIR=${2:-$REPO_ROOT/experiments/outputs}

python3 "$SCRIPT_DIR/run_experiment.py" "$CONFIG" "$OUTDIR"

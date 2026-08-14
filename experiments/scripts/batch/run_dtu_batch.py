#!/usr/bin/env python3
"""[2026-08-10 이후 신규 배치는 run_experiment_batch.py 사용 권장]
이 파일은 DTU 전용/모델 하드코딩 버전으로 남겨둔다(기존 batch_summary.json과의 호환을 위해).
model_registry.py 기반으로 일반화된 버전은 experiments/scripts/run_experiment_batch.py 참고.

여러 scan × seed × view_count에 대해 vanilla_3dgs_runner.py / mvsplat_runner.py를
순회 실행하는 driver.

이 파일의 목적:
- 지금까지는 러너를 한 번에 하나씩 손으로 돌렸다. 파일럿·본 실험 규모에서는 scan/seed
  조합을 자동으로 순회해야 한다.
- 각 method는 서로 다른 conda env(ps3/mvsplat)를 요구하므로(§7.3/§9.1 audit log 사고),
  이 스크립트는 subprocess로 해당 env의 python 바이너리를 직접 호출한다 — 이 파일 자체는
  아무 env에서나 실행 가능하다(표준 라이브러리만 사용).
- 중단·재개가 가능해야 한다(계획서 STEP3): 이미 로그 파일이 존재하는 조합은 건너뛴다.
  실패한 run은 전체를 죽이지 않고 계속 진행하며, 끝나면 성공/실패 목록을 요약 출력한다.

예시:
    python3 experiments/scripts/run_dtu_batch.py \\
        --methods vanilla3dgs mvsplat \\
        --scans 1 34 \\
        --seeds 0 \\
        --view-counts 2 \\
        --vanilla-budget-seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DTU_ROOT = Path("/data/Re-feem/datasets/dtu")

# MVSplat 공식 sparse-view test split (convert_dtu.py get_example_keys(), §9.2 audit log).
OFFICIAL_DTU_SPLIT = [1, 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]

PS3_PYTHON = "/opt/conda/envs/ps3/bin/python3"
MVSPLAT_PYTHON = "/opt/conda/envs/mvsplat/bin/python3"


def _env_with_bin(conda_env_bin: str) -> dict:
    # ps3 env의 gsplat은 첫 CUDA 커널 사용 시 ninja를 PATH에서 찾아 JIT 컴파일한다.
    # subprocess는 부모의 PATH를 그대로 물려받으므로, 해당 env의 bin을 명시적으로 앞에 붙여야
    # `RuntimeError: Ninja is required` 로 조용히 실패하는 걸 막을 수 있다 (오늘 발견한 버그).
    env = os.environ.copy()
    env["PATH"] = f"{conda_env_bin}:{env.get('PATH', '')}"
    return env


def run_vanilla3dgs(scan_id: int, seed: int, view_count: int, output_dir: Path, budget_seconds: float) -> dict:
    scan_dir = DTU_ROOT / f"scan{scan_id}"
    scene = f"dtu_scan{scan_id}"
    log_path = output_dir / "logs" / f"{scene}_Vanilla3DGS_{view_count}view_seed{seed}.json"
    if log_path.exists():
        return {"status": "skipped_exists", "log": str(log_path)}

    cmd = [
        PS3_PYTHON, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
        "--scan-dir", str(scan_dir),
        "--scene", scene,
        "--seed", str(seed),
        "--view-count", str(view_count),
        "--max-budget-seconds", str(budget_seconds),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin("/opt/conda/envs/ps3/bin"))
    if result.returncode != 0:
        return {"status": "failed", "stderr_tail": result.stderr[-2000:]}
    return {"status": "ok", "log": str(log_path)}


def run_mvsplat(scan_id: int, seed: int, view_count: int, output_dir: Path) -> dict:
    scan_dir = DTU_ROOT / f"scan{scan_id}"
    scene = f"dtu_scan{scan_id}"
    log_path = output_dir / "logs" / f"{scene}_MVSplat_{view_count}view_seed{seed}.json"
    if log_path.exists():
        return {"status": "skipped_exists", "log": str(log_path)}

    cmd = [
        MVSPLAT_PYTHON, str(REPO_ROOT / "experiments/scripts/runners/mvsplat_runner.py"),
        "--scan-dir", str(scan_dir),
        "--scene", scene,
        "--seed", str(seed),
        "--view-count", str(view_count),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin("/opt/conda/envs/mvsplat/bin"))
    if result.returncode != 0:
        return {"status": "failed", "stderr_tail": result.stderr[-2000:]}
    return {"status": "ok", "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run vanilla3dgs/mvsplat over DTU scan x seed x view_count.")
    parser.add_argument("--methods", nargs="+", choices=["vanilla3dgs", "mvsplat"], default=["vanilla3dgs", "mvsplat"])
    parser.add_argument("--scans", nargs="+", type=int, default=OFFICIAL_DTU_SPLIT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--view-counts", nargs="+", type=int, default=[2])
    parser.add_argument("--vanilla-budget-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = []
    t_start = time.time()

    combos = [
        (scan, seed, vc)
        for scan in args.scans
        for seed in args.seeds
        for vc in args.view_counts
    ]
    print(f"[batch] {len(combos)} scan x seed x view_count combos, methods={args.methods}")

    for scan, seed, vc in combos:
        scan_dir = DTU_ROOT / f"scan{scan}"
        if not scan_dir.exists():
            print(f"[skip] scan{scan} not downloaded yet ({scan_dir})")
            summary.append({"scan": scan, "seed": seed, "view_count": vc, "method": "-", "status": "missing_data"})
            continue

        if "vanilla3dgs" in args.methods:
            t0 = time.time()
            result = run_vanilla3dgs(scan, seed, vc, output_dir, args.vanilla_budget_seconds)
            result.update({"scan": scan, "seed": seed, "view_count": vc, "method": "Vanilla3DGS", "elapsed": time.time() - t0})
            print(f"[vanilla3dgs] scan{scan} seed{seed} {vc}view -> {result['status']} ({result['elapsed']:.1f}s)")
            summary.append(result)

        if "mvsplat" in args.methods:
            t0 = time.time()
            result = run_mvsplat(scan, seed, vc, output_dir)
            result.update({"scan": scan, "seed": seed, "view_count": vc, "method": "MVSplat", "elapsed": time.time() - t0})
            print(f"[mvsplat] scan{scan} seed{seed} {vc}view -> {result['status']} ({result['elapsed']:.1f}s)")
            summary.append(result)

    summary_path = output_dir / "logs" / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_path.write_text(json.dumps(existing + summary, indent=2), encoding="utf-8")

    ok = sum(1 for s in summary if s["status"] == "ok")
    skipped = sum(1 for s in summary if s["status"] == "skipped_exists")
    failed = sum(1 for s in summary if s["status"] == "failed")
    missing = sum(1 for s in summary if s["status"] == "missing_data")
    print(f"[batch] done in {time.time()-t_start:.1f}s. ok={ok} skipped={skipped} failed={failed} missing_data={missing}")
    print(f"[batch] summary written to {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

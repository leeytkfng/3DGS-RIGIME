#!/usr/bin/env python3
"""C2(depth-noise sensitivity) 실행 전 필요한 depth cache를 미리 뽑는다.

DTU 8scan × 4 representative_conditions(2view_low, 4view_low, 4view_high, 12view_high)
= 32개 cache. `depth` conda env에서 `precompute_depth_maps.py --overlap-level`을 호출한다
(2026-08-16 추가한 DTU overlap-level 지원, `generate_dtu_overlap_candidates.py` 출력 재사용).

이미 있는 cache는 건너뛴다(재개 가능).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPTH_PY = "/opt/conda/envs/depth/bin/python3"
DTU_ROOT = Path("/data/Re-feem/datasets/dtu")

REPRESENTATIVE_CONDITIONS = [
    (2, "low"),
    (4, "low"),
    (4, "high"),
    (12, "high"),
]


def _env_with_bin(conda_env_python: str) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{Path(conda_env_python).parent}:{env.get('PATH', '')}"
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtu-overlap-candidates-index", default=str(REPO_ROOT / "experiments/outputs/dtu_overlap_candidates/dtu_overlap_candidates.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/c2_depth_cache"))
    args = parser.parse_args()

    candidates = json.loads(Path(args.dtu_overlap_candidates_index).read_text())
    scenes = sorted(candidates.keys(), key=lambda s: int(s.replace("scan", "")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * len(REPRESENTATIVE_CONDITIONS)
    step = 0
    t_start = time.time()
    failed = []

    for scene in scenes:
        scan_id = scene.replace("scan", "")
        for view_count, level in REPRESENTATIVE_CONDITIONS:
            step += 1
            out_path = output_dir / f"{scene}_{view_count}view_{level}.pt"
            if out_path.exists():
                print(f"[skip {step}/{total}] {out_path.name} exists")
                continue
            print(f"[{step}/{total}] {scene} {view_count}view {level} ...")
            t0 = time.time()
            cmd = [
                DEPTH_PY, str(REPO_ROOT / "experiments/scripts/analysis/precompute_depth_maps.py"),
                "--dataset", "dtu", "--scan-dir", str(DTU_ROOT / scene),
                "--overlap-level", level, "--view-count", str(view_count),
                "--dtu-overlap-candidates-index", args.dtu_overlap_candidates_index,
                "--output", str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(DEPTH_PY))
            if result.returncode != 0:
                print(f"  [FAIL] {result.stderr[-1500:]}")
                failed.append(str(out_path))
            else:
                print(f"  done in {time.time()-t0:.1f}s")

    print(f"\n[done] {(time.time()-t_start)/60:.1f}min. failed={len(failed)}")
    for f in failed:
        print(f"  FAILED: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

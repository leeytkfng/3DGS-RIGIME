#!/usr/bin/env python3
"""C2: 통제된 depth-noise/scale-bias 개입 실험.

`experiment_config.yaml`의 `c2:` 블록 그대로 재현 — DTU 8scan(외부 검증) ×
representative_conditions(4: 2view_low, 4view_low, 4view_high, 12view_high) × seed{0,1} ×
{depth_noise_levels(5, scale_bias=1.0 고정) + global_scale_bias(5, sigma=0.0 고정)}
= 8*2*4*(5+5) = 640 run. `run_experiment.py::build_plan()`의 C2 매니페스트 생성 로직과
동일 축(두 교란은 따로 sweep, 5x5 cross 아님).

각 run은 `vanilla_3dgs_runner.py --depth-cache-path ...`(2026-08-15 구현)로 `precompute_c2_
depth_caches.py`가 만든 base depth cache를 perturb_depth()로 교란한 뒤 back-projection
init으로 사용한다. sigma/scale 조합마다 로그 파일명이 같은 (scene,view_count,seed,level)
조합끼리 충돌하므로, 조합마다 별도 --output-dir 하위 폴더를 쓴다(run_dl3dv_c1a_main.py의
scene별 vanilla_output 패턴과 동일 이유).

budget/trajectory는 main phase/C1-b와 동일(단일 300s + [1,10,60,300] snapshot,
overall.md §5.9 2026-08-12 결정 그대로).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PS3_PY = "/opt/conda/envs/ps3/bin/python3"
DTU_ROOT = Path("/data/Re-feem/datasets/dtu")

DEPTH_NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.10]
GLOBAL_SCALE_BIAS = [0.9, 0.95, 1.0, 1.05, 1.1]
SEEDS = [0, 1]
BUDGET_SNAPSHOTS = [1.0, 10.0, 60.0, 300.0]
MAX_BUDGET = 300.0

REPRESENTATIVE_CONDITIONS = {
    "2view_low_overlap": (2, "low"),
    "4view_low_overlap": (4, "low"),
    "4view_high_overlap": (4, "high"),
    "12view_high_overlap": (12, "high"),
}


def _env_with_bin(conda_env_python: str) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{Path(conda_env_python).parent}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(cmd[0]))
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] {label}: {result.stderr[-1500:]}")
    return ok


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_path = output_dir / "c2_sensitivity_summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 dtu_overlap_candidates.json의 8scan 전체")
    parser.add_argument("--conditions", nargs="+", default=list(REPRESENTATIVE_CONDITIONS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--depth-cache-dir", default=str(REPO_ROOT / "experiments/outputs/c2_depth_cache"))
    parser.add_argument("--dtu-overlap-candidates-index", default=str(REPO_ROOT / "experiments/outputs/dtu_overlap_candidates/dtu_overlap_candidates.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/c2_sensitivity_sweep"))
    args = parser.parse_args()

    candidates = json.loads(Path(args.dtu_overlap_candidates_index).read_text())
    scenes = args.scenes or sorted(candidates.keys(), key=lambda s: int(s.replace("scan", "")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # (phase, sigma, scale) 조합 생성 — run_experiment.py::build_plan()의 C2 두 서브페이즈와 동일.
    phase_rows = [("c2_depth_noise", sigma, 1.0) for sigma in DEPTH_NOISE_LEVELS]
    phase_rows += [("c2_depth_scale_bias", 0.0, scale) for scale in GLOBAL_SCALE_BIAS]

    rows: list[dict] = []
    t_start = time.time()
    combos = list(product(scenes, args.conditions, phase_rows, args.seeds))
    total = len(combos)
    step = 0

    for scene, condition, (phase, sigma, scale), seed in combos:
        step += 1
        view_count, level = REPRESENTATIVE_CONDITIONS[condition]
        cache_path = Path(args.depth_cache_dir) / f"{scene}_{view_count}view_{level}.pt"
        if not cache_path.exists():
            print(f"[skip {step}/{total}] {scene} {condition} sigma={sigma} scale={scale}: depth cache 없음({cache_path})")
            continue

        cond_dir = output_dir / phase / f"sigma{sigma}_scale{scale}"
        log_path = cond_dir / "logs" / f"dtu_{scene}_Vanilla3DGS_{view_count}view_seed{seed}_{level}.json"
        if not log_path.exists():
            print(f"[{step}/{total}] {scene} {condition} {phase} sigma={sigma} scale={scale} seed={seed}")
            t0 = time.time()
            run_step(
                [
                    PS3_PY, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
                    "--dataset", "dtu", "--scan-dir", str(DTU_ROOT / scene), "--scene", f"dtu_{scene}",
                    "--overlap-level", level, "--dtu-overlap-candidates-index", args.dtu_overlap_candidates_index,
                    "--view-count", str(view_count), "--seed", str(seed),
                    "--depth-cache-path", str(cache_path),
                    "--depth-noise-sigma", str(sigma), "--depth-scale-bias", str(scale),
                    "--max-budget-seconds", str(MAX_BUDGET),
                    "--budget-snapshots", *[str(b) for b in BUDGET_SNAPSHOTS],
                    "--output-dir", str(cond_dir),
                ],
                "vanilla_3dgs_runner(c2)",
            )
            print(f"  done in {time.time()-t0:.1f}s")

        if log_path.exists():
            trajectory = json.loads(log_path.read_text())
            by_budget = {r["wall_clock"]: r for r in trajectory}
            for budget in BUDGET_SNAPSHOTS:
                r = by_budget.get(budget)
                rows.append({
                    "scene": scene, "condition": condition, "view_count": view_count, "overlap_level": level,
                    "phase": phase, "sigma": sigma, "scale_bias": scale, "seed": seed, "budget": budget,
                    "status": "ok" if r else "no_result",
                    "test_psnr": r["test_psnr"] if r else None,
                    "gaussian_count": r["gaussian_count"] if r else None,
                })
            write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    print(f"\n[done] {len(rows)} rows in {(time.time()-t_start)/3600:.2f}h. summary: {output_dir / 'c2_sensitivity_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""C1-a(품질-시간 Regime Map) 첫 파일럿 — RE10K 5 scene × 2/4/8/12-view, Vanilla3DGS vs MVSplat.

overall.md §5.4의 "파일럿: 장면 5개, 조건 전 조합 1회전" 규모를 그대로 따른다. 오늘 밤 연결한
두 경로를 처음으로 정면 비교한다:
- Vanilla3DGS(optimization): `vanilla_3dgs_runner.py --dataset re10k`(COLMAP/random init,
  warm-start 아님) — max_budget=60s, snapshot=[1,10,60]
- MVSplat(feed-forward): `mvsplat_re10k_runner.py` — 단일 추론, wall_clock이 budget보다
  크면 그 budget 칸은 §5.7 규칙대로 No result

예상 결과: output_dir/c1a_pilot_summary.json에 (scene, view_count, budget, method) 별 PSNR.
아직 seed 1회, 5 scene뿐이라 통계적 결론용이 아니라 "파이프라인이 실제로 두 방법을 같은
조건에서 비교 가능한 표를 만들어내는가"를 확인하는 첫 파일럿이다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MVSPLAT_PY = "/opt/conda/envs/mvsplat/bin/python3"
PS3_PY = "/opt/conda/envs/ps3/bin/python3"
BUDGETS = [1.0, 10.0, 60.0]


def _env_with_bin(conda_env_python: str) -> dict:
    conda_bin = str(Path(conda_env_python).parent)
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], label: str) -> tuple[bool, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(cmd[0]))
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] {label}: {result.stderr[-800:]}")
    return ok, result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-index", default=str(REPO_ROOT / "experiments/outputs/re10k_main_subset/re10k_main_subset.json"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 subset의 첫 5개(§5.4 파일럿 규모)")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Vanilla3DGS만 seed별로 반복(MVSplat은 deterministic이라 seed 무관, 1회만 실행)")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/re10k_c1a_pilot"))
    args = parser.parse_args()

    subset = json.loads(Path(args.subset_index).read_text())
    scene_keys = args.scenes if args.scenes else list(subset.keys())[:5]
    output_dir = Path(args.output_dir)

    rows = []
    t_start = time.time()
    for si, scene_key in enumerate(scene_keys):
        for vi, view_count in enumerate(args.view_counts):
            candidate = subset[scene_key]["view_candidates"].get(str(view_count), {})
            if candidate.get("context") is None:
                print(f"[skip] {scene_key} {view_count}view: candidate 없음")
                continue
            print(f"[{si*len(args.view_counts)+vi+1}/{len(scene_keys)*len(args.view_counts)}] {scene_key} {view_count}view")
            t0 = time.time()

            # --- MVSplat (feed-forward, single inference) ---
            mv_log = output_dir / "logs" / f"{scene_key}_MVSplat_{view_count}view.json"
            if not mv_log.exists():
                run_step(
                    [
                        MVSPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/mvsplat_re10k_runner.py"),
                        "--subset-index", args.subset_index, "--scene-key", scene_key,
                        "--view-count", str(view_count), "--output-dir", str(output_dir),
                    ],
                    "mvsplat_re10k_runner",
                )
            if mv_log.exists():
                mv_row = json.loads(mv_log.read_text())
                for budget in BUDGETS:
                    rows.append({
                        "scene": scene_key, "view_count": view_count, "budget": budget,
                        "method": "MVSplat",
                        "status": "ok" if mv_row["wall_clock"] <= budget else "no_result_budget_exceeded",
                        "test_psnr": mv_row["test_psnr"] if mv_row["wall_clock"] <= budget else None,
                        "wall_clock": mv_row["wall_clock"],
                    })

            # --- Vanilla3DGS (optimization, COLMAP/random init), seed별 반복 ---
            vanilla_output = output_dir / "vanilla_runs" / scene_key
            for seed in args.seeds:
                log_path = vanilla_output / "logs" / f"re10k_{scene_key}_c1a_Vanilla3DGS_{view_count}view_seed{seed}.json"
                if not log_path.exists():
                    run_step(
                        [
                            PS3_PY, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
                            "--dataset", "re10k", "--re10k-scene-key", scene_key,
                            "--re10k-subset-index", args.subset_index,
                            "--scene", f"re10k_{scene_key}_c1a", "--view-count", str(view_count), "--seed", str(seed),
                            "--image-shape", "256", "256",
                            "--max-budget-seconds", str(max(BUDGETS)),
                            "--budget-snapshots", *[str(b) for b in BUDGETS],
                            "--output-dir", str(vanilla_output),
                        ],
                        "vanilla_3dgs_runner(re10k general)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in BUDGETS:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "budget": budget, "seed": seed,
                            "method": "Vanilla3DGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                            "init_source": r["init_source"] if r else None,
                        })

            print(f"  done in {time.time()-t0:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "c1a_pilot_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[done] {len(rows)} rows in {time.time()-t_start:.1f}s. summary: {summary_path}")

    # 요약 표 출력: view_count x budget 별 각 method 평균 PSNR
    print(f"\n{'view':>5} {'budget':>7} {'MVSplat':>9} {'Vanilla3DGS':>12}")
    for view_count in args.view_counts:
        for budget in BUDGETS:
            mv_vals = [r["test_psnr"] for r in rows if r["method"] == "MVSplat" and r["view_count"] == view_count and r["budget"] == budget and r["test_psnr"] is not None]
            va_vals = [r["test_psnr"] for r in rows if r["method"] == "Vanilla3DGS" and r["view_count"] == view_count and r["budget"] == budget and r["test_psnr"] is not None]
            mv_str = f"{sum(mv_vals)/len(mv_vals):.2f}" if mv_vals else "N/A"
            va_str = f"{sum(va_vals)/len(va_vals):.2f}" if va_vals else "N/A"
            print(f"{view_count:>5} {budget:>7.0f} {mv_str:>9} {va_str:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

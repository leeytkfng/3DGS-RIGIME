#!/usr/bin/env python3
"""Overlap 축 보조 실험 — RE10K 8-scene(seed=0 결정론적 서브샘플) × 2/4/8/12-view ×
high/low overlap × budget[1,10,60,300]s, seed=0 (단일).

C1-a 본 실험(`run_re10k_c1a_main.py`)이 view_count×budget 축만 완주하고 남겨뒀던 overlap 축을
채운다(원래 설계에 있었지만 selector 검증 지연으로 미뤄졌던 것 — 새 축 아님). 규모는 GPU 시간
절약을 위해 30-scene 전체가 아니라 8-scene으로 축소(2026-08-17 결정), `re10k_overlap_candidates.json`
(30-scene 전체 이미 생성됨, A-2 진단에도 쓴 파일)에서 seed=0으로 결정론적 서브샘플.

세 방법 모두 `--overlap-level`을 지원한다(MVSplat은 원래부터, Vanilla3DGS/FSGS는 A-2/이번
세션에 배선 완료). Vanilla3DGS/FSGS는 단일 seed=0만 돈다 — 소규모 보조실험이라 scene cluster
bootstrap CI 폭이 넓겠지만, "high/low 방향이 실제로 존재하는가"를 보는 게 목적이라 seed
반복보다 scene 8개 커버가 우선.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MVSPLAT_PY = "/opt/conda/envs/mvsplat/bin/python3"
PS3_PY = "/opt/conda/envs/ps3/bin/python3"
FSGS_PY = "/opt/conda/envs/fsgs/bin/python3"
BUDGETS = [1.0, 10.0, 60.0, 300.0]
SEED = 0
NUM_SCENES = 8


def _env_with_bin(conda_env_python: str) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{Path(conda_env_python).parent}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(cmd[0]))
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] {label}: {result.stderr[-1200:]}")
    return ok


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_path = output_dir / "overlap_supplement_summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)


def select_scenes(candidates_path: Path) -> list[str]:
    all_scenes = sorted(json.loads(candidates_path.read_text()).keys())
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(all_scenes, size=NUM_SCENES, replace=False).tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-index", default=str(REPO_ROOT / "experiments/outputs/re10k_main_subset/re10k_main_subset.json"))
    parser.add_argument("--overlap-candidates-index", default=str(REPO_ROOT / "experiments/outputs/re10k_overlap_candidates/re10k_overlap_candidates.json"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 seed=0으로 8-scene 결정론적 선택")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--levels", nargs="+", default=["high", "low"])
    parser.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/re10k_overlap_supplement"))
    args = parser.parse_args()

    scene_keys = args.scenes or select_scenes(Path(args.overlap_candidates_index))
    print(f"[scenes] {len(scene_keys)}: {scene_keys}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()
    total = len(scene_keys) * len(args.view_counts) * len(args.levels)
    step = 0

    for scene_key in scene_keys:
        for view_count in args.view_counts:
            for level in args.levels:
                step += 1
                print(f"[{step}/{total}] {scene_key} {view_count}view {level}")
                t0 = time.time()

                # --- MVSplat (feed-forward, single deterministic inference) ---
                mv_log = output_dir / "logs" / f"{scene_key}_MVSplat_{view_count}view_{level}.json"
                if not mv_log.exists():
                    run_step(
                        [
                            MVSPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/mvsplat_re10k_runner.py"),
                            "--subset-index", args.subset_index, "--scene-key", scene_key,
                            "--overlap-level", level, "--overlap-candidates-index", args.overlap_candidates_index,
                            "--view-count", str(view_count), "--output-dir", str(output_dir),
                        ],
                        "mvsplat_re10k_runner",
                    )
                if mv_log.exists():
                    mv_row = json.loads(mv_log.read_text())
                    for budget in args.budgets:
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "overlap_level": level, "budget": budget,
                            "method": "MVSplat", "seed": None,
                            "status": "ok" if mv_row["wall_clock"] <= budget else "no_result_budget_exceeded",
                            "test_psnr": mv_row["test_psnr"] if mv_row["wall_clock"] <= budget else None,
                        })

                # --- Vanilla3DGS ---
                vanilla_output = output_dir / "vanilla_runs" / scene_key
                log_path = vanilla_output / "logs" / f"re10k_{scene_key}_ovl_Vanilla3DGS_{view_count}view_seed{SEED}_{level}.json"
                if not log_path.exists():
                    run_step(
                        [
                            PS3_PY, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
                            "--dataset", "re10k", "--re10k-scene-key", scene_key,
                            "--re10k-subset-index", args.subset_index,
                            "--overlap-level", level, "--overlap-candidates-index", args.overlap_candidates_index,
                            "--scene", f"re10k_{scene_key}_ovl", "--view-count", str(view_count), "--seed", str(SEED),
                            "--image-shape", "256", "256",
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(vanilla_output),
                        ],
                        "vanilla_3dgs_runner(re10k overlap)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "overlap_level": level, "budget": budget,
                            "seed": SEED, "method": "Vanilla3DGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                        })

                # --- FSGS ---
                fsgs_output = output_dir / "fsgs_runs" / scene_key
                log_path = fsgs_output / "logs" / f"re10k_{scene_key}_ovl_FSGS_{view_count}view_seed{SEED}_{level}.json"
                if not log_path.exists():
                    run_step(
                        [
                            FSGS_PY, str(REPO_ROOT / "experiments/scripts/runners/fsgs_runner.py"),
                            "--dataset", "re10k", "--re10k-scene-key", scene_key,
                            "--re10k-subset-index", args.subset_index,
                            "--overlap-level", level, "--overlap-candidates-index", args.overlap_candidates_index,
                            "--scene", f"re10k_{scene_key}_ovl", "--view-count", str(view_count), "--seed", str(SEED),
                            "--image-shape", "256", "256",
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(fsgs_output),
                        ],
                        "fsgs_runner(re10k overlap)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "overlap_level": level, "budget": budget,
                            "seed": SEED, "method": "FSGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                        })

                print(f"  done in {time.time()-t0:.1f}s ({len(rows)} rows so far)")
                write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    print(f"\n[done] {len(rows)} rows in {(time.time()-t_start)/3600:.2f}h. summary: {output_dir / 'overlap_supplement_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

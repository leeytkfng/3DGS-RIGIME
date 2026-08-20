#!/usr/bin/env python3
"""C1-a 본 실험 — DL3DV 25 scene × 2/4/8/12-view × budget[1,10,60,300]s, seed{0,1}.

`run_re10k_c1a_main.py`(RE10K, FSGS 버그 수정 후 재검증 완료, 2026-08-14)를 DL3DV로 그대로
이식한 것. RE10K와 다른 점은 딱 하나 — feed-forward가 MVSplat이 아니라 **DepthSplat**이다
(DL3DV로 학습된 in-domain 체크포인트, §4.2 "각 데이터셋에는 그 데이터셋으로 학습된
in-domain feed-forward 모델을 짝짓는다"). Vanilla3DGS/FSGS는 RE10K와 완전히 동일한 로직
(`--dataset dl3dv` 분기, 어제 통제 스모크로 실제 검증 완료).

**착수 시점(2026-08-14 작성)**: RE10K 본 실험이 아직 GPU를 쓰고 있어 이 스크립트는 코드만
준비해두고, RE10K가 끝난 뒤 착수한다(사용자 결정, 2026-08-14). DepthSplat은 우리 체크포인트
기준 2~6-view 학습 분포라 8/12-view는 분포 밖(OOD)이다 — RE10K의 MVSplat(2-view 전용)과
같은 이유로, 그래도 실행은 하되 §5.2 기준 OOD로 표기해 분석한다(기존 처리 방식과 동일).

overlap_level(고/저) 축은 RE10K와 동일하게 이번 착수 범위에서 제외한다 — DL3DV용
co-visibility selector(`generate_dl3dv_overlap_candidates.py`)는 25-scene 전체 검증까지는
끝났지만(94/100) 러너에 아직 연결 안 함.

summary는 (scene, view_count) 조합 하나 끝날 때마다 즉시 디스크에 다시 쓴다(atomic) —
`run_re10k_c1a_main.py`와 동일 패턴, 대시보드가 중간에 읽을 수 있게.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPTHSPLAT_PY = "/opt/conda/envs/depthsplat/bin/python3"
PS3_PY = "/opt/conda/envs/ps3/bin/python3"
FSGS_PY = "/opt/conda/envs/fsgs/bin/python3"
BUDGETS = [1.0, 10.0, 60.0, 300.0]
SEEDS = [0, 1]
IMAGE_SHAPE = ["256", "448"]  # DepthSplat DL3DV 체크포인트 학습 해상도(§protocol, 입력 해상도 상속 원칙)


def _env_with_bin(conda_env_python: str) -> dict:
    conda_bin = str(Path(conda_env_python).parent)
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(cmd[0]))
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] {label}: {result.stderr[-1200:]}")
    return ok


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_path = output_dir / "c1a_main_summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-summary", default=str(REPO_ROOT / "experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 overlap-summary에 있는 scene 전체(25개)")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/dl3dv_c1a_main"))
    args = parser.parse_args()

    overlap_rows = json.loads(Path(args.overlap_summary).read_text())
    available = {(r["scene"], r["view_count"]) for r in overlap_rows if r.get("context_indices") is not None}
    scene_keys = args.scenes if args.scenes else sorted({r["scene"] for r in overlap_rows})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()
    total = len(scene_keys) * len(args.view_counts)
    step = 0

    for scene_key in scene_keys:
        for view_count in args.view_counts:
            step += 1
            if (scene_key, view_count) not in available:
                print(f"[skip {step}/{total}] {scene_key} {view_count}view: candidate 없음(window보다 view 수가 큼)")
                continue
            print(f"[{step}/{total}] {scene_key} {view_count}view")
            t0 = time.time()

            # --- DepthSplat (feed-forward, single deterministic inference, DL3DV in-domain) ---
            ds_log = output_dir / "logs" / f"{scene_key}_DepthSplat_{view_count}view.json"
            if not ds_log.exists():
                run_step(
                    [
                        DEPTHSPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/depthsplat_dl3dv_runner.py"),
                        "--overlap-summary", args.overlap_summary, "--scene-key", scene_key,
                        "--view-count", str(view_count), "--output-dir", str(output_dir),
                    ],
                    "depthsplat_dl3dv_runner",
                )
            if ds_log.exists():
                ds_row = json.loads(ds_log.read_text())
                for budget in args.budgets:
                    rows.append({
                        "scene": scene_key, "view_count": view_count, "budget": budget,
                        "method": "DepthSplat", "seed": None,
                        "status": "ok" if ds_row["wall_clock"] <= budget else "no_result_budget_exceeded",
                        "test_psnr": ds_row["test_psnr"] if ds_row["wall_clock"] <= budget else None,
                        "wall_clock": ds_row["wall_clock"],
                    })

            # --- Vanilla3DGS (optimization, COLMAP/random init), seed별 반복 ---
            vanilla_output = output_dir / "vanilla_runs" / scene_key
            for seed in args.seeds:
                log_path = vanilla_output / "logs" / f"dl3dv_{scene_key}_c1a_Vanilla3DGS_{view_count}view_seed{seed}.json"
                if not log_path.exists():
                    run_step(
                        [
                            PS3_PY, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
                            "--dataset", "dl3dv", "--dl3dv-scene-key", scene_key,
                            "--dl3dv-overlap-summary", args.overlap_summary,
                            "--scene", f"dl3dv_{scene_key}_c1a", "--view-count", str(view_count), "--seed", str(seed),
                            "--image-shape", *IMAGE_SHAPE,
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(vanilla_output),
                        ],
                        "vanilla_3dgs_runner(dl3dv general)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "budget": budget, "seed": seed,
                            "method": "Vanilla3DGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                            "init_source": r["init_source"] if r else None,
                        })

            # --- FSGS (sparse-view 특화 optimization, sparse-init 대체), seed별 반복 ---
            fsgs_output = output_dir / "fsgs_runs" / scene_key
            for seed in args.seeds:
                log_path = fsgs_output / "logs" / f"dl3dv_{scene_key}_c1a_FSGS_{view_count}view_seed{seed}.json"
                if not log_path.exists():
                    run_step(
                        [
                            FSGS_PY, str(REPO_ROOT / "experiments/scripts/runners/fsgs_runner.py"),
                            "--dataset", "dl3dv", "--dl3dv-scene-key", scene_key,
                            "--dl3dv-overlap-summary", args.overlap_summary,
                            "--scene", f"dl3dv_{scene_key}_c1a", "--view-count", str(view_count), "--seed", str(seed),
                            "--image-shape", *IMAGE_SHAPE,
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(fsgs_output),
                        ],
                        "fsgs_runner(dl3dv)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "budget": budget, "seed": seed,
                            "method": "FSGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                            "init_source": r["init_source"] if r else None,
                        })

            print(f"  done in {time.time()-t0:.1f}s ({len(rows)} rows so far)")
            write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    print(f"\n[done] {len(rows)} rows in {(time.time()-t_start)/3600:.2f}h. summary: {output_dir / 'c1a_main_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

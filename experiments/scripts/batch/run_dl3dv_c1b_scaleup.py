#!/usr/bin/env python3
"""V3(C1-b) 파이프라인을 DepthSplat + DL3DV pilot 25 scene 전체로 스케일업한다.

`run_re10k_c1b_scaleup.py`(MVSplat/RE10K)와 완전히 같은 구조, 모델/데이터셋만 다르다:
1. DepthSplat 추론(depthsplat env) -> gaussians.pt/render_reference.pt
2. 렌더 등가성 gate 체크(ps3 env)
3. vanilla_3dgs_runner.py --dataset dl3dv warm-start: refinement off(0s) + on(10s, 60s)

목적: RE10K에서 MVSplat으로 4/8/12-view C1-b를 돌렸을 때 나온 "view가 늘수록 refinement
효과가 커진다"는 패턴이, MVSplat의 2-view 전용(분포 밖) 학습 때문인지 아니면 진짜 view-count
효과인지 분리한다. DepthSplat은 2~6-view가 분포 안이므로 2/4-view는 공정한 비교가 된다.
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
    parser.add_argument("--overlap-summary", default=str(REPO_ROOT / "experiments/outputs/dl3dv_overlap/all_scenes_summary.json"))
    parser.add_argument("--view-count", type=int, default=2)
    parser.add_argument("--refine-budgets", type=float, nargs="+", default=[10.0, 60.0])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/dl3dv_c1b_scaleup"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 overlap summary에 있는 scene 전체")
    args = parser.parse_args()

    overlap_rows = json.loads(Path(args.overlap_summary).read_text())
    all_scenes = sorted({r["scene"] for r in overlap_rows if r["view_count"] == args.view_count})
    scene_keys = args.scenes if args.scenes else all_scenes
    output_dir = Path(args.output_dir)
    max_budget = max(args.refine_budgets)

    summary = []
    t_start = time.time()
    for i, scene_key in enumerate(scene_keys):
        t0 = time.time()
        print(f"[{i+1}/{len(scene_keys)}] {scene_key}")
        row = {"scene": scene_key, "view_count": args.view_count}

        ckpt_dir = output_dir / "checkpoints" / scene_key / f"{args.view_count}view"
        gaussians_path = ckpt_dir / "gaussians.pt"
        if not gaussians_path.exists():
            ok, _ = run_step(
                [
                    DEPTHSPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/depthsplat_dl3dv_runner.py"),
                    "--overlap-summary", args.overlap_summary, "--scene-key", scene_key,
                    "--view-count", str(args.view_count), "--output-dir", str(output_dir),
                ],
                "depthsplat_dl3dv_runner",
            )
            row["depthsplat_status"] = "ok" if ok else "failed"
            if not ok:
                summary.append(row)
                continue
        else:
            row["depthsplat_status"] = "skipped_exists"

        gate_path = ckpt_dir / "renderer_equivalence_gate.json"
        ok, _ = run_step(
            [
                PS3_PY, str(REPO_ROOT / "experiments/scripts/analysis/check_renderer_equivalence.py"),
                "--checkpoint-dir", str(ckpt_dir), "--sh-degree", "2",
            ],
            "check_renderer_equivalence",
        )
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {"gate_passed": False}
        row["gate_passed"] = gate.get("gate_passed", False)
        if not row["gate_passed"]:
            print(f"  [warn] {scene_key}: 렌더 등가성 gate 실패 — C1-b refinement 스킵")
            summary.append(row)
            continue

        vanilla_output = output_dir / "vanilla_runs" / scene_key
        run_step(
            [
                PS3_PY, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
                "--dataset", "dl3dv", "--dl3dv-scene-key", scene_key,
                "--dl3dv-overlap-summary", args.overlap_summary,
                "--scene", f"dl3dv_{scene_key}_c1b", "--view-count", str(args.view_count), "--seed", "0",
                "--warm-start-checkpoint", str(gaussians_path),
                "--image-shape", "256", "448",
                "--initial-sh-degree", "2", "--sh-degree", "2",
                "--max-budget-seconds", str(max_budget),
                "--budget-snapshots", *[str(b) for b in args.refine_budgets],
                # RE10K 스케일업에서 발견한 것과 동일한 이유로 opacity reset 비활성화.
                "--reset-every", "1000000",
                "--output-dir", str(vanilla_output),
            ],
            "vanilla_3dgs_runner(dl3dv warm-start)",
        )

        log_path = vanilla_output / "logs" / f"dl3dv_{scene_key}_c1b_Vanilla3DGS_{args.view_count}view_seed0.json"
        if log_path.exists():
            trajectory = json.loads(log_path.read_text())
            by_label = {}
            for r in trajectory:
                if r["iteration"] == 0:
                    by_label["off"] = r["test_psnr"]
                else:
                    by_label[f"on_{r['wall_clock']:.0f}s"] = r["test_psnr"]
            row.update(by_label)
            if "off" in by_label:
                on_keys = [k for k in by_label if k.startswith("on_")]
                if on_keys:
                    last_on = sorted(on_keys, key=lambda k: float(k[3:-1]))[-1]
                    row["delta_final_minus_off"] = by_label[last_on] - by_label["off"]
        else:
            row["vanilla_status"] = "no_log"

        summary.append(row)
        print(f"  done in {time.time()-t0:.1f}s: {row}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "c1b_scaleup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {len(summary)} scenes in {time.time()-t_start:.1f}s. summary: {summary_path}")

    deltas = [r["delta_final_minus_off"] for r in summary if "delta_final_minus_off" in r]
    if deltas:
        improved = sum(1 for d in deltas if d > 0)
        print(f"[stats] {len(deltas)} scenes with delta: {improved} improved, {len(deltas)-improved} regressed. "
              f"mean delta={sum(deltas)/len(deltas):.3f}dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

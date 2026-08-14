#!/usr/bin/env python3
"""V3(C1-b) 파이프라인을 RE10K main subset 20 scene 전체로 스케일업한다.

어제(2026-08-12) scene 1개(`0588138dfec165a1`)로 증명한 파이프라인을 그대로 반복한다:
1. MVSplat 추론(mvsplat env) -> gaussians.pt/render_reference.pt
2. 렌더 등가성 gate 체크(ps3 env)
3. vanilla_3dgs_runner.py warm-start: refinement off(0s) + on(10s, 60s)

표준 라이브러리만 써서 어떤 env에서 실행해도 되고, 각 단계는 맞는 conda env로 subprocess
격리한다(§7.3/§9.1 사고 재발 방지 — 기존 run_experiment_batch.py와 같은 원칙).

예상 결과: output_dir/c1b_scaleup_summary.json에 scene별 gate 통과 여부와
off/on(10s)/on(60s) PSNR, delta를 남긴다.
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


def _env_with_bin(conda_env_python: str) -> dict:
    # gsplat이 첫 호출 시 CUDA 커널을 JIT 컴파일하며 ninja를 PATH에서 찾는다. subprocess는
    # 부모 PATH만 물려받으므로 대상 env의 bin을 명시적으로 앞에 붙여야 한다
    # (run_experiment_batch.py와 동일한 2026-08-10 발견 버그/수정).
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
    parser.add_argument("--view-count", type=int, default=2)
    parser.add_argument("--refine-budgets", type=float, nargs="+", default=[10.0, 60.0])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/re10k_c1b_scaleup"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 subset 전체")
    args = parser.parse_args()

    subset = json.loads(Path(args.subset_index).read_text())
    scene_keys = args.scenes if args.scenes else list(subset.keys())
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
                    MVSPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/mvsplat_re10k_runner.py"),
                    "--subset-index", args.subset_index, "--scene-key", scene_key,
                    "--view-count", str(args.view_count), "--output-dir", str(output_dir),
                ],
                "mvsplat_re10k_runner",
            )
            row["mvsplat_status"] = "ok" if ok else "failed"
            if not ok:
                summary.append(row)
                continue
        else:
            row["mvsplat_status"] = "skipped_exists"

        gate_path = ckpt_dir / "renderer_equivalence_gate.json"
        ok, _ = run_step(
            [
                PS3_PY, str(REPO_ROOT / "experiments/scripts/analysis/check_renderer_equivalence.py"),
                "--checkpoint-dir", str(ckpt_dir),
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
                "--dataset", "re10k", "--re10k-scene-key", scene_key,
                "--re10k-subset-index", args.subset_index,
                "--scene", f"re10k_{scene_key}_c1b", "--view-count", str(args.view_count), "--seed", "0",
                "--warm-start-checkpoint", str(gaussians_path),
                "--image-shape", "256", "256",
                "--initial-sh-degree", "4", "--sh-degree", "4",
                "--max-budget-seconds", str(max_budget),
                "--budget-snapshots", *[str(b) for b in args.refine_budgets],
                # gsplat DefaultStrategy의 opacity reset(기본 reset_every=3000)은 "처음부터
                # 학습"을 가정한 메커니즘이다 — 2026-08-12 실측: 짧은 warm-start 예산(60s,
                # ~5000 iter) 안에서 iter~3000 reset을 맞으면 이미 좋은 FF 초기값의 opacity가
                # 전부 날아가고 남은 iteration으로 복구를 못 해 PSNR이 26→6.7dB로 붕괴한다
                # (reset_every를 크게 주면 사라짐, 재현 확인함). C1-b가 측정하려는 건
                # "refinement 자체의 효과"이지 "짧은 예산에 reset 타이밍이 잘못 걸리는 효과"가
                # 아니므로, C1-b warm-start 경로에서는 reset을 사실상 비활성화한다.
                "--reset-every", "1000000",
                "--output-dir", str(vanilla_output),
            ],
            "vanilla_3dgs_runner(re10k warm-start)",
        )

        log_path = vanilla_output / "logs" / f"re10k_{scene_key}_c1b_Vanilla3DGS_{args.view_count}view_seed0.json"
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

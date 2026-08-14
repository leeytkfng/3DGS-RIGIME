#!/usr/bin/env python3
"""데이터셋·모델에 무관하게 scan/scene × seed × view_count를 순회 실행하는 driver.

`run_dtu_batch.py`(2026-08-10, DTU 전용)를 일반화한 버전. model_registry.py의
`conda_env_python`/`runner_script`를 lookup해서 subprocess로 dispatch하므로, 새 모델을
추가할 때 이 파일을 고칠 필요 없이 model_registry.py에 항목만 추가하면 된다.

이 파일의 목적:
- "메인 실험 장소를 하나 정하고 모델/데이터셋은 갖다 쓴다" 방향의 아키텍처화 (2026-08-10 논의).
- 각 method는 서로 다른 conda env를 요구하므로(§7.3/§9.1 audit log 사고) 항상 subprocess로
  격리 실행한다 — 이 파일 자체는 표준 라이브러리만 써서 어떤 env에서도 실행 가능하다.
- 중단·재개 가능(계획서 STEP3): 로그 파일이 이미 있으면 건너뛴다. 실패해도 나머지는 계속 진행.

예시:
    python3 experiments/scripts/run_experiment_batch.py \\
        --dataset-root /data/Re-feem/datasets/dtu --scene-prefix dtu_scan --scenes 1 34 \\
        --methods Vanilla3DGS MVSplat --seeds 0 --view-counts 2 --vanilla-budget-seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from model_registry import MODEL_REGISTRY, ModelSpec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

# MVSplat 공식 sparse-view test split (convert_dtu.py get_example_keys(), §9.2 audit log).
OFFICIAL_DTU_SPLIT = [1, 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]


def _env_with_bin(conda_env_python: str) -> dict:
    # 첫 CUDA 커널 사용 시 ninja를 PATH에서 찾아 JIT 컴파일하는 모델이 있다(gsplat).
    # subprocess는 부모의 PATH를 그대로 물려받으므로, env의 bin을 명시적으로 앞에 붙여야
    # `RuntimeError: Ninja is required` 로 조용히 실패하는 걸 막을 수 있다 (2026-08-10 발견 버그).
    conda_bin = str(Path(conda_env_python).parent)
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    return env


def run_one(
    spec: ModelSpec,
    scene_dir: Path,
    scene_name: str,
    seed: int,
    view_count: int,
    output_dir: Path,
    vanilla_budget_seconds: float,
    densification: str,
) -> dict:
    if spec.conda_env_python is None or spec.runner_script is None:
        return {"status": "no_runner", "reason": f"{spec.name}: conda_env_python/runner_script 미등록 (model_registry.py 확인)"}

    # densification=off는 vanilla_3dgs_runner.py가 별도 log 파일(_densoff suffix)로 남긴다.
    # feed-forward 모델은 densification 개념이 없으므로 항상 "on" 취급(suffix 없음).
    suffix = "_densoff" if (spec.family == "optimization" and densification == "off") else ""
    log_path = output_dir / "logs" / f"{scene_name}_{spec.name}_{view_count}view_seed{seed}{suffix}.json"
    if log_path.exists():
        return {"status": "skipped_exists", "log": str(log_path)}

    cmd = [
        spec.conda_env_python, str(REPO_ROOT / spec.runner_script),
        "--scan-dir", str(scene_dir),
        "--scene", scene_name,
        "--seed", str(seed),
        "--view-count", str(view_count),
        "--output-dir", str(output_dir),
    ]
    if spec.family == "optimization":
        cmd += ["--max-budget-seconds", str(vanilla_budget_seconds), "--densification", densification]

    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(spec.conda_env_python))
    if result.returncode != 0:
        return {"status": "failed", "stderr_tail": result.stderr[-2000:]}
    return {"status": "ok", "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run registered models over scene x seed x view_count, any dataset.")
    parser.add_argument("--dataset-root", default="/data/Re-feem/datasets/dtu", help="scene 폴더들이 들어있는 루트 (예: .../dtu, .../re10k)")
    parser.add_argument("--scene-prefix", default="dtu_scan", help="scene 폴더명 -> 로그용 scene 이름 접두사 (scene<번호> 폴더를 가정)")
    parser.add_argument("--scenes", nargs="+", default=[str(s) for s in OFFICIAL_DTU_SPLIT], help="scene 번호/id 목록")
    parser.add_argument("--methods", nargs="+", default=["Vanilla3DGS", "MVSplat"], choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--view-counts", nargs="+", type=int, default=[2])
    parser.add_argument("--vanilla-budget-seconds", type=float, default=300.0)
    parser.add_argument(
        "--densification",
        choices=["on", "off"],
        default="on",
        help="optimization 계열 method에만 적용(C1-b ablation). feed-forward는 무시된다.",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs"))
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    summary: list[dict] = []
    t_start = time.time()

    combos = [(scene, seed, vc) for scene in args.scenes for seed in args.seeds for vc in args.view_counts]
    print(f"[batch] {len(combos)} scene x seed x view_count combos, methods={args.methods}, dataset_root={dataset_root}")

    for scene_id, seed, vc in combos:
        scene_dir = dataset_root / f"scan{scene_id}" if args.scene_prefix.endswith("scan") else dataset_root / str(scene_id)
        scene_name = f"{args.scene_prefix}{scene_id}"
        if not scene_dir.exists():
            print(f"[skip] {scene_name} not found ({scene_dir})")
            summary.append({"scene": scene_id, "seed": seed, "view_count": vc, "method": "-", "status": "missing_data"})
            continue

        for method in args.methods:
            spec = MODEL_REGISTRY[method]
            t0 = time.time()
            result = run_one(spec, scene_dir, scene_name, seed, vc, output_dir, args.vanilla_budget_seconds, args.densification)
            result.update({"scene": scene_id, "seed": seed, "view_count": vc, "method": method, "elapsed": time.time() - t0})
            print(f"[{method}] {scene_name} seed{seed} {vc}view -> {result['status']} ({result['elapsed']:.1f}s)")
            summary.append(result)

    summary_path = output_dir / "logs" / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    summary_path.write_text(json.dumps(existing + summary, indent=2), encoding="utf-8")

    counts = {}
    for s in summary:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print(f"[batch] done in {time.time()-t_start:.1f}s. {counts}")
    print(f"[batch] summary written to {summary_path}")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())

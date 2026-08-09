#!/usr/bin/env python3
"""실험 config를 읽어 전체 run manifest를 생성하는 스캐폴드.

이 파일의 목적:
- 실제 DepthSplat/MVSplat/3DGS runner를 붙이기 전에 실험 격자를 먼저 고정한다.
- 어떤 scene, seed, view 수, overlap level, budget, method를 실행할지 manifest로 남긴다.
- 예상 출력은 `experiments/outputs/experiment_manifest.json`이다.
"""

import json
import sys
from itertools import product
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 실행 환경에서 PyYAML 누락 시 바로 안내한다.
    raise SystemExit(f"PyYAML is required to run this script: {exc}") from exc

from model_registry import DATASET_REGISTRY, MODEL_REGISTRY
from protocol_utils import compute_tau


def build_experiment_plan(config: dict) -> list[dict]:
    """config의 축을 조합해 실행 계획 list를 만든다.

    목적:
    - 본 실험 전에 전체 실험 공간을 눈으로 확인하고 동결한다.
    - 나중에 실제 runner가 붙으면 이 plan의 각 row를 하나의 작업 단위로 삼는다.

    예상 결과:
    - main phase: regime map용 전체 grid.
    - c1b phase: feed-forward 초기값 고정 후 refinement off/on 비교.
    - c2 phase: depth noise와 scale bias 민감도 분석.
    """

    protocol = config.get("protocol", {})
    methods_cfg = config.get("methods", {})
    runtime_cfg = config.get("runtime", {})

    primary_dataset = protocol.get("dataset_primary", "RE10K")
    view_counts = protocol.get("view_counts", [])
    overlap_levels = protocol.get("overlap_levels", [])
    budgets = protocol.get("budgets_seconds", [])
    seeds = protocol.get("seeds", [0])
    scene_count = int(protocol.get("scenes_primary", 0))

    # 실제 scene id 목록이 붙기 전까지는 deterministic placeholder를 쓴다.
    # 예상 결과: re10k_scene_000, re10k_scene_001 ... 형태의 manifest row가 생성된다.
    scenes = [f"{primary_dataset.lower()}_scene_{index:03d}" for index in range(scene_count)]
    feedforward_methods = methods_cfg.get("feedforward", [])
    optimization_methods = methods_cfg.get("optimization", [])
    all_methods = feedforward_methods + optimization_methods

    plan = []

    # Main regime-map grid.
    # 중요: scene은 통계적 독립 단위, seed는 scene 내부 반복 측정이다.
    # 중요: checkpoint_selection은 budget_end_checkpoint로 고정해 test oracle 선택을 막는다.
    for scene, seed, view_count, overlap, budget, method in product(scenes, seeds, view_counts, overlap_levels, budgets, all_methods):
        plan.append(
            {
                "phase": "main",
                "dataset": primary_dataset,
                "scene": scene,
                "seed": seed,
                "view_count": view_count,
                "overlap": overlap,
                "budget_seconds": budget,
                "method": method,
                "checkpoint_selection": protocol.get("checkpoint_rule", "budget_end_checkpoint"),
                "oracle_allowed": False,
                "data_root": runtime_cfg.get("data_root", "/data"),
            }
        )

    c1b_cfg = config.get("c1b", {})
    if c1b_cfg.get("enabled", False):
        # C1-b 목적:
        # - feed-forward Gaussian 초기값은 그대로 둔다.
        # - standard 3DGS refinement를 끄거나 켰을 때의 순효과만 본다.
        # 예상 결과: refinement=off 0초 row와 refinement=on 10/60/300초 row가 생성된다.
        refinement_budgets = c1b_cfg.get("refinement_budget_seconds", [10, 60, 300])
        if isinstance(refinement_budgets, (int, float)):
            refinement_budgets = [refinement_budgets]

        for scene, seed, view_count, overlap, method in product(scenes, seeds, view_counts, overlap_levels, feedforward_methods):
            plan.append(
                {
                    "phase": "c1b",
                    "dataset": primary_dataset,
                    "scene": scene,
                    "seed": seed,
                    "view_count": view_count,
                    "overlap": overlap,
                    "budget_seconds": 0,
                    "method": method,
                    "refinement": "off",
                    "renderer_equivalence_gate": True,
                    "data_root": runtime_cfg.get("data_root", "/data"),
                }
            )
            for refinement_budget in refinement_budgets:
                plan.append(
                    {
                        "phase": "c1b",
                        "dataset": primary_dataset,
                        "scene": scene,
                        "seed": seed,
                        "view_count": view_count,
                        "overlap": overlap,
                        "budget_seconds": refinement_budget,
                        "method": method,
                        "refinement": "on",
                        "renderer_equivalence_gate": True,
                        "data_root": runtime_cfg.get("data_root", "/data"),
                    }
                )

    c2_cfg = config.get("c2", {})
    if c2_cfg.get("enabled", False):
        # C2 목적:
        # - 초기 depth uncertainty가 refinement 동역학에 미치는 민감도를 본다.
        # - 현실의 모든 depth 오류를 재현한다는 주장이 아니라 sensitivity analysis다.
        # 예상 결과: DTU 대표 조건마다 iid noise row와 scale bias row가 생성된다.
        external_dataset = protocol.get("dataset_external", "DTU")
        external_scenes = [
            f"{external_dataset.lower()}_scene_{index:03d}"
            for index in range(int(protocol.get("scenes_external", 0)))
        ]
        for scene, seed, condition, noise_sigma in product(
            external_scenes,
            seeds,
            c2_cfg.get("representative_conditions", []),
            c2_cfg.get("depth_noise_levels", []),
        ):
            plan.append(
                {
                    "phase": "c2_depth_noise",
                    "dataset": external_dataset,
                    "scene": scene,
                    "seed": seed,
                    "condition": condition,
                    "depth_perturbation": "iid_multiplicative",
                    "sigma": noise_sigma,
                    "claim_scope": c2_cfg.get("claim_scope", "sensitivity_analysis"),
                    "data_root": runtime_cfg.get("data_root", "/data"),
                }
            )
        for scene, seed, condition, scale in product(
            external_scenes,
            seeds,
            c2_cfg.get("representative_conditions", []),
            c2_cfg.get("global_scale_bias", []),
        ):
            plan.append(
                {
                    "phase": "c2_depth_scale_bias",
                    "dataset": external_dataset,
                    "scene": scene,
                    "seed": seed,
                    "condition": condition,
                    "depth_perturbation": "global_scale_bias",
                    "scale": scale,
                    "claim_scope": c2_cfg.get("claim_scope", "sensitivity_analysis"),
                    "data_root": runtime_cfg.get("data_root", "/data"),
                }
            )

    return plan


def validate_config(config: dict) -> list[str]:
    """config가 논문 프로토콜을 벗어나는지 가벼운 경고를 만든다.

    목적:
    - 지원하지 않는 view 수 요청, oracle peak 사용 같은 위험한 drift를 빨리 발견한다.

    예상 결과:
    - 문제가 없으면 빈 list.
    - 문제가 있으면 manifest의 `config_warnings`와 터미널 출력에 남길 문자열 list.
    """

    warnings = []
    protocol = config.get("protocol", {})
    methods_cfg = config.get("methods", {})
    requested_methods = methods_cfg.get("feedforward", []) + methods_cfg.get("optimization", [])

    for method in requested_methods:
        if method not in MODEL_REGISTRY:
            warnings.append(f"Unknown method in config: {method}")
            continue
        supported = set(MODEL_REGISTRY[method].supports_views)
        unsupported = [view_count for view_count in protocol.get("view_counts", []) if view_count not in supported]
        if unsupported:
            warnings.append(f"{method} does not list support for views: {unsupported}")

    if protocol.get("checkpoint_rule") != "budget_end_checkpoint":
        warnings.append("Main checkpoint rule should be budget_end_checkpoint to avoid test leakage.")

    if config.get("protocol", {}).get("use_oracle_peak", False):
        warnings.append("use_oracle_peak is true; oracle peak must stay diagnostic-only.")

    return warnings


def main() -> int:
    """manifest 생성 CLI 진입점.

    실행 흐름:
    1. YAML config를 읽는다.
    2. output directory와 하위 결과 폴더를 만든다.
    3. 전체 실험 plan과 protocol guard를 manifest JSON으로 저장한다.
    4. 터미널에는 plan 수와 등록된 모델/데이터셋을 출력한다.

    예상 결과:
    - `experiments/outputs/experiment_manifest.json` 생성.
    - 정상 종료 시 0 반환.
    """

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else repo_root / "experiments/configs/experiment_config.yaml"
    output_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else repo_root / "experiments/outputs"

    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in config.get("logging", {}).get("output_dirs", {}).values():
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    runtime_cfg = config.get("runtime", {})
    data_root = runtime_cfg.get("data_root", "/data")
    plan = build_experiment_plan(config)
    warnings = validate_config(config)
    tau_cfg = config.get("analysis", {}).get("tau", {})
    tau = compute_tau(
        seed_variability=float(tau_cfg.get("pilot_seed_variability_psnr", 0.0)),
        practical_min_delta=float(tau_cfg.get("practical_min_delta_psnr", 0.5)),
    )

    manifest_path = output_dir / "experiment_manifest.json"
    # protocol_guards는 config가 나중에 바뀌더라도 특정 결과 폴더가 어떤 규칙으로
    # 생성됐는지 복원할 수 있게 manifest 안에 복제해 둔다.
    manifest_payload = {
        "data_root": data_root,
        "protocol_guards": {
            "pose_mode": runtime_cfg.get("pose_mode", "pose_given"),
            "main_checkpoint_rule": config.get("protocol", {}).get("checkpoint_rule", "budget_end_checkpoint"),
            "oracle_peak_storage": config.get("logging", {}).get("output_dirs", {}).get("oracle_results"),
            "tau_psnr": tau,
            "cluster_unit": config.get("analysis", {}).get("cluster_unit", "scene"),
            "renderer_equivalence_tolerance": config.get("c1b", {}).get("renderer_equivalence_tolerance"),
        },
        "datasets": {
            name: {
                **spec,
                "resolved_path": spec["path"],
            }
            for name, spec in DATASET_REGISTRY.items()
        },
        "models": {
            name: {
                "family": spec.family,
                "requires_pose": spec.requires_pose,
                "supports_views": spec.supports_views,
                "notes": spec.notes,
            }
            for name, spec in MODEL_REGISTRY.items()
        },
        "overlap_protocol": config.get("overlap", {}),
        "analysis_protocol": config.get("analysis", {}),
        "config_warnings": warnings,
        "plan": plan,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"Loaded config: {config_path}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_dir}")
    print(f"Prepared {len(plan)} planned experiment runs.")
    print("Experiment manifest written to:", manifest_path)
    if warnings:
        print("Config warnings:")
        for warning in warnings:
            print(f" ! {warning}")
    print("Registered models and datasets for the scaffold:")
    for name in sorted(MODEL_REGISTRY):
        print(f" - model: {name}")
    for name in sorted(DATASET_REGISTRY):
        print(f" - dataset: {name} -> {DATASET_REGISTRY[name]['path']}")
    print("Recommendation: use RE10K as the primary benchmark, DTU as the external geometry validation set, and DL3DV as a secondary candidate if available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

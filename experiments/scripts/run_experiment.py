#!/usr/bin/env python3
import json
import sys
from itertools import product
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime safeguard
    raise SystemExit(f"PyYAML is required to run this script: {exc}") from exc

from model_registry import DATASET_REGISTRY, MODEL_REGISTRY
from protocol_utils import compute_tau


def build_experiment_plan(config: dict) -> list[dict]:
    # This scaffold creates an auditable run manifest before real model runners
    # are connected. The manifest is the contract that keeps the paper protocol
    # stable across later implementation work.
    protocol = config.get("protocol", {})
    methods_cfg = config.get("methods", {})
    runtime_cfg = config.get("runtime", {})

    primary_dataset = protocol.get("dataset_primary", "RE10K")
    view_counts = protocol.get("view_counts", [])
    overlap_levels = protocol.get("overlap_levels", [])
    budgets = protocol.get("budgets_seconds", [])
    seeds = protocol.get("seeds", [0])
    scene_count = int(protocol.get("scenes_primary", 0))
    scenes = [f"{primary_dataset.lower()}_scene_{index:03d}" for index in range(scene_count)]
    feedforward_methods = methods_cfg.get("feedforward", [])
    optimization_methods = methods_cfg.get("optimization", [])
    all_methods = feedforward_methods + optimization_methods

    plan = []
    # Main regime-map grid: scene is the statistical unit, seed is a repeated
    # measurement, and budget uses the fixed budget-end checkpoint rule.
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
        # C1-b isolates the effect of standard 3DGS refinement by holding the
        # feed-forward Gaussian initialization fixed and toggling refinement.
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
        # C2 is intentionally a sensitivity analysis, not a claim that these
        # perturbations fully reproduce real monocular-depth errors.
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
    # These warnings catch protocol drift early, especially accidental oracle
    # checkpoint use or unsupported view-count requests.
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
    # Protocol guards are duplicated into the manifest so result folders remain
    # interpretable even if the YAML config changes later.
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

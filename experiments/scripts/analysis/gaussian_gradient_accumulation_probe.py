#!/usr/bin/env python3
"""Per-Gaussian gradient accumulation/count 분포를 view 수 축에서 실측한다 — v2.

배경(2026-08-13 대화): gsplat `DefaultStrategy._grow_gs()`는 densification 판단을

    grad2d[g] / count[g] > tau_pos   (tau_pos = grow_grad2d = 0.0002)

로 한다 — `count[g]`는 그 Gaussian이 최근 refine window(기본 100 step) 동안 "보인"
횟수(effective observation)다. 가설: sparse-view/저-overlap일수록 각 Gaussian이 보이는
view 수가 줄어 `count[g]` 분포가 낮은 쪽으로 치우치고, 그 결과 gradient 평균 추정이
더 적은 표본에 기반해 노이즈가 커진다.

**v1(densification off로 index 고정 후 관측)은 null 결과였다** — 초기 COLMAP triangulation
점은 정의상 여러 view에서 correspondence가 맞아야만 존재하므로(survivorship bias),
count가 거의 항상 포화(≈n_steps)됐다. 낮은 count가 나올 진짜 후보군은 densification이
"새로" 만들어내는(grow/split) 아직 검증 안 된 점들이다 — 그래서 v2는 densification을
켜고, gsplat이 실제로 판단을 내리는 바로 그 순간(매 `refine_every`=100 step, `_grow_gs`
호출 직전)에 `state["count"]`/`state["grad2d"]`를 스냅샷한다. `step_post_backward()`를
그대로 호출하지 않고 내부 로직(`_update_state`→(refine 조건 만족 시) 스냅샷→`_grow_gs`/
`_prune_gs`→reset)을 우리가 직접 오케스트레이션한다 — 함수 자체(`_update_state`,
`_grow_gs`, `_prune_gs`)는 gsplat 원본을 그대로 쓰고, "스냅샷을 껴넣는 지점"만 다르다.
opacity reset(`reset_every`=3000)은 우리 step 범위(<2000) 밖이라 생략.

2-view 등 초기 triangulation이 부실한 조건은 `vanilla_3dgs_runner.py`와 동일한
`MIN_SFM_POINTS`+random-sphere fallback을 그대로 재사용한다 — 이 fallback 점들 자체가
"correspondence 검증을 거치지 않은 점"이라 count-skew 가설을 테스트할 두 번째 후보군이다.

데이터: RE10K main subset(실제 scene), COLMAP sparse triangulation init(다른 러너와 동일 코어).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from re10k_dataset import get_scene_item, load_views  # noqa: E402
from vanilla_3dgs_runner import (  # noqa: E402
    MIN_SFM_POINTS,
    _colmap_init_from_loaded_views,
    _random_points_in_sphere,
    build_camera_tensors,
    build_optimizers,
    init_gaussians,
    ssim,
)

RE10K_ROOT = Path("/data/Re-feem/datasets/re10k/test")
TAU = 0.0002  # gsplat DefaultStrategy 기본 grow_grad2d


def _snapshot_stats(count: np.ndarray, grad2d: np.ndarray, step: int) -> dict:
    grad_avg = grad2d / np.maximum(count, 1)
    low_mask = count <= 2
    high_mask = ~low_mask
    return {
        "step": step,
        "n_gaussian": int(count.shape[0]),
        "count_median": float(np.median(count)),
        "count_p10": float(np.percentile(count, 10)),
        "frac_count_eq0": float((count == 0).mean()),
        "frac_count_le2": float(low_mask.mean()),
        "frac_count_le5": float((count <= 5).mean()),
        "frac_above_tau_low_count": float(grad_avg[low_mask].mean() > TAU) if low_mask.any() else None,
        "frac_above_tau_overall": float((grad_avg > TAU).mean()),
        "n_low_count": int(low_mask.sum()),
        "n_high_count": int(high_mask.sum()),
        # low-count 집단과 high-count 집단이 "densify 대상으로 뽑힐 확률" 자체를 비교 —
        # 이게 가설의 핵심 예측: count가 적을수록 grad_avg 추정이 노이즈에 취약해 tau를
        # 넘는 비율이 count 무관하게 비슷하거나(=추정 노이즈로 인한 우연한 초과) 오히려
        # 왜곡될 것이다.
        "frac_above_tau_by_count_bucket": {
            "count_le2": float((grad_avg[low_mask] > TAU).mean()) if low_mask.any() else None,
            "count_3_10": float((grad_avg[(count > 2) & (count <= 10)] > TAU).mean()) if ((count > 2) & (count <= 10)).any() else None,
            "count_gt10": float((grad_avg[count > 10] > TAU).mean()) if (count > 10).any() else None,
        },
    }


def run_one(scene_key: str, entry: dict, view_count: int, n_steps: int, seed: int, device: torch.device) -> dict:
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    candidate = entry["view_candidates"][str(view_count)]
    if candidate.get("context") is None:
        return {"status": "no_candidate"}
    train_ids = candidate["context"]

    item = get_scene_item(RE10K_ROOT / entry["chunk_file"], scene_key)
    train_views = load_views(item, train_ids, target_shape=(256, 256))

    workdir = Path("/tmp") / "gaussian_grad_probe_v2" / scene_key / f"{view_count}view"
    sfm_points, sfm_colors = _colmap_init_from_loaded_views(train_views, workdir)

    centers = np.stack([v["center"] for v in train_views])
    center = centers.mean(axis=0)
    radius = float(np.median(np.linalg.norm(centers - center, axis=1))) or 1.0

    if sfm_points.shape[0] < MIN_SFM_POINTS:
        init_source = "random_sphere_fallback"
        points, colors = _random_points_in_sphere(center, radius, 100_000, seed)
    else:
        init_source = "colmap_sfm"
        points, colors = sfm_points, sfm_colors

    params = init_gaussians(points, colors, device)
    n_gaussian_init = params["means"].shape[0]

    optimizers = build_optimizers(params, scene_scale=radius)

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=radius)

    train_cams = [build_camera_tensors(v, device) for v in train_views]
    height, width = train_views[0]["height"], train_views[0]["width"]

    rng = np.random.default_rng(seed)
    order = list(range(len(train_cams)))
    snapshots: list[dict] = []

    for step in range(1, n_steps + 1):
        if (step - 1) % len(order) == 0:
            rng.shuffle(order)
        viewmat, k, gt_image = train_cams[order[(step - 1) % len(order)]]

        colors_sh = torch.cat([params["sh0"], params["shN"]], dim=1)
        render, _, info = rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors_sh,
            viewmats=viewmat[None],
            Ks=k[None],
            width=width,
            height=height,
            sh_degree=0,
            packed=False,
        )
        info["means2d"].retain_grad()
        pred = render[0].permute(2, 0, 1).clamp(0.0, 1.0)
        loss = 0.8 * torch.abs(pred - gt_image).mean() + 0.2 * (1.0 - ssim(pred, gt_image))

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()

        strategy._update_state(params, state, info)

        is_refine_step = step > strategy.refine_start_iter and step % strategy.refine_every == 0
        if is_refine_step:
            count_np = state["count"].detach().cpu().numpy().copy()
            grad2d_np = state["grad2d"].detach().cpu().numpy().copy()
            snapshots.append(_snapshot_stats(count_np, grad2d_np, step))

            strategy._grow_gs(params, optimizers, state, step)
            strategy._prune_gs(params, optimizers, state, step)
            state["grad2d"].zero_()
            state["count"].zero_()

        for optimizer in optimizers.values():
            optimizer.step()

    return {
        "status": "ok",
        "init_source": init_source,
        "n_gaussian_init": int(n_gaussian_init),
        "n_gaussian_final": int(params["means"].shape[0]),
        "n_steps": n_steps,
        "n_train_views": len(train_ids),
        "n_snapshots": len(snapshots),
        "snapshots": snapshots,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-index", default="experiments/outputs/re10k_main_subset/re10k_main_subset.json")
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 subset 처음 3개 scene.")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--n-steps", type=int, default=2000, help="refine_start_iter=500 이후 여러 refine window(100 step)를 보려면 충분히 커야 한다.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/outputs/gaussian_grad_probe/summary_v2.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = json.loads(Path(args.subset_index).read_text())
    scene_keys = args.scenes or list(subset.keys())[:3]

    results = []
    for scene_key in scene_keys:
        entry = subset[scene_key]
        for view_count in args.view_counts:
            print(f"[run] scene={scene_key} view_count={view_count}")
            row = run_one(scene_key, entry, view_count, args.n_steps, args.seed, device)
            row.update({"scene": scene_key, "view_count": view_count})
            results.append(row)
            last = row["snapshots"][-1] if row.get("snapshots") else None
            print(f"  -> status={row['status']} init={row.get('init_source')} n_gaussian_final={row.get('n_gaussian_final')} last_snapshot={last}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")

    print("\n=== view_count별 마지막 snapshot pooled 평균 ===")
    for view_count in args.view_counts:
        rows = [r for r in results if r["view_count"] == view_count and r["status"] == "ok" and r["snapshots"]]
        if not rows:
            print(f"view_count={view_count}: no data")
            continue
        lasts = [r["snapshots"][-1] for r in rows]
        frac_le2 = [s["frac_count_le2"] for s in lasts]
        frac_low_tau = [s["frac_above_tau_by_count_bucket"]["count_le2"] for s in lasts if s["frac_above_tau_by_count_bucket"]["count_le2"] is not None]
        frac_high_tau = [s["frac_above_tau_by_count_bucket"]["count_gt10"] for s in lasts if s["frac_above_tau_by_count_bucket"]["count_gt10"] is not None]
        print(
            f"view_count={view_count}: n_scenes={len(rows)} "
            f"frac_count_le2={np.mean(frac_le2):.3f} "
            f"frac_above_tau(count<=2)={np.mean(frac_low_tau) if frac_low_tau else float('nan'):.3f} "
            f"frac_above_tau(count>10)={np.mean(frac_high_tau) if frac_high_tau else float('nan'):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

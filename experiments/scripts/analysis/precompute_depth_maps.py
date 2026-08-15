#!/usr/bin/env python3
"""C2용 depth map 사전계산 — train view마다 Depth Anything V2 Metric을 돌려 raw depth를
캐시 파일(.pt)로 저장한다. 노이즈/scale-bias는 여기서 적용하지 않는다 — 그건
`vanilla_3dgs_runner.py`가 소비할 때(§5.9 sigma/scale sweep) 적용해서, 같은 base depth를
여러 교란 조건에 재사용하고 모델을 sweep마다 다시 돌리지 않는다.

전용 conda env `depth`에서 실행: /opt/conda/envs/depth/bin/python3 analysis/precompute_depth_maps.py ...

출력 스키마 (torch.save):
{
  "view_ids": [int, ...],
  "depths": [np.ndarray(H,W), ...],       # 카메라 기하로 스케일 교정된 metric depth, perturb 전
  "images": [np.ndarray(H,W,3) float01, ...],
  "K": [np.ndarray(3,3), ...], "R": [np.ndarray(3,3), ...], "t": [np.ndarray(3,), ...],
}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from depth_model import DepthEstimator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def compute_scene_center(args: argparse.Namespace, train_views: list[dict]) -> "np.ndarray":
    """카메라 기하로만(=depth 예측과 무관하게) scene 중심을 추정한다. 이 값은 뒤에서 monocular
    depth의 절대 스케일을 교정하는 기준으로 쓰인다 — `vanilla_3dgs_runner.py`의 DTU
    `estimate_scene_sphere()` / RE10K·DL3DV centroid-median 휴리스틱과 동일한 로직."""

    if args.dataset == "dtu":
        from dtu_dataset import estimate_scene_sphere, load_camera

        scan_dir = Path(args.scan_dir)
        cameras = [load_camera(scan_dir / "cameras", v) for v in range(1, 50)]
        center, _radius = estimate_scene_sphere(cameras)
        return center

    centers = np.stack([v["center"] for v in train_views])
    return centers.mean(axis=0)


def calibrate_depth(depth: "np.ndarray", camera_position: "np.ndarray", scene_center: "np.ndarray") -> tuple["np.ndarray", float]:
    """monocular metric depth 모델은 DTU 같은 근접(macro) 촬영에서는 학습 분포 밖이라 절대
    스케일이 종종 크게 틀린다(2026-08-15 스모크 테스트에서 실측: DTU scan1 카메라-중심 거리는
    ~608 world-unit인데 모델 원출력은 0.5~3.9 범위로 나와 view마다 제각각 스케일이 어긋났고,
    그 결과 여러 view를 합친 back-projection point cloud의 중심이 실제 장면 중심에서
    scene radius의 2배 넘게 벗어났다). 카메라 pose는 이미 정확히 알고 있으므로(pose-given
    track), 카메라-장면중심 거리를 기준으로 view별 median depth를 재조정해 스케일을 바로잡는다.
    이 교정은 sigma=0/scale_bias=1.0(교란 없음) 기준선을 "그럭저럭 쓸만한 depth"로 만들기
    위한 전처리이며, C2가 실제로 통제하는 sigma/scale_bias 교란과는 별개다."""

    target_distance = float(np.linalg.norm(camera_position - scene_center))
    valid = depth[np.isfinite(depth) & (depth > 0)]
    median_depth = float(np.median(valid)) if valid.size > 0 else 1.0
    calibration_scale = target_distance / median_depth if median_depth > 0 else 1.0
    return depth * calibration_scale, calibration_scale


def load_train_views(args: argparse.Namespace) -> list[dict]:
    target_shape = tuple(args.image_shape) if args.image_shape else None

    if args.dataset == "dtu":
        from dtu_dataset import load_scan

        if not args.scan_dir:
            raise SystemExit("--dataset dtu는 --scan-dir가 필요하다.")
        scan_dir = Path(args.scan_dir)
        all_view_ids = list(range(1, 50))
        test_ids = set(all_view_ids[::7])
        train_pool = [v for v in all_view_ids if v not in test_ids]

        rng = np.random.default_rng(args.seed)
        train_ids = sorted(rng.choice(train_pool, size=min(args.view_count, len(train_pool)), replace=False).tolist())
        return load_scan(scan_dir, train_ids, target_shape=target_shape)

    if args.dataset == "re10k":
        from re10k_dataset import get_scene_item, load_views

        subset = json.loads(Path(args.re10k_subset_index).read_text())
        entry = subset[args.re10k_scene_key]
        candidate = entry["view_candidates"][str(args.view_count)]
        if candidate.get("context") is None:
            raise SystemExit(f"{args.re10k_scene_key} view_count={args.view_count}: candidate 없음")
        item = get_scene_item(Path("/data/Re-feem/datasets/re10k/test") / entry["chunk_file"], args.re10k_scene_key)
        return load_views(item, candidate["context"], target_shape=target_shape)

    if args.dataset == "dl3dv":
        from dl3dv_dataset import load_metadata, load_views

        overlap_summary = json.loads(Path(args.dl3dv_overlap_summary).read_text())
        row = next(r for r in overlap_summary if r["scene"] == args.dl3dv_scene_key and r["view_count"] == args.view_count)
        scene_dir = Path("/data/Re-feem/datasets/dl3dv") / args.dl3dv_scene_key
        meta = load_metadata(scene_dir)
        return load_views(scene_dir, meta, row["context_indices"], target_shape=target_shape)

    raise SystemExit(f"unknown dataset {args.dataset}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dtu", "re10k", "dl3dv"], required=True)
    parser.add_argument("--scan-dir", default=None)
    parser.add_argument("--re10k-scene-key", default=None)
    parser.add_argument("--re10k-subset-index", default=str(REPO_ROOT / "experiments/outputs/re10k_main_subset/re10k_main_subset.json"))
    parser.add_argument("--dl3dv-scene-key", default=None)
    parser.add_argument("--dl3dv-overlap-summary", default=str(REPO_ROOT / "experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json"))
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0, help="dtu 전용: train view 샘플링 seed. runner와 반드시 동일해야 같은 view가 쓰인다.")
    parser.add_argument("--image-shape", type=int, nargs=2, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_views = load_train_views(args)
    print(f"[data] {len(train_views)} train views loaded (dataset={args.dataset}, view_count={args.view_count})")

    scene_center = compute_scene_center(args, train_views)
    print(f"[calib] scene_center={scene_center}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = DepthEstimator(device=device)
    print(f"[setup] depth estimator ready (device={device})")

    view_ids, depths, images, Ks, Rs, ts = [], [], [], [], [], []
    t0 = time.time()
    for view in train_views:
        raw_depth = estimator.predict(view["image"])
        cam_position = -view["R"].T @ view["t"]
        depth, calib_scale = calibrate_depth(raw_depth, cam_position, scene_center)
        view_ids.append(view["view_id"])
        depths.append(depth)
        images.append(view["image"])
        Ks.append(view["K"])
        Rs.append(view["R"])
        ts.append(view["t"])
        print(
            f"  view {view['view_id']}: raw depth [{raw_depth.min():.3f}, {raw_depth.max():.3f}] "
            f"-> calibrated [{depth.min():.3f}, {depth.max():.3f}] (scale={calib_scale:.3f})"
        )
    print(f"[done] {len(view_ids)} views in {time.time()-t0:.1f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".pt.tmp")
    torch.save(
        {"view_ids": view_ids, "depths": depths, "images": images, "K": Ks, "R": Rs, "t": ts},
        tmp_path,
    )
    tmp_path.replace(output_path)
    print(f"[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

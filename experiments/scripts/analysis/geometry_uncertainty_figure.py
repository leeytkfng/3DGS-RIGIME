#!/usr/bin/env python3
"""§5.3(co-visibility 기반 overlap 정의)와 D 항목(공부방향.md)을 정당화하는 실측 figure 생성.

이론적 배경 (공부방향.md A-1):
- Gauss-Newton 삼각측량의 공분산은 σ²(JᵀJ)⁻¹로 근사된다. baseline이 좁을수록(시선이
  거의 평행할수록) JᵀJ가 ill-conditioned해지고 깊이 방향 분산이 커진다.
- 이 파일은 그 근사를 "이론값"이 아니라, 이미 갖고 있는 COLMAP triangulation 결과에서
  직접 계산해 baseline·overlap과 함께 그린다. 새 실험이 아니라 기존 산출물 재활용이다.

입력: pycolmap Reconstruction (scan1 dense-sanity에서 이미 만들어진 sparse_triangulated).
출력:
- baseline_vs_overlap.png (공부방향.md C-1: 두 힘이 반대 방향이라는 것의 실측 확인)
- overlap_vs_depth_uncertainty.png (§5.3 overlap 정의를 정당화하는 핵심 figure)
- baseline_vs_depth_uncertainty.png
- pairwise_geometry.csv (view pair별 baseline/overlap/JᵀJ 기반 불확실성/COLMAP reprojection error)

실행: ps3 conda env (pycolmap, protocol_utils가 여기 있음).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from protocol_utils import compute_pairwise_overlaps  # noqa: E402


def load_reconstruction(path: Path):
    import pycolmap

    return pycolmap.Reconstruction(str(path))


def build_view_points(reconstruction) -> dict[str, set[str]]:
    """image name -> 관측한 point3D id 집합. generate_overlap.py와 동일한 입력 형태."""

    view_points: dict[str, set[str]] = {}
    for image in reconstruction.images.values():
        visible = set()
        for p2d in image.points2D:
            if p2d.has_point3D():
                visible.add(str(p2d.point3D_id))
        view_points[image.name] = visible
    return view_points


def two_view_depth_uncertainty(
    point_xyz: np.ndarray,
    cam_i: dict,
    cam_j: dict,
    pixel_sigma: float = 1.0,
) -> float:
    """두 view만으로 X를 삼각측량했다고 가정할 때의 depth-direction 표준편차.

    Gauss-Newton 재투영 오차의 Jacobian(J: 4x3, view당 2행)을 실제 카메라 파라미터로
    직접 계산하고, Cov(X) ≈ pixel_sigma^2 * (JᵀJ)^-1 을 구한 뒤 view i의 시선 방향으로
    투영한 성분을 depth 불확실성으로 쓴다 (공부방향.md A-1의 그 근사).
    """

    J_rows = []
    for cam in (cam_i, cam_j):
        R, t, K = cam["R"], cam["t"], cam["K"]
        x_cam = R @ point_xyz + t
        x, y, z = x_cam
        fx, fy = K[0, 0], K[1, 1]
        # d(u,v)/d(x_cam)
        d_uv_d_xcam = np.array(
            [
                [fx / z, 0.0, -fx * x / z**2],
                [0.0, fy / z, -fy * y / z**2],
            ]
        )
        # d(x_cam)/dX = R
        J_view = d_uv_d_xcam @ R
        J_rows.append(J_view)

    J = np.vstack(J_rows)  # 4x3
    JtJ = J.T @ J
    try:
        cov = pixel_sigma**2 * np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        return float("nan")

    # 깊이 방향: view i의 시선 방향(world frame)으로 공분산을 투영.
    R_i = cam_i["R"]
    view_dir_world = R_i.T @ np.array([0.0, 0.0, 1.0])
    view_dir_world /= np.linalg.norm(view_dir_world)
    depth_var = float(view_dir_world @ cov @ view_dir_world)
    return float(np.sqrt(max(depth_var, 0.0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reconstruction",
        default="/root/task 5/experiments/outputs_dense_sanity/colmap_work/dtu_scan1_dense_sanity/49view_seed0/sparse_triangulated",
    )
    parser.add_argument("--out-dir", default="/root/task 5/experiments/outputs/geometry_figures")
    parser.add_argument("--min-shared-points", type=int, default=5, help="이 개수 미만으로 공유하는 pair는 uncertainty 계산에서 제외(노이즈 방지)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = load_reconstruction(Path(args.reconstruction))
    print(f"[data] {reconstruction.summary()}")

    view_points = build_view_points(reconstruction)
    pairwise_overlap = compute_pairwise_overlaps(view_points)
    print(f"[overlap] {len(pairwise_overlap)} pairs computed (non-edge included as 0, protocol_utils 재사용)")

    # 카메라 파라미터 캐시: name -> {R, t, K, center}
    cams: dict[str, dict] = {}
    for image in reconstruction.images.values():
        cfw = image.cam_from_world()
        camera = reconstruction.cameras[image.camera_id]
        cams[image.name] = {
            "R": cfw.rotation.matrix(),
            "t": cfw.translation,
            "K": camera.calibration_matrix(),
            "center": image.projection_center(),
        }

    points3D = reconstruction.points3D

    rows = []
    for item in pairwise_overlap:
        cam_i, cam_j = cams[item.view_i], cams[item.view_j]
        baseline = float(np.linalg.norm(cam_i["center"] - cam_j["center"]))

        shared_ids = view_points[item.view_i] & view_points[item.view_j]
        depth_uncertainties = []
        reproj_errors = []
        for pid_str in shared_ids:
            p3d = points3D[int(pid_str)]
            du = two_view_depth_uncertainty(p3d.xyz, cam_i, cam_j)
            if np.isfinite(du):
                depth_uncertainties.append(du)
            reproj_errors.append(p3d.error)

        if len(shared_ids) < args.min_shared_points:
            mean_depth_uncertainty = float("nan")
            mean_reproj_error = float("nan")
        else:
            mean_depth_uncertainty = float(np.mean(depth_uncertainties)) if depth_uncertainties else float("nan")
            mean_reproj_error = float(np.mean(reproj_errors)) if reproj_errors else float("nan")

        rows.append(
            {
                "view_i": item.view_i,
                "view_j": item.view_j,
                "baseline": baseline,
                "overlap": item.overlap,
                "shared_points": len(shared_ids),
                "mean_depth_uncertainty": mean_depth_uncertainty,
                "mean_reproj_error": mean_reproj_error,
            }
        )

    csv_path = out_dir / "pairwise_geometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {csv_path} ({len(rows)} pairs)")

    make_plots(rows, out_dir)
    return 0


def make_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = np.array([r["baseline"] for r in rows])
    overlap = np.array([r["overlap"] for r in rows])
    uncertainty = np.array([r["mean_depth_uncertainty"] for r in rows])
    reproj = np.array([r["mean_reproj_error"] for r in rows])
    valid = np.isfinite(uncertainty) & (overlap > 0)  # overlap=0 pair는 uncertainty 정의 자체가 안 됨(공유점 없음)

    # 1. baseline vs overlap (C-1: 두 힘이 반대 방향)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(baseline, overlap, s=10, alpha=0.5, c="#3b6fa0")
    ax.set_xlabel("Baseline (camera center distance, world units)")
    ax.set_ylabel("Co-visibility overlap  $O_{ij} = 2|P_i \\cap P_j| / (|P_i|+|P_j|)$")
    ax.set_title("scan1 (42-view): baseline vs overlap — trade-off (§C-1)")
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_vs_overlap.png", dpi=150)
    plt.close(fig)

    # 2. overlap vs depth uncertainty (핵심: §5.3 overlap 정의 정당화)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(overlap[valid], uncertainty[valid], s=10, alpha=0.5, c="#c0533e")
    ax.set_yscale("log")
    ax.set_xlabel("Co-visibility overlap $O_{ij}$")
    ax.set_ylabel(r"Two-view depth uncertainty $\sqrt{\hat{d}^\top (\sigma^2 (J^\top J)^{-1}) \hat{d}}$ (log scale)")
    ax.set_title("scan1 (42-view): overlap vs Gauss-Newton depth uncertainty")
    fig.tight_layout()
    fig.savefig(out_dir / "overlap_vs_depth_uncertainty.png", dpi=150)
    plt.close(fig)

    # 3. baseline vs depth uncertainty
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(baseline[valid], uncertainty[valid], s=10, alpha=0.5, c="#4a9e6b")
    ax.set_yscale("log")
    ax.set_xlabel("Baseline (camera center distance, world units)")
    ax.set_ylabel("Two-view depth uncertainty (log scale)")
    ax.set_title("scan1 (42-view): baseline vs Gauss-Newton depth uncertainty")
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_vs_depth_uncertainty.png", dpi=150)
    plt.close(fig)

    # 4. sanity check: COLMAP 자체 reprojection error vs 우리가 계산한 depth uncertainty
    # (같은 방향으로 움직여야 우리 JᵀJ 계산이 COLMAP의 실제 최적화 결과와 일관됨을 확인)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(reproj[valid], uncertainty[valid], s=10, alpha=0.5, c="#7a5ea8")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("COLMAP mean reprojection error (px, log scale)")
    ax.set_ylabel("Our two-view depth uncertainty (log scale)")
    ax.set_title("sanity check: COLMAP reprojection error vs our JᵀJ estimate")
    fig.tight_layout()
    fig.savefig(out_dir / "sanity_reproj_vs_uncertainty.png", dpi=150)
    plt.close(fig)

    print(f"[plots] 4 figures written to {out_dir}")

    corr = np.corrcoef(overlap[valid], np.log(uncertainty[valid]))[0, 1]
    print(f"[stat] corr(overlap, log(depth_uncertainty)) = {corr:.3f} (음수면 예상대로: overlap↑ → uncertainty↓)")


if __name__ == "__main__":
    raise SystemExit(main())

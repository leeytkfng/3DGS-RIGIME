#!/usr/bin/env python3
"""C2 depth 개입 실험용 — 예측 depth map을 교란(perturb)한 뒤 3D point로 back-projection한다.

overall.md §5.9의 두 교란을 그대로 구현한다:
- (a) iid 오차: d' = d(1 + eps), eps ~ N(0, sigma^2)
- (b) global scale bias: d' = s * d

두 교란은 함께 줄 수도 있다: d' = s * d * (1 + eps). sigma=0, s=1.0이면 원본 depth 그대로다
(C2의 sigma=0/s=1.0 조건 = "교란 없음" 기준선).

좌표계는 `vanilla_3dgs_runner.py::build_camera_tensors`와 동일하다 — world-to-camera는
`X_cam = R @ X_world + t` (viewmat 관례), 따라서 역변환은 `X_world = R^T @ (X_cam - t)`.
"""
from __future__ import annotations

import numpy as np


def perturb_depth(depth: np.ndarray, sigma: float, scale_bias: float, seed: int) -> np.ndarray:
    """§5.9의 d' = s * d * (1 + eps)를 적용한다. sigma=0, scale_bias=1.0이면 항등."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=depth.shape) if sigma > 0.0 else np.zeros_like(depth)
    return scale_bias * depth * (1.0 + eps)


def back_project_depth_map(
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    rgb_image: np.ndarray,
    stride: int = 1,
    max_points: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """depth map(H,W, camera-space Z, meters) 하나를 world-space 3D point로 back-projection한다.

    Args:
        depth: (H, W) camera-space depth(Z). 이미 perturb_depth()로 교란 적용된 상태여야 한다.
        K: (3, 3) intrinsics, depth map과 같은 해상도 기준.
        R, t: world-to-camera 회전/이동 (viewmat 관례, X_cam = R @ X_world + t).
        rgb_image: (H, W, 3) float [0,1] 또는 uint8 [0,255] — 색상 샘플링용, depth와 같은 해상도.
        stride: 픽셀을 다 쓰지 않고 stride 간격으로 서브샘플 (밀도가 과할 때).
        max_points: 그래도 많으면 무작위로 이만큼만 남긴다(재현성 위해 seed 사용).

    Returns:
        (points_world [N,3] float32, colors_uint8 [N,3])
    """
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    ys, xs = ys.ravel(), xs.ravel()
    z = depth[ys, xs]

    valid = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[valid], xs[valid], z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs.astype(np.float64) - cx) / fx * z
    y_cam = (ys.astype(np.float64) - cy) / fy * z
    points_cam = np.stack([x_cam, y_cam, z], axis=1)  # (N, 3)

    # X_cam = R @ X_world + t  =>  X_world = R^T @ (X_cam - t)
    points_world = (points_cam - t[None, :]) @ R  # R^T applied via right-multiply of R (row vectors)

    if rgb_image.dtype != np.uint8:
        colors = np.clip(rgb_image, 0.0, 1.0)
        colors = (colors * 255.0).astype(np.uint8)
    else:
        colors = rgb_image
    colors = colors[ys, xs]

    if max_points is not None and points_world.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points_world.shape[0], size=max_points, replace=False)
        points_world, colors = points_world[keep], colors[keep]

    return points_world.astype(np.float32), colors


def back_project_multi_view(
    depths: list[np.ndarray],
    cameras: list[tuple[np.ndarray, np.ndarray, np.ndarray]],  # (K, R, t) per view
    rgb_images: list[np.ndarray],
    sigma: float,
    scale_bias: float,
    seed: int,
    stride: int = 4,
    max_points_per_view: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """여러 view의 depth map을 각각 교란·back-projection한 뒤 하나의 point cloud로 합친다."""
    all_points, all_colors = [], []
    for i, (depth, (K, R, t), rgb) in enumerate(zip(depths, cameras, rgb_images)):
        perturbed = perturb_depth(depth, sigma=sigma, scale_bias=scale_bias, seed=seed + i)
        pts, cols = back_project_depth_map(
            perturbed, K, R, t, rgb, stride=stride, max_points=max_points_per_view, seed=seed + i
        )
        all_points.append(pts)
        all_colors.append(cols)
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)

#!/usr/bin/env python3
"""gsplat 기반 vanilla 3DGS per-scene optimization 러너 (DTU pose-given track).

이 파일의 목적:
- 계획서(연구계획서_정리.md §4.1, overall.md §5.1/§5.7/§5.10)에서 정의한 프로토콜대로
  DTU 한 scan을 실제로 최적화해서, 스캐폴드(protocol_utils.py)가 다루는 체크포인트/로그
  스키마가 진짜 학습 곡선에서도 동작하는지 검증한다.
- H200의 큰 VRAM을 쓰기 위해 기본값을 downsample 없는 원본 해상도(1600x1200), 큰 초기
  Gaussian 수, 상한 없는 growth로 둔다.

지금 단계에서 의도적으로 비워둔 것 (스모크 테스트 범위를 넘는 부분):
- 초기화는 COLMAP SfM이 아니라 카메라 기하로 추정한 bounding sphere 안의 random point다.
  §5.2/§8 계획대로 실제 실험에서는 COLMAP triangulation으로 교체해야 한다.
- LPIPS는 사전학습 가중치 다운로드가 필요해 이번 스모크 테스트에서는 계산하지 않는다
  (로그에는 null로 남긴다).
- checkpoint_rule=budget_end_checkpoint / oracle_peak 분리는 protocol_utils의 기존 함수를
  그대로 재사용한다. 이 파일은 새 규칙을 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtu_dataset import estimate_scene_sphere, load_camera, load_scan  # noqa: E402
from protocol_utils import budget_checkpoint, oracle_checkpoint  # noqa: E402

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# SSIM (single-scale, gaussian window) — 외부 패키지 없이 학습 loss/평가에 함께 쓴다.
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.outer(g)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """img1, img2: [C, H, W] in [0, 1]. 채널별 gaussian-window SSIM의 평균을 반환한다."""

    channels = img1.shape[0]
    window = _gaussian_window(window_size, 1.5, img1.device).repeat(channels, 1, 1, 1)
    pad = window_size // 2

    img1b = img1.unsqueeze(0)
    img2b = img2.unsqueeze(0)
    mu1 = F.conv2d(img1b, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2b, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1b * img1b, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2b * img2b, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1b * img2b, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


# ---------------------------------------------------------------------------
# Camera / init helpers
# ---------------------------------------------------------------------------
def build_camera_tensors(view: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """view dict(dtu_dataset.load_scan 결과 원소 하나)에서 rasterization() 입력을 만든다."""

    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = torch.from_numpy(view["R"]).float()
    viewmat[:3, 3] = torch.from_numpy(view["t"]).float()
    k = torch.from_numpy(view["K"]).float().to(device)
    image = torch.from_numpy(view["image"]).float().to(device).permute(2, 0, 1)  # [3,H,W]
    return viewmat, k, image


def init_gaussians(
    center: np.ndarray,
    radius: float,
    num_points: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.nn.Parameter]:
    """카메라 기하로 추정한 bounding sphere 안에 random Gaussian을 초기화한다."""

    rng = np.random.default_rng(seed)
    # 구 내부 uniform sampling: 방향은 정규분포 후 정규화, 반경은 세제곱근으로 보정.
    directions = rng.normal(size=(num_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * (rng.uniform(size=(num_points, 1)) ** (1.0 / 3.0))
    points = center[None, :] + directions * radii

    # nearest-neighbor 간격 근사값으로 초기 scale을 잡는다 (원본 3DGS의 SfM 기반 초기화를
    # random 초기화로 대체할 때 흔히 쓰는 근사: 부피/점수 기반 평균 간격).
    avg_spacing = radius / (num_points ** (1.0 / 3.0))
    scales = np.full((num_points, 3), np.log(max(avg_spacing, 1e-4)), dtype=np.float32)

    means = torch.tensor(points, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    quats = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity quaternion (w, x, y, z)
    opacities = torch.full((num_points,), -2.1972246, dtype=torch.float32, device=device)  # inverse_sigmoid(0.1)

    gray = torch.full((num_points, 1, 3), (0.5 - 0.5) / SH_C0, dtype=torch.float32, device=device)
    sh_rest_dims = (3 + 1) ** 2 - 1  # sh_degree=3 최대 계수까지 미리 확보
    shN = torch.zeros((num_points, sh_rest_dims, 3), dtype=torch.float32, device=device)

    return {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales_t),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(gray),
        "shN": torch.nn.Parameter(shN),
    }


def build_optimizers(params: dict[str, torch.nn.Parameter], scene_scale: float) -> dict[str, torch.optim.Optimizer]:
    """원본 3DGS 기본 learning rate를 scene_scale로 스케일해 param별 Adam optimizer를 만든다."""

    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    return {key: torch.optim.Adam([params[key]], lr=lrs[key], eps=1e-15) for key in params}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_dir = Path(args.scan_dir)

    all_view_ids = list(range(1, 50))
    test_ids = all_view_ids[::7]  # 1,8,15,...,43 -> 고정 held-out test split
    train_pool = [v for v in all_view_ids if v not in test_ids]

    rng = np.random.default_rng(args.seed)
    train_ids = sorted(rng.choice(train_pool, size=min(args.view_count, len(train_pool)), replace=False).tolist())

    print(f"[data] train views ({len(train_ids)}): {train_ids}")
    print(f"[data] test views ({len(test_ids)}): {test_ids}")

    train_views = load_scan(scan_dir, train_ids)
    test_views = load_scan(scan_dir, test_ids)

    cameras_for_sphere = [load_camera(scan_dir / "cameras", v) for v in all_view_ids]
    center, radius = estimate_scene_sphere(cameras_for_sphere)
    print(f"[init] estimated scene center={center}, radius={radius:.3f}")

    params = init_gaussians(center, radius, args.num_init_points, device, args.seed)
    optimizers = build_optimizers(params, scene_scale=radius)

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=radius)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints" / args.scene / f"{args.view_count}view_seed{args.seed}"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_cams = [build_camera_tensors(v, device) for v in train_views]
    test_cams = [build_camera_tensors(v, device) for v in test_views]
    height, width = train_views[0]["height"], train_views[0]["width"]

    # --- CUDA warm-up: 프로토콜(runtime.cuda_warmup=true)에 따라 최초 컴파일 시간은 측정에서 뺀다.
    _warmup_forward(params, train_cams[0], rasterization, height, width, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    trajectory: list[dict[str, object]] = []
    order = list(range(len(train_cams)))
    step = 0
    elapsed = 0.0
    active_sh_degree = 0
    next_snapshot_targets = sorted(set(args.budget_snapshots))
    snapshot_idx = 0

    print(f"[train] starting optimization, budget={args.max_budget_seconds}s, init_points={args.num_init_points}")
    while elapsed < args.max_budget_seconds:
        if step % len(order) == 0:
            rng.shuffle(order)
        cam_idx = order[step % len(order)]
        viewmat, k, gt_image = train_cams[cam_idx]

        if step > 0 and step % 1000 == 0 and active_sh_degree < args.sh_degree:
            active_sh_degree += 1

        step_start = time.perf_counter()
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        render, _, info = rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat[None],
            Ks=k[None],
            width=width,
            height=height,
            sh_degree=active_sh_degree,
            packed=False,
        )
        info["means2d"].retain_grad()
        pred = render[0].permute(2, 0, 1).clamp(0.0, 1.0)

        l1 = torch.abs(pred - gt_image).mean()
        ssim_val = ssim(pred, gt_image)
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        for optimizer in optimizers.values():
            optimizer.step()

        torch.cuda.synchronize()
        elapsed += time.perf_counter() - step_start
        step += 1

        if snapshot_idx < len(next_snapshot_targets) and elapsed >= next_snapshot_targets[snapshot_idx]:
            budget_label = next_snapshot_targets[snapshot_idx]
            snapshot_idx += 1
            row = _evaluate_and_checkpoint(
                params=params,
                rasterization=rasterization,
                test_cams=test_cams,
                height=height,
                width=width,
                active_sh_degree=args.sh_degree,
                step=step,
                elapsed=elapsed,
                train_loss=float(loss.item()),
                checkpoints_dir=checkpoints_dir,
                args=args,
                budget_label=budget_label,
            )
            trajectory.append(row)
            print(
                f"[ckpt] budget={budget_label}s iter={step} elapsed={elapsed:.1f}s "
                f"gaussians={row['gaussian_count']} test_psnr={row['test_psnr']:.3f} "
                f"peak_vram_mb={row['peak_vram']:.0f}"
            )

        if step > args.max_iterations:
            print(f"[train] hit max_iterations={args.max_iterations} before budget exhausted, stopping.")
            break

    log_path = logs_dir / f"{args.scene}_{args.method}_{args.view_count}view_seed{args.seed}.json"
    log_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"[done] trajectory written to {log_path}")

    if trajectory:
        main_row = budget_checkpoint(trajectory, args.max_budget_seconds)
        oracle_row = oracle_checkpoint(trajectory, metric="test_psnr")
        print(f"[protocol] budget_end_checkpoint (main, leakage-safe): {main_row}")
        print(f"[protocol] oracle_checkpoint (diagnostic only): {oracle_row}")


def _warmup_forward(params, cam, rasterization, height, width, device):
    viewmat, k, _ = cam
    with torch.no_grad():
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat[None],
            Ks=k[None],
            width=width,
            height=height,
            sh_degree=0,
            packed=False,
        )


def _evaluate_and_checkpoint(
    *,
    params,
    rasterization,
    test_cams,
    height,
    width,
    active_sh_degree,
    step,
    elapsed,
    train_loss,
    checkpoints_dir,
    args,
    budget_label,
) -> dict[str, object]:
    with torch.no_grad():
        psnrs, ssims = [], []
        for viewmat, k, gt_image in test_cams:
            colors = torch.cat([params["sh0"], params["shN"]], dim=1)
            render, _, _ = rasterization(
                means=params["means"],
                quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=colors,
                viewmats=viewmat[None],
                Ks=k[None],
                width=width,
                height=height,
                sh_degree=active_sh_degree,
                packed=False,
            )
            pred = render[0].permute(2, 0, 1).clamp(0.0, 1.0)
            psnrs.append(psnr(pred, gt_image))
            ssims.append(float(ssim(pred, gt_image)))

    checkpoint_path = checkpoints_dir / f"budget_{budget_label}s_iter{step}.pt"
    torch.save({key: value.detach().cpu() for key, value in params.items()}, checkpoint_path)

    return {
        "experiment_id": args.experiment_id,
        "scene": args.scene,
        "seed": args.seed,
        "method": args.method,
        "iteration": step,
        "wall_clock": elapsed,
        "train_loss": train_loss,
        "validation_metric": None,
        "test_psnr": float(np.mean(psnrs)),
        "test_ssim": float(np.mean(ssims)),
        "test_lpips": None,  # pending: lpips 가중치 다운로드 필요 (audit log 참고)
        "gaussian_count": int(params["means"].shape[0]),
        "peak_vram": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "checkpoint_path": str(checkpoint_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vanilla 3DGS (gsplat) runner for a single DTU scan.")
    parser.add_argument("--scan-dir", required=True, help="e.g. /data/Re-feem/datasets/dtu/scan1")
    parser.add_argument("--scene", required=True, help="scene id used in logs, e.g. dtu_scan1")
    parser.add_argument("--method", default="Vanilla3DGS")
    parser.add_argument("--experiment-id", default="regime-map-20260806")
    parser.add_argument("--view-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-budget-seconds", type=float, default=300.0)
    parser.add_argument(
        "--budget-snapshots",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 60.0, 300.0],
        help="config.protocol.budgets_seconds와 맞춘다.",
    )
    parser.add_argument("--max-iterations", type=int, default=200_000, help="budget 안에서도 무한 루프 방지용 상한.")
    parser.add_argument("--num-init-points", type=int, default=100_000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.budget_snapshots = [b for b in sorted(args.budget_snapshots) if b <= args.max_budget_seconds]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

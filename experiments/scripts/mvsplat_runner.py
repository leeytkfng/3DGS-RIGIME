#!/usr/bin/env python3
"""MVSplat(feed-forward) 러너 — DTU pose-given track, protocol_utils 스키마 준수.

이 파일의 목적:
- `/data/Re-feem/code/mvsplat/run_on_custom_dtu.py`(1회성 검증 스크립트)를 실제 실험에서
  재사용 가능한 형태로 감싼다. vanilla_3dgs_runner.py와 같은 로깅 스키마·held-out split·
  seed 규칙을 공유해서 두 method의 결과가 바로 비교 가능하게 만든다.

실행 환경 주의:
- 반드시 `mvsplat` conda env로 실행해야 한다 (torch 2.1.2 + 커스텀 diff-gaussian-rasterization).
  `ps3` env(vanilla_3dgs_runner.py가 쓰는 env)와는 호환되지 않는다 — 섞으면 §7.3 사고가
  재발한다.
  예: /opt/conda/envs/mvsplat/bin/python3 experiments/scripts/mvsplat_runner.py ...

의도적으로 비워둔 것:
- MVSplat은 RE10K에서 2-view context로 학습됐다. 다른 view_count는 분포 밖 사용이며,
  §5.2 지원 view 수 표가 채워지기 전까지는 참고용으로만 취급해야 한다.
- checkpoint_rule=budget_end_checkpoint는 optimization용 개념이라 feed-forward에는
  "예산 안에 추론이 끝났는가"로 단순화된다 (§5.7). protocol_utils.budget_checkpoint()를
  단일 row에 그대로 적용해 동일한 leakage 방지 규칙을 재사용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _add_repo_paths(mvsplat_repo: Path) -> None:
    sys.path.insert(0, str(mvsplat_repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent))


DTU_SCALE_FACTOR = 1.0 / 200.0  # MVSplat convert_dtu.py와 동일 (§9.3 audit log)
IMAGE_SHAPE = (256, 256)
DTU_NEAR, DTU_FAR = 2.125, 4.525  # config/experiment/dtu.yaml 고정값


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def build_view(dtu_dataset_mod, scan_dir: Path, view_id: int) -> dict:
    cam = dtu_dataset_mod.load_camera(scan_dir / "cameras", view_id)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = cam.R
    w2c[:3, 3] = cam.t
    w2c[:3, 3] *= DTU_SCALE_FACTOR
    cam2world = np.linalg.inv(w2c)

    width, height = 1600, 1200
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = cam.K[0, 0] / width
    intrinsics[1, 1] = cam.K[1, 1] / height
    intrinsics[0, 2], intrinsics[1, 2] = 0.5, 0.5

    from PIL import Image

    image = Image.open(scan_dir / "images" / f"{view_id:03d}.png").convert("RGB")
    image = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

    return {"extrinsics": torch.from_numpy(cam2world), "intrinsics": torch.from_numpy(intrinsics), "image": image}


def build_batched_views(dtu_dataset_mod, crop_shim, scan_dir: Path, view_ids: list[int]) -> dict:
    views = [build_view(dtu_dataset_mod, scan_dir, v) for v in view_ids]
    batch = {
        "extrinsics": torch.stack([v["extrinsics"] for v in views])[None],
        "intrinsics": torch.stack([v["intrinsics"] for v in views])[None],
        "image": torch.stack([v["image"] for v in views])[None],
        "near": torch.full((1, len(view_ids)), DTU_NEAR),
        "far": torch.full((1, len(view_ids)), DTU_FAR),
    }
    return crop_shim(batch, IMAGE_SHAPE)


def run(args: argparse.Namespace) -> None:
    mvsplat_repo = Path(args.mvsplat_repo)
    _add_repo_paths(mvsplat_repo)

    import dtu_dataset  # 우리 프로젝트의 pose 파서 (experiments/scripts/dtu_dataset.py)
    from protocol_utils import budget_checkpoint, oracle_checkpoint  # noqa: F401 (oracle: FF는 사실상 main==oracle)
    from hydra import compose, initialize_config_dir
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_dir = Path(args.scan_dir)

    # vanilla_3dgs_runner.py와 완전히 동일한 held-out split/seed 규칙.
    # 같은 seed면 두 method가 정확히 같은 입력 view 조합을 보게 만들어 비교 가능성을 지킨다.
    all_view_ids = list(range(1, 50))
    test_ids = all_view_ids[::7]
    train_pool = [v for v in all_view_ids if v not in test_ids]
    rng = np.random.default_rng(args.seed)
    context_ids = sorted(rng.choice(train_pool, size=min(args.view_count, len(train_pool)), replace=False).tolist())

    print(f"[data] context views ({len(context_ids)}): {context_ids}")
    print(f"[data] test views ({len(test_ids)}): {test_ids}")
    if args.view_count != 2:
        print("[warn] MVSplat은 2-view context로 학습됐다. 이 run은 분포 밖(§5.2) 사용이다.")

    with initialize_config_dir(version_base=None, config_dir=str(mvsplat_repo / "config"), job_name="mvsplat_runner"):
        cfg = compose(config_name="main", overrides=["+experiment=dtu", "mode=test"])
    set_cfg(cfg)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    encoder.to(device).eval()
    decoder.to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder_sd = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")

    from lpips import LPIPS

    lpips_model = LPIPS(net="alex").to(device).eval()

    context = build_batched_views(dtu_dataset, apply_crop_shim_to_views, scan_dir, context_ids)
    target = build_batched_views(dtu_dataset, apply_crop_shim_to_views, scan_dir, test_ids)
    context = {k: v.to(device) for k, v in context.items()}
    target_on_device = {k: v.to(device) for k, v in target.items()}

    # --- CUDA warm-up: runtime.cuda_warmup=true 프로토콜에 따라 최초 컴파일 시간은 측정에서 뺀다.
    with torch.no_grad():
        _ = encoder(context, global_step=0, deterministic=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        gaussians = encoder(context, global_step=0, deterministic=True)
        output = decoder.forward(
            gaussians,
            target_on_device["extrinsics"],
            target_on_device["intrinsics"],
            target_on_device["near"],
            target_on_device["far"],
            IMAGE_SHAPE,
        )
    torch.cuda.synchronize()
    wall_clock = time.perf_counter() - t0

    pred = output.color[0].clamp(0, 1)
    gt = target_on_device["image"][0]
    psnrs = [psnr(pred[i], gt[i]) for i in range(pred.shape[0])]
    ssims, lpipss = [], []
    with torch.no_grad():
        for i in range(pred.shape[0]):
            ssims.append(_ssim(pred[i], gt[i]))
            lpipss.append(float(lpips_model((pred[i] * 2 - 1)[None], (gt[i] * 2 - 1)[None]).item()))

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints" / args.scene / f"{args.view_count}view_seed{args.seed}"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "gaussians.pt"
    torch.save(
        {
            "means": gaussians.means.detach().cpu(),
            "covariances": gaussians.covariances.detach().cpu(),
            "harmonics": gaussians.harmonics.detach().cpu(),
            "opacities": gaussians.opacities.detach().cpu(),
        },
        checkpoint_path,
    )

    row = {
        "experiment_id": args.experiment_id,
        "scene": args.scene,
        "seed": args.seed,
        "method": args.method,
        "iteration": 0,  # feed-forward: 학습 iteration 개념 없음, 단일 forward pass
        "wall_clock": wall_clock,
        "train_loss": None,
        "validation_metric": None,
        "test_psnr": float(np.mean(psnrs)),
        "test_ssim": float(np.mean(ssims)),
        "test_lpips": float(np.mean(lpipss)),
        "gaussian_count": int(gaussians.means.shape[1]),
        "peak_vram": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "checkpoint_path": str(checkpoint_path),
    }

    # §5.7: feed-forward 추론 시간이 budget을 넘으면 그 budget 칸은 No result.
    # optimization과 동일한 leakage-safe 함수(protocol_utils.budget_checkpoint)를 재사용.
    budget_results = {}
    for budget in args.budgets:
        selected = budget_checkpoint([row], budget)
        budget_results[budget] = "completed" if selected is not None else "no_result_budget_exceeded"

    log_path = logs_dir / f"{args.scene}_{args.method}_{args.view_count}view_seed{args.seed}.json"
    log_path.write_text(json.dumps({"row": row, "budget_status": budget_results}, indent=2), encoding="utf-8")

    print(f"[eval] wall_clock={wall_clock:.3f}s gaussians={row['gaussian_count']}")
    print(f"[eval] test_psnr={row['test_psnr']:.3f} test_ssim={row['test_ssim']:.3f} test_lpips={row['test_lpips']:.3f}")
    print(f"[protocol] budget status: {budget_results}")
    print(f"[done] log written to {log_path}")


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    import torch.nn.functional as F

    channels = img1.shape[0]
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = g.outer(g).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    pad = window_size // 2
    img1b, img2b = img1.unsqueeze(0), img2.unsqueeze(0)
    mu1 = F.conv2d(img1b, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2b, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1b * img1b, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2b * img2b, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1b * img2b, window, padding=pad, groups=channels) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVSplat runner for a single DTU scan (mvsplat conda env only).")
    parser.add_argument("--scan-dir", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", default="MVSplat")
    parser.add_argument("--experiment-id", default="regime-map-20260806")
    parser.add_argument("--view-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=float, nargs="+", default=[1.0, 10.0, 60.0, 300.0])
    parser.add_argument("--mvsplat-repo", default="/data/Re-feem/code/mvsplat")
    parser.add_argument("--checkpoint", default="/data/Re-feem/code/mvsplat/checkpoints/re10k.ckpt")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

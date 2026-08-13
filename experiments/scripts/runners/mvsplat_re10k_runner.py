#!/usr/bin/env python3
"""MVSplat(feed-forward) 러너 — RE10K main subset, protocol_utils 스키마 준수.

`mvsplat_runner.py`(DTU)의 RE10K 버전. 차이점:
- view 후보를 seed 기반으로 새로 뽑지 않고 `generate_re10k_main_subset.py`가 만든
  `re10k_main_subset.json`의 context/target을 그대로 쓴다(재현성 유지 — overlap 계산도
  같은 candidate로 이미 했으므로 여기서 다시 뽑으면 서로 다른 view를 보게 된다).
- DTU_SCALE_FACTOR 같은 좌표계 보정이 없다. RE10K raw pose가 이미 MVSplat 학습에 쓰인
  그 좌표계다(`mvsplat_re10k_probe.py`가 이미 이렇게 검증: mean PSNR 25.6dB).
- C1-b(V3)용으로 `gaussians.pt`/`render_reference.pt`를 저장한다 — DTU 경로와 완전히 같은
  포맷이라 `check_renderer_equivalence.py`/`vanilla_3dgs_runner.py --warm-start-checkpoint`를
  그대로 재사용할 수 있다.

실행 환경: mvsplat conda env 필요.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RE10K_ROOT = Path("/data/Re-feem/datasets/re10k/test")
IMAGE_SHAPE = (256, 256)
NEAR, FAR = 1.0, 100.0  # mvsplat_re10k_probe.py(2026-08-10 검증)와 동일


def _add_repo_paths(mvsplat_repo: Path) -> None:
    sys.path.insert(0, str(mvsplat_repo))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def run(args: argparse.Namespace) -> None:
    mvsplat_repo = Path(args.mvsplat_repo)
    _add_repo_paths(mvsplat_repo)

    from protocol_utils import budget_checkpoint  # noqa: F401
    from re10k_dataset import get_scene_item, load_views
    from hydra import compose, initialize_config_dir
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subset = json.loads(Path(args.subset_index).read_text())
    entry = subset[args.scene_key]
    candidate = entry["view_candidates"][str(args.view_count)]
    if candidate.get("context") is None:
        raise SystemExit(f"{args.scene_key} view_count={args.view_count}: candidate 없음(too short)")
    context_ids = candidate["context"]
    target_ids = candidate["target"]
    print(f"[data] scene={args.scene_key} context={context_ids} target={target_ids}")
    if args.view_count != 2:
        print("[warn] MVSplat은 2-view context로 학습됐다. 이 run은 분포 밖(§5.2) 사용이다.")

    item = get_scene_item(RE10K_ROOT / entry["chunk_file"], args.scene_key)

    def build_batch(frame_ids: list[int]) -> dict:
        views = load_views(item, frame_ids)  # crop_shim이 리사이즈까지 하므로 여기선 원본 그대로
        images = torch.stack([torch.from_numpy(v["image"]).permute(2, 0, 1).float() for v in views])
        # RE10K raw pose(정규화 K, w2c)를 MVSplat 자체 convert_poses() convention으로 다시 조립한다
        # (load_views가 이미 픽셀 K로 바꿔놨으므로, crop_shim에 넣을 정규화 K를 원본 pose에서 새로 뽑는다).
        poses = torch.stack([torch.from_numpy(item["cameras"][fid].numpy()) for fid in frame_ids])
        fx, fy, cx, cy = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(len(frame_ids), 1, 1)
        intrinsics[:, 0, 0], intrinsics[:, 1, 1] = fx, fy
        intrinsics[:, 0, 2], intrinsics[:, 1, 2] = cx, cy
        w2c = torch.eye(4, dtype=torch.float32).repeat(len(frame_ids), 1, 1)
        w2c[:, :3] = poses[:, 6:].reshape(len(frame_ids), 3, 4)
        extrinsics = w2c.inverse()
        batch = {
            "extrinsics": extrinsics[None],
            "intrinsics": intrinsics[None],
            "image": images[None],
            "near": torch.full((1, len(frame_ids)), NEAR),
            "far": torch.full((1, len(frame_ids)), FAR),
        }
        return apply_crop_shim_to_views(batch, IMAGE_SHAPE)

    context = build_batch(context_ids)
    target = build_batch(target_ids)
    context = {k: v.to(device) for k, v in context.items()}
    target_on_device = {k: v.to(device) for k, v in target.items()}

    with initialize_config_dir(version_base=None, config_dir=str(mvsplat_repo / "config"), job_name="mvsplat_re10k_runner"):
        cfg = compose(config_name="main", overrides=["+experiment=re10k", "mode=test"])
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

    with torch.no_grad():
        _ = encoder(context, global_step=0, deterministic=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        gaussians = encoder(context, global_step=0, deterministic=True)
        output = decoder.forward(
            gaussians, target_on_device["extrinsics"], target_on_device["intrinsics"],
            target_on_device["near"], target_on_device["far"], IMAGE_SHAPE,
        )
    torch.cuda.synchronize()
    wall_clock = time.perf_counter() - t0

    pred = output.color[0].clamp(0, 1)
    gt = target_on_device["image"][0]
    psnrs = [psnr(pred[i], gt[i]) for i in range(pred.shape[0])]
    lpipss = [float(lpips_model((pred[i] * 2 - 1)[None], (gt[i] * 2 - 1)[None]).item()) for i in range(pred.shape[0])]

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints" / args.scene_key / f"{args.view_count}view"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "means": gaussians.means.detach().cpu(),
            "covariances": gaussians.covariances.detach().cpu(),
            "harmonics": gaussians.harmonics.detach().cpu(),
            "opacities": gaussians.opacities.detach().cpu(),
        },
        checkpoints_dir / "gaussians.pt",
    )
    torch.save(
        {
            "pred": pred.detach().cpu(),
            "view_ids": target_ids,
            "extrinsics": target_on_device["extrinsics"].detach().cpu(),
            "intrinsics": target_on_device["intrinsics"].detach().cpu(),
            "near": target_on_device["near"].detach().cpu(),
            "far": target_on_device["far"].detach().cpu(),
            "image_shape": IMAGE_SHAPE,
            "dtu_scale_factor": 1.0,  # RE10K는 좌표계 보정 없음(§docstring)
        },
        checkpoints_dir / "render_reference.pt",
    )

    row = {
        "experiment_id": args.experiment_id,
        "scene": args.scene_key,
        "seed": 0,
        "method": "MVSplat",
        "iteration": 0,
        "wall_clock": wall_clock,
        "test_psnr": float(np.mean(psnrs)),
        "test_lpips": float(np.mean(lpipss)),
        "gaussian_count": int(gaussians.means.shape[1]),
        "peak_vram": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    }
    log_path = logs_dir / f"{args.scene_key}_MVSplat_{args.view_count}view.json"
    # 원자적 쓰기 — vanilla_3dgs_runner.py/fsgs_runner.py와 동일 이유.
    tmp_path = log_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    tmp_path.replace(log_path)

    print(f"[eval] wall_clock={wall_clock:.3f}s gaussians={row['gaussian_count']}")
    print(f"[eval] test_psnr={row['test_psnr']:.3f} test_lpips={row['test_lpips']:.3f}")
    print(f"[done] gaussians.pt/render_reference.pt written to {checkpoints_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVSplat runner for RE10K main subset (mvsplat conda env only).")
    parser.add_argument("--subset-index", default="experiments/outputs/re10k_main_subset/re10k_main_subset.json")
    parser.add_argument("--scene-key", required=True)
    parser.add_argument("--view-count", type=int, default=2)
    parser.add_argument("--experiment-id", default="regime-map-20260806")
    parser.add_argument("--mvsplat-repo", default="/data/Re-feem/code/mvsplat")
    parser.add_argument("--checkpoint", default="/data/Re-feem/code/mvsplat/checkpoints/re10k.ckpt")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "re10k_main_subset"))
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

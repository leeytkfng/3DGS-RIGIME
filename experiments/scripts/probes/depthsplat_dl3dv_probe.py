#!/usr/bin/env python3
"""DepthSplat을 DL3DV 공식 test subset(.torch chunk)에서 in-domain으로 검증하는 스크립트.

mvsplat_re10k_probe.py와 같은 목적: DepthSplat이 DL3DV in-domain 체크포인트로 정상 범위
PSNR을 내는지 확인해, 나중에 우리 자체 DL3DV probe(raw 480P zip)에 붙일 때 기준점을 만든다.

데이터: haofeixu/depthsplat HF repo의 dl3dv_960p_test_subset.zip (공식 2-scene quick test set).
체크포인트: depthsplat-gs-base-dl3dv-256x448-randview2-6 (DL3DV in-domain, 2-6 view).
카메라 변환: dataset_dl3dv.py convert_poses()와 동일(포즈 18값, RE10K와 같은 포맷).

실행 환경: depthsplat conda env 필요 (별도 격리, ps3/mvsplat env와 안 섞음).
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import torch


def convert_poses(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = poses.shape[0]
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
    fx, fy, cx, cy = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
    w2c[:, :3] = poses[:, 6:].reshape(b, 3, 4)
    return w2c.inverse(), intrinsics


def decode_images(raw_images: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as tf

    to_tensor = tf.ToTensor()
    images = []
    for i in indices:
        img = Image.open(BytesIO(raw_images[i].numpy().tobytes()))
        images.append(to_tensor(img))
    return torch.stack(images)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default="/data/Re-feem/raw_downloads/dl3dv_960p_test_subset/dl3dv_960p_test_subset/test/000000.torch")
    parser.add_argument("--scene-index", type=int, default=0)
    # DL3DV scene은 수백 프레임짜리 긴 walkthrough라 RE10K와 달리 0.2/0.8처럼 넓게 잡으면
    # context 두 장의 baseline이 너무 커져서(거의 scene 전체 스팬) 저품질로 나온다(12dB대).
    # 좁은 구간(약 30프레임 이내)으로 좁혀야 학습 분포에 맞는 정상적인 baseline이 된다.
    parser.add_argument("--context-frac", type=float, nargs=2, default=[0.46, 0.54])
    parser.add_argument("--num-targets", type=int, default=3)
    parser.add_argument("--depthsplat-repo", default="/data/Re-feem/code/depthsplat")
    parser.add_argument(
        "--checkpoint",
        default="/data/Re-feem/code/depthsplat/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
    )
    parser.add_argument("--image-shape", type=int, nargs=2, default=[256, 448])
    args = parser.parse_args()

    import sys

    sys.path.insert(0, args.depthsplat_repo)
    from hydra import compose, initialize_config_dir
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views
    from src.dataset.shims.patch_shim import apply_patch_shim_to_views
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunks = torch.load(args.chunk, map_location="cpu", weights_only=False)
    example = chunks[args.scene_index]
    print(f"[data] scene key={example['key']}, num frames={len(example['images'])}")

    extrinsics, intrinsics = convert_poses(example["cameras"])
    n = extrinsics.shape[0]
    ctx_idx = [int(args.context_frac[0] * n), int(args.context_frac[1] * n)]
    # target은 context 두 장 "사이"에서 뽑는다 (interpolation) — 바깥에서 뽑으면
    # extrapolation이 되어 baseline이 큰 것과 같은 이유로 저품질로 나온다.
    remaining = [i for i in range(ctx_idx[0] + 1, ctx_idx[1]) if i not in ctx_idx]
    rng = np.random.default_rng(0)
    target_idx = sorted(rng.choice(remaining, size=min(args.num_targets, len(remaining)), replace=False).tolist())
    print(f"[data] context idx={ctx_idx}, target idx={target_idx}")

    # README §Evaluation/DL3DV "Table 7, 6 input views"가 이 체크포인트(base, dl3dv-only)에
    # 실제로 사용하는 override 조합. 기본 dl3dv.yaml 설정은 "small" 계열이라 채널 수가 안 맞는다.
    with initialize_config_dir(version_base=None, config_dir=str(Path(args.depthsplat_repo) / "config"), job_name="dl3dv_probe"):
        cfg = compose(
            config_name="main",
            overrides=[
                "+experiment=dl3dv",
                "mode=test",
                "model.encoder.num_scales=2",
                "model.encoder.upsample_factor=4",
                "model.encoder.lowest_feature_resolution=8",
                "model.encoder.monodepth_vit_type=vitb",
                "model.encoder.return_depth=false",
            ],
        )
    set_cfg(cfg)
    patch_size = cfg.model.encoder.shim_patch_size

    def build_views(indices):
        imgs = decode_images(example["images"], indices)
        return {
            "extrinsics": extrinsics[indices][None],
            "intrinsics": intrinsics[indices][None],
            "image": imgs[None],
            "near": torch.full((1, len(indices)), 0.5),
            "far": torch.full((1, len(indices)), 200.0),
        }

    shape = tuple(args.image_shape)
    context = apply_crop_shim_to_views(build_views(ctx_idx), shape)
    target = apply_crop_shim_to_views(build_views(target_idx), shape)
    context = apply_patch_shim_to_views(context, patch_size)
    context = {k: v.to(device) for k, v in context.items()}
    target = {k: v.to(device) for k, v in target.items()}

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    encoder.to(device).eval()
    decoder.to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder_sd = {k[len("encoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    with torch.no_grad():
        gaussians = encoder(context, global_step=0, deterministic=True)
        output = decoder.forward(gaussians, target["extrinsics"], target["intrinsics"], target["near"], target["far"], shape)

    pred = output.color[0].clamp(0, 1)
    gt = target["image"][0]
    psnrs = [psnr(pred[i], gt[i]) for i in range(pred.shape[0])]
    print(f"[eval] per-target PSNR: {[round(p, 3) for p in psnrs]}")
    print(f"[eval] mean PSNR: {sum(psnrs) / len(psnrs):.3f}")

    out_dir = Path("/tmp/claude-0/-root-task-5/43ccff02-44e8-40ba-a45e-9ff6091fb164/scratchpad/depthsplat_dl3dv_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image as PILImage

    for i, tid in enumerate(target_idx):
        PILImage.fromarray((pred[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out_dir / f"pred_{tid}.png")
        PILImage.fromarray((gt[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out_dir / f"gt_{tid}.png")
    print(f"[done] renders saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

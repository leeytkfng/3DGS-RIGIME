#!/usr/bin/env python3
"""MVSplat을 RE10K probe(.torch chunk)에서 in-domain으로 검증하는 스크립트.

목적:
- 어제/오늘 DTU에서 확인한 MVSplat zero-shot(RE10K checkpoint -> DTU) PSNR(4.8~11.8dB)은
  cross-dataset OOD 상황이라 낮게 나오는 게 정상이다 (critical_path_2026-08-10.md §위험2).
- RE10K는 MVSplat의 실제 학습 도메인이므로, 같은 체크포인트를 RE10K probe에서 돌려보면
  "파이프라인이 문제냐, OOD가 문제냐"를 가르는 두 번째 참조점이 된다.
- config/experiment/re10k.yaml 기준으로 재현: image_shape=256x256, near=1.0, far=100.0,
  baseline 정규화 없음(make_baseline_1=false). 카메라 변환은 dataset_re10k.py의
  convert_poses()를 그대로 재현한다(포즈 18값 = [fx,fy,cx,cy,0,0] + w2c 3x4 flatten).

실행 환경: mvsplat conda env 필요 (mvsplat_runner.py와 동일한 이유).
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import torch


def load_scene(chunk_path: Path, scene_key: str | None = None) -> dict:
    chunks = torch.load(chunk_path, map_location="cpu", weights_only=False)
    example = chunks[0] if scene_key is None else next(c for c in chunks if c["key"] == scene_key)
    return example


def convert_poses(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """dataset_re10k.py DatasetRE10k.convert_poses()와 동일한 로직."""
    b = poses.shape[0]
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
    fx, fy, cx, cy = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
    w2c[:, :3] = poses[:, 6:].reshape(b, 3, 4)
    return w2c.inverse(), intrinsics  # extrinsics(cam2world), intrinsics


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
    parser.add_argument("--chunk", default="/data/Re-feem/datasets/re10k/test/000000.torch")
    parser.add_argument("--scene-index", type=int, default=0, help="chunk 안에서 몇 번째 scene을 쓸지")
    parser.add_argument("--context-frac", type=float, nargs=2, default=[0.25, 0.75], help="context view를 고를 상대 위치(0~1)")
    parser.add_argument("--num-targets", type=int, default=3)
    parser.add_argument("--mvsplat-repo", default="/data/Re-feem/code/mvsplat")
    parser.add_argument("--checkpoint", default="/data/Re-feem/code/mvsplat/checkpoints/re10k.ckpt")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, args.mvsplat_repo)
    from hydra import compose, initialize_config_dir
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views
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
    remaining = [i for i in range(n) if i not in ctx_idx]
    rng = np.random.default_rng(0)
    target_idx = sorted(rng.choice(remaining, size=min(args.num_targets, len(remaining)), replace=False).tolist())
    print(f"[data] context idx={ctx_idx}, target idx={target_idx}")

    def build_views(indices):
        imgs = decode_images(example["images"], indices)
        return {
            "extrinsics": extrinsics[indices][None],
            "intrinsics": intrinsics[indices][None],
            "image": imgs[None],
            "near": torch.full((1, len(indices)), 1.0),
            "far": torch.full((1, len(indices)), 100.0),
        }

    context = apply_crop_shim_to_views(build_views(ctx_idx), (256, 256))
    target = apply_crop_shim_to_views(build_views(target_idx), (256, 256))
    context = {k: v.to(device) for k, v in context.items()}
    target = {k: v.to(device) for k, v in target.items()}

    with initialize_config_dir(version_base=None, config_dir=str(Path(args.mvsplat_repo) / "config"), job_name="re10k_probe"):
        cfg = compose(config_name="main", overrides=["+experiment=re10k", "mode=test"])
    set_cfg(cfg)

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
        output = decoder.forward(gaussians, target["extrinsics"], target["intrinsics"], target["near"], target["far"], (256, 256))

    pred = output.color[0].clamp(0, 1)
    gt = target["image"][0]
    psnrs = [psnr(pred[i], gt[i]) for i in range(pred.shape[0])]
    print(f"[eval] per-target PSNR: {[round(p, 3) for p in psnrs]}")
    print(f"[eval] mean PSNR: {sum(psnrs) / len(psnrs):.3f}  (참고: DTU zero-shot 4.8~11.8dB)")

    out_dir = Path("/tmp/claude-0/-root-task-5/43ccff02-44e8-40ba-a45e-9ff6091fb164/scratchpad/mvsplat_re10k_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image as PILImage

    for i, tid in enumerate(target_idx):
        PILImage.fromarray((pred[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out_dir / f"pred_{tid}.png")
        PILImage.fromarray((gt[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out_dir / f"gt_{tid}.png")
    print(f"[done] renders saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""DepthSplat(feed-forward) 러너 — DL3DV pilot, C1-b용 gaussians.pt/render_reference.pt 생성.

`mvsplat_re10k_runner.py`의 DepthSplat/DL3DV 버전. DepthSplat을 C1-b에 연결하는 이유:
RE10K에서 MVSplat으로 4/8/12-view C1-b를 돌려보니 view 수가 늘수록 refinement 효과가
커지는 뚜렷한 패턴이 나왔는데(§V3 20-scene 스케일업, 2026-08-12), MVSplat이 2-view
전용 학습이라 4/8/12-view가 전부 분포 밖(§5.2)이라는 교란요인이 섞여 있었다. DepthSplat은
이 체크포인트(`depthsplat-gs-base-dl3dv-256x448-randview2-6`) 기준 **2~6-view가 분포 안**
이므로, 같은 실험을 DepthSplat으로 반복하면 "view 수 자체의 효과"와 "모델이 분포 밖이라
망가지는 효과"를 분리할 수 있다.

RE10K가 아니라 DL3DV를 쓰는 이유: 로컬에 받아둔 DepthSplat 체크포인트가 DL3DV 전용이고
(RE10K+DL3DV 혼합 체크포인트는 아직 미다운로드), DL3DV는 오늘 이미 overlap 파이프라인과
view candidate(`experiments/outputs/dl3dv_overlap/all_scenes_summary.json`)를 만들어뒀다.

view 후보는 새로 안 뽑고 `generate_dl3dv_view_overlap.py`가 저장한 `context_indices`/
`target_indices`를 그대로 쓴다(overlap 계산에 쓴 것과 같은 view라야 비교가 일관됨).

실행 환경: depthsplat conda env 필요.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DL3DV_ROOT = Path("/data/Re-feem/datasets/dl3dv")
IMAGE_SHAPE = (256, 448)
NEAR, FAR = 0.5, 200.0
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _add_repo_paths(depthsplat_repo: Path) -> None:
    sys.path.insert(0, str(depthsplat_repo))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def build_batch(scene_dir: Path, meta: dict, frame_indices: list[int], undo_applied_transform: bool = True) -> dict:
    """RE10K 러너의 build_batch()와 같은 역할 — 정규화 intrinsics + opencv c2w를 만든다.
    crop_shim이 리사이즈를 알아서 하므로 여기서는 원본 이미지를 그대로 쓴다.
    """

    store_w, store_h = meta["w"], meta["h"]
    saved_fx, saved_fy = meta["fl_x"] / store_w, meta["fl_y"] / store_h
    saved_cx, saved_cy = meta["cx"] / store_w, meta["cy"] / store_h

    applied = np.eye(4)
    if undo_applied_transform and "applied_transform" in meta:
        applied[:3, :4] = np.array(meta["applied_transform"])

    image_dir = scene_dir / "images_8"
    images, extrinsics = [], []
    for idx in frame_indices:
        frame = meta["frames"][idx]
        pil_image = Image.open(image_dir / Path(frame["file_path"]).name).convert("RGB")
        images.append(torch.from_numpy(np.asarray(pil_image, dtype=np.float32) / 255.0).permute(2, 0, 1))
        c2w = applied @ np.array(frame["transform_matrix"]) @ BLENDER_TO_OPENCV
        extrinsics.append(torch.from_numpy(c2w).float())  # camera-to-world, decoder/crop_shim convention

    n = len(frame_indices)
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(n, 1, 1)
    intrinsics[:, 0, 0], intrinsics[:, 1, 1] = saved_fx, saved_fy
    intrinsics[:, 0, 2], intrinsics[:, 1, 2] = saved_cx, saved_cy

    batch = {
        "extrinsics": torch.stack(extrinsics)[None],
        "intrinsics": intrinsics[None],
        "image": torch.stack(images)[None],
        "near": torch.full((1, n), NEAR),
        "far": torch.full((1, n), FAR),
    }
    return batch


def run(args: argparse.Namespace) -> None:
    depthsplat_repo = Path(args.depthsplat_repo)
    _add_repo_paths(depthsplat_repo)

    from dl3dv_dataset import load_metadata
    from hydra import compose, initialize_config_dir
    from src.dataset.shims.crop_shim import apply_crop_shim_to_views
    from src.dataset.shims.patch_shim import apply_patch_shim_to_views
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.overlap_level:
        overlap_data = json.loads(Path(args.overlap_candidates_index).read_text())
        ov_entry = overlap_data[args.scene_key][str(args.view_count)][args.overlap_level]
        context_ids, target_ids = ov_entry["context"], ov_entry["target"]
        print(f"[data] scene={args.scene_key} view_count={args.view_count} overlap_level={args.overlap_level} "
              f"context={context_ids} target={target_ids}")
    else:
        overlap_summary = json.loads(Path(args.overlap_summary).read_text())
        row = next(
            r for r in overlap_summary if r["scene"] == args.scene_key and r["view_count"] == args.view_count
        )
        context_ids, target_ids = row["context_indices"], row["target_indices"]
        print(f"[data] scene={args.scene_key} view_count={args.view_count} context={context_ids} target={target_ids}")

    scene_dir = DL3DV_ROOT / args.scene_key
    meta = load_metadata(scene_dir)

    with initialize_config_dir(version_base=None, config_dir=str(depthsplat_repo / "config"), job_name="depthsplat_dl3dv_runner"):
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

    context = apply_crop_shim_to_views(build_batch(scene_dir, meta, context_ids), IMAGE_SHAPE)
    target = apply_crop_shim_to_views(build_batch(scene_dir, meta, target_ids), IMAGE_SHAPE)
    context = apply_patch_shim_to_views(context, patch_size)
    context = {k: v.to(device) for k, v in context.items()}
    target_on_device = {k: v.to(device) for k, v in target.items()}

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
    overlap_suffix = f"_{args.overlap_level}" if args.overlap_level else ""
    checkpoints_dir = output_dir / "checkpoints" / args.scene_key / f"{args.view_count}view{overlap_suffix}"
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
            "dtu_scale_factor": 1.0,  # DL3DV도 좌표계 보정 없음 (RE10K와 동일한 이유)
        },
        checkpoints_dir / "render_reference.pt",
    )

    row_out = {
        "experiment_id": args.experiment_id,
        "scene": args.scene_key,
        "method": "DepthSplat",
        "wall_clock": wall_clock,
        "test_psnr": float(np.mean(psnrs)),
        "test_lpips": float(np.mean(lpipss)),
        "gaussian_count": int(gaussians.means.shape[1]),
        "peak_vram": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    }
    log_path = logs_dir / f"{args.scene_key}_DepthSplat_{args.view_count}view{overlap_suffix}.json"
    # 원자적 쓰기 — vanilla_3dgs_runner.py/fsgs_runner.py/mvsplat_re10k_runner.py와 동일 이유.
    tmp_path = log_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(row_out, indent=2), encoding="utf-8")
    tmp_path.replace(log_path)

    print(f"[eval] wall_clock={wall_clock:.3f}s gaussians={row_out['gaussian_count']}")
    print(f"[eval] test_psnr={row_out['test_psnr']:.3f} test_lpips={row_out['test_lpips']:.3f}")
    print(f"[done] gaussians.pt/render_reference.pt written to {checkpoints_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DepthSplat runner for DL3DV pilot scenes (depthsplat conda env only).")
    parser.add_argument("--overlap-summary", default="experiments/outputs/dl3dv_overlap/all_scenes_summary.json",
                         help="--overlap-level 미지정 시 사용.")
    parser.add_argument("--overlap-level", choices=["high", "low"], default=None,
                         help="지정하면 --overlap-candidates-index에서 co-visibility selector 후보를 쓴다.")
    parser.add_argument("--overlap-candidates-index",
                         default="experiments/outputs/dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json")
    parser.add_argument("--scene-key", required=True)
    parser.add_argument("--view-count", type=int, default=2)
    parser.add_argument("--experiment-id", default="regime-map-20260806")
    parser.add_argument("--depthsplat-repo", default="/data/Re-feem/code/depthsplat")
    parser.add_argument(
        "--checkpoint",
        default="/data/Re-feem/code/depthsplat/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "dl3dv_c1b"))
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

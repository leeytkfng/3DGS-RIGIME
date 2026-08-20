#!/usr/bin/env python3
"""ReSplat(feed-forward, recurrent refinement) 러너 — DL3DV, 탐색적 확장 비교 전용.

`depthsplat_dl3dv_runner.py`의 ReSplat 버전. `cvg/resplat`(Xu et al., ECCV 2026) —
DepthSplat과 같은 연구실/1저자(Haofei Xu) 계보라 RE10K/DL3DV chunk 포맷·카메라 규약이
그대로 호환된다(2026-08-16 확인). 우리 프로토콜에서의 위치는 **메인 regime map이 아니라
별도 절의 탐색적 확장 비교**다 — 사전 등록 비교군이 아니고(single-pass FF vs per-scene
optimization의 두 축이 본 비교), recurrent refinement라는 다른 패러다임이기 때문
(`main.tex` Limitations 절 참고).

view 주입: ReSplat도 pixelSplat/MVSplat 계보라 `dataset/view_sampler=evaluation` +
index_path 방식을 쓰지만, 우리는 그 로더를 아예 안 타고(MVSplat/DepthSplat 러너와 동일
패턴) `overlap_summary`/`overlap_candidates_index`에서 직접 context/target을 읽어
batch를 우리가 만든다 — monkey-patch 불필요(2026-08-16 확인).

iteration 스냅샷: `encoder.forward_update()`가 refine iteration별 Gaussians를 리스트로
반환한다(`refine_output["gaussian"][i]`) — 한 번의 forward pass로 여러 refine 단계를
동시에 얻을 수 있다. 여기서는 체크포인트 학습에 쓰인 `num_refine`(기본 4, 학습 스크립트의
`train_max_refine=4`와 일치)의 최종 단계만 기록한다 — 다른 FF 러너(MVSplat/DepthSplat)와
동일하게 "결정론적 단일 추론" 한 지점만 budget 축 전체에 반복 기록하는 방식을 따른다.

체크포인트/해상도: DL3DV는 `pretrained/resplat-base-dl3dv-256x448-view{8,16}-*.pth`
(view8/16 base, 256x448 — 기존 DepthSplat 파이프라인과 동일 해상도라 별도 리사이즈 로직
불필요). view_count가 8/16보다 크게 벗어나면(예: 2,4-view) 분포 밖(OOD)이다 — 기존
MVSplat/DepthSplat과 동일하게 실행은 하되 §5.2 기준 OOD로 표기해 분석한다.

실행 환경: resplat conda env 필요(`/opt/conda/envs/resplat`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DL3DV_ROOT = Path("/data/Re-feem/datasets/dl3dv")
NEAR, FAR = 0.05, 20.0  # 실제 원인은 near/far가 아니라 --overlap-summary 기본값이 v1(버그
# 있는 view 선택)을 가리키고 있던 것이었다(아래 --overlap-summary 기본값 주석 참고) — v2로
# 고치니 PSNR 26dB대로 정상화됨. near/far 자체는 0.05~20 범위면 충분히 잘 작동한다(우리
# raw pose 스케일의 실제 camera baseline이 O(1)이라 이 범위가 알맞다).
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
IMAGE_SHAPE = (256, 448)


def _add_repo_paths(resplat_repo: Path) -> None:
    sys.path.insert(0, str(resplat_repo))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def load_frames(scene_dir: Path, meta: dict, frame_indices: list[int], undo_applied_transform: bool = True):
    """`dl3dv_dataset.load_views()`와 동일 로직(정규화 intrinsics + opencv c2w) + IMAGE_SHAPE로
    resize_and_crop — ReSplat encoder는 고정 (256,448) 해상도를 기대한다(2026-08-16 스모크
    테스트에서 native 480x270을 그대로 넣었다가 multi-scale feature concat에서 텐서 크기
    불일치로 실패 확인). `dtu_dataset.resize_and_crop()`(MVSplat crop_shim과 동일 convention:
    짧은 변 기준 리사이즈 -> center crop -> cx/cy 정중앙 강제)을 그대로 재사용한다.

    이미지는 [3,H,W] float01 텐서 리스트, 카메라는 c2w(4x4) 텐서 리스트, K는 **정규화**
    (row0/w, row1/h — ReSplat README의 카메라 규약) 3x3 텐서 리스트로 반환한다.
    """

    from dtu_dataset import resize_and_crop

    store_w, store_h = meta["w"], meta["h"]
    saved_fx, saved_fy = meta["fl_x"] / store_w, meta["fl_y"] / store_h
    saved_cx, saved_cy = meta["cx"] / store_w, meta["cy"] / store_h

    applied = np.eye(4)
    if undo_applied_transform and "applied_transform" in meta:
        applied[:3, :4] = np.array(meta["applied_transform"])

    image_dir = scene_dir / "images_8"
    images, c2ws, Ks = [], [], []
    h_out, w_out = IMAGE_SHAPE
    for idx in frame_indices:
        frame = meta["frames"][idx]
        pil_image = Image.open(image_dir / Path(frame["file_path"]).name).convert("RGB")
        image = np.asarray(pil_image, dtype=np.float32) / 255.0
        width, height = pil_image.size
        K_pixel = np.array(
            [[saved_fx * width, 0.0, saved_cx * width], [0.0, saved_fy * height, saved_cy * height], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        image, K_pixel = resize_and_crop(image, K_pixel, IMAGE_SHAPE)
        K_norm = K_pixel.copy()
        K_norm[0, :] /= w_out
        K_norm[1, :] /= h_out

        images.append(torch.from_numpy(image.copy()).permute(2, 0, 1).float())
        Ks.append(torch.from_numpy(K_norm).float())
        c2w = applied @ np.array(frame["transform_matrix"]) @ BLENDER_TO_OPENCV
        c2ws.append(torch.from_numpy(c2w).float())

    return torch.stack(images), torch.stack(c2ws), torch.stack(Ks)


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """`infer_colmap.py::camera_normalization` 그대로 — 기준 pose(보통 context 중앙)에
    상대적으로 전체 pose를 재정렬한다(순수 world-frame 재중심화, scale 변화 없음).
    학습 시 `dataset.pose_align_middle_view=true`와 동일 규약이라 반드시 맞춰야 한다."""

    camera_norm_matrix = torch.inverse(pivotal_pose)
    return torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)


def build_model(resplat_repo: Path, experiment: str, checkpoint: str, num_refine: int,
                 image_shape: tuple[int, int], device: torch.device):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(resplat_repo / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                f"+experiment={experiment}",
                "mode=test",
                f"model.encoder.num_refine={num_refine}",
                f"dataset.image_shape=[{image_shape[0]},{image_shape[1]}]",
                f"dataset.ori_image_shape=[{image_shape[0]},{image_shape[1]}]",
                "output_dir=outputs/resplat_dl3dv_runner",
            ],
        )

    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, None, decoder, [], None)

    ckpt = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model_wrapper.load_state_dict(ckpt, strict=False)
    model_wrapper = model_wrapper.to(device).eval()
    return model_wrapper.encoder, model_wrapper.decoder


def run(args: argparse.Namespace) -> None:
    resplat_repo = Path(args.resplat_repo)
    _add_repo_paths(resplat_repo)

    from dl3dv_dataset import load_metadata

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.overlap_level:
        overlap_data = json.loads(Path(args.overlap_candidates_index).read_text())
        ov_entry = overlap_data[args.scene_key][str(args.view_count)][args.overlap_level]
        context_ids, target_ids = ov_entry["context"], ov_entry["target"]
        print(f"[data] scene={args.scene_key} view_count={args.view_count} overlap_level={args.overlap_level} "
              f"context={context_ids} target={target_ids}")
    else:
        overlap_summary = json.loads(Path(args.overlap_summary).read_text())
        row = next(r for r in overlap_summary if r["scene"] == args.scene_key and r["view_count"] == args.view_count)
        context_ids, target_ids = row["context_indices"], row["target_indices"]
        print(f"[data] scene={args.scene_key} view_count={args.view_count} context={context_ids} target={target_ids}")

    scene_dir = DL3DV_ROOT / args.scene_key
    meta = load_metadata(scene_dir)

    context_images, context_c2w, context_K = load_frames(scene_dir, meta, context_ids)
    target_images, target_c2w, target_K = load_frames(scene_dir, meta, target_ids)

    Vc = len(context_ids)
    all_c2w = torch.cat([context_c2w, target_c2w], dim=0)
    mid_idx = Vc // 2
    all_c2w = camera_normalization(context_c2w[mid_idx : mid_idx + 1], all_c2w)
    context_c2w = all_c2w[:Vc]
    target_c2w = all_c2w[Vc:]

    batch = {
        "context": {
            "image": context_images.unsqueeze(0).to(device),
            "extrinsics": context_c2w.unsqueeze(0).to(device),
            "intrinsics": context_K.unsqueeze(0).to(device),
            "near": torch.full((1, Vc), NEAR, device=device),
            "far": torch.full((1, Vc), FAR, device=device),
            "index": torch.arange(Vc, device=device).unsqueeze(0),
        },
        "target": {
            "image": target_images.unsqueeze(0).to(device),
            "extrinsics": target_c2w.unsqueeze(0).to(device),
            "intrinsics": target_K.unsqueeze(0).to(device),
            "near": torch.full((1, len(target_ids)), NEAR, device=device),
            "far": torch.full((1, len(target_ids)), FAR, device=device),
            "index": torch.arange(Vc, Vc + len(target_ids), device=device).unsqueeze(0),
        },
        "scene": [args.scene_key],
    }

    encoder, decoder = build_model(resplat_repo, "dl3dv", args.checkpoint, args.num_refine, IMAGE_SHAPE, device)

    from lpips import LPIPS

    lpips_model = LPIPS(net="alex").to(device).eval()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        gaussians_out = encoder(batch["context"], global_step=0, deterministic=False)
        condition_features = gaussians_out.get("condition_features") if isinstance(gaussians_out, dict) else None
        gaussians = gaussians_out["gaussians"] if isinstance(gaussians_out, dict) else gaussians_out
        if args.num_refine > 0 and condition_features is not None:
            refine_output = encoder.forward_update(
                batch["context"], batch["target"], condition_features, gaussians, decoder, None,
            )
            gaussians = refine_output["gaussian"][-1]
        output = decoder.forward(
            gaussians, batch["target"]["extrinsics"], batch["target"]["intrinsics"],
            batch["target"]["near"], batch["target"]["far"], IMAGE_SHAPE,
        )
    torch.cuda.synchronize()
    wall_clock = time.perf_counter() - t0

    pred = output.color[0].clamp(0, 1)
    gt = target_images.to(device)
    psnrs = [psnr(pred[i], gt[i]) for i in range(pred.shape[0])]
    lpipss = [float(lpips_model((pred[i] * 2 - 1)[None], (gt[i] * 2 - 1)[None]).item()) for i in range(pred.shape[0])]

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    row_out = {
        "experiment_id": args.experiment_id,
        "scene": args.scene_key,
        "method": "ReSplat",
        "num_refine": args.num_refine,
        "checkpoint": Path(args.checkpoint).name,
        "wall_clock": wall_clock,
        "test_psnr": float(np.mean(psnrs)),
        "test_lpips": float(np.mean(lpipss)),
        "gaussian_count": int(gaussians.means.shape[1]),
        "peak_vram": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    }
    log_path = logs_dir / f"{args.scene_key}_ReSplat_{args.view_count}view.json"
    tmp_path = log_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(row_out, indent=2), encoding="utf-8")
    tmp_path.replace(log_path)

    print(f"[eval] wall_clock={wall_clock:.3f}s gaussians={row_out['gaussian_count']}")
    print(f"[eval] test_psnr={row_out['test_psnr']:.3f} test_lpips={row_out['test_lpips']:.3f}")
    print(f"[done] {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReSplat runner for DL3DV (resplat conda env only).")
    parser.add_argument("--overlap-summary", default="experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json",
                         help="--overlap-level 미지정 시 사용. **v2를 기본값으로 쓴다** — v1"
                         "(dl3dv_overlap/all_scenes_summary.json)은 알려진 버그(부실한 view 선택,"
                         "예: 이 스크립트 디버깅 중 발견한 sfm_points=1/has_isolated_view=True 사례)가"
                         " 있어 실제 C1-a 파이프라인(run_dl3dv_c1a_main.py)도 v2를 명시적으로 쓴다"
                         "(2026-08-16 발견 — ReSplat PSNR이 14~17dB로 붕괴한 원인이 바로 이 기본값"
                         " 오사용이었음, v2로 바꾸니 26dB대로 정상화).")
    parser.add_argument("--overlap-level", choices=["high", "low"], default=None)
    parser.add_argument("--overlap-candidates-index",
                         default="experiments/outputs/dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json")
    parser.add_argument("--scene-key", required=True)
    parser.add_argument("--view-count", type=int, default=8)
    parser.add_argument("--num-refine", type=int, default=4, help="학습 스크립트 train_max_refine=4와 일치.")
    parser.add_argument("--experiment-id", default="resplat-exploratory-20260817")
    parser.add_argument("--resplat-repo", default="/data/Re-feem/code/resplat")
    parser.add_argument(
        "--checkpoint",
        default="/data/Re-feem/code/resplat/pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "resplat_dl3dv"))
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

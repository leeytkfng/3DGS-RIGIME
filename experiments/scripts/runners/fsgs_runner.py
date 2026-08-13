#!/usr/bin/env python3
"""FSGS(sparse-view 특화 3DGS optimization) 러너 — protocol_utils 스키마 준수, DTU pose-given track.

FSGS(`/data/Re-feem/code/fsgs`)는 외부 repo다. `train.py`를 subprocess로 그대로 부르지 않고,
이 러너 안에서 FSGS의 Scene/GaussianModel/render()/loss들을 직접 import해서 우리 프로토콜에
맞는 바깥 루프(wall-clock budget checkpoint)로 감싼다 — `vanilla_3dgs_runner.py`가 gsplat을
감싸는 것과 같은 패턴이다. FSGS 자체 loss 구성(L1+D-SSIM+MiDaS depth correlation+pseudo-view
depth loss)과 densification/opacity-reset 스케줄(`train.py::training()`)은 그대로 재사용한다 —
우리가 바꾸는 건 딱 두 곳이다.

1. **View 선택**: `scene/dataset_readers.py::readColmapSceneInfo()`는 자체 llffhold(=8) split
   + n_views linspace subsample로 train/test view를 "다시" 고른다 — `prep_dtu_for_fsgs.py`가
   seed로 미리 골라둔 train_ids/test_ids를 완전히 무시한다(직접 읽어서 확인함, 2026-08-13).
   다른 러너(Vanilla3DGS/MVSplat)와 같은 view로 학습시켜야 (scene, view_count, seed) 조건이
   공정하게 비교되므로, `scene.dataset_readers.sceneLoadTypeCallbacks["Colmap"]`을 우리 함수로
   monkey-patch한다 — 카메라 로딩 코드(`readColmapCameras`/`getNerfppNorm`/`fetchPly`)는 원본을
   그대로 재사용하고, split 로직만 우리 train_ids/test_ids로 바꾼다.
2. **바깥 루프 제어**: `train.py`는 iteration 기반(`test_iterations`)이라 우리 프로토콜의
   wall-clock budget(1/10/60/300s) 개념이 없다 — loop 본문은 거의 그대로 옮기되 종료·체크포인트
   조건만 elapsed 기반으로 바꾼다. FSGS 자체 iteration-indexed 스케줄 상수(start_sample_pseudo,
   densify_until_iter 등)는 바꾸지 않는다 — 짧은 budget에서 densification/pseudo-view sampling이
   충분히 못 도는 건 Vanilla3DGS와 마찬가지로 "그 방법의 실제 동작"이지 보정할 대상이 아니다.

seed: FSGS 자체 `utils.general_utils.safe_state()`는 seed=1을 하드코딩해서 우리 --seed를
무시한다 — 여기서는 `safe_state()`를 호출하지 않고 numpy/torch/random 시드를 직접 우리
seed로 설정한다.

초기화(§overall.md §4.2, 2026-08-13 결정): FSGS 원 프로토콜은 dense-MVS point cloud를
기대하지만 우리 시스템엔 COLMAP CLI의 dense MVS 모듈이 없어 Vanilla3DGS와 동일한 sparse
COLMAP triangulation으로 대체한다(`prep_dtu_for_fsgs.py`) — 논문 methods/limitations 양쪽에
명시해야 하는 의도적 편차.

RE10K/DL3DV 지원(2026-08-13 확장): `prep_dtu_for_fsgs.py::prepare_views_for_fsgs()`(dataset-
agnostic 버전)로 이미 메모리에 로드된 view를 FSGS-ready 디렉토리로 만든다 —
`vanilla_3dgs_runner.py::_colmap_init_from_loaded_views()`와 같은 패턴. view 후보는 다른
러너와 동일하게 `re10k_main_subset.json`/`dl3dv_overlap_v2` summary에서 가져와 seed/overlap
축이 일관되게 유지된다.

실행 환경: fsgs conda env 필요(`/opt/conda/envs/fsgs/bin/python3`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_ROOT / "core"))
sys.path.insert(0, str(_SCRIPTS_ROOT / "analysis"))

from protocol_utils import budget_checkpoint, oracle_checkpoint  # noqa: E402

RE10K_NEAR_FAR = (1.0, 100.0)  # mvsplat_re10k_runner.py와 동일
DL3DV_NEAR_FAR = (0.5, 200.0)  # depthsplat_dl3dv_runner.py와 동일


def _make_patched_colmap_loader(train_names: list[str], test_names: list[str]):
    """`sceneLoadTypeCallbacks["Colmap"]`을 대체할 함수를 만든다.

    FSGS 원본(`readColmapSceneInfo`)과 다른 점은 view 선택 로직뿐이다 — 카메라/포인트클라우드
    로딩(`readColmapCameras`, `getNerfppNorm`, `fetchPly`)은 원본을 그대로 재사용한다. 이 함수는
    FSGS repo가 이미 sys.path에 있는 상태에서(=`_add_fsgs_repo` 이후) 호출해야 한다.
    """

    from scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from scene.dataset_readers import SceneInfo, fetchPly, getNerfppNorm, readColmapCameras

    def _load(path, images, eval_, n_views):
        del eval_  # train_ids/test_ids가 이미 split을 결정한다 — FSGS의 llffhold는 쓰지 않는다.
        ply_path = os.path.join(path, f"{n_views}_views/dense/fused.ply")
        try:
            cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0/images.bin"))
            cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0/cameras.bin"))
        except Exception:
            cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0/images.txt"))
            cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0/cameras.txt"))

        pcd = fetchPly(ply_path)
        reading_dir = "images" if images is None else images
        rgb_mapping = sorted(
            f
            for f in glob.glob(os.path.join(path, reading_dir, "*"))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        )
        cam_extrinsics = {cam_extrinsics[k].name: cam_extrinsics[k] for k in cam_extrinsics}
        cam_infos = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            path=path,
            rgb_mapping=rgb_mapping,
        )
        by_name = {c.image_name: c for c in cam_infos}
        train_cam_infos = [by_name[name] for name in train_names]
        test_cam_infos = [by_name[name] for name in test_names]
        assert len(train_cam_infos) == len(train_names), (
            f"expected {len(train_names)} train cams, got {len(train_cam_infos)} "
            f"(image_name 매칭 실패 — prep_dtu_for_fsgs.py 파일명 규칙 확인 필요)"
        )

        nerf_normalization = getNerfppNorm(train_cam_infos)
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path,
        )

    return _load


def _build_fsgs_args(data_dir: Path, model_path: Path, view_count: int):
    """FSGS 자체 ArgumentParser(ModelParams/OptimizationParams/PipelineParams)를 그대로 써서
    모든 기본값(LR, densify threshold, depth_weight 등)을 원본과 동일하게 가져온다 — 우리가
    override하는 건 source_path/model_path/images/n_views/eval뿐이다."""

    from arguments import ModelParams, OptimizationParams, PipelineParams

    fsgs_parser = argparse.ArgumentParser()
    lp = ModelParams(fsgs_parser)
    op = OptimizationParams(fsgs_parser)
    pp = PipelineParams(fsgs_parser)
    fsgs_args = fsgs_parser.parse_args(
        [
            "--source_path", str(data_dir),
            "--model_path", str(model_path),
            "--images", "images",
            "--n_views", str(view_count),
            "--eval",
        ]
    )
    # train.py의 최상위 parser에만 있고 ModelParams/OptimizationParams/PipelineParams엔 없는
    # 플래그 — GaussianModel.create_from_pcd()/training_setup()이 참조한다(train_bg=False가
    # FSGS 기본 동작, 우리는 이 옵션을 쓰지 않는다).
    fsgs_args.train_bg = False
    return fsgs_args, lp.extract(fsgs_args), op.extract(fsgs_args), pp.extract(fsgs_args)


def _evaluate(gaussians, scene, render, pipe, background) -> dict[str, float]:
    """FSGS 자체 psnr/ssim/lpips 구현을 재사용해 test set 평가 — FSGS가 자기 논문에서
    보고하는 것과 같은 척도로 우리도 비교한다."""

    from lpipsPyTorch import lpips as fsgs_lpips
    from utils.image_utils import psnr as fsgs_psnr
    from utils.loss_utils import ssim as fsgs_ssim

    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for cam in scene.getTestCameras():
            image = torch.clamp(render(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            psnrs.append(float(fsgs_psnr(image, gt).mean()))
            ssims.append(float(fsgs_ssim(image, gt).mean()))
            lpipss.append(float(fsgs_lpips(image, gt, net_type="vgg")))
    return {
        "test_psnr": float(np.mean(psnrs)),
        "test_ssim": float(np.mean(ssims)),
        "test_lpips": float(np.mean(lpipss)),
    }


def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(args.fsgs_repo).resolve()))

    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    import scene.dataset_readers as dataset_readers
    from torchmetrics.functional.regression import pearson_corrcoef
    from utils.depth_utils import estimate_depth
    from utils.loss_utils import l1_loss_mask, ssim

    from prep_dtu_for_fsgs import prepare_dtu_for_fsgs, prepare_views_for_fsgs

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.output_dir) / "fsgs_data" / f"{args.scene}_{args.view_count}view_seed{args.seed}"

    if args.dataset == "dtu":
        train_ids, test_ids, init_source = prepare_dtu_for_fsgs(Path(args.scan_dir), args.view_count, args.seed, data_dir)
        train_names = [f"{v:03d}" for v in train_ids]
        test_names = [f"{v:03d}" for v in test_ids]
    elif args.dataset == "re10k":
        from re10k_dataset import get_scene_item, load_views

        target_shape = tuple(args.image_shape) if args.image_shape else None
        subset = json.loads(Path(args.re10k_subset_index).read_text())
        entry = subset[args.re10k_scene_key]
        candidate = entry["view_candidates"][str(args.view_count)]
        if candidate.get("context") is None:
            raise SystemExit(f"{args.re10k_scene_key} view_count={args.view_count}: candidate 없음(too short)")
        train_ids, test_ids = candidate["context"], candidate["target"]
        item = get_scene_item(Path("/data/Re-feem/datasets/re10k/test") / entry["chunk_file"], args.re10k_scene_key)
        train_views = load_views(item, train_ids, target_shape=target_shape)
        test_views = load_views(item, test_ids, target_shape=target_shape)
        near, far = RE10K_NEAR_FAR
        train_names_ext, test_names_ext, init_source = prepare_views_for_fsgs(train_views, test_views, args.seed, data_dir, near, far)
        train_names = [Path(n).stem for n in train_names_ext]
        test_names = [Path(n).stem for n in test_names_ext]
    else:  # dl3dv
        from dl3dv_dataset import load_metadata, load_views

        target_shape = tuple(args.image_shape) if args.image_shape else None
        overlap_summary = json.loads(Path(args.dl3dv_overlap_summary).read_text())
        row = next(r for r in overlap_summary if r["scene"] == args.dl3dv_scene_key and r["view_count"] == args.view_count)
        train_ids, test_ids = row["context_indices"], row["target_indices"]
        scene_dir = Path("/data/Re-feem/datasets/dl3dv") / args.dl3dv_scene_key
        meta = load_metadata(scene_dir)
        train_views = load_views(scene_dir, meta, train_ids, target_shape=target_shape)
        test_views = load_views(scene_dir, meta, test_ids, target_shape=target_shape)
        near, far = DL3DV_NEAR_FAR
        train_names_ext, test_names_ext, init_source = prepare_views_for_fsgs(train_views, test_views, args.seed, data_dir, near, far)
        train_names = [Path(n).stem for n in train_names_ext]
        test_names = [Path(n).stem for n in test_names_ext]

    dataset_readers.sceneLoadTypeCallbacks["Colmap"] = _make_patched_colmap_loader(train_names, test_names)

    checkpoints_dir = Path(args.output_dir) / "checkpoints" / args.scene / f"{args.view_count}view_seed{args.seed}_fsgs"
    logs_dir = Path(args.output_dir) / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    fsgs_args, dataset, opt, pipe = _build_fsgs_args(data_dir, checkpoints_dir, args.view_count)

    gaussians = GaussianModel(fsgs_args)
    scene = Scene(fsgs_args, gaussians, shuffle=False)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    if not train_cams:
        raise SystemExit("no train cameras loaded — data prep이 잘못됐을 가능성")

    # CUDA warm-up: 프로토콜(runtime.cuda_warmup=true)대로 최초 컴파일 시간은 측정에서 뺀다.
    with torch.no_grad():
        render(train_cams[0], gaussians, pipe, background)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print(
        f"[train] starting FSGS optimization, budget={args.max_budget_seconds}s, "
        f"init_points={gaussians.get_xyz.shape[0]} (init_source={init_source})"
    )

    trajectory: list[dict[str, object]] = []
    viewpoint_stack, pseudo_stack = None, None
    step = 0
    elapsed = 0.0
    next_snapshot_targets = sorted(set(args.budget_snapshots))
    snapshot_idx = 0
    depth_weight = fsgs_args.depth_weight  # end_sample_pseudo 이후 0.001로 낮아지는 mutable 사본

    while elapsed < args.max_budget_seconds:
        step += 1
        if step % 500 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = train_cams.copy()

        step_start = time.perf_counter()

        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        l1 = l1_loss_mask(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        rendered_depth = render_pkg["depth"][0].reshape(-1, 1)
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda().reshape(-1, 1)
        depth_loss = min(
            1 - pearson_corrcoef(-midas_depth, rendered_depth),
            1 - pearson_corrcoef(1 / (midas_depth + 200.0), rendered_depth),
        )
        loss = loss + depth_weight * depth_loss
        if step > fsgs_args.end_sample_pseudo:
            depth_weight = 0.001

        if (
            step % fsgs_args.sample_pseudo_interval == 0
            and fsgs_args.start_sample_pseudo < step < fsgs_args.end_sample_pseudo
        ):
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(random.randint(0, len(pseudo_stack) - 1))
            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0].reshape(-1, 1)
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode="train").reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()
            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((step - fsgs_args.start_sample_pseudo) / 500.0, 1)
                loss = loss + loss_scale * fsgs_args.depth_pseudo_weight * depth_loss_pseudo

        loss.backward()

        with torch.no_grad():
            if step < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if step > opt.densify_from_iter and step % opt.densification_interval == 0:
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, None, step
                    )

            if step < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.update_learning_rate(step)

            if (
                step > fsgs_args.start_sample_pseudo
                and (step - fsgs_args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0
            ):
                gaussians.reset_opacity()

        torch.cuda.synchronize()
        elapsed += time.perf_counter() - step_start

        if snapshot_idx < len(next_snapshot_targets) and elapsed >= next_snapshot_targets[snapshot_idx]:
            budget_label = next_snapshot_targets[snapshot_idx]
            snapshot_idx += 1
            metrics = _evaluate(gaussians, scene, render, pipe, background)
            scene.save(step)
            checkpoint_path = Path(scene.model_path) / "point_cloud" / f"iteration_{step}" / "point_cloud.ply"
            row = {
                "experiment_id": args.experiment_id,
                "scene": args.scene,
                "seed": args.seed,
                "method": args.method,
                "iteration": step,
                "wall_clock": min(elapsed, budget_label),
                "train_loss": float(loss.item()),
                "validation_metric": None,
                "test_psnr": metrics["test_psnr"],
                "test_ssim": metrics["test_ssim"],
                "test_lpips": metrics["test_lpips"],
                "gaussian_count": int(gaussians.get_xyz.shape[0]),
                "peak_vram": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                "checkpoint_path": str(checkpoint_path),
                "init_source": init_source,  # colmap_sfm_sparse(dense-MVS 대체, §overall.md §4.2) 또는 random_sphere_fallback
                "densification": "fsgs_default",
            }
            trajectory.append(row)
            print(
                f"[ckpt] budget={budget_label}s iter={step} elapsed={elapsed:.1f}s "
                f"gaussians={row['gaussian_count']} test_psnr={row['test_psnr']:.3f} "
                f"test_lpips={row['test_lpips']:.3f} peak_vram_mb={row['peak_vram']:.0f}"
            )

        if step >= args.max_iterations:
            print(f"[train] hit max_iterations={args.max_iterations} before budget exhausted, stopping.")
            break

    log_path = logs_dir / f"{args.scene}_{args.method}_{args.view_count}view_seed{args.seed}.json"
    # 원자적 쓰기 — vanilla_3dgs_runner.py와 동일 이유(배치 driver의 resume 판단이
    # log_path.exists()에 의존하므로 write 도중 kill되면 잘린 파일이 "완료"로 오인됨).
    tmp_path = log_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    tmp_path.replace(log_path)
    print(f"[done] trajectory written to {log_path}")

    if trajectory:
        main_row = budget_checkpoint(trajectory, args.max_budget_seconds)
        oracle_row = oracle_checkpoint(trajectory, metric="test_psnr")
        print(f"[protocol] budget_end_checkpoint (main, leakage-safe): {main_row}")
        print(f"[protocol] oracle_checkpoint (diagnostic only): {oracle_row}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FSGS runner (protocol_utils schema), DTU/RE10K/DL3DV.")
    parser.add_argument("--dataset", choices=["dtu", "re10k", "dl3dv"], default="dtu")
    parser.add_argument("--scan-dir", default="/data/Re-feem/datasets/dtu/scan1", help="dataset=dtu일 때만 필요.")
    parser.add_argument(
        "--re10k-subset-index",
        default="experiments/outputs/re10k_main_subset/re10k_main_subset.json",
        help="dataset=re10k일 때만 사용. generate_re10k_main_subset.py 출력.",
    )
    parser.add_argument("--re10k-scene-key", default=None, help="dataset=re10k일 때 필요.")
    parser.add_argument(
        "--dl3dv-overlap-summary",
        default="experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json",
        help="dataset=dl3dv일 때만 사용. generate_dl3dv_view_overlap.py(v2) 출력.",
    )
    parser.add_argument("--dl3dv-scene-key", default=None, help="dataset=dl3dv일 때 필요.")
    parser.add_argument(
        "--image-shape", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"),
        help="dataset=re10k/dl3dv일 때 리사이즈 목표 해상도(vanilla_3dgs_runner.py와 동일 convention). 기본은 원본 해상도.",
    )
    parser.add_argument("--scene", required=True, help="scene id used in logs, e.g. dtu_scan1 or RE10K/DL3DV scene key")
    parser.add_argument("--method", default="FSGS")
    parser.add_argument("--experiment-id", default="regime-map-20260806")
    parser.add_argument("--view-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-budget-seconds", type=float, default=300.0)
    parser.add_argument(
        "--budget-snapshots", type=float, nargs="+", default=[1.0, 10.0, 60.0, 300.0],
        help="config.protocol.budgets_seconds와 맞춘다.",
    )
    parser.add_argument("--max-iterations", type=int, default=200_000, help="budget 안에서도 무한 루프 방지용 상한.")
    parser.add_argument("--data-dir", default=None, help="FSGS-ready 데이터 출력 경로. 기본은 output-dir/fsgs_data/<scene>_<view>view_seed<seed>.")
    parser.add_argument("--fsgs-repo", default="/data/Re-feem/code/fsgs")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "outputs"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.budget_snapshots = [b for b in sorted(args.budget_snapshots) if b <= args.max_budget_seconds]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

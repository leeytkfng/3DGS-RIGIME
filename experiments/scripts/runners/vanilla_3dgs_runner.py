#!/usr/bin/env python3
"""gsplat 기반 vanilla 3DGS per-scene optimization 러너 (DTU pose-given track).

이 파일의 목적:
- 계획서(연구계획서_정리.md §4.1, overall.md §5.1/§5.7/§5.10)에서 정의한 프로토콜대로
  DTU 한 scan을 실제로 최적화해서, 스캐폴드(protocol_utils.py)가 다루는 체크포인트/로그
  스키마가 진짜 학습 곡선에서도 동작하는지 검증한다.
- H200의 큰 VRAM을 쓰기 위해 기본값을 downsample 없는 원본 해상도(1600x1200), 큰 초기
  Gaussian 수, 상한 없는 growth로 둔다.

초기화는 colmap_init.triangulate_sfm_points()로 학습 view만 이용해 known-pose triangulation을
수행한다 (§5.2/§8에서 요구하는 COLMAP SfM init). GT point cloud는 참조하지 않는다. Triangulation이
너무 빈약하면(예: 극단적으로 낮은 overlap) 카메라 기하로 추정한 bounding sphere 안의 random
point로 fallback한다 — 이 fallback 경로를 탄 run은 로그에 `init_source`로 남긴다.

지금 단계에서 의도적으로 비워둔 것:
- checkpoint_rule=budget_end_checkpoint / oracle_peak 분리는 protocol_utils의 기존 함수를
  그대로 재사용한다. 이 파일은 새 규칙을 만들지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from colmap_init import CameraParams, triangulate_sfm_points, triangulate_sfm_points_from_cameras  # noqa: E402
from depth_backprojection import back_project_multi_view  # noqa: E402
from dtu_dataset import estimate_scene_sphere, load_camera, load_scan  # noqa: E402
from protocol_utils import budget_checkpoint, oracle_checkpoint  # noqa: E402

SH_C0 = 0.28209479177387814
MIN_SFM_POINTS = 200  # 이보다 triangulated point가 적으면 random-sphere init으로 fallback


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
def build_camera_tensors(
    view: dict, device: torch.device, pose_scale_factor: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """view dict(dtu_dataset.load_scan 결과 원소 하나)에서 rasterization() 입력을 만든다.

    pose_scale_factor: C1-b FF warm-start 전용. MVSplat DTU 경로(mvsplat_runner.py)는
    world-to-camera translation에 DTU_SCALE_FACTOR(1/200)를 곱해서 씀 — Gaussian means도
    그 축소된 좌표계로 나온다. 여기서 로드하는 카메라가 원본(축소 안 된) DTU 좌표계라서,
    warm-start로 불러온 Gaussians와 좌표계를 맞추려면 같은 배율을 translation에 적용해야
    한다. 기본값 1.0(원본 DTU 좌표계, 기존 동작 그대로)이면 아무 영향 없다.
    """

    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = torch.from_numpy(view["R"]).float()
    viewmat[:3, 3] = torch.from_numpy(view["t"]).float() * pose_scale_factor
    k = torch.from_numpy(view["K"]).float().to(device)
    image = torch.from_numpy(view["image"]).float().to(device).permute(2, 0, 1)  # [3,H,W]
    return viewmat, k, image


def _load_warm_start_params(checkpoint_path: Path, device: torch.device) -> tuple[dict[str, torch.nn.Parameter], str]:
    """mvsplat_runner.py(또는 향후 depthsplat_runner.py)가 저장한 `gaussians.pt`를 로드해
    gsplat 파라미터화로 변환한다. §5.8 렌더 등가성 gate 통과 여부는 호출측 책임."""

    from ff_gaussian_convert import gaussians_to_gsplat_params

    raw = torch.load(checkpoint_path, map_location="cpu")
    means, covariances = raw["means"][0], raw["covariances"][0]
    harmonics, opacities = raw["harmonics"][0], raw["opacities"][0]
    params = gaussians_to_gsplat_params(means, covariances, harmonics, opacities, device)
    return params, "ff_warm_start"


def _colmap_init_from_loaded_views(train_views: list[dict], workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """RE10K/DL3DV처럼 이미 메모리에 로드된 view(image/K/R/t/width/height)로 known-pose
    COLMAP triangulation을 돌린다. DTU의 `triangulate_sfm_points()`와 달리 원본 이미지
    파일을 다시 찾지 않고, `load_views()`가 만든(리사이즈까지 끝난) 이미지를 그대로
    임시 디렉토리에 써서 쓴다 — `generate_re10k_view_overlap.py`/`generate_dl3dv_view_overlap.py`
    가 쓰는 것과 같은 공용 코어(`triangulate_sfm_points_from_cameras`)를 재사용한다.
    """

    from PIL import Image

    image_dir = workdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_names = []
    camera_by_name: dict[str, CameraParams] = {}
    for view in train_views:
        name = f"{view['view_id']:04d}.png"
        image_names.append(name)
        image_uint8 = (np.clip(view["image"], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(image_uint8).save(image_dir / name)
        camera_by_name[name] = CameraParams(
            K=view["K"], R=view["R"], t=view["t"], width=view["width"], height=view["height"]
        )

    return triangulate_sfm_points_from_cameras(image_dir, image_names, camera_by_name, workdir)


def _random_points_in_sphere(center: np.ndarray, radius: float, num_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * (rng.uniform(size=(num_points, 1)) ** (1.0 / 3.0))
    points = center[None, :] + directions * radii
    colors = np.full((num_points, 3), 128, dtype=np.uint8)
    return points, colors


def init_gaussians(
    points: np.ndarray,
    colors_rgb: np.ndarray,
    device: torch.device,
) -> dict[str, torch.nn.Parameter]:
    """3D point(+RGB color)로부터 Gaussian을 초기화한다 (SfM 또는 random-sphere fallback 공용).

    scale은 원본 3DGS 관례대로 각 점의 최근접 이웃까지 거리로 잡는다 (KNN, k=3 평균).
    """

    num_points = points.shape[0]
    tree = cKDTree(points)
    # k=4: 자기 자신(distance 0) + 최근접 3개.
    k = min(4, num_points)
    dists, _ = tree.query(points, k=k)
    if k > 1:
        avg_nn_dist = dists[:, 1:].mean(axis=1)
    else:
        avg_nn_dist = np.full(num_points, 1e-3)
    avg_nn_dist = np.clip(avg_nn_dist, 1e-4, None)
    scales = np.log(avg_nn_dist)[:, None].repeat(3, axis=1).astype(np.float32)

    means = torch.tensor(points, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    quats = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity quaternion (w, x, y, z)
    opacities = torch.full((num_points,), -2.1972246, dtype=torch.float32, device=device)  # inverse_sigmoid(0.1)

    rgb01 = colors_rgb.astype(np.float32) / 255.0
    sh0_np = (rgb01 - 0.5) / SH_C0
    sh0 = torch.tensor(sh0_np, dtype=torch.float32, device=device).unsqueeze(1)
    sh_rest_dims = (3 + 1) ** 2 - 1  # sh_degree=3 최대 계수까지 미리 확보
    shN = torch.zeros((num_points, sh_rest_dims, 3), dtype=torch.float32, device=device)

    return {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales_t),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }


class LPIPSMetric:
    """`lpips` 패키지를 lazy하게 로드하는 wrapper. 첫 호출에서 AlexNet 사전학습 가중치를 받는다."""

    def __init__(self, device: torch.device):
        self._model = lpips.LPIPS(net="alex").to(device)
        self._model.eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # lpips는 [-1, 1] 범위의 [1,3,H,W] 입력을 기대한다.
        pred_n = (pred.unsqueeze(0) * 2.0 - 1.0)
        target_n = (target.unsqueeze(0) * 2.0 - 1.0)
        return float(self._model(pred_n, target_n).item())


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
    target_shape = tuple(args.image_shape) if args.image_shape else None

    if args.dataset == "re10k":
        from re10k_dataset import get_scene_item, load_views

        subset = json.loads(Path(args.re10k_subset_index).read_text())
        entry = subset[args.re10k_scene_key]
        if args.overlap_level:
            # co-visibility selector가 만든 low/high 후보 사용 — chunk_file은 여전히
            # re10k_main_subset.json에서 가져온다(overlap_candidates_index엔 없음).
            overlap_data = json.loads(Path(args.overlap_candidates_index).read_text())
            ov_entry = overlap_data[args.re10k_scene_key][str(args.view_count)]
            train_ids, test_ids = ov_entry[args.overlap_level]["context"], ov_entry["target"]
            print(f"[data] RE10K scene={args.re10k_scene_key} overlap_level={args.overlap_level} "
                  f"train(context)={train_ids} test(target)={test_ids}")
        else:
            candidate = entry["view_candidates"][str(args.view_count)]
            if candidate.get("context") is None:
                raise SystemExit(f"{args.re10k_scene_key} view_count={args.view_count}: candidate 없음(too short)")
            train_ids, test_ids = candidate["context"], candidate["target"]
            print(f"[data] RE10K scene={args.re10k_scene_key} train(context)={train_ids} test(target)={test_ids}")

        item = get_scene_item(Path("/data/Re-feem/datasets/re10k/test") / entry["chunk_file"], args.re10k_scene_key)
        train_views = load_views(item, train_ids, target_shape=target_shape)
        test_views = load_views(item, test_ids, target_shape=target_shape)

        # DTU의 estimate_scene_sphere() 대응물 — RE10K는 calibration 기반 sphere-fit이 아직
        # 없어서, context view 카메라 중심들의 산포로 간단히 scene scale을 근사한다
        # (optimizer LR 스케일링에만 쓰이므로 정밀할 필요는 없다).
        centers = np.stack([v["center"] for v in train_views])
        center = centers.mean(axis=0)
        radius = float(np.median(np.linalg.norm(centers - center, axis=1))) or 1.0
        print(f"[init] RE10K scene scale(centroid-median) radius={radius:.4f}")
    elif args.dataset == "dl3dv":
        # view 후보는 generate_dl3dv_view_overlap.py가 저장한 context/target을 그대로
        # 재사용한다(overlap 계산에 쓴 것과 같은 view라야 함). warm-start와 COLMAP/random init
        # 둘 다 지원(RE10K와 동일).
        from dl3dv_dataset import load_metadata, load_views

        overlap_summary = json.loads(Path(args.dl3dv_overlap_summary).read_text())
        row = next(
            r for r in overlap_summary if r["scene"] == args.dl3dv_scene_key and r["view_count"] == args.view_count
        )
        train_ids, test_ids = row["context_indices"], row["target_indices"]
        print(f"[data] DL3DV scene={args.dl3dv_scene_key} train(context)={train_ids} test(target)={test_ids}")

        scene_dir = Path("/data/Re-feem/datasets/dl3dv") / args.dl3dv_scene_key
        meta = load_metadata(scene_dir)
        train_views = load_views(scene_dir, meta, train_ids, target_shape=target_shape)
        test_views = load_views(scene_dir, meta, test_ids, target_shape=target_shape)

        centers = np.stack([v["center"] for v in train_views])
        center = centers.mean(axis=0)
        radius = float(np.median(np.linalg.norm(centers - center, axis=1))) or 1.0
        print(f"[init] DL3DV scene scale(centroid-median) radius={radius:.4f}")
    else:
        if not args.scan_dir:
            raise SystemExit("--dataset dtu는 --scan-dir가 필요하다.")
        scan_dir = Path(args.scan_dir)
        all_view_ids = list(range(1, 50))
        test_ids = all_view_ids[::7]  # 1,8,15,...,43 -> 고정 held-out test split
        train_pool = [v for v in all_view_ids if v not in test_ids]

        rng = np.random.default_rng(args.seed)
        train_ids = sorted(rng.choice(train_pool, size=min(args.view_count, len(train_pool)), replace=False).tolist())

        print(f"[data] train views ({len(train_ids)}): {train_ids}")
        print(f"[data] test views ({len(test_ids)}): {test_ids}")

        train_views = load_scan(scan_dir, train_ids, target_shape=target_shape)
        test_views = load_scan(scan_dir, test_ids, target_shape=target_shape)

        cameras_for_sphere = [load_camera(scan_dir / "cameras", v) for v in all_view_ids]
        center, radius = estimate_scene_sphere(cameras_for_sphere)
        print(f"[init] estimated scene center={center}, radius={radius:.3f}")

    if args.warm_start_checkpoint and args.depth_cache_path:
        raise SystemExit("--warm-start-checkpoint와 --depth-cache-path는 동시에 줄 수 없다(둘 다 init 단계 전체를 대체하는 경로).")

    if args.warm_start_checkpoint:
        # C1-b: FF(MVSplat/DepthSplat) Gaussian 출력을 그대로 최적화 시작점으로 쓴다.
        # COLMAP triangulation/random init은 건너뛴다 — §5.8 렌더 등가성 gate를
        # `check_renderer_equivalence.py`로 먼저 통과시킨 체크포인트만 여기 넣어야 한다.
        params, init_source = _load_warm_start_params(Path(args.warm_start_checkpoint), device)
        print(
            f"[init] source={init_source}, num_points={params['means'].shape[0]} "
            f"(FF warm-start, pose_scale_factor={args.pose_scale_factor})"
        )
    elif args.depth_cache_path:
        # C2: analysis/precompute_depth_maps.py가 미리 뽑아둔 raw depth를 교란(§5.9)한 뒤
        # back-projection해서 초기값으로 쓴다. COLMAP triangulation은 건너뛴다 — C2의 목적
        # 자체가 "depth 품질을 통제된 방식으로 나쁘게 만들었을 때 무슨 일이 일어나는가"이므로,
        # COLMAP(=깨끗한 기하)로 되돌아가면 개입 효과가 섞인다.
        cache = torch.load(args.depth_cache_path, map_location="cpu")
        cache_view_ids = set(cache["view_ids"])
        train_view_ids = {v["view_id"] for v in train_views}
        if cache_view_ids != train_view_ids:
            raise SystemExit(
                f"depth cache의 view_id({sorted(cache_view_ids)})가 train view({sorted(train_view_ids)})와 다르다"
                " — precompute_depth_maps.py를 이 run과 같은 --view-count/--seed로 다시 돌려야 한다."
            )
        cameras = [(K, R, t) for K, R, t in zip(cache["K"], cache["R"], cache["t"])]
        points, colors = back_project_multi_view(
            cache["depths"], cameras, cache["images"],
            sigma=args.depth_noise_sigma, scale_bias=args.depth_scale_bias, seed=args.seed,
            stride=args.depth_stride, max_points_per_view=args.depth_max_points_per_view,
        )
        init_source = "depth_backprojection"
        print(
            f"[init] source={init_source}, num_points={points.shape[0]} "
            f"(sigma={args.depth_noise_sigma}, scale_bias={args.depth_scale_bias})"
        )
        params = init_gaussians(points, colors, device)
    else:
        overlap_suffix = f"_{args.overlap_level}" if args.overlap_level else ""
        colmap_workdir = Path(args.output_dir) / "colmap_work" / args.scene / f"{args.view_count}view_seed{args.seed}{overlap_suffix}"
        if args.dataset == "dtu":
            sfm_points, sfm_colors = triangulate_sfm_points(scan_dir, train_ids, colmap_workdir)
        else:
            # RE10K/DL3DV: 이미 메모리에 로드된 train_views(image/K/R/t)를 그대로 임시
            # 디렉토리에 써서 known-pose triangulation을 돌린다 — generate_re10k_view_overlap.py/
            # generate_dl3dv_view_overlap.py와 같은 공용 코어(triangulate_sfm_points_from_cameras)
            # 를 재사용한다.
            sfm_points, sfm_colors = _colmap_init_from_loaded_views(train_views, colmap_workdir)
        if sfm_points.shape[0] >= MIN_SFM_POINTS:
            init_source = "colmap_sfm"
            points, colors = sfm_points, sfm_colors
        else:
            init_source = "random_sphere_fallback"
            points, colors = _random_points_in_sphere(center, radius, args.num_init_points, args.seed)
        print(f"[init] source={init_source}, num_points={points.shape[0]}")
        params = init_gaussians(points, colors, device)

    optimizers = build_optimizers(params, scene_scale=radius * args.pose_scale_factor)

    # C1-b densification on/off ablation (paper_scaffold_audit_log.md §12.1):
    # densification=off는 refine_stop_iter=0으로 강제해 adaptive density control(grow/prune)을
    # 전부 끈다. step은 0부터 시작하고 strategy 내부 조건이 `step < refine_stop_iter`이므로
    # 0으로 두면 어떤 step에서도 densify/prune가 트리거되지 않는다.
    refine_stop_iter = 0 if args.densification == "off" else args.refine_stop_iter
    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=args.refine_start_iter,
        refine_stop_iter=refine_stop_iter,
        reset_every=args.reset_every,
        grow_grad2d=args.grow_grad2d,
        grow_scale3d=args.grow_scale3d,
        prune_opa=args.prune_opa,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=radius * args.pose_scale_factor)

    densification_dir_suffix = "" if args.densification == "on" else "_densoff"
    overlap_dir_suffix = f"_{args.overlap_level}" if args.overlap_level else ""
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints" / args.scene / f"{args.view_count}view_seed{args.seed}{densification_dir_suffix}{overlap_dir_suffix}"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("re10k", "dl3dv"):
        # DTU 분기는 view 선택에 쓴 rng를 셔플에도 이어서 쓰지만, re10k/dl3dv 분기는 view를
        # seed 기반 rng로 뽑지 않으므로(overlap summary의 candidate 재사용) 셔플 전용 rng가
        # 따로 필요하다.
        rng = np.random.default_rng(args.seed)

    train_cams = [build_camera_tensors(v, device, args.pose_scale_factor) for v in train_views]
    test_cams = [build_camera_tensors(v, device, args.pose_scale_factor) for v in test_views]
    height, width = train_views[0]["height"], train_views[0]["width"]

    # --- CUDA warm-up: 프로토콜(runtime.cuda_warmup=true)에 따라 최초 컴파일 시간은 측정에서 뺀다.
    _warmup_forward(params, train_cams[0], rasterization, height, width, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    trajectory: list[dict[str, object]] = []
    order = list(range(len(train_cams)))
    step = 0
    elapsed = 0.0
    active_sh_degree = args.initial_sh_degree
    next_snapshot_targets = sorted(set(args.budget_snapshots))
    snapshot_idx = 0
    # Dense-view sanity check처럼 "30k iteration에서 정확히 평가"해야 하는 경우를 위해
    # 시간 budget snapshot과 별도로 iteration snapshot도 지원한다.
    next_iteration_targets = sorted(set(args.iteration_snapshots))
    iteration_snapshot_idx = 0

    lpips_metric = LPIPSMetric(device) if args.compute_lpips else None

    print(
        f"[train] starting optimization, budget={args.max_budget_seconds}s, "
        f"init_points={params['means'].shape[0]} ({init_source})"
    )

    if init_source == "ff_warm_start":
        # C1-b refinement=off(0s) baseline: 방금 불러온 FF Gaussian을 최적화 스텝 없이
        # 그대로 평가한다. max_budget_seconds가 0이어도(=refinement off 조건) 이 한 줄이
        # 있어야 trajectory가 비어있지 않다 — while 루프는 elapsed(0.0) < 0.0이라 한 번도
        # 안 돈다.
        baseline_row = _evaluate_and_checkpoint(
            params=params,
            rasterization=rasterization,
            test_cams=test_cams,
            height=height,
            width=width,
            active_sh_degree=args.sh_degree,
            step=0,
            elapsed=0.0,
            train_loss=None,
            checkpoints_dir=checkpoints_dir,
            args=args,
            checkpoint_label="ff_warm_start_baseline",
            lpips_metric=lpips_metric,
            init_source=init_source,
        )
        trajectory.append(baseline_row)
        lpips_str = f"{baseline_row['test_lpips']:.3f}" if baseline_row["test_lpips"] is not None else "n/a"
        print(
            f"[ckpt] ff_warm_start_baseline (refinement=off 기준점) test_psnr={baseline_row['test_psnr']:.3f} "
            f"test_lpips={lpips_str}"
        )

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
            # elapsed는 이 스텝이 끝난 "후"에 확인하므로 budget_label을 살짝 넘긴 값이다.
            # protocol_utils.budget_checkpoint()는 wall_clock <= budget만 인정하므로,
            # 넘친 실측치 대신 budget_label로 clamp해 해당 budget의 유효 체크포인트로 기록한다.
            row = _evaluate_and_checkpoint(
                params=params,
                rasterization=rasterization,
                test_cams=test_cams,
                height=height,
                width=width,
                active_sh_degree=args.sh_degree,
                step=step,
                elapsed=min(elapsed, budget_label),
                train_loss=float(loss.item()),
                checkpoints_dir=checkpoints_dir,
                args=args,
                checkpoint_label=f"budget_{budget_label}s",
                lpips_metric=lpips_metric,
                init_source=init_source,
            )
            trajectory.append(row)
            lpips_str = f"{row['test_lpips']:.3f}" if row["test_lpips"] is not None else "n/a"
            print(
                f"[ckpt] budget={budget_label}s iter={step} elapsed={elapsed:.1f}s "
                f"gaussians={row['gaussian_count']} test_psnr={row['test_psnr']:.3f} "
                f"test_lpips={lpips_str} peak_vram_mb={row['peak_vram']:.0f}"
            )

        if iteration_snapshot_idx < len(next_iteration_targets) and step >= next_iteration_targets[iteration_snapshot_idx]:
            iteration_label = next_iteration_targets[iteration_snapshot_idx]
            iteration_snapshot_idx += 1
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
                checkpoint_label=f"iter_{iteration_label}",
                lpips_metric=lpips_metric,
                init_source=init_source,
            )
            row["iteration_snapshot"] = iteration_label
            trajectory.append(row)
            lpips_str = f"{row['test_lpips']:.3f}" if row["test_lpips"] is not None else "n/a"
            print(
                f"[ckpt] iter_snapshot={iteration_label} iter={step} elapsed={elapsed:.1f}s "
                f"gaussians={row['gaussian_count']} test_psnr={row['test_psnr']:.3f} "
                f"test_lpips={lpips_str} peak_vram_mb={row['peak_vram']:.0f}"
            )

        if step >= args.max_iterations:
            print(f"[train] hit max_iterations={args.max_iterations} before budget exhausted, stopping.")
            break

    # densification=off는 별도 log 파일로 남긴다 — 같은 scene/seed/view_count에서 on/off 두
    # 조건을 모두 돌리는 게 C1-b 설계 자체이므로, 파일명이 같으면 batch driver의
    # "log 있으면 skip" 로직이 두 번째 조건 실행을 막아버린다.
    densification_suffix = "" if args.densification == "on" else "_densoff"
    overlap_suffix = f"_{args.overlap_level}" if args.overlap_level else ""
    log_path = logs_dir / f"{args.scene}_{args.method}_{args.view_count}view_seed{args.seed}{densification_suffix}{overlap_suffix}.json"
    # 원자적 쓰기(temp + rename) — 배치 driver가 log_path.exists()로 "이미 끝남"을 판단해
    # 재실행을 건너뛰므로, write 도중 죽으면 잘린 JSON이 "완료"로 오인돼 다음 재개 시
    # json.loads()가 깨진 파일에서 죽는다. write_summary()(run_re10k_c1a_main.py)와 같은 패턴.
    tmp_path = log_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    tmp_path.replace(log_path)
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
    checkpoint_label,
    lpips_metric=None,
    init_source=None,
) -> dict[str, object]:
    with torch.no_grad():
        psnrs, ssims, lpipss = [], [], []
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
            if lpips_metric is not None:
                lpipss.append(lpips_metric(pred, gt_image))

    checkpoint_path = checkpoints_dir / f"{checkpoint_label}_iter{step}.pt"
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
        "test_lpips": float(np.mean(lpipss)) if lpipss else None,
        "gaussian_count": int(params["means"].shape[0]),
        "peak_vram": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "checkpoint_path": str(checkpoint_path),
        "init_source": init_source,
        "densification": args.densification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vanilla 3DGS (gsplat) runner for a single DTU scan.")
    parser.add_argument("--dataset", choices=["dtu", "re10k", "dl3dv"], default="dtu")
    parser.add_argument("--scan-dir", default=None, help="dataset=dtu일 때만 필요. e.g. /data/Re-feem/datasets/dtu/scan1")
    parser.add_argument("--scene", required=True, help="scene id used in logs, e.g. dtu_scan1")
    parser.add_argument(
        "--re10k-subset-index",
        default="experiments/outputs/re10k_main_subset/re10k_main_subset.json",
        help="dataset=re10k일 때만 사용. generate_re10k_main_subset.py 출력.",
    )
    parser.add_argument("--re10k-scene-key", default=None, help="dataset=re10k일 때 필요. re10k_main_subset.json의 scene key.")
    parser.add_argument(
        "--overlap-level", choices=["high", "low"], default=None,
        help="지정하면 --overlap-candidates-index에서 co-visibility selector가 만든 low/high 후보를 쓴다"
        "(view_selector.py, 2026-08-13). 미지정 시 --re10k-subset-index의 단일 후보 사용(기존 동작, 하위호환).",
    )
    parser.add_argument(
        "--overlap-candidates-index",
        default="experiments/outputs/re10k_overlap_candidates/re10k_overlap_candidates.json",
        help="--overlap-level 지정 시 사용. generate_re10k_overlap_candidates.py 출력.",
    )
    parser.add_argument(
        "--dl3dv-overlap-summary",
        default="experiments/outputs/dl3dv_overlap/all_scenes_summary.json",
        help="dataset=dl3dv일 때만 사용. generate_dl3dv_view_overlap.py 출력.",
    )
    parser.add_argument("--dl3dv-scene-key", default=None, help="dataset=dl3dv일 때 필요. DL3DV scene 폴더명(hash).")
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
    parser.add_argument(
        "--iteration-snapshots",
        type=int,
        nargs="+",
        default=[],
        help="Dense sanity check처럼 특정 iteration에서 평가/checkpoint를 남길 때 사용한다.",
    )
    parser.add_argument("--num-init-points", type=int, default=100_000, help="COLMAP triangulation이 실패할 때만 쓰는 random-fallback 점 개수.")
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="지정하면 dtu_dataset.resize_and_crop()으로 이 해상도에 맞춘다(MVSplat"
        " crop_shim과 동일 convention). FF checkpoint 학습 해상도를 상속해야 하는"
        " C1-b/warm-start에 필요. 기본(미지정)은 기존 동작 그대로 네이티브 해상도.",
    )
    parser.add_argument(
        "--densification",
        choices=["on", "off"],
        default="on",
        help="off면 DefaultStrategy의 refine_stop_iter=0으로 강제해 adaptive density control"
        "(grow/prune)을 전부 끈다. C1-b densification on/off ablation용 (§12.1).",
    )
    parser.add_argument("--refine-start-iter", type=int, default=500, help="densification=on일 때만 적용.")
    parser.add_argument("--refine-stop-iter", type=int, default=15_000, help="densification=on일 때만 적용.")
    parser.add_argument("--reset-every", type=int, default=3_000)
    parser.add_argument("--grow-grad2d", type=float, default=0.0002)
    parser.add_argument("--grow-scale3d", type=float, default=0.01)
    parser.add_argument("--prune-opa", type=float, default=0.005)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument(
        "--initial-sh-degree",
        type=int,
        default=0,
        help="training loop 시작 시 active_sh_degree. FF warm-start는 이미 최종 SH 계수를"
        " 갖고 있으므로 0에서부터 다시 ramp-up하면 안 된다 — warm-start 시 --sh-degree와"
        " 같은 값으로 준다(예: MVSplat sh_degree=4).",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        default=None,
        help="C1-b: mvsplat_runner.py 등이 저장한 gaussians.pt 경로. 주어지면 COLMAP/random"
        " init을 건너뛰고 이 Gaussian을 그대로 최적화 시작점으로 쓴다. §5.8 렌더 등가성"
        " gate(check_renderer_equivalence.py)를 먼저 통과한 체크포인트만 써야 한다.",
    )
    parser.add_argument(
        "--pose-scale-factor",
        type=float,
        default=1.0,
        help="카메라 translation에 곱할 배율. warm-start 소스가 좌표계를 스케일해서 썼다면"
        "(예: mvsplat_runner.py의 DTU_SCALE_FACTOR=1/200) 반드시 같은 값을 줘야 Gaussian과"
        " 카메라 좌표계가 맞는다. 기본 1.0은 기존 동작(원본 DTU 좌표계)과 동일.",
    )
    parser.add_argument(
        "--depth-cache-path",
        default=None,
        help="C2: analysis/precompute_depth_maps.py가 저장한 .pt 캐시 경로. 주어지면 COLMAP/random"
        " init을 건너뛰고 이 depth map을 교란(--depth-noise-sigma/--depth-scale-bias)한 뒤"
        " back-projection해서 초기값으로 쓴다. --warm-start-checkpoint와는 동시에 못 쓴다"
        " (둘 다 init 단계를 완전히 대체하는 경로라 상호배타적).",
    )
    parser.add_argument(
        "--depth-noise-sigma", type=float, default=0.0,
        help="C2 (a) iid 오차: d' = d(1+eps), eps~N(0,sigma^2). --depth-cache-path와 함께 쓴다."
        " 0.0이면 노이즈 없음(overall.md §5.9 sigma 후보: 0/0.01/0.03/0.05/0.10).",
    )
    parser.add_argument(
        "--depth-scale-bias", type=float, default=1.0,
        help="C2 (b) global scale bias: d' = s*d. --depth-cache-path와 함께 쓴다. 1.0이면 편향 없음"
        " (overall.md §5.9 s 후보: 0.9/0.95/1.0/1.05/1.1).",
    )
    parser.add_argument(
        "--depth-stride", type=int, default=4,
        help="depth back-projection 시 픽셀을 몇 칸 간격으로 샘플링할지(밀도 조절, COLMAP 수준 점"
        " 개수에 맞추기 위한 서브샘플).",
    )
    parser.add_argument(
        "--depth-max-points-per-view", type=int, default=50_000,
        help="view당 back-projection 점 상한(그 이상이면 무작위 서브샘플).",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "outputs"))
    parser.add_argument("--compute-lpips", action="store_true", default=True, help="AlexNet 기반 LPIPS도 함께 계산.")
    parser.add_argument("--no-lpips", dest="compute_lpips", action="store_false")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.budget_snapshots = [b for b in sorted(args.budget_snapshots) if b <= args.max_budget_seconds]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

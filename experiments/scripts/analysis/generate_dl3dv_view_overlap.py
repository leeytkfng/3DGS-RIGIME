#!/usr/bin/env python3
"""DL3DV pilot 25 scene에 2/4/8/12-view candidate + co-visibility overlap을 계산한다.

RE10K용 `generate_re10k_view_overlap.py`와 같은 패턴(known-pose COLMAP triangulation ->
zero-included overlap report)을 DL3DV로 이식한 것. RE10K와의 차이:
- MVSplat 같은 공식 2-view context/target index가 DL3DV엔 없다. 대신 DepthSplat이
  test-time에 실제로 쓰는 view selection 알고리즘을 코드로 확인해서(`view_sampler_bounded_v2.py`
  ::sample(), 2026-08-12) 그대로 재현한다 — §Ⅴ 참고.
- 이미지가 이미 디스크 파일(`images_8/frame_*.png`)이라 RE10K처럼 `.torch`에서 풀어낼
  필요가 없다.

## Ⅴ. DepthSplat test-time view selection 재현 (2026-08-12, v2)

최초 버전(v1, 2026-08-12 오전)은 전체 영상에서 순수 랜덤으로 context를 뽑았는데, 그 결과
4-view도 56%(14/25 scene)가 SfM overlap 0으로 나왔다. 원인을 코드로 추적한 결과, DepthSplat
자신의 test-time 샘플러는 전체 영상 랜덤이 아니라:
1. `index_context_left = 0`으로 고정(항상 scene 시작 프레임부터)
2. `context_gap = max_distance_between_context_views`로 고정 (우리 config 기준 50)
3. window `[0, context_gap]` 안에서 카메라 위치 기준 **farthest-point sampling**으로
   `num_context_views`개를 뽑는다(`farthest_point_sample()`, 원본 로직을 numpy로 재현)

이 스크립트는 이제 이 알고리즘을 재현한다. target(held-out 3-view)은 DepthSplat 원본처럼
context window 안의 전체 프레임을 다 쓰지 않고(우리 프로토콜은 고정 3-view 필요), 같은
window 안에서 context와 겹치지 않게 seed 기반으로 3개를 뽑는다(window이 너무 짧으면
window 밖까지 허용).

예상 결과:
- output_dir/<scene>/<N>view_seed0/{summary.json, visibility.json, pairwise_overlap.csv}
- output_dir/all_scenes_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from colmap_init import triangulate_sfm_points_from_cameras  # noqa: E402
from dl3dv_dataset import load_metadata, load_views  # noqa: E402
from protocol_utils import build_overlap_report  # noqa: E402

DL3DV_ROOT = Path("/data/Re-feem/datasets/dl3dv")
VIEW_COUNTS = [2, 4, 8, 12]
NUM_TARGET = 3
MIN_FRAMES = 60  # 12-view 후보를 뽑기에 너무 짧은 scene 제외
SEED = 0
MAX_CONTEXT_GAP = 50  # boundedv2_360.yaml의 max_distance_between_context_views(공식값)
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def list_scenes() -> list[str]:
    return sorted(p.name for p in DL3DV_ROOT.iterdir() if p.is_dir())


def camera_center(meta: dict, idx: int, undo_applied_transform: bool = True) -> np.ndarray:
    """c2w[:3,3]이 곧 world 좌표계에서의 카메라 위치다 — R,t로 분해할 필요 없음."""

    applied = np.eye(4)
    if undo_applied_transform and "applied_transform" in meta:
        applied[:3, :4] = np.array(meta["applied_transform"])
    frame = meta["frames"][idx]
    c2w = applied @ np.array(frame["transform_matrix"]) @ BLENDER_TO_OPENCV
    return c2w[:3, 3]


def farthest_point_sample(positions: np.ndarray, npoint: int) -> list[int]:
    """DepthSplat `view_sampler_bounded_v2.py::farthest_point_sample()`을 numpy/단일 배치로
    재현. centroid에서 가장 먼 점부터 시작해, 매번 이미 뽑힌 집합에서 가장 먼 점을 추가한다."""

    n = positions.shape[0]
    distance = np.full(n, 1e10)
    centroid = positions.mean(axis=0)
    farthest = int(np.argmax(np.sum((positions - centroid) ** 2, axis=1)))
    selected: list[int] = []
    for _ in range(npoint):
        selected.append(farthest)
        dist = np.sum((positions - positions[farthest]) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    return selected


def select_view_candidates(
    meta: dict, num_frames: int, view_count: int, seed: int = SEED
) -> tuple[list[int], list[int]] | tuple[None, None]:
    """DepthSplat test-time 알고리즘 재현: window=[0, MAX_CONTEXT_GAP] 안에서 farthest-point로
    context를 뽑고, 같은 window 안에서 겹치지 않게 target을 시드 기반으로 뽑는다."""

    window_right = min(MAX_CONTEXT_GAP, num_frames - 1)
    window_indices = list(range(window_right + 1))
    if view_count > len(window_indices):
        return None, None

    centers = np.stack([camera_center(meta, i) for i in window_indices])
    fps_local = farthest_point_sample(centers, view_count)
    context_indices = sorted(window_indices[i] for i in fps_local)

    rng = np.random.default_rng(seed)
    target_pool = [i for i in window_indices if i not in set(context_indices)]
    if len(target_pool) < NUM_TARGET:
        target_pool = [i for i in range(num_frames) if i not in set(context_indices)]
    target_indices = sorted(rng.choice(target_pool, size=min(NUM_TARGET, len(target_pool)), replace=False).tolist())
    return context_indices, target_indices


def visibility_from_reconstruction(model_dir: Path) -> dict[str, set[str]]:
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    visibility: dict[str, set[str]] = {}
    for image in reconstruction.images.values():
        visible = {str(int(p.point3D_id)) for p in image.points2D if p.point3D_id != -1}
        visibility[image.name] = visible
    return visibility


def write_pairwise_csv(pairwise: list[dict[str, object]], path: Path) -> None:
    import csv

    fieldnames = ["view_i", "view_j", "overlap", "shared_points", "points_i", "points_j", "connected"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairwise:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-counts", type=int, nargs="+", default=VIEW_COUNTS)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--output-dir", default="experiments/outputs/dl3dv_overlap")
    parser.add_argument("--colmap-work-dir", default="experiments/outputs/dl3dv_overlap/colmap_work")
    parser.add_argument("--min-common-points", type=int, default=1)
    args = parser.parse_args()

    scenes = args.scenes if args.scenes else list_scenes()
    output_root = Path(args.output_dir)
    colmap_root = Path(args.colmap_work_dir)

    all_summaries = []
    for scene in scenes:
        scene_dir = DL3DV_ROOT / scene
        meta = load_metadata(scene_dir)
        num_frames = len(meta["frames"])
        if num_frames < MIN_FRAMES:
            print(f"[skip] {scene}: num_frames={num_frames} < {MIN_FRAMES}")
            continue

        for view_count in args.view_counts:
            label = f"{view_count}view_seed{SEED}"
            context_indices, target_indices = select_view_candidates(meta, num_frames, view_count)
            if context_indices is None:
                print(f"[skip] {scene} {label}: window(0~{MAX_CONTEXT_GAP})보다 view_count가 큼")
                continue

            report_dir = output_root / scene / label
            workdir = colmap_root / scene / label
            image_dir = scene_dir / "images_8"

            views = load_views(scene_dir, meta, context_indices)
            image_names = [v["image_name"] for v in views]
            camera_by_name = {
                v["image_name"]: __import__("colmap_init").CameraParams(
                    K=v["K"], R=v["R"], t=v["t"], width=v["width"], height=v["height"]
                )
                for v in views
            }

            points, _colors = triangulate_sfm_points_from_cameras(image_dir, image_names, camera_by_name, workdir)
            model_dir = workdir / "sparse_triangulated"
            if points.shape[0] > 0 and (model_dir / "images.bin").exists():
                visibility = visibility_from_reconstruction(model_dir)
            else:
                visibility = {name: set() for name in image_names}

            report = build_overlap_report(visibility, min_common_points=args.min_common_points)
            report["metadata"] = {
                "scene": scene,
                "view_count_label": label,
                "min_common_points": args.min_common_points,
                "non_edge_policy": "zero_included",
                "primary_statistic": "mean_overlap",
                "source": "pycolmap sparse_triangulated image point tracks (DL3DV images_8)",
            }
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (report_dir / "visibility.json").write_text(
                json.dumps({k: sorted(v) for k, v in visibility.items()}, indent=2), encoding="utf-8"
            )
            write_pairwise_csv(report["pairwise"], report_dir / "pairwise_overlap.csv")

            summary = report["summary"]
            row = {
                "scene": scene,
                "view_count": view_count,
                "context_indices": context_indices,
                "target_indices": target_indices,
                "sfm_points": int(points.shape[0]),
                "mean_overlap": summary["mean_overlap"],
                "q25_overlap": summary["q25_overlap"],
                "zero_pair_ratio": summary["zero_pair_ratio"],
                "has_isolated_view": summary["has_isolated_view"],
            }
            all_summaries.append(row)
            print(
                f"[{scene}][{view_count}view] sfm_points={row['sfm_points']} "
                f"mean_overlap={row['mean_overlap']:.4f} zero_pair_ratio={row['zero_pair_ratio']:.3f}"
            )

    summary_path = output_root / "all_scenes_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"[done] {len(all_summaries)} rows written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""DL3DV pilot 25 scene에 2/4/8/12-view candidate + co-visibility overlap을 계산한다.

RE10K용 `generate_re10k_view_overlap.py`와 같은 패턴(known-pose COLMAP triangulation ->
zero-included overlap report)을 DL3DV로 이식한 것. RE10K와의 차이:
- MVSplat 같은 공식 2-view context/target index가 DL3DV엔 없다(DepthSplat은 애초에
  4-view 위주로 학습됨, `boundedv2_360.yaml` 기본 num_context_views=4). 그래서 DTU smoke와
  같은 방식(seed 기반 rng)으로 view candidate를 직접 뽑는다.
- 이미지가 이미 디스크 파일(`images_8/frame_*.png`)이라 RE10K처럼 `.torch`에서 풀어낼
  필요가 없다.

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


def list_scenes() -> list[str]:
    return sorted(p.name for p in DL3DV_ROOT.iterdir() if p.is_dir())


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

        rng = np.random.default_rng(SEED)
        # 비디오 프레임이라 인접 프레임끼리는 사실상 동일 view다 - 전체 범위에서 목표(target)
        # 3장을 먼저 고정하고, 나머지 pool에서 context를 뽑는다(RE10K와 동일한 원칙).
        target_indices = sorted(rng.choice(num_frames, size=NUM_TARGET, replace=False).tolist())
        pool = [i for i in range(num_frames) if i not in set(target_indices)]

        for view_count in args.view_counts:
            label = f"{view_count}view_seed{SEED}"
            if view_count > len(pool):
                print(f"[skip] {scene} {label}: pool보다 view_count가 큼")
                continue
            context_indices = sorted(rng.choice(pool, size=view_count, replace=False).tolist())

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

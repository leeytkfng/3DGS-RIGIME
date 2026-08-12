#!/usr/bin/env python3
"""RE10K main subset의 2/4/8/12-view candidate에 co-visibility overlap을 계산한다.

DTU용 `generate_dtu_view_overlap_smoke.py`와 같은 패턴(known-pose COLMAP triangulation ->
zero-included overlap report)을 RE10K에 이식한 것. 차이는 입력뿐이다:
- view 후보는 새로 뽑지 않고 `generate_re10k_main_subset.py`가 만든
  `re10k_main_subset.json`의 context view를 그대로 쓴다(재현성 유지, 두 스크립트가
  서로 다른 view를 뽑는 사고를 방지).
- 이미지가 디스크 파일이 아니라 `.torch` chunk 안에 있으므로, scene마다 필요한 context
  프레임만 임시 디렉토리에 풀어서 COLMAP에 넘긴다(re10k_dataset.extract_frames_to_disk).

예상 결과:
- output_dir/<scene>/<N>view_seed0/{summary.json, visibility.json, pairwise_overlap.csv}
- output_dir/all_scenes_summary.json (scene x view_count 전체 요약, low/high overlap bucket
  경계를 나중에 정할 때 쓸 원본 데이터)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from colmap_init import triangulate_sfm_points_from_cameras  # noqa: E402
from protocol_utils import build_overlap_report  # noqa: E402
from re10k_dataset import extract_frames_to_disk, get_scene_item  # noqa: E402

RE10K_ROOT = Path("/data/Re-feem/datasets/re10k/test")


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


def process_scene_view_count(
    scene_key: str,
    chunk_file: str,
    context_frames: list[int],
    output_root: Path,
    colmap_root: Path,
    view_count: int,
    min_common_points: int,
) -> dict[str, object]:
    label = f"{view_count}view_seed0"
    report_dir = output_root / scene_key / label
    image_dir = colmap_root / scene_key / label / "images"
    workdir = colmap_root / scene_key / label

    item = get_scene_item(RE10K_ROOT / chunk_file, scene_key)
    camera_by_name = extract_frames_to_disk(item, context_frames, image_dir)
    image_names = sorted(camera_by_name.keys())

    points, _colors = triangulate_sfm_points_from_cameras(image_dir, image_names, camera_by_name, workdir)
    model_dir = workdir / "sparse_triangulated"
    if points.shape[0] > 0 and (model_dir / "images.bin").exists():
        visibility = visibility_from_reconstruction(model_dir)
    else:
        visibility = {name: set() for name in image_names}

    report = build_overlap_report(visibility, min_common_points=min_common_points)
    report["metadata"] = {
        "scene": scene_key,
        "view_count_label": label,
        "min_common_points": min_common_points,
        "non_edge_policy": "zero_included",
        "primary_statistic": "mean_overlap",
        "source": "pycolmap sparse_triangulated image point tracks (RE10K context frames)",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    serializable_visibility = {k: sorted(v) for k, v in visibility.items()}
    (report_dir / "visibility.json").write_text(json.dumps(serializable_visibility, indent=2), encoding="utf-8")
    write_pairwise_csv(report["pairwise"], report_dir / "pairwise_overlap.csv")

    summary = report["summary"]
    return {
        "scene": scene_key,
        "view_count": view_count,
        "context_frames": context_frames,
        "sfm_points": int(points.shape[0]),
        "mean_overlap": summary["mean_overlap"],
        "q25_overlap": summary["q25_overlap"],
        "zero_pair_ratio": summary["zero_pair_ratio"],
        "has_isolated_view": summary["has_isolated_view"],
        "report_dir": str(report_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute co-visibility overlap for RE10K main-subset view candidates.")
    parser.add_argument("--subset-index", default="experiments/outputs/re10k_main_subset/re10k_main_subset.json")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 subset index의 전체 scene을 돈다.")
    parser.add_argument("--output-dir", default="experiments/outputs/re10k_main_subset/overlap")
    parser.add_argument("--colmap-work-dir", default="experiments/outputs/re10k_main_subset/colmap_work")
    parser.add_argument("--min-common-points", type=int, default=1)
    args = parser.parse_args()

    subset = json.loads(Path(args.subset_index).read_text())
    scenes = args.scenes if args.scenes else list(subset.keys())
    output_root = Path(args.output_dir)
    colmap_root = Path(args.colmap_work_dir)

    all_summaries = []
    for scene_key in scenes:
        entry = subset[scene_key]
        for view_count in args.view_counts:
            cand = entry["view_candidates"][str(view_count)]
            if cand.get("context") is None:
                print(f"[skip] {scene_key} view_count={view_count}: candidate 생성 불가(too short)")
                continue
            row = process_scene_view_count(
                scene_key=scene_key,
                chunk_file=entry["chunk_file"],
                context_frames=cand["context"],
                output_root=output_root,
                colmap_root=colmap_root,
                view_count=view_count,
                min_common_points=args.min_common_points,
            )
            all_summaries.append(row)
            print(
                f"[{scene_key}][{view_count}view] frames={row['context_frames']} "
                f"sfm_points={row['sfm_points']} mean_overlap={row['mean_overlap']:.4f} "
                f"zero_pair_ratio={row['zero_pair_ratio']:.3f}"
            )

    summary_path = output_root / "all_scenes_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"[done] {len(all_summaries)} scene x view_count rows written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

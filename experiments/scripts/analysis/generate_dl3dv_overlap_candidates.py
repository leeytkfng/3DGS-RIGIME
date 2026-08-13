#!/usr/bin/env python3
"""DL3DV에 co-visibility selector(`core/view_selector.py`)를 적용해 scene별/view_count별
high/low overlap 후보를 만들고 실제 overlap을 측정해 방향이 맞는지 검증한다.

RE10K용 `generate_re10k_overlap_candidates.py`와 같은 목적/패턴. **기존
`dl3dv_overlap_v2/`는 건드리지 않는다** — DepthSplat 실제 알고리즘(좁은 window FPS)을
그대로 재현한 기존 "high-ish" 단일 후보이자 이미 여러 실측(C1-b scaleup 등)이 참조하는
데이터라, 별도 출력(`dl3dv_overlap_lowhigh/`)으로 남긴다.
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
from view_selector import select_overlap_candidates  # noqa: E402

DL3DV_ROOT = Path("/data/Re-feem/datasets/dl3dv")
VIEW_COUNTS = [2, 4, 8, 12]
NUM_TARGET = 3
MIN_FRAMES = 60
SEED = 0
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def camera_center(meta: dict, idx: int) -> np.ndarray:
    applied = np.eye(4)
    if "applied_transform" in meta:
        applied[:3, :4] = np.array(meta["applied_transform"])
    frame = meta["frames"][idx]
    c2w = applied @ np.array(frame["transform_matrix"]) @ BLENDER_TO_OPENCV
    return c2w[:3, 3]


def visibility_from_reconstruction(model_dir: Path) -> dict[str, set[str]]:
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    return {
        image.name: {str(int(p.point3D_id)) for p in image.points2D if p.point3D_id != -1}
        for image in reconstruction.images.values()
    }


def measure_overlap(scene_dir: Path, meta: dict, context_indices: list[int], workdir: Path) -> dict:
    views = load_views(scene_dir, meta, context_indices)
    image_names = [v["image_name"] for v in views]
    from colmap_init import CameraParams

    camera_by_name = {
        v["image_name"]: CameraParams(K=v["K"], R=v["R"], t=v["t"], width=v["width"], height=v["height"])
        for v in views
    }
    image_dir = scene_dir / "images_8"
    points, _ = triangulate_sfm_points_from_cameras(image_dir, image_names, camera_by_name, workdir)
    model_dir = workdir / "sparse_triangulated"
    if points.shape[0] > 0 and (model_dir / "images.bin").exists():
        visibility = visibility_from_reconstruction(model_dir)
    else:
        visibility = {name: set() for name in image_names}
    report = build_overlap_report(visibility, min_common_points=1)
    return {"sfm_points": int(points.shape[0]), **report["summary"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 처음 3개 scene(파일럿 검증).")
    parser.add_argument("--view-counts", type=int, nargs="+", default=VIEW_COUNTS)
    parser.add_argument("--output", default="experiments/outputs/dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json")
    parser.add_argument("--workdir", default="experiments/outputs/dl3dv_overlap_lowhigh/colmap_work")
    args = parser.parse_args()

    scenes = args.scenes or sorted(p.name for p in DL3DV_ROOT.iterdir() if p.is_dir())[:3]

    output: dict[str, dict] = {}
    for scene in scenes:
        scene_dir = DL3DV_ROOT / scene
        meta = load_metadata(scene_dir)
        num_frames = len(meta["frames"])
        if num_frames < MIN_FRAMES:
            print(f"[skip] {scene}: num_frames={num_frames} < {MIN_FRAMES}")
            continue

        all_indices = list(range(num_frames))
        positions = np.stack([camera_center(meta, i) for i in all_indices])

        output[scene] = {}
        for view_count in args.view_counts:
            rng = np.random.default_rng(SEED)
            candidates = select_overlap_candidates(all_indices, positions, view_count, seed=SEED)
            row = {}
            for level, context in candidates.items():
                target_pool = [i for i in all_indices if i not in set(context)]
                target = sorted(rng.choice(target_pool, size=min(NUM_TARGET, len(target_pool)), replace=False).tolist())
                workdir = Path(args.workdir) / scene / f"{view_count}view_{level}"
                measured = measure_overlap(scene_dir, meta, context, workdir)
                row[level] = {"context": context, "target": target, **measured}
                print(
                    f"[{scene}][{view_count}view][{level}] context={context} "
                    f"mean_overlap={measured['mean_overlap']:.4f} sfm_points={measured['sfm_points']}"
                )
            output[scene][str(view_count)] = row

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")

    print("\n=== high vs low overlap 방향 검증 ===")
    n_correct, n_total = 0, 0
    for scene, by_view in output.items():
        for view_count, row in by_view.items():
            if "high" not in row or "low" not in row:
                continue
            n_total += 1
            ok = row["high"]["mean_overlap"] >= row["low"]["mean_overlap"]
            n_correct += int(ok)
            print(
                f"{scene} {view_count}view: high={row['high']['mean_overlap']:.4f} "
                f"low={row['low']['mean_overlap']:.4f} {'OK' if ok else 'FAIL(방향 반대)'}"
            )
    print(f"\n{n_correct}/{n_total} 조건에서 high >= low 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

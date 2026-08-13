#!/usr/bin/env python3
"""RE10K에 co-visibility selector(`core/view_selector.py`)를 적용해 scene별/view_count별
high/low overlap 후보를 만들고 실제 overlap을 측정해 방향이 맞는지 검증한다.

**기존 `re10k_main_subset.json`은 건드리지 않는다** — 이번 세션에 이미 쌓인 실측 결과(C1-a
seed×3 파일럿 등)가 그 파일의 기존(단일) 후보를 참조하므로, 여기서는 별도 출력 파일
(`re10k_overlap_candidates.json`)로 남긴다. scene 목록/frame 수/공식 target은 기존 파일에서
그대로 가져와 재사용(같은 scene 정의를 공유).

파일럿 검증: scene 몇 개만 돌려 "high 후보의 mean_overlap이 low 후보보다 실제로 크다"는
selector의 핵심 전제를 실측으로 확인한다. 전체 20-scene 적용은 30-scene 본 실험 착수 시점에.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from colmap_init import CameraParams, triangulate_sfm_points_from_cameras  # noqa: E402
from protocol_utils import build_overlap_report  # noqa: E402
from re10k_dataset import get_scene_item, load_chunk  # noqa: E402
from view_selector import select_overlap_candidates  # noqa: E402

RE10K_ROOT = Path("/data/Re-feem/datasets/re10k/test")
VIEW_COUNTS = [2, 4, 8, 12]
SEED = 0


def camera_centers(item: dict, frame_indices: list[int]) -> np.ndarray:
    """이미지 디코딩 없이 pose만 읽어 camera center를 뽑는다 — selector가 전체 pool을 훑어야
    해서 `load_views()`로 이미지까지 다 읽으면 느리다."""

    centers = []
    for idx in frame_indices:
        pose = item["cameras"][idx].numpy()
        w2c = pose[6:18].reshape(3, 4)
        R, t = w2c[:, :3], w2c[:, 3]
        centers.append(-R.T @ t)
    return np.stack(centers)


def visibility_from_reconstruction(model_dir: Path) -> dict[str, set[str]]:
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    return {
        image.name: {str(int(p.point3D_id)) for p in image.points2D if p.point3D_id != -1}
        for image in reconstruction.images.values()
    }


def measure_overlap(item: dict, context_indices: list[int], workdir: Path) -> dict:
    from PIL import Image
    from io import BytesIO

    image_dir = workdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_names, camera_by_name = [], {}
    for idx in context_indices:
        name = f"{idx:04d}.png"
        raw = item["images"][idx]
        pil_image = Image.open(BytesIO(raw.numpy().tobytes())).convert("RGB")
        pil_image.save(image_dir / name)
        width, height = pil_image.size
        pose = item["cameras"][idx].numpy()
        fx, fy, cx, cy = pose[:4]
        K = np.array([[fx * width, 0, cx * width], [0, fy * height, cy * height], [0, 0, 1]])
        w2c = pose[6:18].reshape(3, 4)
        image_names.append(name)
        camera_by_name[name] = CameraParams(K=K, R=w2c[:, :3], t=w2c[:, 3], width=width, height=height)

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
    parser.add_argument("--subset-index", default="experiments/outputs/re10k_main_subset/re10k_main_subset.json")
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 subset 처음 3개 scene(파일럿 검증).")
    parser.add_argument("--view-counts", type=int, nargs="+", default=VIEW_COUNTS)
    parser.add_argument("--output", default="experiments/outputs/re10k_overlap_candidates/re10k_overlap_candidates.json")
    parser.add_argument("--workdir", default="experiments/outputs/re10k_overlap_candidates/colmap_work")
    args = parser.parse_args()

    subset = json.loads(Path(args.subset_index).read_text())
    scene_keys = args.scenes or list(subset.keys())[:3]

    output: dict[str, dict] = {}
    for scene_key in scene_keys:
        entry = subset[scene_key]
        item = get_scene_item(RE10K_ROOT / entry["chunk_file"], scene_key)
        num_frames = entry["num_frames"]
        target = sorted(entry["view_candidates"]["2"]["target"])
        pool_indices = [i for i in range(num_frames) if i not in set(target)]
        positions = camera_centers(item, pool_indices)

        output[scene_key] = {}
        for view_count in args.view_counts:
            if view_count > len(pool_indices):
                print(f"[skip] {scene_key} view_count={view_count}: pool({len(pool_indices)}) too small")
                continue
            candidates = select_overlap_candidates(pool_indices, positions, view_count, seed=SEED)
            row = {"target": target}
            for level, context in candidates.items():
                workdir = Path(args.workdir) / scene_key / f"{view_count}view_{level}"
                measured = measure_overlap(item, context, workdir)
                row[level] = {"context": context, **measured}
                print(
                    f"[{scene_key}][{view_count}view][{level}] context={context} "
                    f"mean_overlap={measured['mean_overlap']:.4f} sfm_points={measured['sfm_points']}"
                )
            output[scene_key][str(view_count)] = row

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")

    print("\n=== high vs low overlap 방향 검증 ===")
    n_correct, n_total = 0, 0
    for scene_key, by_view in output.items():
        for view_count, row in by_view.items():
            if "high" not in row or "low" not in row:
                continue
            n_total += 1
            ok = row["high"]["mean_overlap"] >= row["low"]["mean_overlap"]
            n_correct += int(ok)
            print(
                f"{scene_key} {view_count}view: high={row['high']['mean_overlap']:.4f} "
                f"low={row['low']['mean_overlap']:.4f} {'OK' if ok else 'FAIL(방향 반대)'}"
            )
    print(f"\n{n_correct}/{n_total} 조건에서 high >= low 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""RE10K main benchmark용 20-scene subset + 2/4/8/12-view candidate index를 만든다.

목적:
- 지금까지 확보한 RE10K test 114 scene 중 config.protocol.scenes_primary(20)개를
  "본 실험 main subset"으로 고정한다. DTU 공식 split처럼 임의 선택이 아니라, MVSplat/
  DepthSplat이 자기 평가에 실제로 쓰는 공식 evaluation index
  (`assets/evaluation_index_re10k.json`, pixelSplat 계열 표준)와 우리 로컬 114 scene의
  교집합에서 뽑는다.
- 2-view는 그 공식 index의 context/target을 그대로 쓴다(재현 가능성 최댓값).
- 4/8/12-view는 공식 index가 2-view만 제공하므로, DTU smoke 스크립트와 같은 방식(seeded rng)
  으로 우리가 직접 만든다. target(held-out test 3-view)은 view 수와 무관하게 고정해
  DTU runner의 "test split은 view_count와 무관하게 고정"이라는 원칙을 그대로 따른다.

출력: experiments/outputs/re10k_main_subset/re10k_main_subset.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RE10K_ROOT = Path("/data/Re-feem/datasets/re10k/test")
MVSPLAT_EVAL_INDEX = Path("/data/Re-feem/code/mvsplat/assets/evaluation_index_re10k.json")
VIEW_COUNTS = [2, 4, 8, 12]
NUM_SCENES = 20
MIN_FRAMES = 50  # 12-view 후보를 뽑기에 너무 짧은 scene(예: 11 frame짜리)은 제외
SEED = 0


def load_frame_counts(local_index: dict[str, str]) -> dict[str, int]:
    """각 scene의 실제 frame(=timestamp) 개수를 .torch chunk에서 직접 읽는다."""

    chunk_cache: dict[str, list[dict]] = {}
    frame_counts: dict[str, int] = {}
    for key, chunk_file in local_index.items():
        if chunk_file not in chunk_cache:
            chunk_cache[chunk_file] = torch.load(RE10K_ROOT / chunk_file, weights_only=False)
        items_by_key = {item["key"]: item for item in chunk_cache[chunk_file]}
        frame_counts[key] = len(items_by_key[key]["timestamps"])
    return frame_counts


def select_main_scenes(local_index: dict[str, str], official_index: dict) -> list[str]:
    """로컬 확보 scene ∩ 공식 index(non-null) ∩ frame 수 조건을 만족하는 scene 중
    20개를 seed=0으로 결정론적으로 뽑는다."""

    frame_counts = load_frame_counts(local_index)

    def is_leakage_free(entry: dict) -> bool:
        # 공식 index 자체에 context와 target이 겹치는 scene이 2개 있었다(실측 확인,
        # 2026-08-12: aadc1e2dc74fd644, cdf439b17a6a98d4). MVSplat 원 프로토콜에서는
        # 허용되는 듯하지만, 우리 §5.7 test-leakage 방지 원칙과 충돌하므로 main subset에서 제외한다.
        return not (set(entry["context"]) & set(entry["target"]))

    usable = sorted(
        key
        for key in local_index
        if key in official_index
        and official_index[key] is not None
        and frame_counts[key] >= MIN_FRAMES
        and is_leakage_free(official_index[key])
    )
    print(f"[select] local scenes={len(local_index)}, usable(공식 index + frame>={MIN_FRAMES} + context/target 겹침없음)={len(usable)}")

    rng = np.random.default_rng(SEED)
    selected = sorted(rng.choice(usable, size=min(NUM_SCENES, len(usable)), replace=False).tolist())
    return selected, frame_counts


def build_view_candidates(
    scene_key: str,
    num_frames: int,
    official_entry: dict,
    rng: np.random.Generator,
) -> dict[int, dict[str, list[int]]]:
    """view_count별 context/target 후보를 만든다.

    - target(held-out test 3-view)은 공식 index의 target을 그대로 쓰고, view_count와
      무관하게 고정한다(§5.7 test leakage 방지 원칙과 동일한 이유 — 조건마다 test set이
      바뀌면 조건 간 비교가 오염된다).
    - 2-view는 공식 context를 그대로 쓴다.
    - 4/8/12-view는 target을 제외한 나머지 frame pool에서 seeded sampling으로 뽑는다.
    """

    target = sorted(official_entry["target"])
    pool = [i for i in range(num_frames) if i not in set(target)]

    candidates: dict[int, dict[str, list[int]]] = {}
    for view_count in VIEW_COUNTS:
        if view_count == 2:
            context = sorted(official_entry["context"])
        else:
            if view_count > len(pool):
                candidates[view_count] = {"context": None, "target": target, "note": "pool보다 view_count가 커서 생성 불가"}
                continue
            context = sorted(rng.choice(pool, size=view_count, replace=False).tolist())
        candidates[view_count] = {"context": context, "target": target}
    return candidates


def main() -> int:
    local_index = json.loads((RE10K_ROOT / "index.json").read_text())
    official_index = json.loads(MVSPLAT_EVAL_INDEX.read_text())

    selected, frame_counts = select_main_scenes(local_index, official_index)
    print(f"[select] chosen {len(selected)} scenes (seed={SEED}): {selected}")

    rng = np.random.default_rng(SEED)  # 4/8/12-view 후보 생성용 — scene을 정렬된 순서로 순회하며 재사용
    output = {}
    for scene_key in selected:
        num_frames = frame_counts[scene_key]
        official_entry = official_index[scene_key]
        candidates = build_view_candidates(scene_key, num_frames, official_entry, rng)
        output[scene_key] = {
            "chunk_file": local_index[scene_key],
            "num_frames": num_frames,
            "official_context": official_entry["context"],
            "official_target": official_entry["target"],
            "view_candidates": candidates,
        }

    output_dir = REPO_ROOT / "experiments/outputs/re10k_main_subset"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "re10k_main_subset.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"[done] wrote {len(output)} scenes to {output_path}")
    for view_count in VIEW_COUNTS:
        missing = [k for k, v in output.items() if v["view_candidates"][view_count].get("context") is None]
        print(f"[check] view_count={view_count}: {len(output) - len(missing)}/{len(output)} scenes have valid candidates"
              + (f" (missing: {missing})" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

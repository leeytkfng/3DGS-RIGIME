#!/usr/bin/env python3
"""RE10K main subset을 20 -> 30 scene으로 **추가(additive) 확장**한다 (2026-08-13, seed 3->2/
scene 20->30 grid 결정 반영, overall.md §5.4).

기존 20개 scene은 완전히 그대로 둔다(같은 view_candidates, 같은 순서) — 이번 세션에 이미
쌓인 파일럿 결과(C1-a seed×3 등)가 이 20개 중 일부를 직접 참조하므로, `NUM_SCENES`를
20->30으로 올려 `generate_re10k_main_subset.py`를 재실행하면 numpy `rng.choice`가 다른
size 인자에서 prefix-stability를 보장하지 않아 **완전히 다른 20개**가 나올 위험이 있다.
그래서 여기서는 기존 20개를 뺀 나머지 "usable" pool에서 10개를 새로 뽑아 **추가만** 한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_re10k_main_subset import (  # noqa: E402
    MIN_FRAMES,
    MVSPLAT_EVAL_INDEX,
    RE10K_ROOT,
    build_view_candidates,
    load_frame_counts,
)

EXTEND_SEED = 1  # 원래 20개는 SEED=0으로 뽑았다 — 다른 seed로 겹치지 않게 10개를 새로 뽑는다.
NUM_NEW = 10


def main() -> int:
    subset_path = Path("experiments/outputs/re10k_main_subset/re10k_main_subset.json")
    subset = json.loads(subset_path.read_text())
    existing_keys = set(subset.keys())
    print(f"[extend] 기존 {len(existing_keys)} scene 유지: {sorted(existing_keys)}")

    local_index = json.loads((RE10K_ROOT / "index.json").read_text())
    official_index = json.loads(MVSPLAT_EVAL_INDEX.read_text())
    frame_counts = load_frame_counts(local_index)

    def is_leakage_free(entry: dict) -> bool:
        return not (set(entry["context"]) & set(entry["target"]))

    usable = sorted(
        key
        for key in local_index
        if key in official_index
        and official_index[key] is not None
        and frame_counts[key] >= MIN_FRAMES
        and is_leakage_free(official_index[key])
    )
    remaining = sorted(set(usable) - existing_keys)
    print(f"[extend] usable={len(usable)}, 기존 제외 후 남은 후보={len(remaining)}")

    rng = np.random.default_rng(EXTEND_SEED)
    new_scenes = sorted(rng.choice(remaining, size=min(NUM_NEW, len(remaining)), replace=False).tolist())
    print(f"[extend] 새로 뽑은 {len(new_scenes)} scene: {new_scenes}")

    rng2 = np.random.default_rng(EXTEND_SEED)  # build_view_candidates 내부 4/8/12-view 샘플링용
    for scene_key in new_scenes:
        num_frames = frame_counts[scene_key]
        official_entry = official_index[scene_key]
        candidates = build_view_candidates(scene_key, num_frames, official_entry, rng2)
        subset[scene_key] = {
            "chunk_file": local_index[scene_key],
            "num_frames": num_frames,
            "official_context": official_entry["context"],
            "official_target": official_entry["target"],
            "view_candidates": candidates,
        }

    subset_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    print(f"[done] {subset_path}: {len(subset)} scene total ({len(existing_keys)} 기존 + {len(new_scenes)} 신규)")

    subset = json.loads(subset_path.read_text())  # 검증은 실제로 디스크에 쓰인(=string key) 형태로 다시 읽어서 한다.
    for view_count in [2, 4, 8, 12]:
        missing = [k for k in new_scenes if subset[k]["view_candidates"][str(view_count)].get("context") is None]
        print(f"[check] view_count={view_count}: 신규 {len(new_scenes) - len(missing)}/{len(new_scenes)} 유효"
              + (f" (missing: {missing})" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""재실행 0 분석 두 개를 한 스크립트에 담는다 — 둘 다 이미 저장된 C1-a 로그만 읽는다.

1. **지표 불일치(PCC) 분석**: (scene, view_count, budget) 셀마다 방법 쌍의 승자를 PSNR
   기준/LPIPS 기준(가능하면 SSIM 기준도)으로 각각 정하고, 서로 다른 지표가 다른 승자를
   가리키는 "지표 불일치" 빈도와 그게 몰리는 regime(주로 view_count)을 찾는다. 승자 자체의
   binary 일치/불일치 대신, ΔPSNR과 ΔLPIPS(부호 반전, LPIPS는 낮을수록 좋음) 사이의 Pearson
   상관계수(PCC)를 셀 전체에 대해 계산해 "두 지표가 전반적으로 같은 방향을 가리키는지"도 함께
   본다. FF 방법(MVSplat/DepthSplat)은 test_ssim을 저장하지 않으므로, SSIM 비교는 양쪽 다
   SSIM이 있는 FSGS-vs-Vanilla3DGS 쌍에서만 한다.

2. **Gaussian 수 / VRAM Pareto**: 모든 방법·모든 로그에 이미 저장된 gaussian_count/peak_vram을
   test_psnr과 함께 뽑아 "품질 대비 온디바이스 비용"을 보여주는 Pareto 자료를 만든다(로그가
   방법마다 다른 스키마라 통일해서 뽑는 게 이 스크립트의 실질적 작업).

입력: `experiments/outputs/{re10k,dl3dv}_c1a_main/{logs,vanilla_runs,fsgs_runs}/**/*.json`
(이미 완료된 C1-a 결과, 재실행 없음).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from protocol_utils import scene_cluster_bootstrap_ci  # noqa: E402

BUDGETS = [1.0, 10.0, 60.0, 300.0]
FF_METHOD = {"RE10K": "MVSplat", "DL3DV": "DepthSplat"}

TRAJ_NAME_RE = re.compile(r"_(Vanilla3DGS|FSGS)_(\d+)view_seed(\d+)\.json$")
FF_NAME_RE = re.compile(r"^(?P<scene>.+)_(?P<method>MVSplat|DepthSplat)_(?P<vc>\d+)view\.json$")


def load_ff_rows(root: Path) -> list[dict]:
    """`logs/*.json`(FF, 단일 dict, budget 없음 -> 4 budget에 동일 값 복제)."""

    rows = []
    for path in (root / "logs").glob("*.json"):
        m = FF_NAME_RE.match(path.name)
        if not m:
            continue
        d = json.loads(path.read_text())
        for budget in BUDGETS:
            rows.append(
                {
                    "scene": m.group("scene"),
                    "view_count": int(m.group("vc")),
                    "budget": budget,
                    "method": m.group("method"),
                    "test_psnr": d.get("test_psnr"),
                    "test_ssim": None,
                    "test_lpips": d.get("test_lpips"),
                    "gaussian_count": d.get("gaussian_count"),
                    "peak_vram": d.get("peak_vram"),
                }
            )
    return rows


def load_trajectory_rows(root: Path, subdir: str) -> list[dict]:
    """`{vanilla,fsgs}_runs/*/logs/*.json`(체크포인트 리스트, seed 평균은 호출측에서)."""

    rows = []
    for scene_dir in (root / subdir).glob("*"):
        for path in (scene_dir / "logs").glob("*.json"):
            m = TRAJ_NAME_RE.search(path.name)
            if not m:
                continue
            method, view_count, seed = m.group(1), int(m.group(2)), int(m.group(3))
            trajectory = json.loads(path.read_text())
            by_wall_clock = {r["wall_clock"]: r for r in trajectory}
            for budget in BUDGETS:
                r = by_wall_clock.get(budget)
                if r is None:
                    continue
                rows.append(
                    {
                        "scene": scene_dir.name,
                        "view_count": view_count,
                        "budget": budget,
                        "method": method,
                        "seed": seed,
                        "test_psnr": r.get("test_psnr"),
                        "test_ssim": r.get("test_ssim"),
                        "test_lpips": r.get("test_lpips"),
                        "gaussian_count": r.get("gaussian_count"),
                        "peak_vram": r.get("peak_vram"),
                    }
                )
    return rows


def average_seeds(rows: list[dict]) -> list[dict]:
    """FSGS/Vanilla3DGS는 seed 0/1을 (scene,view_count,budget,method) 단위로 먼저 평균."""

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    out = []
    for r in rows:
        if "seed" not in r:
            out.append(r)
            continue
        key = (r["scene"], r["view_count"], r["budget"], r["method"])
        grouped[key].append(r)
    for (scene, vc, budget, method), group in grouped.items():
        def avg(field):
            vals = [g[field] for g in group if g.get(field) is not None]
            return float(np.mean(vals)) if vals else None

        out.append(
            {
                "scene": scene, "view_count": vc, "budget": budget, "method": method,
                "test_psnr": avg("test_psnr"), "test_ssim": avg("test_ssim"), "test_lpips": avg("test_lpips"),
                "gaussian_count": avg("gaussian_count"), "peak_vram": avg("peak_vram"),
            }
        )
    return out


def load_dataset(dataset: str) -> list[dict]:
    root = Path(f"experiments/outputs/{dataset.lower()}_c1a_main")
    rows = load_ff_rows(root)
    rows += average_seeds(load_trajectory_rows(root, "vanilla_runs"))
    rows += average_seeds(load_trajectory_rows(root, "fsgs_runs"))
    for r in rows:
        r["dataset"] = dataset
    return rows


# ---------------------------------------------------------------------------
# 1. 지표 불일치(PCC) 분석
# ---------------------------------------------------------------------------

PAIRS = [("Vanilla3DGS", "FF"), ("FSGS", "FF"), ("FSGS", "Vanilla3DGS")]


def pearson(x: list[float], y: list[float]) -> tuple[float, int]:
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), int(len(x))
    return float(np.corrcoef(x, y)[0, 1]), int(len(x))


def metric_disagreement(rows: list[dict], dataset: str) -> None:
    ff_method = FF_METHOD[dataset]
    by_key: dict[tuple, dict] = {}
    for r in rows:
        method = "FF" if r["method"] == ff_method else r["method"]
        by_key[(r["scene"], r["view_count"], r["budget"], method)] = r

    print(f"\n=== {dataset}: 지표 불일치(PCC) — ΔPSNR vs Δ(-LPIPS), 방법쌍 x view_count ===")
    for a, b in PAIRS:
        scenes = sorted({k[0] for k in by_key if k[3] in (a, b)})
        view_counts = sorted({k[1] for k in by_key if k[3] in (a, b)})
        for vc in view_counts:
            d_psnr, d_neg_lpips, d_ssim = [], [], []
            disagree_n, total_n = 0, 0
            for scene in scenes:
                for budget in BUDGETS:
                    ra = by_key.get((scene, vc, budget, a))
                    rb = by_key.get((scene, vc, budget, b))
                    if not ra or not rb or ra["test_psnr"] is None or rb["test_psnr"] is None:
                        continue
                    dp = ra["test_psnr"] - rb["test_psnr"]
                    if ra["test_lpips"] is not None and rb["test_lpips"] is not None:
                        dl = -(ra["test_lpips"] - rb["test_lpips"])  # LPIPS 낮을수록 좋음 -> 부호 반전
                        d_psnr.append(dp)
                        d_neg_lpips.append(dl)
                        total_n += 1
                        if (dp > 0) != (dl > 0) and abs(dp) > 1e-6 and abs(dl) > 1e-6:
                            disagree_n += 1
                    if ra["test_ssim"] is not None and rb["test_ssim"] is not None:
                        d_ssim.append((dp, ra["test_ssim"] - rb["test_ssim"]))
            if total_n == 0:
                continue
            r_psnr_lpips, n = pearson(d_psnr, d_neg_lpips)
            rate = disagree_n / total_n if total_n else float("nan")
            ssim_note = ""
            if d_ssim:
                dp_s = [x[0] for x in d_ssim]
                ds_s = [x[1] for x in d_ssim]
                r_psnr_ssim, n_s = pearson(dp_s, ds_s)
                ssim_note = f"  | PSNR-SSIM r={r_psnr_ssim:+.3f} (n={n_s})"
            print(
                f"  [{a} vs {b}][{vc}view] PSNR-LPIPS r={r_psnr_lpips:+.3f} (n={n}), "
                f"승자 불일치율={rate:.1%} ({disagree_n}/{total_n}){ssim_note}"
            )


# ---------------------------------------------------------------------------
# 2. Gaussian 수 / VRAM Pareto
# ---------------------------------------------------------------------------

def gaussian_vram_pareto(rows: list[dict], dataset: str) -> None:
    print(f"\n=== {dataset}: Gaussian 수 / VRAM Pareto — 방법 x view_count (budget=300s) ===")
    by_method_vc: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["budget"] != 300.0 or r["gaussian_count"] is None:
            continue
        by_method_vc[(r["method"], r["view_count"])].append(r)

    for (method, vc), group in sorted(by_method_vc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        psnr = scene_cluster_bootstrap_ci(group, lambda r: r["test_psnr"])
        gauss = scene_cluster_bootstrap_ci(group, lambda r: r["gaussian_count"])
        vram = scene_cluster_bootstrap_ci(group, lambda r: r["peak_vram"])
        print(
            f"  [{vc}view][{method:12s}] PSNR={psnr['mean']:.2f}dB  "
            f"gaussians={gauss['mean']:,.0f}  peak_vram={vram['mean']:,.0f}MB  (n={psnr['scene_count']})"
        )


def main() -> int:
    all_rows = []
    for dataset in ("RE10K", "DL3DV"):
        rows = load_dataset(dataset)
        all_rows += rows
        print(f"[loaded] {dataset}: {len(rows)} rows")
        metric_disagreement(rows, dataset)
        gaussian_vram_pareto(rows, dataset)

    out_path = Path("experiments/outputs/metric_disagreement_pareto/rows.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\n[done] {out_path} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

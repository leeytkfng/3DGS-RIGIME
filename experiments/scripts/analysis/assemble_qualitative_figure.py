#!/usr/bin/env python3
"""`render_qualitative_gsplat.py`/`render_qualitative_fsgs.py`가 만든 PNG들을 논문
Figure(§Results 정성적 비교)용 그리드 하나로 합친다. 순수 이미지 조합, 재렌더링 없음.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

METHODS = ["GT", "MVSplat", "Vanilla3DGS", "FSGS"]
LABEL_H = 28
ROW_LABEL_W = 190
PAD = 6


def load_font(size: int):
    for path in (
        "/root/task 2/assets/fonts/NanumGothic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", required=True, help="예: 0588138dfec165a1_8view_300s:역전 경계(8-view) 0588138dfec165a1_12view_300s:명확한 승패(12-view)")
    parser.add_argument("--in-dir", default="experiments/outputs/qualitative_comparison")
    parser.add_argument("--out", default="experiments/docs/paper/overleaf_draft/figures/qualitative_comparison.png")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    rows = []
    for cond in args.conditions:
        tag, row_label = cond.split(":", 1)
        images = [Image.open(in_dir / f"{tag}_{m}.png") for m in METHODS]
        rows.append((row_label, images))

    cell_w, cell_h = rows[0][1][0].size
    n_cols = len(METHODS)
    n_rows = len(rows)
    total_w = ROW_LABEL_W + n_cols * cell_w + (n_cols - 1) * PAD
    total_h = LABEL_H + n_rows * cell_h + (n_rows - 1) * PAD

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)
    font_row = load_font(14)

    for c, method in enumerate(METHODS):
        x = ROW_LABEL_W + c * (cell_w + PAD)
        bbox = draw.textbbox((0, 0), method, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell_w - tw) // 2, 4), method, fill="black", font=font)

    for r, (row_label, images) in enumerate(rows):
        y = LABEL_H + r * (cell_h + PAD)
        draw.text((4, y + cell_h // 2 - 8), row_label, fill="black", font=font_row)
        for c, img in enumerate(images):
            x = ROW_LABEL_W + c * (cell_w + PAD)
            canvas.paste(img, (x, y))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[done] {out_path} ({total_w}x{total_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""`render_qualitative_gsplat.py`/`render_qualitative_fsgs.py`가 만든 PNG들을 논문
Figure(§Results 정성적 비교)용 그리드 하나로 합친다. 순수 이미지 조합, 재렌더링 없음.

최신 NVS 논문(MVSplat/DepthSplat/3DGS 계열) 스타일을 따른다: 각 패널 하단에 PSNR 라벨,
가장 오차가 큰 영역을 자동으로 찾아 확대 inset + 원본에 위치 표시 박스, 얇은 그리드
구분선, 최고 PSNR 방법은 라벨을 강조색으로 표시.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

LABEL_H = 40
ROW_LABEL_W = 210
GRID = "#dcd8cc"
INK = "#1d2225"
ACCENT = "#0d6e68"
INSET_BORDER = "#c63f3f"
PSNR_BAR_H = 30
INSET_SCALE = 3
INSET_SIZE = 44  # crop 정사각형 한 변(원본 px)


def load_font(size: int, bold: bool = False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return 99.0
    return float(-10.0 * np.log10(mse / (255.0**2)))


def find_detail_crop(gt: np.ndarray, other: np.ndarray, size: int = INSET_SIZE, stride: int = 8) -> tuple[int, int]:
    """GT에 디테일(edge/텍스처)이 있으면서 동시에 방법들 사이 오차도 있는 영역을 고른다 —
    오차만 보면 GT 자체가 어둡거나 텍스처 없는 영역(예: 그늘진 구석)이 뽑혀서 확대해도
    다른 방법 패널이 그냥 새까맣게만 나오는 문제가 있었다(2026-08-17 첫 시도에서 발견).
    GT gradient 크기 x GT-other 오차를 같이 곱해 "디테일이 있고 실제로 차이도 나는" 곳을
    찾는다."""

    gt_gray = gt.astype(np.float32).mean(axis=-1)
    gy, gx = np.gradient(gt_gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    diff = np.abs(gt.astype(np.float32) - other.astype(np.float32)).mean(axis=-1)

    h, w = diff.shape
    best = (-1.0, 0, 0)
    for y in range(0, h - size, stride):
        for x in range(0, w - size, stride):
            detail = float(grad_mag[y : y + size, x : x + size].mean())
            err = float(diff[y : y + size, x : x + size].mean())
            score = detail * (1.0 + err)
            if score > best[0]:
                best = (score, x, y)
    return best[1], best[2]


def annotate_cell(img: Image.Image, label: str, crop_xy: tuple[int, int], is_best: bool, is_ref: bool) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    if not is_ref:
        cx, cy = crop_xy
        draw.rectangle([cx, cy, cx + INSET_SIZE, cy + INSET_SIZE], outline=INSET_BORDER, width=2)
        crop = img.crop((cx, cy, cx + INSET_SIZE, cy + INSET_SIZE)).resize(
            (INSET_SIZE * INSET_SCALE, INSET_SIZE * INSET_SCALE), Image.NEAREST
        )
        crop = ImageOps.expand(crop, border=2, fill=INSET_BORDER)
        img.paste(crop, (w - crop.width - 4, h - crop.height - 4))

    bar = Image.new("RGBA", (w, PSNR_BAR_H), (15, 15, 15, 195))
    img.paste(Image.alpha_composite(img.crop((0, h - PSNR_BAR_H, w, h)).convert("RGBA"), bar), (0, h - PSNR_BAR_H))
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(18, bold=True)
    color = "#8CFFC7" if is_best else "#ffffff"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, h - PSNR_BAR_H + (PSNR_BAR_H - th) // 2 - bbox[1]), label, fill=color, font=font)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", nargs="+", required=True,
        help="tag:행라벨 형식, 예: 0588138dfec165a1_8view_300s:'8-view, 300s (역전 경계)'",
    )
    parser.add_argument("--methods", nargs="+", default=["GT", "MVSplat", "Vanilla3DGS", "FSGS"])
    parser.add_argument("--in-dir", default="experiments/outputs/qualitative_comparison")
    parser.add_argument("--out", default="experiments/docs/paper/overleaf_draft/figures/qualitative_comparison.png")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    methods = args.methods
    rows = []
    for cond in args.conditions:
        tag, row_label = cond.split(":", 1)
        images = {m: Image.open(in_dir / f"{tag}_{m}.png").convert("RGB") for m in methods}
        gt_arr = np.array(images["GT"])
        psnrs = {m: (psnr(gt_arr, np.array(images[m])) if m != "GT" else None) for m in methods}
        non_gt = [m for m in methods if m != "GT"]
        best_method = max(non_gt, key=lambda m: psnrs[m]) if non_gt else None
        mvsplat_ref = images.get("MVSplat", images[non_gt[0]] if non_gt else images["GT"])
        crop_xy = find_detail_crop(gt_arr, np.array(mvsplat_ref))
        annotated = []
        for m in methods:
            label = "Reference" if m == "GT" else f"{m}  {psnrs[m]:.2f} dB"
            annotated.append(annotate_cell(images[m], label, crop_xy, m == best_method, m == "GT"))
        rows.append((row_label, annotated))

    cell_w, cell_h = rows[0][1][0].size
    n_cols, n_rows = len(methods), len(rows)
    gap = 2
    total_w = ROW_LABEL_W + n_cols * cell_w + (n_cols - 1) * gap
    total_h = LABEL_H + n_rows * cell_h + (n_rows - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(21, bold=True)
    row_font = load_font(17, bold=True)

    for c, method in enumerate(methods):
        x = ROW_LABEL_W + c * (cell_w + gap)
        bbox = draw.textbbox((0, 0), method, font=header_font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell_w - tw) // 2, (LABEL_H - 21) // 2), method, fill=INK, font=header_font)

    for r, (row_label, images) in enumerate(rows):
        y = LABEL_H + r * (cell_h + gap)
        lines = row_label.split(", ", 1)
        line_h = 24
        ty = y + cell_h // 2 - (len(lines) * line_h) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=row_font)
            tw = bbox[2] - bbox[0]
            draw.text(((ROW_LABEL_W - tw) // 2, ty), line, fill=INK, font=row_font)
            ty += line_h
        for c, img in enumerate(images):
            x = ROW_LABEL_W + c * (cell_w + gap)
            canvas.paste(img, (x, y))

    draw.line([(ROW_LABEL_W - 1, 0), (ROW_LABEL_W - 1, total_h)], fill=GRID, width=1)
    draw.line([(0, LABEL_H - 1), (total_w, LABEL_H - 1)], fill=GRID, width=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[done] {out_path} ({total_w}x{total_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("缺少 Pillow。请运行：python -m pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
PAIRS_ROOT = ROOT / "results" / "pairs"
FINAL_DIR = ROOT / "results" / "final_figures"
SUMMARY = ROOT / "results" / "summary_vina_results.csv"

CANVAS_W, CANVAS_H = 3900, 1600
TITLE_H = 115
MARGIN = 55
GAP = 70
LEFT_W = 1650
RIGHT_W = CANVAS_W - 2 * MARGIN - GAP - LEFT_W
PANEL_H = CANVAS_H - TITLE_H - 2 * MARGIN
LABEL_COLUMN = 300


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend([
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ])
    else:
        candidates.extend([
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ])
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


FONT_TITLE = load_font(42, bold=True)
FONT_PANEL = load_font(42, bold=True)
FONT_LABEL = load_font(34, bold=True)
FONT_META = load_font(30, bold=False)


def summary_scores() -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not SUMMARY.exists():
        return result
    with SUMMARY.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pair_id = str(row.get("pair_id", "")).strip()
            score = str(row.get("best_affinity_kcal_mol", "")).strip()
            if pair_id:
                result[pair_id] = score
    return result


def white_crop_bbox(image: Image.Image, threshold: int = 247, padding: int = 24) -> Tuple[int, int, int, int]:
    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg).convert("L")
    # Ignore near-white antialiasing.
    mask = diff.point(lambda value: 255 if value > (255 - threshold) else 0)
    bbox = mask.getbbox()
    if not bbox:
        return (0, 0, image.width, image.height)
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)
    return left, top, right, bottom


def fit_with_transform(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    crop_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Image.Image, dict]:
    if crop_bbox is None:
        crop_bbox = white_crop_bbox(image)
    cropped = image.crop(crop_bbox).convert("RGB")
    x0, y0, x1, y1 = box
    target_w, target_h = x1 - x0, y1 - y0
    scale = min(target_w / cropped.width, target_h / cropped.height)
    new_w = max(1, round(cropped.width * scale))
    new_h = max(1, round(cropped.height * scale))
    fitted = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    paste_x = x0 + (target_w - new_w) // 2
    paste_y = y0 + (target_h - new_h) // 2
    transform = {
        "crop": crop_bbox,
        "scale": scale,
        "paste_x": paste_x,
        "paste_y": paste_y,
        "width": new_w,
        "height": new_h,
    }
    return fitted, transform


def apply_transform(image: Image.Image, transform: dict) -> Image.Image:
    crop = image.crop(transform["crop"]).convert("RGB")
    return crop.resize((transform["width"], transform["height"]), Image.Resampling.NEAREST)


def detect_cyan_bbox(image: Image.Image, offset: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    rgb = image.convert("RGB")
    small_w = min(650, rgb.width)
    scale = small_w / rgb.width
    small_h = max(1, round(rgb.height * scale))
    small = rgb.resize((small_w, small_h), Image.Resampling.BILINEAR)
    xs: List[int] = []
    ys: List[int] = []
    pixels = small.load()
    for y in range(small.height):
        for x in range(small.width):
            r, g, b = pixels[x, y]
            if b > 125 and g > 105 and r < 125 and b - r > 55 and g - r > 35 and abs(b - g) < 105:
                xs.append(x)
                ys.append(y)
    if len(xs) < 8:
        return None
    inv = 1.0 / scale
    left = int(min(xs) * inv) + offset[0]
    top = int(min(ys) * inv) + offset[1]
    right = int((max(xs) + 1) * inv) + offset[0]
    bottom = int((max(ys) + 1) * inv) + offset[1]
    pad_x = max(40, int((right - left) * 0.8))
    pad_y = max(40, int((bottom - top) * 0.8))
    return left - pad_x, top - pad_y, right + pad_x, bottom + pad_y


def color_center(image: Image.Image, rgb: Sequence[int], tolerance: int = 34) -> Optional[Tuple[float, float]]:
    pixels = image.convert("RGB")
    target = tuple(map(int, rgb))
    xs: List[int] = []
    ys: List[int] = []
    step = 2 if pixels.width * pixels.height > 1_500_000 else 1
    px = pixels.load()
    for y in range(0, pixels.height, step):
        for x in range(0, pixels.width, step):
            current = px[x, y]
            distance = math.sqrt(sum((current[i] - target[i]) ** 2 for i in range(3)))
            if distance <= tolerance:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def distribute_y(items: List[dict], top: int, bottom: int, gap: int = 72) -> None:
    if not items:
        return
    items.sort(key=lambda item: item["anchor"][1])
    desired = [max(top, min(bottom, int(item["anchor"][1]))) for item in items]
    assigned = desired[:]
    for i in range(1, len(assigned)):
        assigned[i] = max(assigned[i], assigned[i - 1] + gap)
    overflow = assigned[-1] - bottom
    if overflow > 0:
        assigned = [value - overflow for value in assigned]
    for i in range(len(assigned) - 2, -1, -1):
        assigned[i] = min(assigned[i], assigned[i + 1] - gap)
    underflow = top - assigned[0]
    if underflow > 0:
        assigned = [value + underflow for value in assigned]
    for item, y in zip(items, assigned):
        item["label_y"] = y


def draw_callouts(
    draw: ImageDraw.ImageDraw,
    records: List[dict],
    anchor_image: Image.Image,
    anchor_offset: Tuple[int, int],
    pocket_box: Tuple[int, int, int, int],
) -> None:
    items: List[dict] = []
    center_x = (pocket_box[0] + pocket_box[2]) / 2
    for record in records:
        center = color_center(anchor_image, record.get("color", [0, 0, 0]))
        if center is None:
            continue
        absolute = (center[0] + anchor_offset[0], center[1] + anchor_offset[1])
        items.append({"label": record.get("label", "RES"), "anchor": absolute})
    left_items = [item for item in items if item["anchor"][0] < center_x]
    right_items = [item for item in items if item["anchor"][0] >= center_x]
    distribute_y(left_items, pocket_box[1] + 55, pocket_box[3] - 55)
    distribute_y(right_items, pocket_box[1] + 55, pocket_box[3] - 55)
    for side, group in (("left", left_items), ("right", right_items)):
        for item in group:
            label = item["label"]
            bbox = draw.textbbox((0, 0), label, font=FONT_LABEL)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if side == "left":
                text_x = pocket_box[0] - LABEL_COLUMN + 15
                line_end = (text_x + text_w + 12, item["label_y"])
            else:
                text_x = pocket_box[2] + 20
                line_end = (text_x - 12, item["label_y"])
            text_y = item["label_y"] - text_h // 2
            anchor = item["anchor"]
            elbow_x = pocket_box[0] - 18 if side == "left" else pocket_box[2] + 18
            draw.line([anchor, (elbow_x, item["label_y"]), line_end], fill=(92, 92, 92), width=3)
            draw.ellipse([anchor[0] - 5, anchor[1] - 5, anchor[0] + 5, anchor[1] + 5], fill=(55, 55, 55))
            draw.text((text_x, text_y), label, fill=(20, 20, 20), font=FONT_LABEL)


def compose_pair(pair_id: str, images_dir: Path, score: str) -> Path:
    overview_path = images_dir / f"{pair_id}__overview_raw.png"
    pocket_path = images_dir / f"{pair_id}__pocket_raw.png"
    anchor_path = images_dir / f"{pair_id}__pocket_anchor_map.png"
    callout_path = images_dir / f"{pair_id}__pocket_callouts.json"
    missing = [path.name for path in (overview_path, pocket_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少：" + ", ".join(missing))

    overview = Image.open(overview_path).convert("RGB")
    pocket = Image.open(pocket_path).convert("RGB")
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(canvas)

    left_box = (MARGIN, TITLE_H + MARGIN, MARGIN + LEFT_W, TITLE_H + MARGIN + PANEL_H)
    right_area_x0 = left_box[2] + GAP
    pocket_box = (
        right_area_x0 + LABEL_COLUMN,
        TITLE_H + MARGIN,
        CANVAS_W - MARGIN - LABEL_COLUMN,
        TITLE_H + MARGIN + PANEL_H,
    )

    overview_fitted, overview_transform = fit_with_transform(overview, left_box)
    canvas.paste(overview_fitted, (overview_transform["paste_x"], overview_transform["paste_y"]))
    pocket_crop = white_crop_bbox(pocket)
    pocket_fitted, pocket_transform = fit_with_transform(pocket, pocket_box, pocket_crop)
    canvas.paste(pocket_fitted, (pocket_transform["paste_x"], pocket_transform["paste_y"]))

    draw.rectangle(left_box, outline=(65, 65, 65), width=3)
    draw.rectangle(pocket_box, outline=(65, 65, 65), width=3)
    draw.text((left_box[0] + 18, left_box[1] + 12), "A", fill=(0, 0, 0), font=FONT_PANEL)
    draw.text((pocket_box[0] + 18, pocket_box[1] + 12), "B", fill=(0, 0, 0), font=FONT_PANEL)

    ligand_box = detect_cyan_bbox(overview_fitted, (overview_transform["paste_x"], overview_transform["paste_y"]))
    if ligand_box:
        lx0 = max(left_box[0] + 10, ligand_box[0])
        ly0 = max(left_box[1] + 10, ligand_box[1])
        lx1 = min(left_box[2] - 10, ligand_box[2])
        ly1 = min(left_box[3] - 10, ligand_box[3])
        draw.rectangle((lx0, ly0, lx1, ly1), outline=(210, 25, 25), width=7)
        draw.line([(lx1, ly0), (pocket_box[0], pocket_box[1] + 70)], fill=(210, 25, 25), width=4)
        draw.line([(lx1, ly1), (pocket_box[0], pocket_box[3] - 70)], fill=(210, 25, 25), width=4)

    if anchor_path.exists() and callout_path.exists():
        anchor_original = Image.open(anchor_path).convert("RGB")
        anchor_fitted = apply_transform(anchor_original, pocket_transform)
        try:
            records = json.loads(callout_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            records = []
        draw_callouts(
            draw, records, anchor_fitted,
            (pocket_transform["paste_x"], pocket_transform["paste_y"]), pocket_box,
        )

    title = pair_id.replace("_", " ")
    draw.text((MARGIN, 38), title, fill=(10, 10, 10), font=FONT_TITLE)
    if score:
        meta = f"Vina affinity: {score} kcal/mol"
        bbox = draw.textbbox((0, 0), meta, font=FONT_META)
        draw.text((CANVAS_W - MARGIN - (bbox[2] - bbox[0]), 48), meta, fill=(45, 45, 45), font=FONT_META)

    output = images_dir / f"{pair_id}__composite_B.png"
    canvas.save(output, dpi=(300, 300))
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, FINAL_DIR / output.name)
    return output


def main() -> int:
    scores = summary_scores()
    manual_files = sorted(PAIRS_ROOT.rglob("*_manual_review.pml"))
    pair_ids = sorted({path.stem.replace("_manual_review", "") for path in manual_files})
    if not pair_ids:
        pair_ids = sorted({path.name.replace("__overview_raw.png", "") for path in PAIRS_ROOT.rglob("*__overview_raw.png")})
    if not pair_ids:
        print("没有找到可拼图的组合。请先完成 PyMOL 图片导出。")
        return 1
    failures = 0
    for pair_id in pair_ids:
        images_dir = PAIRS_ROOT / pair_id / "images"
        try:
            output = compose_pair(pair_id, images_dir, scores.get(pair_id, ""))
            print(f"[OK] {output}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {pair_id}: {exc}")
    print(f"\n最终图片目录：{FINAL_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

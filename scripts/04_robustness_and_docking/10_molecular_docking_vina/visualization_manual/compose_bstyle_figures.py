from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bstyle_common import load_manifest, project_root_from_arg

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required to compose the final figure. Install it with:\n"
        "  python -m pip install Pillow"
    ) from exc


CANVAS_SIZE = (1640, 888)
RED = (255, 20, 20)


def trim_white(image: Image.Image, threshold: int = 248, padding: int = 4) -> Image.Image:
    rgb = image.convert("RGB")
    pixels = rgb.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(rgb.height):
        for x in range(rgb.width):
            r, g, b = pixels[x, y]
            if min(r, g, b) < threshold:
                xs.append(x)
                ys.append(y)
    if not xs:
        return rgb
    box = (
        max(0, min(xs) - padding),
        max(0, min(ys) - padding),
        min(rgb.width, max(xs) + padding + 1),
        min(rgb.height, max(ys) + padding + 1),
    )
    return rgb.crop(box)


def fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    fill=RED,
    width: int = 4,
    dash: int = 15,
    gap: int = 10,
) -> None:
    import math

    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return
    ux, uy = dx / length, dy / length
    cursor = 0.0
    while cursor < length:
        end = min(cursor + dash, length)
        draw.line(
            (x1 + ux * cursor, y1 + uy * cursor, x1 + ux * end, y1 + uy * end),
            fill=fill,
            width=width,
        )
        cursor += dash + gap


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill=RED,
    width: int = 4,
    dash: int = 15,
    gap: int = 10,
) -> None:
    x1, y1, x2, y2 = box
    draw_dashed_line(draw, (x1, y1), (x2, y1), fill, width, dash, gap)
    draw_dashed_line(draw, (x2, y1), (x2, y2), fill, width, dash, gap)
    draw_dashed_line(draw, (x2, y2), (x1, y2), fill, width, dash, gap)
    draw_dashed_line(draw, (x1, y2), (x1, y1), fill, width, dash, gap)


def ligand_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    xs: list[int] = []
    ys: list[int] = []
    for y in range(rgb.height):
        for x in range(rgb.width):
            r, g, b = rgb.getpixel((x, y))
            # Cyan ligand carbons plus blue/red hetero atoms tightly connected to it.
            is_cyan = b > 120 and g > 105 and r < 110 and (g + b - 2 * r) > 150
            if is_cyan:
                xs.append(x)
                ys.append(y)
    if len(xs) < 8:
        return None
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def expand_box(
    box: tuple[int, int, int, int],
    margin_x: int,
    margin_y: int,
    bounds: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = bounds
    return (
        max(4, x1 - margin_x),
        max(4, y1 - margin_y),
        min(width - 4, x2 + margin_x),
        min(height - 4, y2 + margin_y),
    )


def compose_pair(pair: dict) -> Path:
    overview_path = Path(pair["overview_png"])
    pocket_path = Path(pair["pocket_png"])
    output_path = Path(pair["composite_png"])
    if not overview_path.exists() or not pocket_path.exists():
        missing = [str(p) for p in (overview_path, pocket_path) if not p.exists()]
        raise FileNotFoundError(
            "Panel image(s) missing:\n  " + "\n  ".join(missing) + "\nRun 05_generate_pymol_scripts.bat first."
        )

    overview = trim_white(Image.open(overview_path))
    pocket = trim_white(Image.open(pocket_path), padding=0)

    canvas = Image.new("RGB", CANVAS_SIZE, "white")
    left = fit(overview, 600, 760)
    right = fit(pocket, 970, 850)

    left_xy = (18, (CANVAS_SIZE[1] - left.height) // 2)
    right_xy = (650, 18)
    canvas.paste(left, left_xy)
    canvas.paste(right, right_xy)

    draw = ImageDraw.Draw(canvas)

    local_ligand = ligand_bbox(left)
    if local_ligand is None:
        local_ligand = (
            int(left.width * 0.55),
            int(left.height * 0.38),
            int(left.width * 0.72),
            int(left.height * 0.58),
        )
    local_crop = expand_box(local_ligand, margin_x=42, margin_y=42, bounds=left.size)
    crop_box = (
        local_crop[0] + left_xy[0],
        local_crop[1] + left_xy[1],
        local_crop[2] + left_xy[0],
        local_crop[3] + left_xy[1],
    )
    draw_dashed_rectangle(draw, crop_box, width=4, dash=16, gap=9)

    right_box = (
        right_xy[0] - 12,
        right_xy[1] - 2,
        right_xy[0] + right.width + 12,
        right_xy[1] + right.height + 2,
    )
    draw_dashed_rectangle(draw, right_box, width=4, dash=16, gap=9)

    elbow_x = right_box[0] - 10
    draw_dashed_line(draw, (crop_box[2], crop_box[1]), (elbow_x, right_box[1]), width=4, dash=16, gap=9)
    draw_dashed_line(draw, (crop_box[2], crop_box[3]), (elbow_x, right_box[3]), width=4, dash=16, gap=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(300, 300), optimize=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose B-style overview plus binding-pocket figures.")
    parser.add_argument("--project", default=".", help="Project root directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = project_root_from_arg(args.project)
    manifest = load_manifest(project)
    pairs = list(manifest.get("pairs", []))
    if not pairs:
        raise RuntimeError("No pairs were found in the B-style scene manifest.")
    for pair in pairs:
        output = compose_pair(pair)
        print(f"[OK] Composite figure: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise SystemExit(1)

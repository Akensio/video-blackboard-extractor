"""Contact-sheet montage of snapshot thumbnails with caption strings."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def make_montage(
    image_paths: list[Path],
    captions: list[str],
    out_path: str | Path,
    cols: int = 3,
    thumb_w: int = 480,
    pad: int = 8,
    caption_h: int = 20,
) -> Path:
    out_path = Path(out_path)
    if not image_paths:
        return out_path

    thumbs = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        h = round(thumb_w * im.height / im.width)
        thumbs.append(im.resize((thumb_w, h)))

    cell_w = thumb_w
    cell_h = max(t.height for t in thumbs) + caption_h
    rows = (len(thumbs) + cols - 1) // cols
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (thumb, cap) in enumerate(zip(thumbs, captions)):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        draw.text((x + 2, y + 2), cap, fill=(240, 240, 240))
        sheet.paste(thumb, (x, y + caption_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path

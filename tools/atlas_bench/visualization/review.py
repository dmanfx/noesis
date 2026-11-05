"""Make a small local RGB/depth/confidence contact sheet after parsing."""
from __future__ import annotations
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from pathlib import Path


def contact_sheet(run: Path, artifacts: list[dict]) -> Path:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    rgb_rows = [row for row in artifacts if row["field"].endswith(".imageAsset")]
    sheet = Image.new("RGB", (960, 204 * len(rgb_rows)), "#161b22")
    draw = ImageDraw.Draw(sheet)
    lookup = {row["field"]: run / row["path"] for row in artifacts}
    for index, row in enumerate(rgb_rows):
        prefix = row["field"].removesuffix(".imageAsset")
        with Image.open(run / row["path"]) as image:
            sheet.paste(image.convert("RGB").resize((320, 180)), (0, index * 204 + 24))
        draw.text((6, index * 204 + 5), f"{prefix}: returned RGB", fill="white")
        depth_path = lookup.get(prefix + ".depth.depthAsset")
        if depth_path:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            valid = np.isfinite(depth) & (depth > 0)
            lo, hi = np.percentile(depth[valid], [5, 95])
            shade = np.zeros(depth.shape, np.uint8)
            shade[valid] = (255 * (1 - np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0, 1))).astype(np.uint8)
            sheet.paste(Image.fromarray(shade).convert("RGB").resize((320, 180)), (320, index * 204 + 24))
            draw.text((326, index * 204 + 5), "depth: per-view 5-95% range", fill="white")
        conf = lookup.get(prefix + ".depth.confidenceAsset")
        if conf:
            with Image.open(conf) as image:
                sheet.paste(image.convert("RGB").resize((320, 180)), (640, index * 204 + 24))
            draw.text((646, index * 204 + 5), "provider confidence", fill="white")
    destination = run / "review.png"
    sheet.save(destination)
    destination.chmod(0o600)
    return destination

"""Printable Basalt 0.1.7 AprilGrid; physical scale must be checked after printing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def generate(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(output), pagesize=letter, pageCompression=1)
    document.setTitle("RoomWalk camera and IMU calibration target")
    document.setAuthor("Noesis RoomWalk")
    width, height = letter
    document.setFont("Helvetica-Bold", 17)
    document.drawString(36, height - 42, "RoomWalk calibration target")
    document.setFont("Helvetica", 9)
    document.drawString(36, height - 58, "US Letter | Print at 100% / Actual size | Disable Fit to page")
    document.drawString(36, height - 73, "Mount flat on a rigid surface. Keep all 36 tags visible during the motion recording.")
    tag = 25.4 * mm
    gap = tag * 0.3
    grid = 6 * tag + 5 * gap
    left, bottom = (width - grid) / 2, 118
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for y in range(6):
        for x in range(6):
            marker_id = 6 * y + x
            image = cv2.aruco.generateImageMarker(dictionary, marker_id, 10, borderBits=2)
            # Pinned Basalt detector defines p0 at the physical bottom-left.
            image = cv2.rotate(image, cv2.ROTATE_180)
            cell = tag / 10
            for row in range(10):
                for column in range(10):
                    if image[row, column] == 0:
                        document.rect(left + x * (tag + gap) + column * cell,
                                      bottom + y * (tag + gap) + (9 - row) * cell,
                                      cell, cell, stroke=0, fill=1)
    document.setFont("Helvetica", 8)
    document.drawCentredString(width / 2, 100, "6 x 6 AprilTag 36h11 | Two-cell black border | IDs 0-5 in the bottom row")
    document.drawCentredString(width / 2, 87, "Each outer black tag edge: 25.4 mm   |   White gap: 7.62 mm   |   Grid width: 190.5 mm")
    start = (width - 100 * mm) / 2
    document.setLineWidth(0.7)
    document.line(start, 65, start + 100 * mm, 65)
    for x in (start, start + 100 * mm):
        document.line(x, 61, x, 69)
    document.drawCentredString(width / 2, 49, "This line must measure 100 mm after printing.")
    document.drawCentredString(width / 2, 36, "Measure the line and several tag edges with a ruler. Report any scale difference before calibration.")
    document.save()
    config = {"tagCols": 6, "tagRows": 6, "tagSize": 0.0254, "tagSpacing": 0.3}
    output.with_suffix('.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)

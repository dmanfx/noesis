#!/usr/bin/env python3
"""Render four RGB perspectives of a standalone point cloud."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageDraw


VIEWS = (
    ("front_isometric", 24.0, -58.0),
    ("rear_isometric", 22.0, 122.0),
    ("side", 18.0, 24.0),
    ("top", 84.0, -90.0),
)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path) as row:
            return (
                np.asarray(row["points"], dtype=np.float32),
                np.asarray(row["colors"], dtype=np.uint8),
            )
    loaded = trimesh.load(path, force="scene", process=False)
    point_geometries = [
        geometry
        for geometry in loaded.geometry.values()
        if isinstance(geometry, trimesh.points.PointCloud)
    ]
    if not point_geometries:
        raise RuntimeError(f"no point geometry found in {path}")
    cloud = max(point_geometries, key=lambda geometry: len(geometry.vertices))
    colors = np.asarray(cloud.colors, dtype=np.uint8)
    if len(colors) != len(cloud.vertices):
        colors = np.full((len(cloud.vertices), 4), 190, dtype=np.uint8)
    return np.asarray(cloud.vertices, dtype=np.float32), colors[:, :3]


def render(path: Path, output_dir: Path, title: str) -> list[Path]:
    native_points, colors = _load(path)
    # DA3 phone-world is Y-down. Display it as conventional Z-up.
    points = native_points[:, [0, 2, 1]].copy()
    points[:, 2] *= -1.0
    lower, upper = np.percentile(points, (0.5, 99.5), axis=0)
    center = (lower + upper) / 2.0
    span = np.maximum(upper - lower, 0.5)
    lower = center - span * 0.54
    upper = center + span * 0.54

    if len(points) > 220_000:
        indices = np.linspace(0, len(points) - 1, 220_000, dtype=np.int64)
        display_points = points[indices]
        display_colors = colors[indices]
    else:
        display_points = points
        display_colors = colors

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, elevation, azimuth in VIEWS:
        figure = plt.figure(figsize=(9.6, 7.2), dpi=140, facecolor="#091018")
        axis = figure.add_subplot(111, projection="3d", facecolor="#091018")
        axis.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            c=display_colors.astype(np.float32) / 255.0,
            s=0.48,
            alpha=0.86,
            linewidths=0,
            depthshade=False,
        )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_zlim(lower[2], upper[2])
        axis.set_box_aspect(span)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(
            f"{title} — {name.replace('_', ' ')}\n"
            f"{len(native_points):,} points · 2 cm weighted fusion",
            color="#f2f5f8",
            fontsize=13,
            pad=12,
        )
        output = output_dir / f"{name}.png"
        figure.savefig(
            output,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(figure)
        written.append(output)

    images = [Image.open(output).convert("RGB") for output in written]
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    sheet = Image.new(
        "RGB", (tile_width * 2, tile_height * 2 + 84), "#091018"
    )
    ImageDraw.Draw(sheet).text(
        (24, 26),
        f"{title} — four RGB point-cloud perspectives",
        fill="#f2f5f8",
    )
    for index, image in enumerate(images):
        x = index % 2 * tile_width
        y = 84 + math.floor(index / 2) * tile_height
        sheet.paste(image, (x, y))
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    written.append(sheet_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--title", default="Point cloud")
    args = parser.parse_args()
    for output in render(
        args.cloud.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.title,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render Roomform object point clusters without the room shell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageDraw


BOX_COLOR = "#ffad38"


def _box_edges(size: np.ndarray) -> list[np.ndarray]:
    half = size / 2.0
    corners = np.asarray(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return [corners[[start, end]] for start, end in edges]


def _load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scene = trimesh.load(path, force="scene")
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        points.append(trimesh.transform_points(geometry.vertices, transform))
        vertex_colors = getattr(getattr(geometry, "visual", None), "vertex_colors", None)
        if vertex_colors is None or len(vertex_colors) != len(geometry.vertices):
            colors.append(np.full((len(geometry.vertices), 4), 190, dtype=np.uint8))
        else:
            colors.append(np.asarray(vertex_colors, dtype=np.uint8))
    if not points:
        raise RuntimeError(f"object GLB contains no point geometry: {path}")
    return np.vstack(points).astype(np.float32), np.vstack(colors)


def _render_object(
    object_id: int,
    row: dict,
    points: np.ndarray,
    colors: np.ndarray,
    output: Path,
) -> None:
    size = np.asarray(row["size"], dtype=np.float32)
    fig = plt.figure(figsize=(4.8, 4.8), dpi=100, facecolor="#091018")
    axis = fig.add_subplot(111, projection="3d", facecolor="#091018")
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=np.asarray(colors[:, :3], dtype=np.float32) / 255.0,
        s=max(1.2, min(7.0, 18_000 / max(len(points), 1))),
        alpha=0.94,
        linewidths=0,
        depthshade=True,
    )
    for edge in _box_edges(size):
        axis.plot(
            edge[:, 0], edge[:, 1], edge[:, 2],
            color=BOX_COLOR, linewidth=1.8, alpha=0.95,
        )
    span = np.maximum(size, 0.15)
    axis.set_xlim(-span[0] * 0.62, span[0] * 0.62)
    axis.set_ylim(-span[1] * 0.62, span[1] * 0.62)
    axis.set_zlim(-span[2] * 0.62, span[2] * 0.62)
    axis.set_box_aspect(span)
    axis.view_init(elev=24, azim=-55)
    axis.set_axis_off()
    support = row.get("qa", {}).get("floor_support_m")
    support_text = "n/a" if support is None else f"{float(support):.2f}m"
    axis.set_title(
        f"#{object_id}  {row['cls']}\n{len(points):,} pts  •  "
        f"{size[0]:.2f}×{size[1]:.2f}×{size[2]:.2f}m  •  floor {support_text}",
        color="#f2f5f8", fontsize=10, pad=8,
    )
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render(run_dir: Path, output_dir: Path) -> tuple[Path, Path, list[Path]]:
    scene_doc = json.loads((run_dir / "scene.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    object_scene = trimesh.Scene()
    box_points: list[np.ndarray] = []

    for object_id, row in enumerate(scene_doc.get("objects", [])):
        object_path = run_dir / "objects" / f"object-{object_id}.glb"
        if not object_path.is_file():
            continue
        points, colors = _load_points(object_path)
        image_path = output_dir / f"object-{object_id:02d}-{row['cls']}.png"
        _render_object(object_id, row, points, colors, image_path)
        images.append(image_path)

        heading = float(row["heading"])
        cosine, sine = math.cos(heading), math.sin(heading)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        center = np.asarray(row["center"], dtype=np.float32)
        world_points = points @ rotation.T + center
        object_scene.add_geometry(
            trimesh.PointCloud(world_points, colors=colors),
            node_name=f"object-{object_id}-{row['cls']}",
        )
        for edge in _box_edges(np.asarray(row["size"], dtype=np.float32)):
            transformed = edge @ rotation.T + center
            t = np.linspace(0.0, 1.0, 10)[:, None]
            box_points.append(
                transformed[0] + t * (transformed[1] - transformed[0])
            )

    if box_points:
        object_scene.add_geometry(
            trimesh.PointCloud(
                np.concatenate(box_points), colors=[255, 173, 56, 255]
            ),
            node_name="object-boxes",
        )
    glb_path = output_dir / "objects-standalone.glb"
    object_scene.export(glb_path)

    opened = [Image.open(path).convert("RGB") for path in images]
    columns = 4
    tile_width = max(image.width for image in opened)
    tile_height = max(image.height for image in opened)
    rows = math.ceil(len(opened) / columns)
    sheet = Image.new(
        "RGB", (columns * tile_width, 74 + rows * tile_height), "#091018"
    )
    ImageDraw.Draw(sheet).text(
        (22, 24),
        "Roomform 2 cm PTv3 — standalone boxed RGB point clusters",
        fill="#f2f5f8",
    )
    for index, image in enumerate(opened):
        x = index % columns * tile_width
        y = 74 + index // columns * tile_height
        sheet.paste(image, (x, y))
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    return glb_path, sheet_path, images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "standalone_objects"
    )
    glb, sheet, images = render(run_dir, output_dir)
    print(glb)
    print(sheet)
    print(f"{len(images)} individual object renders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

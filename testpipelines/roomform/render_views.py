#!/usr/bin/env python3
"""Render repeatable multi-angle PNGs from a Roomform run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw


VIEWS = (
    ("isometric_front", 24.0, -58.0),
    ("isometric_rear", 22.0, 122.0),
    ("side", 14.0, 28.0),
    ("top", 82.0, -90.0),
)
CLASS_COLORS = (
    ("wall", "#4694e6"),
    ("floor", "#74ca88"),
    ("ceiling", "#d2c86e"),
)
OBJECT_COLOR = "#f0a33b"


def _sample(values: np.ndarray, budget: int) -> np.ndarray:
    if len(values) <= budget:
        return values
    return values[:: max(1, len(values) // budget + 1)]


def _axis_limits(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    span = np.maximum(upper - lower, 0.5)
    return center - span * 0.53, center + span * 0.53, span


def _box_edges(center: np.ndarray, size: np.ndarray, heading: float) -> list[np.ndarray]:
    cosine, sine = np.cos(heading), np.sin(heading)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    half = size / 2.0
    corners = np.asarray(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    corners = corners @ rotation.T + center
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return [corners[[start, end]] for start, end in edges]


def render(run_dir: Path, output_dir: Path) -> list[Path]:
    evidence = np.load(run_dir / "evidence.npz")
    patchgraph = np.load(run_dir / "patchgraph.npz")
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    scene_doc = json.loads((run_dir / "scene.json").read_text(encoding="utf-8"))
    vox = float(report["evidence"]["vox_m"])
    occ_cells = np.argwhere(evidence["occ"] > 0)
    occ_cells = _sample(occ_cells, 90_000)
    evidence_points = (occ_cells.astype(np.float32) + 0.5) * vox
    features = np.asarray(evidence["features"], dtype=np.float32)
    evidence_colors = np.stack(
        [features[channel][tuple(occ_cells.T)] for channel in (1, 2, 3)], axis=1
    )
    evidence_colors = np.clip(evidence_colors, 0.0, 1.0)
    nodes = np.asarray(patchgraph["node_probs"], dtype=np.float32)
    offsets = np.asarray(patchgraph["offsets"], dtype=np.float32)
    shell_rows: list[tuple[str, str, np.ndarray]] = []
    for class_index, (name, color) in enumerate(CLASS_COLORS):
        cells = np.argwhere(nodes[class_index] >= 0.5)
        cells = _sample(cells, 100_000)
        points = (cells.astype(np.float32) + 0.5) * vox
        if len(cells):
            displacement = np.stack(
                [offsets[axis][tuple(cells.T)] for axis in range(3)], axis=1
            )
            points += displacement * vox
        shell_rows.append((name, color, points))
    object_rows = [
        (
            str(row["cls"]),
            np.asarray(row["center"], dtype=np.float32),
            np.asarray(row["size"], dtype=np.float32),
            float(row["heading"]),
        )
        for row in scene_doc.get("objects", [])
    ]
    semantic_points = np.empty((0, 3), dtype=np.float32)
    semantic_colors = np.empty((0, 4), dtype=np.float32)
    labels_path = run_dir / "labels.npz"
    if labels_path.is_file():
        labels = np.load(labels_path)
        classes = np.asarray([str(value) for value in labels["classes"]])
        label_ids = np.asarray(labels["label"], dtype=np.int64)
        keep = ~np.isin(classes[label_ids], ("wall", "floor"))
        raw_points = np.asarray(labels["pts"], dtype=np.float32)[keep]
        raw_ids = label_ids[keep]
        shift = np.asarray(scene_doc["frame_shift"], dtype=np.float32)
        semantic_points = raw_points - shift
        if len(semantic_points) > 80_000:
            stride = len(semantic_points) // 80_000 + 1
            semantic_points = semantic_points[::stride]
            raw_ids = raw_ids[::stride]
        palette = plt.get_cmap("tab20")(np.arange(len(classes)) % 20)
        semantic_colors = palette[raw_ids]
        semantic_colors[:, 3] = 0.48
    combined = [evidence_points, semantic_points]
    combined.extend(points for _, _, points in shell_rows if len(points))
    combined.extend(center[None] for _, center, _, _ in object_rows)
    all_points = np.concatenate(combined, axis=0)
    lower, upper, span = _axis_limits(all_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, elevation, azimuth in VIEWS:
        fig = plt.figure(figsize=(13.333, 10.0), dpi=120, facecolor="#0b1017")
        axis = fig.add_subplot(111, projection="3d", facecolor="#0b1017")
        axis.scatter(
            evidence_points[:, 0],
            evidence_points[:, 1],
            evidence_points[:, 2],
            c=evidence_colors,
            s=0.45,
            alpha=0.22,
            linewidths=0,
            depthshade=False,
        )
        for class_name, color, points in shell_rows:
            if not len(points):
                continue
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=color,
                s=3.5,
                alpha=0.92,
                linewidths=0,
                depthshade=True,
                label=class_name,
            )
        if len(semantic_points):
            axis.scatter(
                semantic_points[:, 0],
                semantic_points[:, 1],
                semantic_points[:, 2],
                c=semantic_colors,
                s=1.8,
                alpha=0.48,
                linewidths=0,
                depthshade=False,
            )
        for _, center, size, heading in object_rows:
            for edge in _box_edges(center, size, heading):
                axis.plot(
                    edge[:, 0],
                    edge[:, 1],
                    edge[:, 2],
                    color=OBJECT_COLOR,
                    linewidth=1.35,
                    alpha=0.95,
                )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_zlim(lower[2], upper[2])
        axis.set_box_aspect(span)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_xlabel("X (m)", color="#ccd5df")
        axis.set_ylabel("Y (m)", color="#ccd5df")
        axis.set_zlabel("Z up (m)", color="#ccd5df")
        axis.tick_params(colors="#8998a8", labelsize=8)
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor((0.08, 0.11, 0.15, 0.5))
            pane.set_edgecolor((0.3, 0.35, 0.4, 0.35))
        axis.grid(True, alpha=0.18)
        axis.set_title(
            "Roomform 55M + local PTv3 — living-room phone walk — "
            f"{name.replace('_', ' ')}",
            color="#f2f5f8",
            fontsize=14,
            pad=18,
        )
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=label, markersize=8)
            for label, color in CLASS_COLORS
        ]
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#aab4bf", label="RGB evidence", markersize=6)
        )
        if object_rows:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=OBJECT_COLOR,
                    label=f"PTv3 objects ({len(object_rows)})",
                    linewidth=2,
                )
            )
        legend = axis.legend(handles=handles, loc="upper right", framealpha=0.75)
        legend.get_frame().set_facecolor("#101820")
        for text in legend.get_texts():
            text.set_color("#e5ebf1")
        path = output_dir / f"{name}.png"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        written.append(path)

    images = [Image.open(path).convert("RGB") for path in written]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width * 2, height * 2 + 90), "#0b1017")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (24, 24),
        "Roomform + local PTv3 living-room phone walk — four model-space views",
        fill="#f2f5f8",
    )
    for index, image in enumerate(images):
        x = (index % 2) * width
        y = 90 + (index // 2) * height
        sheet.paste(image, (x, y))
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path, quality=95)
    written.append(sheet_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "renders"
    )
    for path in render(run_dir, output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

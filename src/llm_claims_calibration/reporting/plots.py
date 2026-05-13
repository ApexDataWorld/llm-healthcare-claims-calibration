from __future__ import annotations

from pathlib import Path
import struct
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _write_png(image: np.ndarray, output_path: str) -> None:
    height, width, _ = image.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png_bytes = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw_rows, level=9)),
            chunk(b"IEND", b""),
        ]
    )
    Path(output_path).write_bytes(png_bytes)


def _draw_line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        for offset_x in range(-thickness, thickness + 1):
            for offset_y in range(-thickness, thickness + 1):
                px = min(max(x0 + offset_x, 0), canvas.shape[1] - 1)
                py = min(max(y0 + offset_y, 0), canvas.shape[0] - 1)
                canvas[py, px] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _plot_line_chart(
    x_values: list[float],
    y_values: list[float],
    output_path: str,
    title: str,
    diagonal: bool = False,
) -> None:
    width = 900
    height = 700
    margin = 80
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    _draw_line(canvas, margin, height - margin, width - margin, height - margin, (0, 0, 0), thickness=2)
    _draw_line(canvas, margin, margin, margin, height - margin, (0, 0, 0), thickness=2)

    if diagonal:
        _draw_line(canvas, margin, height - margin, width - margin, margin, (170, 170, 170), thickness=1)

    if len(x_values) > 1:
        for idx in range(len(x_values) - 1):
            x0 = int(margin + x_values[idx] * (width - 2 * margin))
            y0 = int(height - margin - y_values[idx] * (height - 2 * margin))
            x1 = int(margin + x_values[idx + 1] * (width - 2 * margin))
            y1 = int(height - margin - y_values[idx + 1] * (height - 2 * margin))
            _draw_line(canvas, x0, y0, x1, y1, (31, 119, 180), thickness=2)

    for x_value, y_value in zip(x_values, y_values):
        x_coord = int(margin + x_value * (width - 2 * margin))
        y_coord = int(height - margin - y_value * (height - 2 * margin))
        _draw_line(canvas, x_coord, y_coord, x_coord, y_coord, (214, 39, 40), thickness=5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _write_png(canvas, output_path)


def save_reliability_diagram(frame: pd.DataFrame, output_path: str, n_bins: int) -> None:
    confidence = frame["calibrated_confidence"].to_numpy(dtype=float)
    correct = frame["correct"].to_numpy(dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_confidence = []
    bin_accuracy = []
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        bin_confidence.append(confidence[mask].mean())
        bin_accuracy.append(correct[mask].mean())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(bin_confidence, bin_accuracy, marker="o", linewidth=2, label="Temperature scaling")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_risk_coverage_curve(curve: pd.DataFrame, output_path: str) -> None:
    x_values = curve["coverage"].to_numpy(dtype=float)
    risk_values = curve["selective_risk"].fillna(0.0).to_numpy(dtype=float)
    ymax = float(np.max(risk_values) * 1.25) if len(risk_values) else 1.0
    if ymax <= 0.0:
        ymax = 1.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_values, risk_values, marker="o", linewidth=2, color="#d62728")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, ymax)
    plt.xlabel("Coverage")
    plt.ylabel("Accepted-case error (selective risk)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

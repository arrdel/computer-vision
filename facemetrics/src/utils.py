"""FaceMetrics – shared utilities, visualization, and helpers."""

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")              # Non-interactive backend for headless servers
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
#  I/O helpers
# ──────────────────────────────────────────────
def ensure_dirs(*paths):
    """Create directories for every path (treats it as a file path and creates its parent)."""
    for p in paths:
        os.makedirs(os.path.dirname(p) if "." in os.path.basename(p) else p, exist_ok=True)


def save_csv(rows: list[dict], path: str):
    """Write a list of dicts to a CSV file."""
    ensure_dirs(path)
    if not rows:
        return
    keys = rows[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Saved CSV → {path}  ({len(rows)} rows)")


def save_json(data: dict, path: str):
    """Write a dict to a JSON file."""
    ensure_dirs(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved JSON → {path}")


# ──────────────────────────────────────────────
#  Math helpers
# ──────────────────────────────────────────────
def euclidean(p1, p2) -> float:
    """Euclidean distance between two (x, y) points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def smooth_signal(values: list[float], window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing (preserves length via 'same' mode)."""
    if window < 2:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ──────────────────────────────────────────────
#  Plotting helpers
# ──────────────────────────────────────────────
def plot_ear_over_time(
    ear_values: list[float],
    blink_frames: list[int],
    threshold: float,
    fps: float,
    save_path: str,
):
    """Plot the EAR signal with blink events highlighted."""
    ensure_dirs(save_path)
    time_axis = np.arange(len(ear_values)) / fps  # seconds

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_axis, ear_values, linewidth=0.5, color="steelblue", label="EAR")
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=0.8, label=f"Threshold ({threshold})")

    # Mark blink events
    for bf in blink_frames:
        t = bf / fps
        ax.axvline(x=t, color="orange", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Eye Aspect Ratio")
    ax.set_title(f"EAR Over Time — {len(blink_frames)} blinks detected")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved EAR plot → {save_path}")


def plot_measurements_summary(measurements: dict, save_path: str):
    """Bar chart of averaged face measurements (mm)."""
    ensure_dirs(save_path)
    labels = list(measurements.keys())
    values = [measurements[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color="teal", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f mm", padding=4)
    ax.set_xlabel("Dimension (mm)")
    ax.set_title("Estimated Face Dimensions")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved measurements plot → {save_path}")

"""
FaceMetrics — Comprehensive Report & Visualization Generator

Produces:
  • Per-video EAR time-series plots with blink annotations
  • Per-video face dimension bar charts
  • Cross-video comparative dashboards
  • Annotated sample frame thumbnails
  • Full HTML report with embedded charts
"""

from __future__ import annotations
import os
import json
import base64
from io import BytesIO
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

from src.utils import ensure_dirs

# ──────────────────────────────────────────────
#  Color palette
# ──────────────────────────────────────────────
COLORS = {
    "primary":    "#2563EB",
    "secondary":  "#7C3AED",
    "accent":     "#F59E0B",
    "success":    "#10B981",
    "danger":     "#EF4444",
    "teal":       "#14B8A6",
    "slate":      "#475569",
    "bg":         "#F8FAFC",
    "card_bg":    "#FFFFFF",
    "text":       "#1E293B",
}

PALETTE_8 = [
    "#2563EB", "#7C3AED", "#EC4899", "#F59E0B",
    "#10B981", "#14B8A6", "#EF4444", "#6366F1",
]


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to base64 PNG for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _fig_save_and_b64(fig, save_path: str) -> str:
    """Save figure to disk AND return base64."""
    ensure_dirs(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    b64 = _fig_to_base64(fig)
    return b64


# ══════════════════════════════════════════════
#  INDIVIDUAL VIDEO PLOTS
# ══════════════════════════════════════════════

def plot_ear_detailed(
    ear_values: list[float],
    blink_frames: list[int],
    threshold: float,
    fps: float,
    video_name: str,
    save_path: str,
) -> str:
    """
    Enhanced EAR plot with blink shading, distribution histogram,
    and statistics annotation.
    """
    time_axis = np.arange(len(ear_values)) / fps
    ear_arr = np.array(ear_values)

    fig = plt.figure(figsize=(16, 6), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)
    ax_main = fig.add_subplot(gs[0, :4])
    ax_hist = fig.add_subplot(gs[0, 4], sharey=ax_main)

    # Main EAR trace
    ax_main.fill_between(time_axis, ear_arr, alpha=0.15, color=COLORS["primary"])
    ax_main.plot(time_axis, ear_arr, linewidth=0.7, color=COLORS["primary"], label="EAR (smoothed)")
    ax_main.axhline(y=threshold, color=COLORS["danger"], linestyle="--",
                     linewidth=1.2, label=f"Threshold = {threshold}", zorder=5)

    # Shade blink regions
    for bf in blink_frames:
        t_start = bf / fps
        # shade ~150ms around blink start
        ax_main.axvspan(t_start - 0.05, t_start + 0.15,
                        color=COLORS["accent"], alpha=0.25, zorder=1)
        ax_main.plot(t_start, threshold, "v", color=COLORS["accent"],
                     markersize=6, zorder=6)

    ax_main.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax_main.set_ylabel("Eye Aspect Ratio (EAR)", fontsize=11, fontweight="bold")
    ax_main.set_title(f"EAR Signal — {video_name}  |  {len(blink_frames)} blinks detected",
                       fontsize=13, fontweight="bold", pad=12)
    ax_main.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_main.set_facecolor(COLORS["bg"])
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, time_axis[-1] if len(time_axis) else 1)

    # EAR distribution histogram (rotated)
    valid = ear_arr[~np.isnan(ear_arr)]
    if len(valid) > 0:
        ax_hist.hist(valid, bins=40, orientation="horizontal",
                     color=COLORS["primary"], alpha=0.6, edgecolor="white", linewidth=0.3)
    ax_hist.axhline(y=threshold, color=COLORS["danger"], linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Count", fontsize=9)
    ax_hist.set_title("Distribution", fontsize=10, fontweight="bold")
    ax_hist.set_facecolor(COLORS["bg"])
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.grid(True, alpha=0.3, axis="x")

    return _fig_save_and_b64(fig, save_path)


def plot_dimensions_detailed(
    measurements_mm: dict,
    video_name: str,
    save_path: str,
) -> str:
    """Enhanced horizontal bar chart with color-coded categories."""
    if not measurements_mm:
        return ""

    # Group by category
    categories = {
        "Eyes": {k: v for k, v in measurements_mm.items() if "eye" in k},
        "Face": {k: v for k, v in measurements_mm.items() if "face" in k},
        "Nose": {k: v for k, v in measurements_mm.items() if "nose" in k},
        "Mouth": {k: v for k, v in measurements_mm.items() if "mouth" in k},
    }
    cat_colors = {"Eyes": COLORS["primary"], "Face": COLORS["secondary"],
                  "Nose": COLORS["teal"], "Mouth": COLORS["accent"]}

    labels, values, colors = [], [], []
    for cat, items in categories.items():
        for name, val in items.items():
            labels.append(name)
            values.append(val)
            colors.append(cat_colors[cat])

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=COLORS["bg"])
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.65)
    ax.bar_label(bars, fmt="%.1f mm", padding=5, fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Dimension (mm)", fontsize=11, fontweight="bold")
    ax.set_title(f"Face Dimensions — {video_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.25 if values else 1)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[c], label=c) for c in categories if categories[c]]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    return _fig_save_and_b64(fig, save_path)


# ══════════════════════════════════════════════
#  CROSS-VIDEO COMPARATIVE PLOTS
# ══════════════════════════════════════════════

def plot_blink_rate_comparison(all_results: list[dict], save_path: str) -> str:
    """Grouped bar chart comparing blink rates across all videos."""
    names = [r["video_name"] for r in all_results]
    blinks = [r["blink_summary"]["total_blinks"] for r in all_results]
    bpm = [r["blink_summary"]["blinks_per_minute"] for r in all_results]
    durations = [r["blink_summary"]["duration_seconds"] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=COLORS["bg"])

    # Panel 1: Total blinks
    bars1 = axes[0].bar(names, blinks, color=PALETTE_8[:len(names)], edgecolor="white")
    axes[0].bar_label(bars1, fontsize=10, fontweight="bold")
    axes[0].set_title("Total Blinks", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=35)

    # Panel 2: Blinks per minute
    bars2 = axes[1].bar(names, bpm, color=PALETTE_8[:len(names)], edgecolor="white")
    axes[1].bar_label(bars2, fmt="%.1f", fontsize=10, fontweight="bold")
    axes[1].set_title("Blink Rate (blinks/min)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Blinks / min")
    axes[1].tick_params(axis="x", rotation=35)
    # Normal range reference band (15-20 blinks/min)
    axes[1].axhspan(15, 20, color=COLORS["success"], alpha=0.12, label="Normal range (15–20)")
    axes[1].legend(fontsize=8)

    # Panel 3: Duration vs Blinks scatter
    axes[2].scatter(durations, blinks, c=PALETTE_8[:len(names)], s=120, edgecolors="white", zorder=5)
    for i, name in enumerate(names):
        axes[2].annotate(name, (durations[i], blinks[i]),
                         textcoords="offset points", xytext=(6, 6), fontsize=8)
    axes[2].set_xlabel("Video Duration (s)")
    axes[2].set_ylabel("Total Blinks")
    axes[2].set_title("Duration vs Blinks", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_facecolor(COLORS["bg"])
        ax.grid(True, alpha=0.3)

    fig.suptitle("Task A — Blink Detection Comparison Across Videos",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_save_and_b64(fig, save_path)


def plot_ear_overlay(all_results: list[dict], save_path: str) -> str:
    """Overlay all EAR signals on one axis (time-normalized to %)."""
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=COLORS["bg"])

    for i, r in enumerate(all_results):
        ear = np.array(r["ear_values"])
        pct = np.linspace(0, 100, len(ear))
        ax.plot(pct, ear, linewidth=0.8, color=PALETTE_8[i % len(PALETTE_8)],
                alpha=0.75, label=r["video_name"])

    threshold = all_results[0]["blink_summary"]["ear_threshold"]
    ax.axhline(y=threshold, color=COLORS["danger"], linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold}")

    ax.set_xlabel("Video Progress (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("EAR", fontsize=11, fontweight="bold")
    ax.set_title("EAR Signals Overlaid (Time-Normalized)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, alpha=0.3)

    return _fig_save_and_b64(fig, save_path)


def plot_ear_heatmap(all_results: list[dict], save_path: str) -> str:
    """Heatmap of EAR over time for all videos (rows=videos, cols=time bins)."""
    n_bins = 100
    names = [r["video_name"] for r in all_results]
    matrix = []
    for r in all_results:
        ear = np.array(r["ear_values"])
        # Resample to n_bins
        indices = np.linspace(0, len(ear) - 1, n_bins).astype(int)
        matrix.append(ear[indices])

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(16, 4), facecolor=COLORS["bg"])
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", interpolation="bilinear",
                   vmin=0.10, vmax=0.35)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Video Progress (%) →", fontsize=11, fontweight="bold")
    ax.set_title("EAR Heatmap Across Videos (Green = eyes open, Red = eyes closing)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="EAR")
    ax.set_facecolor(COLORS["bg"])

    return _fig_save_and_b64(fig, save_path)


def plot_dimensions_comparison(all_results: list[dict], save_path: str) -> str:
    """Grouped bar chart: each measurement across all videos."""
    # Collect all measurement names
    valid = [r for r in all_results if r.get("meas_summary", {}).get("measurements_mm")]
    if not valid:
        return ""

    meas_names = list(valid[0]["meas_summary"]["measurements_mm"].keys())
    names = [r["video_name"] for r in valid]

    x = np.arange(len(meas_names))
    width = 0.8 / len(valid)

    fig, ax = plt.subplots(figsize=(18, 7), facecolor=COLORS["bg"])
    for i, r in enumerate(valid):
        vals = [r["meas_summary"]["measurements_mm"].get(m, 0) for m in meas_names]
        offset = (i - len(valid) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=r["video_name"],
               color=PALETTE_8[i % len(PALETTE_8)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(meas_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Dimension (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Task B — Face Dimensions Comparison Across Videos",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return _fig_save_and_b64(fig, save_path)


def plot_dimensions_radar(all_results: list[dict], save_path: str) -> str:
    """Radar/spider chart comparing dimensions (normalized) across videos."""
    valid = [r for r in all_results if r.get("meas_summary", {}).get("measurements_mm")]
    if not valid:
        return ""

    meas_names = list(valid[0]["meas_summary"]["measurements_mm"].keys())
    short_labels = [n.replace("face_", "").replace("(", "\n(").replace("_", " ") for n in meas_names]
    N = len(meas_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), facecolor=COLORS["bg"])

    # Normalize: for each dimension, max across videos = 1
    all_vals = {m: [] for m in meas_names}
    for r in valid:
        for m in meas_names:
            all_vals[m].append(r["meas_summary"]["measurements_mm"].get(m, 0))
    max_vals = {m: max(vs) if max(vs) > 0 else 1 for m, vs in all_vals.items()}

    for i, r in enumerate(valid):
        vals = [r["meas_summary"]["measurements_mm"].get(m, 0) / max_vals[m] for m in meas_names]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, color=PALETTE_8[i % len(PALETTE_8)],
                label=r["video_name"])
        ax.fill(angles, vals, color=PALETTE_8[i % len(PALETTE_8)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_title("Face Dimensions — Normalized Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.set_facecolor(COLORS["bg"])

    return _fig_save_and_b64(fig, save_path)


def plot_detection_quality(all_results: list[dict], save_path: str) -> str:
    """Detection confidence / quality metrics across videos."""
    names = [r["video_name"] for r in all_results]
    detection_rates = [r["blink_summary"].get("face_detection_rate", 0) for r in all_results]
    frames_proc = [r["blink_summary"]["frames_processed"] for r in all_results]
    throughputs = [r.get("throughput_fps", 0) for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=COLORS["bg"])

    # Face detection rate
    bar_colors = [COLORS["success"] if d > 90 else COLORS["accent"] if d > 70
                  else COLORS["danger"] for d in detection_rates]
    bars1 = axes[0].bar(names, detection_rates, color=bar_colors, edgecolor="white")
    axes[0].bar_label(bars1, fmt="%.1f%%", fontsize=9, fontweight="bold")
    axes[0].set_title("Face Detection Rate", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Detection %")
    axes[0].set_ylim(0, 110)
    axes[0].axhline(y=90, color=COLORS["success"], linestyle="--", alpha=0.5, label="90% threshold")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(fontsize=8)

    # Frames processed
    bars2 = axes[1].bar(names, frames_proc, color=PALETTE_8[:len(names)], edgecolor="white")
    axes[1].bar_label(bars2, fontsize=9, fontweight="bold")
    axes[1].set_title("Frames Processed", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Frames")
    axes[1].tick_params(axis="x", rotation=35)

    # Processing throughput
    bars3 = axes[2].bar(names, throughputs, color=PALETTE_8[:len(names)], edgecolor="white")
    axes[2].bar_label(bars3, fmt="%.0f", fontsize=9, fontweight="bold")
    axes[2].set_title("Processing Throughput", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Frames / sec")
    axes[2].tick_params(axis="x", rotation=35)

    for ax in axes:
        ax.set_facecolor(COLORS["bg"])
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Processing Quality & Performance",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_save_and_b64(fig, save_path)


def plot_summary_dashboard(all_results: list[dict], save_path: str) -> str:
    """Big overview dashboard: 2×2 grid of key insights."""
    fig = plt.figure(figsize=(20, 14), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    names = [r["video_name"] for r in all_results]
    n = len(names)

    # ─── Panel 1: Blink rates stacked with detection rate overlay ───
    ax1 = fig.add_subplot(gs[0, 0])
    bpm = [r["blink_summary"]["blinks_per_minute"] for r in all_results]
    det = [r["blink_summary"].get("face_detection_rate", 0) for r in all_results]

    bars = ax1.bar(names, bpm, color=PALETTE_8[:n], edgecolor="white", zorder=3)
    ax1.bar_label(bars, fmt="%.1f", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Blinks / min", color=COLORS["primary"])
    ax1.set_title("Blink Rate & Detection Quality", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    ax1.axhspan(15, 20, color=COLORS["success"], alpha=0.1, zorder=1)

    ax1b = ax1.twinx()
    ax1b.plot(names, det, "D-", color=COLORS["danger"], markersize=6, linewidth=1.5,
              label="Detection %", zorder=4)
    ax1b.set_ylabel("Detection %", color=COLORS["danger"])
    ax1b.set_ylim(0, 110)
    ax1b.legend(fontsize=8, loc="lower right")

    # ─── Panel 2: EAR statistics box plots ───
    ax2 = fig.add_subplot(gs[0, 1])
    ear_data = []
    for r in all_results:
        arr = np.array(r["ear_values"])
        ear_data.append(arr[~np.isnan(arr)])

    bp = ax2.boxplot(ear_data, labels=names, patch_artist=True, widths=0.6,
                     medianprops=dict(color="white", linewidth=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE_8[i % len(PALETTE_8)])
        patch.set_alpha(0.75)
    ax2.axhline(y=all_results[0]["blink_summary"]["ear_threshold"],
                color=COLORS["danger"], linestyle="--", linewidth=1, label="Threshold")
    ax2.set_ylabel("EAR Value")
    ax2.set_title("EAR Distribution per Video", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(fontsize=8)

    # ─── Panel 3: Face dimensions heatmap ───
    ax3 = fig.add_subplot(gs[1, 0])
    valid = [r for r in all_results if r.get("meas_summary", {}).get("measurements_mm")]
    if valid:
        meas_names = list(valid[0]["meas_summary"]["measurements_mm"].keys())
        short = [m.replace("_", "\n") for m in meas_names]
        matrix = []
        v_names = []
        for r in valid:
            row = [r["meas_summary"]["measurements_mm"].get(m, 0) for m in meas_names]
            matrix.append(row)
            v_names.append(r["video_name"])
        matrix = np.array(matrix)
        im = ax3.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax3.set_xticks(range(len(meas_names)))
        ax3.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
        ax3.set_yticks(range(len(v_names)))
        ax3.set_yticklabels(v_names, fontsize=9)
        # Annotate values
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                ax3.text(xi, yi, f"{matrix[yi, xi]:.1f}", ha="center", va="center",
                         fontsize=7, fontweight="bold", color="black")
        fig.colorbar(im, ax=ax3, shrink=0.75, label="mm")
    ax3.set_title("Dimensions Heatmap (mm)", fontsize=12, fontweight="bold")

    # ─── Panel 4: Processing summary ───
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    total_frames = sum(r["blink_summary"]["frames_processed"] for r in all_results)
    total_blinks = sum(r["blink_summary"]["total_blinks"] for r in all_results)
    total_dur = sum(r["blink_summary"]["duration_seconds"] for r in all_results)
    avg_bpm = np.mean(bpm)
    avg_det = np.mean(det)
    total_proc_time = sum(r.get("processing_time", 0) for r in all_results)

    summary_text = (
        f"{'═' * 42}\n"
        f"  BATCH PROCESSING SUMMARY\n"
        f"{'═' * 42}\n\n"
        f"  Videos processed     :  {n}\n"
        f"  Total frames         :  {total_frames:,}\n"
        f"  Total duration       :  {total_dur:.1f}s ({total_dur/60:.1f} min)\n"
        f"  Total processing     :  {total_proc_time:.1f}s\n\n"
        f"  ── Task A: Blink Detection ──\n"
        f"  Total blinks (all)   :  {total_blinks}\n"
        f"  Avg blinks/min       :  {avg_bpm:.1f}\n"
        f"  Avg detection rate   :  {avg_det:.1f}%\n\n"
        f"  ── Task B: Dimensions ──\n"
        f"  Videos measured      :  {len(valid)}\n"
        f"{'═' * 42}"
    )
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.8", facecolor=COLORS["card_bg"],
                       edgecolor=COLORS["slate"], alpha=0.9))
    ax4.set_title("Summary", fontsize=12, fontweight="bold")

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(COLORS["bg"])
        ax.grid(True, alpha=0.2)

    fig.suptitle("FaceMetrics — Batch Analysis Dashboard",
                 fontsize=17, fontweight="bold", y=0.98,
                 color=COLORS["text"])

    return _fig_save_and_b64(fig, save_path)


# ══════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════

def generate_html_report(
    all_results: list[dict],
    plot_images: dict[str, str],  # name → base64 png
    output_path: str,
):
    """Generate a self-contained HTML report with all charts and data tables."""
    ensure_dirs(output_path)

    n_videos = len(all_results)
    total_blinks = sum(r["blink_summary"]["total_blinks"] for r in all_results)
    total_frames = sum(r["blink_summary"]["frames_processed"] for r in all_results)
    total_dur = sum(r["blink_summary"]["duration_seconds"] for r in all_results)
    avg_bpm = np.mean([r["blink_summary"]["blinks_per_minute"] for r in all_results])
    avg_det = np.mean([r["blink_summary"].get("face_detection_rate", 0) for r in all_results])
    total_time = sum(r.get("processing_time", 0) for r in all_results)

    now = datetime.now().strftime("%B %d, %Y at %H:%M")

    # Build per-video cards
    video_cards = ""
    for i, r in enumerate(all_results):
        bs = r["blink_summary"]
        ms = r.get("meas_summary", {})
        ms_mm = ms.get("measurements_mm", {})

        # Measurement rows
        meas_rows_html = ""
        for name, val in ms_mm.items():
            meas_rows_html += f"<tr><td>{name}</td><td><strong>{val:.2f} mm</strong></td></tr>\n"
        if not meas_rows_html:
            meas_rows_html = '<tr><td colspan="2">No measurements (face not detected)</td></tr>'

        ear_key = f"ear_detail_{r['video_name']}"
        dim_key = f"dim_detail_{r['video_name']}"
        ear_img = f'<img src="data:image/png;base64,{plot_images[ear_key]}" />' if ear_key in plot_images else ""
        dim_img = f'<img src="data:image/png;base64,{plot_images[dim_key]}" />' if dim_key in plot_images else ""

        video_cards += f"""
        <div class="video-card">
            <h3>📹 {r['video_name']}</h3>
            <div class="meta">
                {r['video_info']['resolution']} &nbsp;|&nbsp;
                {r['video_info']['fps']:.1f} fps &nbsp;|&nbsp;
                {r['video_info']['total_frames']} frames &nbsp;|&nbsp;
                {r['video_info']['duration']:.1f}s &nbsp;|&nbsp;
                Processed in {r.get('processing_time', 0):.1f}s
            </div>

            <div class="two-col">
                <div class="col">
                    <h4>🔵 Task A: Blink Detection</h4>
                    <table>
                        <tr><td>Total Blinks</td><td><strong>{bs['total_blinks']}</strong></td></tr>
                        <tr><td>Blinks / second</td><td>{bs['blinks_per_second']:.4f}</td></tr>
                        <tr><td>Blinks / minute</td><td><strong>{bs['blinks_per_minute']:.2f}</strong></td></tr>
                        <tr><td>Duration</td><td>{bs['duration_seconds']:.1f}s ({bs['duration_minutes']:.2f} min)</td></tr>
                        <tr><td>Face Detection Rate</td><td>{bs.get('face_detection_rate', 0):.1f}%</td></tr>
                        <tr><td>Frames Processed</td><td>{bs['frames_processed']}</td></tr>
                    </table>
                </div>
                <div class="col">
                    <h4>🟢 Task B: Face Dimensions</h4>
                    <table>
                        {meas_rows_html}
                    </table>
                </div>
            </div>

            <div class="plot-section">
                <h4>EAR Signal</h4>
                {ear_img}
            </div>
            <div class="plot-section">
                <h4>Face Dimensions</h4>
                {dim_img}
            </div>
        </div>
        """

    # Build comparative plot sections
    def _plot_section(title, key, desc=""):
        if key not in plot_images:
            return ""
        return f"""
        <div class="plot-section">
            <h3>{title}</h3>
            {"<p>" + desc + "</p>" if desc else ""}
            <img src="data:image/png;base64,{plot_images[key]}" />
        </div>
        """

    comparative_html = ""
    comparative_html += _plot_section(
        "📊 Overview Dashboard", "dashboard",
        "High-level summary of all videos: blink rates, EAR distributions, dimension heatmap, and processing stats.")
    comparative_html += _plot_section(
        "👁️ Blink Rate Comparison", "blink_comparison",
        "Total blinks, blink rate (blinks/min), and duration-vs-blinks scatter for all videos. "
        "The green band marks the normal resting blink rate (15–20/min).")
    comparative_html += _plot_section(
        "📈 EAR Signal Overlay", "ear_overlay",
        "All EAR signals overlaid on a time-normalized (0–100%) axis for direct comparison.")
    comparative_html += _plot_section(
        "🌡️ EAR Heatmap", "ear_heatmap",
        "Color-coded heatmap — each row is a video, green means eyes open, red means closing/closed.")
    comparative_html += _plot_section(
        "📐 Face Dimensions Comparison", "dim_comparison",
        "Grouped bar chart comparing all 10 face measurements across videos.")
    comparative_html += _plot_section(
        "🕸️ Dimensions Radar", "dim_radar",
        "Normalized radar chart — shows the proportional shape profile for each subject.")
    comparative_html += _plot_section(
        "⚙️ Detection & Processing Quality", "quality",
        "Face detection rate, frames processed, and pipeline throughput per video.")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FaceMetrics — Batch Analysis Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: {COLORS['bg']};
        color: {COLORS['text']};
        line-height: 1.6;
    }}
    .container {{ max-width: 1300px; margin: 0 auto; padding: 20px 30px; }}

    header {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        padding: 40px 0;
        text-align: center;
        border-radius: 0 0 20px 20px;
    }}
    header h1 {{ font-size: 2.4em; margin-bottom: 8px; }}
    header p {{ font-size: 1.1em; opacity: 0.9; }}

    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin: 30px 0;
    }}
    .kpi-card {{
        background: {COLORS['card_bg']};
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid {COLORS['primary']};
    }}
    .kpi-card .value {{ font-size: 2em; font-weight: 800; color: {COLORS['primary']}; }}
    .kpi-card .label {{ font-size: 0.85em; color: {COLORS['slate']}; margin-top: 4px; }}

    h2 {{
        font-size: 1.6em;
        margin: 40px 0 16px;
        padding-bottom: 8px;
        border-bottom: 3px solid {COLORS['primary']};
        display: inline-block;
    }}
    h3 {{ font-size: 1.25em; margin: 20px 0 10px; }}
    h4 {{ font-size: 1.05em; margin: 14px 0 8px; color: {COLORS['slate']}; }}

    .video-card {{
        background: {COLORS['card_bg']};
        border-radius: 14px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        border-top: 4px solid {PALETTE_8[0]};
    }}
    .video-card:nth-child(2) {{ border-top-color: {PALETTE_8[1]}; }}
    .video-card:nth-child(3) {{ border-top-color: {PALETTE_8[2]}; }}
    .video-card:nth-child(4) {{ border-top-color: {PALETTE_8[3]}; }}
    .video-card:nth-child(5) {{ border-top-color: {PALETTE_8[4]}; }}
    .video-card:nth-child(6) {{ border-top-color: {PALETTE_8[5]}; }}
    .video-card:nth-child(7) {{ border-top-color: {PALETTE_8[6]}; }}
    .video-card:nth-child(8) {{ border-top-color: {PALETTE_8[7]}; }}

    .video-card .meta {{
        font-size: 0.88em;
        color: {COLORS['slate']};
        margin-bottom: 16px;
        padding: 6px 12px;
        background: {COLORS['bg']};
        border-radius: 8px;
        display: inline-block;
    }}

    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92em;
    }}
    table td {{
        padding: 7px 10px;
        border-bottom: 1px solid #e2e8f0;
    }}
    table tr:last-child td {{ border-bottom: none; }}
    table td:first-child {{ color: {COLORS['slate']}; }}

    .plot-section {{
        margin: 20px 0;
        text-align: center;
    }}
    .plot-section img {{
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }}
    .plot-section p {{
        font-size: 0.9em;
        color: {COLORS['slate']};
        margin-bottom: 10px;
    }}

    footer {{
        text-align: center;
        padding: 30px 0;
        color: {COLORS['slate']};
        font-size: 0.85em;
    }}
</style>
</head>
<body>

<header>
    <h1>🔬 FaceMetrics — Batch Analysis Report</h1>
    <p>Comprehensive blink detection &amp; face dimension analysis across {n_videos} videos</p>
    <p style="font-size: 0.9em; opacity: 0.7;">Generated on {now}</p>
</header>

<div class="container">

    <!-- KPI Cards -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="value">{n_videos}</div>
            <div class="label">Videos Analyzed</div>
        </div>
        <div class="kpi-card">
            <div class="value">{total_frames:,}</div>
            <div class="label">Total Frames</div>
        </div>
        <div class="kpi-card">
            <div class="value">{total_dur:.1f}s</div>
            <div class="label">Total Duration</div>
        </div>
        <div class="kpi-card">
            <div class="value">{total_blinks}</div>
            <div class="label">Total Blinks</div>
        </div>
        <div class="kpi-card">
            <div class="value">{avg_bpm:.1f}</div>
            <div class="label">Avg Blinks/min</div>
        </div>
        <div class="kpi-card">
            <div class="value">{avg_det:.1f}%</div>
            <div class="label">Avg Detection Rate</div>
        </div>
        <div class="kpi-card">
            <div class="value">{total_time:.1f}s</div>
            <div class="label">Processing Time</div>
        </div>
    </div>

    <!-- Comparative Analysis -->
    <h2>📊 Comparative Analysis</h2>
    {comparative_html}

    <!-- Per-Video Details -->
    <h2>📹 Per-Video Results</h2>
    {video_cards}

</div>

<footer>
    FaceMetrics Batch Report &mdash; {now} &mdash; Processed {n_videos} videos, {total_frames:,} frames
</footer>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  ✓ Saved HTML report → {output_path}")

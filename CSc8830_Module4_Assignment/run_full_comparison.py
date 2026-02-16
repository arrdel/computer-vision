#!/usr/bin/env python3
"""
================================================================================
CSc 8830: Computer Vision - Module 4 Assignment
Full Comparison Pipeline: Traditional CV Boundary Detection vs SAM2
================================================================================

For each sample image this script:
  1. Runs the traditional CV thermal boundary detection pipeline
     (CLAHE, thresholding, morphology, contour filtering, Canny refinement)
  2. Runs SAM2 automatic mask generation
  3. Produces per-image output:
     - Traditional CV pipeline visualization (all intermediate steps)
     - SAM2 multi-region visualization
     - Side-by-side overlay comparison (CV green vs SAM2 red)
     - Overlap / agreement map with IoU & Dice
     - Boundary contour comparison
     - Per-image metrics table
  4. Produces a final cross-image summary figure and console table

Usage:
    python run_full_comparison.py
================================================================================
"""

import os
import sys
import glob
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import torch

# ---- SAM2 imports ----
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ---- Our traditional CV segmenter ----
from thermal_segmentation import ThermalAnimalSegmenter


# ===========================================================================
# Configuration
# ===========================================================================
IMAGE_PATH     = "images"  # Can be a directory or a single file
OUTPUT_DIR     = "comparison_results"
SAM2_CKPT      = "checkpoints/sam2.1_hiera_small.pt"
SAM2_CFG       = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_AREA_MIN  = 0.01   # min mask area as fraction of image
SAM2_AREA_MAX  = 0.60   # max mask area as fraction of image


# ===========================================================================
# Helpers
# ===========================================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_sam2_model(device):
    """Build SAM2 and wrap in an automatic mask generator."""
    print(f"  Loading SAM2 checkpoint on [{device}] ...")
    sam2 = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=100,
    )
    print("  SAM2 ready.\n")
    return generator


def run_sam2(generator, image_bgr):
    """
    Run SAM2 on an image and return filtered masks.
    Returns: best_mask, combined_mask, filtered_mask_list, raw_mask_list
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    total = h * w

    raw = generator.generate(image_rgb)
    raw.sort(key=lambda x: x['area'], reverse=True)

    filtered = [
        m for m in raw
        if SAM2_AREA_MIN * total < m['area'] < SAM2_AREA_MAX * total
    ]

    combined = np.zeros((h, w), dtype=np.uint8)
    for m in filtered:
        combined[m['segmentation']] = 255

    best = (filtered[0]['segmentation'].astype(np.uint8) * 255) if filtered else np.zeros((h, w), dtype=np.uint8)
    return best, combined, filtered, raw


def compute_metrics(mask_a, mask_b):
    """Compute IoU, Dice, Precision, Recall between two binary masks."""
    a = (mask_a > 127).astype(np.float64)
    b = (mask_b > 127).astype(np.float64)
    inter = (a * b).sum()
    union = ((a + b) > 0).sum()
    iou   = inter / union if union > 0 else 0.0
    dice  = 2 * inter / (a.sum() + b.sum()) if (a.sum() + b.sum()) > 0 else 0.0
    prec  = inter / a.sum() if a.sum() > 0 else 0.0
    rec   = inter / b.sum() if b.sum() > 0 else 0.0
    return dict(
        iou=iou, dice=dice, precision=prec, recall=rec,
        cv_coverage=a.sum() / a.size * 100,
        sam_coverage=b.sum() / b.size * 100,
        cv_area=int(a.sum()), sam_area=int(b.sum()),
    )


# ===========================================================================
# Visualization helpers
# ===========================================================================

def fig_cv_pipeline(image_bgr, cv_results, image_name, save_path):
    """
    Full traditional-CV pipeline figure (4x2 grid):
      Row 1: Original | CLAHE Enhanced | Gaussian Blur | Intensity Threshold
      Row 2: Otsu Threshold | Combined Threshold | Morphological Refinement | Final Boundary Overlay
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Traditional CV Pipeline  -  {image_name}", fontsize=16, fontweight='bold')

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    panels = [
        (axes[0, 0], rgb,                             "Original",              None),
        (axes[0, 1], cv_results['enhanced'],           "CLAHE Enhanced",        'gray'),
        (axes[0, 2], cv_results['blurred'],            "Gaussian Blur",         'gray'),
        (axes[0, 3], cv_results['intensity_thresh'],   "Intensity Threshold",   'gray'),
        (axes[1, 0], cv_results['otsu_thresh'],        "Otsu Threshold",        'gray'),
        (axes[1, 1], cv_results['combined_thresh'],    "Combined Threshold",    'gray'),
        (axes[1, 2], cv_results['refined_mask'],       "Morphological Refine",  'gray'),
        (axes[1, 3], cv2.cvtColor(cv_results['boundary_overlay'], cv2.COLOR_BGR2RGB),
                                                       "Final Boundary Overlay", None),
    ]

    for ax, img, title, cmap in panels:
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Annotate animal count
    n = cv_results['num_animals']
    mask_pct = cv_results['final_mask'].sum() / (cv_results['final_mask'].size) * 100 / 255
    axes[1, 3].text(
        0.02, 0.02,
        f"{n} animal(s)  |  {mask_pct:.1f}% coverage",
        transform=axes[1, 3].transAxes, fontsize=10, color='white',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'),
        verticalalignment='bottom',
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    -> {save_path}")


def fig_sam2_masks(image_bgr, filtered_masks, raw_masks, image_name, save_path):
    """
    SAM2 visualization (1x2): all raw masks colored  |  filtered masks colored.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"SAM2 Automatic Mask Generation  -  {image_name}", fontsize=16, fontweight='bold')
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # All raw masks
    ax1.imshow(rgb)
    np.random.seed(42)
    for m in raw_masks:
        color = np.concatenate([np.random.random(3), [0.35]])
        overlay = np.zeros((*image_bgr.shape[:2], 4))
        overlay[m['segmentation']] = color
        ax1.imshow(overlay)
    ax1.set_title(f"All SAM2 Masks ({len(raw_masks)} total)", fontsize=12)
    ax1.axis('off')

    # Filtered masks (animal-sized)
    ax2.imshow(rgb)
    np.random.seed(0)
    for i, m in enumerate(filtered_masks):
        color = np.concatenate([np.random.random(3), [0.45]])
        overlay = np.zeros((*image_bgr.shape[:2], 4))
        overlay[m['segmentation']] = color
        ax2.imshow(overlay)
        ys, xs = np.where(m['segmentation'])
        if len(xs) > 0:
            ax2.text(int(xs.mean()), int(ys.mean()), str(i + 1),
                     fontsize=11, color='white', ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    area_range = f"{SAM2_AREA_MIN*100:.0f}%-{SAM2_AREA_MAX*100:.0f}%"
    ax2.set_title(f"Filtered Masks ({len(filtered_masks)} regions, area {area_range})", fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    -> {save_path}")


def fig_comparison(image_bgr, mask_cv, mask_sam, metrics, image_name, save_path):
    """
    3-row comparison figure:
      Row 1: Original  |  CV Overlay  |  SAM2 Overlay
      Row 2: CV Mask   |  SAM2 Mask   |  Overlap Map
      Row 3: Boundary Contours (CV blue, SAM2 red) | Metrics text panel | Histogram of mask areas
    """
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.28, wspace=0.15)
    fig.suptitle(f"Comparison: Traditional CV vs SAM2  -  {image_name}",
                 fontsize=18, fontweight='bold', y=0.98)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    # ---------- Row 1 ----------
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(rgb); ax00.set_title("Original", fontsize=13); ax00.axis('off')

    # CV overlay (green)
    ax01 = fig.add_subplot(gs[0, 1])
    cv_over = rgb.copy()
    g = np.zeros_like(rgb); g[mask_cv > 127] = [0, 255, 0]
    cv_over = cv2.addWeighted(cv_over, 0.6, g, 0.4, 0)
    cnts_cv, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cv_over, cnts_cv, -1, (0, 0, 255), 2)
    ax01.imshow(cv_over); ax01.set_title("Traditional CV (green)", fontsize=13, color='green'); ax01.axis('off')

    # SAM2 overlay (red)
    ax02 = fig.add_subplot(gs[0, 2])
    sam_over = rgb.copy()
    r = np.zeros_like(rgb); r[mask_sam > 127] = [255, 60, 60]
    sam_over = cv2.addWeighted(sam_over, 0.6, r, 0.4, 0)
    cnts_sam, _ = cv2.findContours(mask_sam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sam_over, cnts_sam, -1, (0, 0, 255), 2)
    ax02.imshow(sam_over); ax02.set_title("SAM2 (red)", fontsize=13, color='red'); ax02.axis('off')

    # ---------- Row 2 ----------
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(mask_cv, cmap='Greens', vmin=0, vmax=255)
    ax10.set_title(f"CV Mask ({metrics['cv_coverage']:.1f}% cov)", fontsize=12)
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(mask_sam, cmap='Reds', vmin=0, vmax=255)
    ax11.set_title(f"SAM2 Mask ({metrics['sam_coverage']:.1f}% cov)", fontsize=12)
    ax11.axis('off')

    # Overlap map
    ax12 = fig.add_subplot(gs[1, 2])
    ov = np.zeros((h, w, 3), dtype=np.uint8)
    cv_only  = np.logical_and(mask_cv > 127, mask_sam <= 127)
    sam_only = np.logical_and(mask_sam > 127, mask_cv <= 127)
    agree    = np.logical_and(mask_cv > 127, mask_sam > 127)
    ov[cv_only]  = [0, 200, 0]
    ov[sam_only] = [200, 50, 50]
    ov[agree]    = [255, 230, 0]
    ax12.imshow(ov)
    ax12.set_title(f"Overlap  (IoU={metrics['iou']:.3f})", fontsize=12)
    ax12.legend(
        handles=[Patch(fc='green', label='CV only'),
                 Patch(fc='red', label='SAM2 only'),
                 Patch(fc='#FFE600', label='Agreement')],
        loc='lower right', fontsize=9, framealpha=0.85)
    ax12.axis('off')

    # ---------- Row 3 ----------

    # Boundary contour comparison
    ax20 = fig.add_subplot(gs[2, 0])
    contour_canvas = rgb.copy()
    cv2.drawContours(contour_canvas, cnts_cv, -1, (0, 120, 255), 2)   # blue = CV
    cv2.drawContours(contour_canvas, cnts_sam, -1, (255, 0, 0), 2)    # red  = SAM2
    ax20.imshow(contour_canvas)
    ax20.set_title("Boundary Contours (blue=CV, red=SAM2)", fontsize=12)
    ax20.axis('off')

    # Metrics text panel
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.axis('off')
    table_data = [
        ["Metric", "Value"],
        ["IoU", f"{metrics['iou']:.4f}"],
        ["Dice", f"{metrics['dice']:.4f}"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["CV coverage", f"{metrics['cv_coverage']:.1f}%"],
        ["SAM2 coverage", f"{metrics['sam_coverage']:.1f}%"],
        ["CV area (px)", f"{metrics['cv_area']:,}"],
        ["SAM2 area (px)", f"{metrics['sam_area']:,}"],
    ]
    tbl = ax21.table(
        cellText=table_data[1:], colLabels=table_data[0],
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    # Style header
    for j in range(2):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    ax21.set_title("Quantitative Metrics", fontsize=13, fontweight='bold')

    # Coverage bar chart
    ax22 = fig.add_subplot(gs[2, 2])
    methods = ['Traditional CV', 'SAM2']
    coverages = [metrics['cv_coverage'], metrics['sam_coverage']]
    colors = ['#2ca02c', '#d62728']
    bars = ax22.bar(methods, coverages, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, coverages):
        ax22.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                  f"{val:.1f}%", ha='center', fontsize=12, fontweight='bold')
    ax22.set_ylabel("Mask Coverage (%)", fontsize=11)
    ax22.set_title("Coverage Comparison", fontsize=13, fontweight='bold')
    ax22.set_ylim(0, max(coverages) * 1.25 + 5)
    ax22.spines['top'].set_visible(False)
    ax22.spines['right'].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    -> {save_path}")


def fig_summary(all_results, save_path):
    """
    Cross-image summary figure with a grouped bar chart and metrics table.
    """
    names = list(all_results.keys())
    n = len(names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5 + n * 0.3))
    fig.suptitle("Overall Comparison Summary", fontsize=16, fontweight='bold')

    # Grouped bar chart: IoU and Dice per image
    x = np.arange(n)
    width = 0.3
    ious  = [all_results[nm]['iou'] for nm in names]
    dices = [all_results[nm]['dice'] for nm in names]
    ax1.bar(x - width/2, ious, width, label='IoU', color='#4472C4', edgecolor='black')
    ax1.bar(x + width/2, dices, width, label='Dice', color='#ED7D31', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("IoU & Dice per Image", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Summary table
    ax2.axis('off')
    header = ["Image", "IoU", "Dice", "Precision", "Recall", "CV %", "SAM2 %"]
    rows = []
    for nm in names:
        m = all_results[nm]
        rows.append([nm,
                     f"{m['iou']:.4f}", f"{m['dice']:.4f}",
                     f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                     f"{m['cv_coverage']:.1f}", f"{m['sam_coverage']:.1f}"])
    # Average row
    avg = lambda k: np.mean([all_results[nm][k] for nm in names])
    rows.append(["AVERAGE",
                 f"{avg('iou'):.4f}", f"{avg('dice'):.4f}",
                 f"{avg('precision'):.4f}", f"{avg('recall'):.4f}",
                 f"{avg('cv_coverage'):.1f}", f"{avg('sam_coverage'):.1f}"])

    tbl = ax2.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.6)
    for j in range(len(header)):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight average row
    last = len(rows)
    for j in range(len(header)):
        tbl[last, j].set_facecolor('#D9E2F3')
        tbl[last, j].set_text_props(fontweight='bold')
    ax2.set_title("Metrics Table", fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    -> {save_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    start_all = time.time()
    device = get_device()

    print("=" * 65)
    print("  CSc 8830 Module 4 - Full Comparison Pipeline")
    print("  Traditional CV Boundary Detection  vs  SAM2")
    print("=" * 65)
    print(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device : {device}")
    print(f"  Images : {IMAGE_PATH}")
    print(f"  Output : {OUTPUT_DIR}/")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load models ---
    sam_gen = load_sam2_model(device)
    cv_seg  = ThermalAnimalSegmenter()

    # --- Collect images (handle both single files and directories) ---
    if os.path.isfile(IMAGE_PATH):
        # Single file
        paths = [IMAGE_PATH]
    elif os.path.isdir(IMAGE_PATH):
        # Directory - collect all supported image formats
        paths = sorted(
            glob.glob(os.path.join(IMAGE_PATH, "*.png")) +
            glob.glob(os.path.join(IMAGE_PATH, "*.jpg")) +
            glob.glob(os.path.join(IMAGE_PATH, "*.jpeg"))
        )
    else:
        print(f"  [!] IMAGE_PATH '{IMAGE_PATH}' is neither a file nor a directory. Exiting.")
        sys.exit(1)
    
    print(f"  Found {len(paths)} image(s): {[Path(p).name for p in paths]}\n")
    if not paths:
        print("  No images found. Exiting.")
        sys.exit(1)

    all_metrics = {}

    for idx, img_path in enumerate(paths, 1):
        name = Path(img_path).stem
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [!] Could not read {img_path}, skipping.")
            continue

        h, w = image.shape[:2]
        print(f"  [{idx}/{len(paths)}] {name}  ({w}x{h})")
        print(f"  {'-'*55}")
        t0 = time.time()

        # ---- Traditional CV ----
        print("    Running Traditional CV pipeline ...")
        cv_results = cv_seg.segment(image)
        mask_cv = cv_results['final_mask']
        t_cv = time.time() - t0
        print(f"      -> {cv_results['num_animals']} animal(s) detected   [{t_cv:.2f}s]")

        # ---- SAM2 ----
        print("    Running SAM2 automatic segmentation ...")
        t1 = time.time()
        best_sam, combined_sam, filt_masks, raw_masks = run_sam2(sam_gen, image)
        t_sam = time.time() - t1
        print(f"      -> {len(raw_masks)} total masks, {len(filt_masks)} filtered   [{t_sam:.2f}s]")

        # Ensure same shape
        mask_sam = combined_sam
        if mask_cv.shape[:2] != mask_sam.shape[:2]:
            mask_sam = cv2.resize(mask_sam, (mask_cv.shape[1], mask_cv.shape[0]))

        # ---- Metrics ----
        metrics = compute_metrics(mask_cv, mask_sam)
        metrics['cv_time']  = t_cv
        metrics['sam_time'] = t_sam
        all_metrics[name] = metrics

        print(f"      IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")
        print(f"      CV cov={metrics['cv_coverage']:.1f}%  "
              f"SAM2 cov={metrics['sam_coverage']:.1f}%")

        # ---- Generate per-image figures ----
        print("    Generating visualizations ...")

        # 1) CV pipeline
        fig_cv_pipeline(
            image, cv_results, name,
            os.path.join(OUTPUT_DIR, f"{name}_01_cv_pipeline.png"),
        )

        # 2) SAM2 masks
        fig_sam2_masks(
            image, filt_masks, raw_masks, name,
            os.path.join(OUTPUT_DIR, f"{name}_02_sam2_masks.png"),
        )

        # 3) Full comparison
        fig_comparison(
            image, mask_cv, mask_sam, metrics, name,
            os.path.join(OUTPUT_DIR, f"{name}_03_comparison.png"),
        )

        # 4) Save raw masks
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_cv_mask.png"), mask_cv)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_sam2_mask.png"), mask_sam)

        print()

    # ---- Cross-image summary ----
    if all_metrics:
        print("  Generating summary figure ...")
        fig_summary(all_metrics, os.path.join(OUTPUT_DIR, "00_summary.png"))

    # ---- Console summary table ----
    print()
    print("=" * 72)
    print("  FINAL RESULTS")
    print("=" * 72)
    hdr = f"  {'Image':<12} {'IoU':>7} {'Dice':>7} {'Prec':>7} {'Rec':>7} {'CV%':>6} {'SAM%':>6} {'CV(s)':>6} {'SAM(s)':>7}"
    print(hdr)
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for name, m in all_metrics.items():
        print(f"  {name:<12} {m['iou']:>7.4f} {m['dice']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['cv_coverage']:>5.1f}% {m['sam_coverage']:>5.1f}% "
              f"{m['cv_time']:>6.2f} {m['sam_time']:>7.2f}")

    n_imgs = len(all_metrics)
    if n_imgs > 1:
        avg = lambda k: np.mean([m[k] for m in all_metrics.values()])
        print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
        print(f"  {'AVERAGE':<12} {avg('iou'):>7.4f} {avg('dice'):>7.4f} "
              f"{avg('precision'):>7.4f} {avg('recall'):>7.4f} "
              f"{avg('cv_coverage'):>5.1f}% {avg('sam_coverage'):>5.1f}% "
              f"{avg('cv_time'):>6.2f} {avg('sam_time'):>7.2f}")

    total_time = time.time() - start_all
    print("=" * 72)
    print(f"  Total time : {total_time:.1f}s")
    print(f"  Outputs in : {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 72)


if __name__ == '__main__':
    main()

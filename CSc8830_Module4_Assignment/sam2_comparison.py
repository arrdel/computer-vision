"""
CSc 8830 Computer Vision - Module 4 Assignment
SAM2 Segmentation Comparison

Compares traditional CV boundary detection with SAM2.
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from thermal_segmentation import ThermalAnimalSegmenter


def load_sam2(checkpoint_path, model_cfg, device):
    """Load SAM2 model and create automatic mask generator."""
    print(f"  Loading SAM2 on {device}...")
    sam2 = build_sam2(model_cfg, checkpoint_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=100,
    )
    print("  SAM2 loaded successfully.")
    return mask_generator


def sam2_segment(mask_generator, image):
    """Run SAM2 automatic mask generation and pick animal masks."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    total_pixels = h * w

    masks_data = mask_generator.generate(image_rgb)
    masks_data.sort(key=lambda x: x['area'], reverse=True)

    filtered = [
        m for m in masks_data
        if 0.01 * total_pixels < m['area'] < 0.60 * total_pixels
    ]

    all_masks_combined = np.zeros((h, w), dtype=np.uint8)
    for m in filtered:
        all_masks_combined[m['segmentation']] = 255

    if filtered:
        best_mask = (filtered[0]['segmentation'].astype(np.uint8)) * 255
    else:
        best_mask = np.zeros((h, w), dtype=np.uint8)

    return best_mask, all_masks_combined, filtered


def compute_metrics(mask_cv, mask_sam):
    """Compute IoU, Dice, Precision, Recall between two binary masks."""
    cv_bin = (mask_cv > 127).astype(np.uint8)
    sam_bin = (mask_sam > 127).astype(np.uint8)

    intersection = np.logical_and(cv_bin, sam_bin).sum()
    union = np.logical_or(cv_bin, sam_bin).sum()

    iou = intersection / union if union > 0 else 0.0
    dice_denom = cv_bin.sum() + sam_bin.sum()
    dice = (2 * intersection) / dice_denom if dice_denom > 0 else 0.0
    precision = intersection / cv_bin.sum() if cv_bin.sum() > 0 else 0.0
    recall = intersection / sam_bin.sum() if sam_bin.sum() > 0 else 0.0

    cv_coverage = cv_bin.sum() / cv_bin.size * 100
    sam_coverage = sam_bin.sum() / sam_bin.size * 100

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'cv_coverage_pct': cv_coverage,
        'sam_coverage_pct': sam_coverage,
        'cv_area': int(cv_bin.sum()),
        'sam_area': int(sam_bin.sum()),
        'intersection': int(intersection),
        'union': int(union),
    }


def visualize_side_by_side(image, mask_cv, mask_sam, metrics, output_path, image_name):
    """Create 2x3 side-by-side comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f'Traditional CV vs SAM2 - {image_name}',
        fontsize=18, fontweight='bold', y=0.98
    )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Row 1: Original
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Thermal Image', fontsize=13)

    # Row 1: Traditional CV overlay (green)
    cv_overlay = image_rgb.copy()
    green_layer = np.zeros_like(image_rgb)
    green_layer[mask_cv > 127] = [0, 255, 0]
    cv_overlay = cv2.addWeighted(cv_overlay, 0.6, green_layer, 0.4, 0)
    cv_contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cv_overlay, cv_contours, -1, (255, 0, 0), 2)
    axes[0, 1].imshow(cv_overlay)
    axes[0, 1].set_title('Traditional CV Segmentation', fontsize=13, color='green')

    # Row 1: SAM2 overlay (red)
    sam_overlay = image_rgb.copy()
    red_layer = np.zeros_like(image_rgb)
    red_layer[mask_sam > 127] = [255, 50, 50]
    sam_overlay = cv2.addWeighted(sam_overlay, 0.6, red_layer, 0.4, 0)
    sam_contours, _ = cv2.findContours(mask_sam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sam_overlay, sam_contours, -1, (0, 0, 255), 2)
    axes[0, 2].imshow(sam_overlay)
    axes[0, 2].set_title('SAM2 Segmentation', fontsize=13, color='red')

    # Row 2: CV mask
    axes[1, 0].imshow(mask_cv, cmap='Greens', vmin=0, vmax=255)
    axes[1, 0].set_title(f'CV Mask ({metrics["cv_coverage_pct"]:.1f}% coverage)', fontsize=12)

    # Row 2: SAM2 mask
    axes[1, 1].imshow(mask_sam, cmap='Reds', vmin=0, vmax=255)
    axes[1, 1].set_title(f'SAM2 Mask ({metrics["sam_coverage_pct"]:.1f}% coverage)', fontsize=12)

    # Row 2: Overlap visualization
    overlap_img = np.zeros((*mask_cv.shape[:2], 3), dtype=np.uint8)
    cv_only = np.logical_and(mask_cv > 127, mask_sam <= 127)
    sam_only = np.logical_and(mask_sam > 127, mask_cv <= 127)
    both = np.logical_and(mask_cv > 127, mask_sam > 127)
    overlap_img[cv_only] = [0, 200, 0]
    overlap_img[sam_only] = [200, 0, 0]
    overlap_img[both] = [255, 255, 0]
    axes[1, 2].imshow(overlap_img)
    axes[1, 2].set_title(f'Overlap (IoU={metrics["iou"]:.3f}, Dice={metrics["dice"]:.3f})', fontsize=12)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='CV Only'),
        Patch(facecolor='red', label='SAM2 Only'),
        Patch(facecolor='yellow', label='Agreement'),
    ]
    axes[1, 2].legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.8)

    metrics_text = (
        f"Comparison Metrics\n"
        f"IoU:       {metrics['iou']:.4f}\n"
        f"Dice:      {metrics['dice']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"CV Area:   {metrics['cv_area']:>8,} px\n"
        f"SAM Area:  {metrics['sam_area']:>8,} px"
    )
    fig.text(0.01, 0.01, metrics_text, fontsize=10,
             family='monospace', verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {output_path}")


def process_one_image(image_path, cv_segmenter, sam_generator, output_dir):
    """Process a single image through both pipelines and compare."""
    image_name = Path(image_path).stem
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Could not load {image_path}")
        return None

    print(f"\n{'='*50}")
    print(f"  Processing: {image_name}")
    print(f"  Size: {image.shape[1]}x{image.shape[0]}")
    print(f"{'='*50}")

    # Traditional CV
    print("  [1/2] Running Traditional CV segmentation...")
    cv_results = cv_segmenter.segment(image)
    mask_cv = cv_results['final_mask']
    print(f"         Found {cv_results['num_animals']} animal region(s)")

    # SAM2
    print("  [2/2] Running SAM2 segmentation...")
    best_mask_sam, combined_mask_sam, sam_masks = sam2_segment(sam_generator, image)
    print(f"         SAM2 found {len(sam_masks)} region(s)")

    mask_sam = combined_mask_sam

    if mask_cv.shape != mask_sam.shape:
        mask_sam = cv2.resize(mask_sam, (mask_cv.shape[1], mask_cv.shape[0]))

    # Compute Metrics
    metrics = compute_metrics(mask_cv, mask_sam)

    print(f"\n  Results for {image_name}:")
    print(f"    IoU:       {metrics['iou']:.4f}")
    print(f"    Dice:      {metrics['dice']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    CV coverage:  {metrics['cv_coverage_pct']:.1f}%")
    print(f"    SAM coverage: {metrics['sam_coverage_pct']:.1f}%")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f'{image_name}_cv_mask.png'), mask_cv)
    cv2.imwrite(os.path.join(output_dir, f'{image_name}_sam2_mask.png'), mask_sam)

    visualize_side_by_side(
        image, mask_cv, mask_sam, metrics,
        os.path.join(output_dir, f'{image_name}_comparison.png'),
        image_name
    )

    # SAM2 multi-mask visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, m in enumerate(sam_masks):
        color = np.random.random(3)
        mask_overlay = np.zeros((*image.shape[:2], 4))
        mask_overlay[m['segmentation']] = [*color, 0.4]
        ax.imshow(mask_overlay)
        ys, xs = np.where(m['segmentation'])
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            ax.text(cx, cy, f'{i+1}', fontsize=10, color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    ax.set_title(f'SAM2 All Masks ({len(sam_masks)} regions) - {image_name}', fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{image_name}_sam2_all_masks.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description='SAM2 vs Traditional CV Comparison')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image_dir', type=str, default='images', help='Image directory')
    parser.add_argument('--output_dir', type=str, default='sam2_results', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sam2.1_hiera_small.pt',
                        help='SAM2 checkpoint path')
    parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_s.yaml',
                        help='SAM2 model config')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("  Traditional CV vs SAM2 Comparison")
    print("  Thermal Animal Boundary Segmentation")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")

    # Load SAM2
    sam_generator = load_sam2(args.checkpoint, args.model_cfg, device)

    # Create traditional CV segmenter
    cv_segmenter = ThermalAnimalSegmenter()

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    image_paths = sorted(set(image_paths))

    print(f"\n  Images to process: {len(image_paths)}")

    if not image_paths:
        print("  No images found!")
        sys.exit(1)

    # Process each image
    all_metrics = {}
    for img_path in image_paths:
        name = Path(img_path).stem
        metrics = process_one_image(img_path, cv_segmenter, sam_generator, args.output_dir)
        if metrics:
            all_metrics[name] = metrics

    # Summary
    if all_metrics:
        print(f"\n{'='*60}")
        print("  OVERALL COMPARISON SUMMARY")
        print(f"{'='*60}")
        header = f"  {'Image':<15} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}"
        print(header)
        sep = f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
        print(sep)
        for name, m in all_metrics.items():
            print(f"  {name:<15} {m['iou']:>8.4f} {m['dice']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f}")

        avg_iou = np.mean([m['iou'] for m in all_metrics.values()])
        avg_dice = np.mean([m['dice'] for m in all_metrics.values()])
        avg_prec = np.mean([m['precision'] for m in all_metrics.values()])
        avg_rec = np.mean([m['recall'] for m in all_metrics.values()])
        print(sep)
        print(f"  {'AVERAGE':<15} {avg_iou:>8.4f} {avg_dice:>8.4f} "
              f"{avg_prec:>8.4f} {avg_rec:>8.4f}")
        print(f"{'='*60}")
        print(f"  Results saved to: {args.output_dir}/")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()

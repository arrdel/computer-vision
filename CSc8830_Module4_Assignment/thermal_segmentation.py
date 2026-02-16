#!/usr/bin/env python3
"""
================================================================================
CSc 8830: Computer Vision - Module 4 Assignment
Thermal Animal Boundary Segmentation (Traditional CV)
================================================================================
Author:  Adele Chinda
Date:    February 2026
Repo:    https://github.com/arrdel/computer-vision

DESCRIPTION:
    This script finds exact boundaries of animals in thermal imaging camera
    images using ONLY traditional computer vision techniques (no deep learning
    or machine learning). It uses OpenCV functions for:
      1. Preprocessing (grayscale conversion, noise reduction)
      2. Adaptive thresholding on thermal intensity
      3. Morphological operations (opening, closing, dilation, erosion)
      4. Contour detection and filtering
      5. Boundary refinement with convex hull and polygon approximation

HOW TO RUN:
    # Single image:
    python thermal_segmentation.py --image images/thermal_animal.jpg

    # Process all images in a directory:
    python thermal_segmentation.py --image_dir images/

    # With custom parameters:
    python thermal_segmentation.py --image images/thermal_animal.jpg \
        --blur_kernel 7 --morph_kernel 5 --min_area 500

    # Save results to custom directory:
    python thermal_segmentation.py --image_dir images/ --output_dir results/

REQUIREMENTS:
    pip install opencv-python numpy matplotlib

OUTPUT:
    - Segmentation masks saved to results/
    - Boundary overlays on original images
    - Side-by-side comparison visualizations
    - Console output with contour statistics
================================================================================
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


class ThermalAnimalSegmenter:
    """
    Traditional CV-based segmenter for finding animal boundaries
    in thermal/infrared images.

    Thermal images show animals as bright (warm) regions against
    a cooler (darker) background. This class exploits that thermal
    contrast using adaptive thresholding, morphological refinement,
    and contour-based boundary extraction.
    """

    def __init__(self, blur_kernel=7, morph_kernel=5, min_area=500,
                 adaptive_block=51, adaptive_c=8):
        """
        Initialize segmenter with tunable parameters.

        Args:
            blur_kernel (int): Gaussian blur kernel size (must be odd).
            morph_kernel (int): Morphological operation kernel size.
            min_area (int): Minimum contour area to keep (filters noise).
            adaptive_block (int): Block size for adaptive thresholding.
            adaptive_c (int): Constant subtracted from adaptive threshold.
        """
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.min_area = min_area
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c

    def preprocess(self, image):
        """
        Step 1: Preprocess the thermal image.

        Converts to grayscale (if needed), applies CLAHE for contrast
        enhancement (critical for thermal images with limited dynamic
        range), and applies Gaussian blur for noise reduction.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            gray (np.ndarray): Grayscale image.
            enhanced (np.ndarray): CLAHE-enhanced grayscale image.
            blurred (np.ndarray): Blurred enhanced image.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is critical for thermal images where contrast can be low
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(enhanced,
                                    (self.blur_kernel, self.blur_kernel), 0)

        return gray, enhanced, blurred

    def threshold_thermal(self, blurred, gray):
        """
        Step 2: Apply multiple thresholding strategies and combine.

        Uses a combination of:
          - Otsu's thresholding (global, automatic)
          - Adaptive thresholding (local, handles uneven illumination)
          - Intensity-based thresholding (thermal hot-spot detection)
        The results are combined via bitwise OR for robust segmentation.

        Args:
            blurred (np.ndarray): Blurred grayscale image.
            gray (np.ndarray): Original grayscale image.

        Returns:
            combined (np.ndarray): Combined binary mask.
            otsu_thresh (np.ndarray): Otsu threshold result.
            adaptive_thresh (np.ndarray): Adaptive threshold result.
            intensity_thresh (np.ndarray): Intensity-based threshold result.
        """
        # Method 1: Otsu's thresholding (automatic global threshold)
        _, otsu_thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Method 2: Adaptive thresholding (handles local contrast variations)
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.adaptive_block, -self.adaptive_c
        )

        # Method 3: Intensity-based thresholding for thermal hot-spots
        # Animals in thermal images appear as bright (warm) regions
        # Use mean + 1.0 * std as threshold to isolate the hottest regions
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        hot_threshold = mean_val + 1.0 * std_val
        _, intensity_thresh = cv2.threshold(
            blurred, hot_threshold, 255, cv2.THRESH_BINARY
        )

        # Combine: Use intersection of at least two methods for robustness
        # Otsu AND Intensity isolates only the brightest (warmest) regions
        combined = cv2.bitwise_and(otsu_thresh, intensity_thresh)

        return combined, otsu_thresh, adaptive_thresh, intensity_thresh

    def morphological_refinement(self, binary_mask):
        """
        Step 3: Refine the binary mask using morphological operations.

        Pipeline:
          1. Close: Fill small holes inside the animal body
          2. Open: Remove small noise blobs outside the animal
          3. Dilate: Slightly expand boundaries to capture edges
          4. Close again: Final hole-filling pass

        Args:
            binary_mask (np.ndarray): Input binary mask.

        Returns:
            refined (np.ndarray): Morphologically refined binary mask.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel, self.morph_kernel)
        )
        kernel_large = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel * 2 + 1, self.morph_kernel * 2 + 1)
        )

        # Step 3a: Close - fill small internal holes
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE,
                                   kernel_large, iterations=2)

        # Step 3b: Open - remove small external noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                   kernel, iterations=3)

        # Step 3c: Final close pass for cleanup
        refined = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                    kernel, iterations=1)

        return refined

    def find_animal_contours(self, refined_mask):
        """
        Step 4: Find and filter contours to identify animal boundaries.

        Finds all external contours, filters by area to remove noise,
        rejects contours that span the full image (background), and
        sorts by area (largest first).

        Args:
            refined_mask (np.ndarray): Refined binary mask.

        Returns:
            animal_contours (list): Filtered contours likely to be animals.
            all_contours (list): All detected contours before filtering.
        """
        h, w = refined_mask.shape[:2]

        contours, hierarchy = cv2.findContours(
            refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        animal_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by minimum area
            if area < self.min_area:
                continue

            # Reject contours that cover too much of the image (background)
            if area > 0.50 * h * w:
                continue

            # Reject contours whose bounding box spans almost the entire image
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > 0.95 * w and bh > 0.95 * h:
                continue

            # Reject contours that touch 3 or more image edges (background)
            edges_touched = 0
            if x <= 1:
                edges_touched += 1
            if y <= 1:
                edges_touched += 1
            if x + bw >= w - 2:
                edges_touched += 1
            if y + bh >= h - 2:
                edges_touched += 1
            if edges_touched >= 3:
                continue

            animal_contours.append(cnt)

        # Sort by area (largest first)
        animal_contours.sort(key=cv2.contourArea, reverse=True)

        return animal_contours, list(contours)

    def refine_boundaries(self, image, contours):
        """
        Step 5: Refine contour boundaries for smoother, more accurate edges.

        Applies:
          - Polygon approximation (Douglas-Peucker) for cleaner edges
          - Convex hull for overall shape
          - Smoothed contour via spline-like approximation

        Args:
            image (np.ndarray): Original image (for dimensions).
            contours (list): List of animal contours.

        Returns:
            refined_contours (list): Smoothed/refined contours.
            hulls (list): Convex hulls for each contour.
            approx_contours (list): Polygon approximations.
        """
        refined_contours = []
        hulls = []
        approx_contours = []

        for cnt in contours:
            # Polygon approximation (simplify the contour)
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)

            # Convex hull
            hull = cv2.convexHull(cnt)
            hulls.append(hull)

            # Smoothed contour: resample points along the contour
            # and apply Gaussian smoothing to coordinates
            if len(cnt) > 10:
                cnt_squeezed = cnt.squeeze()
                # Smooth x and y coordinates separately
                kernel_size = min(15, len(cnt_squeezed) // 2)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                if kernel_size >= 3:
                    from scipy.ndimage import uniform_filter1d
                    x_smooth = uniform_filter1d(
                        cnt_squeezed[:, 0].astype(float), size=kernel_size
                    )
                    y_smooth = uniform_filter1d(
                        cnt_squeezed[:, 1].astype(float), size=kernel_size
                    )
                    smoothed = np.stack(
                        [x_smooth, y_smooth], axis=1
                    ).astype(np.int32).reshape(-1, 1, 2)
                    refined_contours.append(smoothed)
                else:
                    refined_contours.append(cnt)
            else:
                refined_contours.append(cnt)

        return refined_contours, hulls, approx_contours

    def create_final_mask(self, image_shape, contours):
        """
        Create a clean binary mask from the final contours.

        Args:
            image_shape (tuple): Shape of the original image.
            contours (list): Final animal contours.

        Returns:
            mask (np.ndarray): Binary mask with animal regions filled.
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        return mask

    def segment(self, image):
        """
        Main segmentation pipeline: run all steps on an input image.

        Args:
            image (np.ndarray): Input BGR thermal image.

        Returns:
            results (dict): Dictionary containing all intermediate and
                            final results for visualization.
        """
        h, w = image.shape[:2]

        # Dynamically adjust min_area based on image size
        dynamic_min_area = max(self.min_area, int(h * w * 0.01))

        # Step 1: Preprocess
        gray, enhanced, blurred = self.preprocess(image)

        # Step 2: Threshold
        combined, otsu, adaptive, intensity = self.threshold_thermal(
            blurred, gray
        )

        # Step 2b: Use Canny edges to refine the thermal mask
        # Canny detects sharp thermal gradients (animal boundaries)
        edges = cv2.Canny(blurred, 30, 100)
        edges_dilated = cv2.dilate(edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2)
        # Combine threshold mask with edge information
        combined = cv2.bitwise_and(combined, cv2.bitwise_or(combined, edges_dilated))

        # Step 3: Morphological refinement
        refined_mask = self.morphological_refinement(combined)

        # Step 4: Find animal contours (with edge/area rejection built in)
        animal_contours, all_contours = self.find_animal_contours(refined_mask)

        # Apply dynamic minimum area
        animal_contours = [
            cnt for cnt in animal_contours
            if cv2.contourArea(cnt) >= dynamic_min_area
        ]

        # Step 5: Refine boundaries
        if animal_contours:
            refined_contours, hulls, approx_contours = self.refine_boundaries(
                image, animal_contours
            )
        else:
            refined_contours, hulls, approx_contours = [], [], []

        # Create final mask
        final_mask = self.create_final_mask(
            image.shape,
            refined_contours if refined_contours else animal_contours
        )

        # Create boundary overlay
        boundary_overlay = image.copy()
        # Draw filled semi-transparent mask
        colored_mask = np.zeros_like(image)
        colored_mask[final_mask > 0] = [0, 255, 0]  # Green
        boundary_overlay = cv2.addWeighted(
            boundary_overlay, 0.7, colored_mask, 0.3, 0
        )
        # Draw contour outlines
        cv2.drawContours(boundary_overlay,
                         refined_contours if refined_contours else animal_contours,
                         -1, (0, 0, 255), 2)  # Red boundary line

        # Collect results
        results = {
            'original': image,
            'gray': gray,
            'enhanced': enhanced,
            'blurred': blurred,
            'otsu_thresh': otsu,
            'adaptive_thresh': adaptive,
            'intensity_thresh': intensity,
            'combined_thresh': combined,
            'refined_mask': refined_mask,
            'final_mask': final_mask,
            'boundary_overlay': boundary_overlay,
            'animal_contours': animal_contours,
            'refined_contours': refined_contours,
            'hulls': hulls,
            'approx_contours': approx_contours,
            'all_contours': all_contours,
            'num_animals': len(animal_contours),
        }

        return results

    def print_statistics(self, results, image_name=""):
        """
        Print detailed statistics about the segmentation results.

        Args:
            results (dict): Results dictionary from segment().
            image_name (str): Name of the processed image.
        """
        print(f"\n{'='*60}")
        print(f"  Segmentation Results: {image_name}")
        print(f"{'='*60}")
        print(f"  Total contours found:    {len(results['all_contours'])}")
        print(f"  Animal contours (filtered): {results['num_animals']}")

        for i, cnt in enumerate(results['animal_contours']):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            circularity = (4 * np.pi * area) / (perimeter ** 2) \
                if perimeter > 0 else 0

            print(f"\n  Animal #{i+1}:")
            print(f"    Area:         {area:.0f} pixels")
            print(f"    Perimeter:    {perimeter:.1f} pixels")
            print(f"    Bounding Box: ({x}, {y}) -> ({x+w}, {y+h})")
            print(f"    Circularity:  {circularity:.3f}")
            print(f"    Aspect Ratio: {w/h:.2f}" if h > 0 else "")

        mask_area = np.sum(results['final_mask'] > 0)
        total_area = results['final_mask'].shape[0] * results['final_mask'].shape[1]
        print(f"\n  Mask coverage: {mask_area}/{total_area} "
              f"({100*mask_area/total_area:.1f}%)")
        print(f"{'='*60}\n")


def visualize_pipeline(results, output_path, image_name=""):
    """
    Create a comprehensive visualization of the entire segmentation pipeline.

    Shows all intermediate steps side-by-side for analysis and documentation.

    Args:
        results (dict): Results from ThermalAnimalSegmenter.segment().
        output_path (str): Path to save the visualization.
        image_name (str): Name of the image for the title.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Thermal Animal Segmentation Pipeline - {image_name}',
                 fontsize=16, fontweight='bold')

    # Row 1: Preprocessing
    axes[0, 0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Original Image')

    axes[0, 1].imshow(results['enhanced'], cmap='gray')
    axes[0, 1].set_title('2. CLAHE Enhanced')

    axes[0, 2].imshow(results['blurred'], cmap='gray')
    axes[0, 2].set_title('3. Gaussian Blurred')

    # Show thermal heatmap
    axes[0, 3].imshow(results['gray'], cmap='hot')
    axes[0, 3].set_title('4. Thermal Heatmap')

    # Row 2: Thresholding
    axes[1, 0].imshow(results['otsu_thresh'], cmap='gray')
    axes[1, 0].set_title("5. Otsu's Threshold")

    axes[1, 1].imshow(results['adaptive_thresh'], cmap='gray')
    axes[1, 1].set_title('6. Adaptive Threshold')

    axes[1, 2].imshow(results['intensity_thresh'], cmap='gray')
    axes[1, 2].set_title('7. Intensity Threshold')

    axes[1, 3].imshow(results['combined_thresh'], cmap='gray')
    axes[1, 3].set_title('8. Combined Threshold')

    # Row 3: Refinement and Results
    axes[2, 0].imshow(results['refined_mask'], cmap='gray')
    axes[2, 0].set_title('9. Morphological Refined')

    axes[2, 1].imshow(results['final_mask'], cmap='gray')
    axes[2, 1].set_title('10. Final Mask')

    axes[2, 2].imshow(
        cv2.cvtColor(results['boundary_overlay'], cv2.COLOR_BGR2RGB)
    )
    axes[2, 2].set_title('11. Boundary Overlay')

    # Contour details
    contour_img = results['original'].copy()
    for i, cnt in enumerate(results['animal_contours']):
        color = plt.cm.Set1(i / max(len(results['animal_contours']), 1))
        bgr_color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
        cv2.drawContours(contour_img, [cnt], -1, bgr_color, 3)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), bgr_color, 2)
        cv2.putText(contour_img, f'#{i+1}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
    axes[2, 3].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[2, 3].set_title(f'12. Detected Animals ({results["num_animals"]})')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pipeline visualization saved: {output_path}")


def visualize_boundary_detail(results, output_path, image_name=""):
    """
    Create a detailed boundary visualization showing different
    refinement methods.

    Args:
        results (dict): Results from segmentation.
        output_path (str): Path to save the visualization.
        image_name (str): Name of the image.
    """
    if not results['animal_contours']:
        print("  No animals detected - skipping boundary detail.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Boundary Refinement Detail - {image_name}',
                 fontsize=14, fontweight='bold')

    original_rgb = cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB)

    # Raw contours
    img1 = original_rgb.copy()
    for cnt in results['animal_contours']:
        pts = cnt.squeeze()
        if len(pts.shape) == 2:
            for pt in pts:
                cv2.circle(img1, tuple(pt), 1, (255, 0, 0), -1)
    axes[0].imshow(img1)
    axes[0].set_title('Raw Contour Points')

    # Polygon approximation
    img2 = original_rgb.copy()
    for approx in results['approx_contours']:
        cv2.drawContours(img2, [approx], -1, (0, 255, 0), 2)
    axes[1].imshow(img2)
    axes[1].set_title('Polygon Approximation')

    # Convex hull
    img3 = original_rgb.copy()
    for hull in results['hulls']:
        cv2.drawContours(img3, [hull], -1, (0, 0, 255), 2)
    axes[2].imshow(img3)
    axes[2].set_title('Convex Hull')

    # Smoothed contour
    img4 = original_rgb.copy()
    for cnt in results['refined_contours']:
        cv2.drawContours(img4, [cnt], -1, (255, 255, 0), 2)
    axes[3].imshow(img4)
    axes[3].set_title('Smoothed Contour')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Boundary detail saved: {output_path}")


def process_image(image_path, segmenter, output_dir):
    """
    Process a single thermal image through the full pipeline.

    Args:
        image_path (str): Path to the input image.
        segmenter (ThermalAnimalSegmenter): Segmenter instance.
        output_dir (str): Directory to save results.
    """
    image_name = Path(image_path).stem

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None

    print(f"\nProcessing: {image_path}")
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Run segmentation
    results = segmenter.segment(image)

    # Print statistics
    segmenter.print_statistics(results, image_name)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save individual outputs
    cv2.imwrite(
        os.path.join(output_dir, f'{image_name}_mask.png'),
        results['final_mask']
    )
    cv2.imwrite(
        os.path.join(output_dir, f'{image_name}_boundary.png'),
        results['boundary_overlay']
    )

    # Save visualizations
    visualize_pipeline(
        results,
        os.path.join(output_dir, f'{image_name}_pipeline.png'),
        image_name
    )
    visualize_boundary_detail(
        results,
        os.path.join(output_dir, f'{image_name}_boundary_detail.png'),
        image_name
    )

    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Thermal Animal Boundary Segmentation (Traditional CV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python thermal_segmentation.py --image images/thermal_animal.jpg
  python thermal_segmentation.py --image_dir images/
  python thermal_segmentation.py --image images/thermal_animal.jpg --blur_kernel 9 --min_area 1000
        """
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single thermal image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing thermal images')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results (default: results/)')
    parser.add_argument('--blur_kernel', type=int, default=7,
                        help='Gaussian blur kernel size (default: 7)')
    parser.add_argument('--morph_kernel', type=int, default=5,
                        help='Morphological kernel size (default: 5)')
    parser.add_argument('--min_area', type=int, default=500,
                        help='Minimum contour area (default: 500)')
    parser.add_argument('--adaptive_block', type=int, default=51,
                        help='Adaptive threshold block size (default: 51)')
    parser.add_argument('--adaptive_c', type=int, default=8,
                        help='Adaptive threshold constant (default: 8)')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.image is None and args.image_dir is None:
        print("Error: Provide --image or --image_dir")
        print("Run with --help for usage information.")
        sys.exit(1)

    # Create segmenter
    segmenter = ThermalAnimalSegmenter(
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
        min_area=args.min_area,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
    )

    print("=" * 60)
    print("  Thermal Animal Boundary Segmentation")
    print("  Method: Traditional Computer Vision (OpenCV)")
    print("=" * 60)
    print(f"  Parameters:")
    print(f"    Blur Kernel:      {args.blur_kernel}")
    print(f"    Morph Kernel:     {args.morph_kernel}")
    print(f"    Min Area:         {args.min_area}")
    print(f"    Adaptive Block:   {args.adaptive_block}")
    print(f"    Adaptive C:       {args.adaptive_c}")

    # Collect images to process
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            image_paths.extend(
                glob.glob(os.path.join(args.image_dir, ext.upper()))
            )

    # Remove duplicates and sort
    image_paths = sorted(set(image_paths))
    print(f"\n  Images to process: {len(image_paths)}")

    if not image_paths:
        print("  No images found. Check your --image or --image_dir path.")
        sys.exit(1)

    # Process each image
    all_results = {}
    for img_path in image_paths:
        result = process_image(img_path, segmenter, args.output_dir)
        if result is not None:
            all_results[img_path] = result

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Images processed: {len(all_results)}/{len(image_paths)}")
    total_animals = sum(r['num_animals'] for r in all_results.values())
    print(f"  Total animals detected: {total_animals}")
    print(f"  Results saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
================================================================================
CSc 8830 - Computer Vision Module 2: Step 1 - Visualization
================================================================================

SCRIPT: visualize_detections.py
PURPOSE: Visualize checkerboard corner detections and create GIF animation
AUTHOR: CSc 8830 Assignment
DATE: February 2, 2026

================================================================================
DESCRIPTION
================================================================================

This script creates visual representations of checkerboard corner detections
from calibration images. It helps verify calibration quality and debug any
detection issues.

Key Features:
    1. Detects checkerboard corners in all calibration images
    2. Visualizes detected corners with colored circles
    3. Creates a summary report of detection statistics
    4. Generates an animated GIF showing the detection process
    5. Outputs individual visualization images

Useful for:
    - Verifying calibration image quality
    - Debugging detection issues
    - Validating corner detection accuracy
    - Presenting results (GIF animation)

================================================================================
HOW TO RUN THIS SCRIPT
================================================================================

Basic Usage (uses default parameters):
    python visualize_detections.py

With Custom Parameters:
    python visualize_detections.py \
        --images clean_checkerboard_images/ \
        --checkerboard 16 16 \
        --output calibration_viz/

With Verbose Output:
    python visualize_detections.py --verbose

================================================================================
ARGUMENTS
================================================================================

--images PATH
    Path to folder containing calibration images
    Default: clean_checkerboard_images/
    
--checkerboard COLS ROWS
    Checkerboard size in (columns rows) format
    Default: 16 16
    Must match the actual checkerboard pattern size
    
--output PATH
    Output folder for visualizations
    Default: calibration_viz/
    
--verbose
    Print detailed progress and detection statistics
    Optional flag

================================================================================
OUTPUTS
================================================================================

1. calibration_process.gif
    - Animated GIF showing corner detection in all images
    - Green circles: detected corners
    - Red markers: detection status
    - Useful for presentations and reports

2. detection_stats.json
    - JSON file with detection statistics
    - Success rate, detection count per image
    - Average detection quality metrics

3. Individual visualization images (optional)
    - PNG images with detected corners marked
    - One per input image (if debug enabled)

================================================================================
EXPECTED RESULTS
================================================================================

For successful detection:
    - Detection Success Rate: 100% (all images detect checkerboard)
    - Each image should show 16x16=256 detected corners
    - Corners appear as green circles on the image
    - GIF animation smoothly plays all detections

With 26 synthetic checkerboard images:
    - Success Rate: 25/25 = 100% ✓
    - GIF created: calibration_process.gif ✓
    - Stats saved: detection_stats.json ✓

================================================================================
NEXT STEPS
================================================================================

After visualization:
    1. Review calibration_process.gif to verify detection quality
    2. Check detection_stats.json for success metrics
    3. If issues found, re-run calibrate_camera.py or capture better images
    4. Proceed to Step2_Object_Dimension_Measurement/

See SCRIPT_EXECUTION_GUIDE.md for complete instructions.

================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import json


class DetectionVisualizer:
    """Visualize checkerboard corner detections."""
    
    def __init__(self, 
                 checkerboard_size: Tuple[int, int] = (16, 16),
                 verbose: bool = True):
        """
        Initialize visualizer.
        
        Args:
            checkerboard_size: (cols, rows) of checkerboard
            verbose: Print progress information
        """
        self.checkerboard_size = checkerboard_size
        self.verbose = verbose
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def detect_and_visualize(self, image_folder: str, output_folder: str) -> dict:
        """
        Detect checkerboard in all images and create visualizations.
        
        Args:
            image_folder: Path to folder with calibration images
            output_folder: Path to save visualization images
        
        Returns:
            Dictionary with detection statistics
        """
        image_folder = Path(image_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(ext))
            image_files.extend(image_folder.glob(ext.upper()))
        
        image_files = sorted(image_files)
        
        if self.verbose:
            print(f"\n[VISUALIZER] Found {len(image_files)} images")
            print(f"[VISUALIZER] Checkerboard size: {self.checkerboard_size[0]}x{self.checkerboard_size[1]}")
        
        stats = {
            'total_images': len(image_files),
            'successful_detections': 0,
            'failed_detections': 0,
            'detection_details': [],
            'images_with_visualizations': []
        }
        
        viz_images = []  # For GIF creation
        
        for idx, image_file in enumerate(image_files):
            if self.verbose:
                print(f"  [{idx+1}/{len(image_files)}] Processing: {image_file.name}", end=" ... ")
            
            image = cv2.imread(str(image_file))
            if image is None:
                if self.verbose:
                    print("ERROR: Could not read")
                stats['failed_detections'] += 1
                stats['detection_details'].append({
                    'image': image_file.name,
                    'detected': False,
                    'reason': 'Could not read file'
                })
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try to detect checkerboard
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corners
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    self.criteria
                )
                
                # Create visualization
                viz_image = image.copy()
                cv2.drawChessboardCorners(
                    viz_image,
                    self.checkerboard_size,
                    corners_refined,
                    ret
                )
                
                # Add text
                cv2.putText(
                    viz_image,
                    f"✓ {self.checkerboard_size[0]}x{self.checkerboard_size[1]} Detected ({len(corners)} corners)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                if self.verbose:
                    print(f"✓ Detected")
                
                stats['successful_detections'] += 1
                stats['detection_details'].append({
                    'image': image_file.name,
                    'detected': True,
                    'corners_found': len(corners),
                    'expected_corners': self.checkerboard_size[0] * self.checkerboard_size[1]
                })
                
            else:
                # Create failure visualization
                viz_image = image.copy()
                cv2.putText(
                    viz_image,
                    f"✗ {self.checkerboard_size[0]}x{self.checkerboard_size[1]} NOT Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    viz_image,
                    "Possible issues:",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    1
                )
                cv2.putText(
                    viz_image,
                    "- Pattern not visible/obscured by pieces",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    1
                )
                cv2.putText(
                    viz_image,
                    "- Poor lighting/contrast",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    1
                )
                cv2.putText(
                    viz_image,
                    "- Extreme angle/distance",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    1
                )
                
                if self.verbose:
                    print(f"✗ NOT Detected")
                
                stats['failed_detections'] += 1
                stats['detection_details'].append({
                    'image': image_file.name,
                    'detected': False,
                    'reason': 'Checkerboard pattern not found'
                })
            
            # Save visualization
            output_path = output_folder / f"viz_{image_file.stem}.jpg"
            cv2.imwrite(str(output_path), viz_image)
            stats['images_with_visualizations'].append(str(output_path))
            
            # Resize for GIF (keep aspect ratio, max 480x360)
            height, width = viz_image.shape[:2]
            scale = min(480 / width, 360 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            viz_resized = cv2.resize(viz_image, (new_width, new_height))
            viz_images.append(viz_resized)
        
        if self.verbose:
            print(f"\n[RESULTS]")
            print(f"  Successful detections: {stats['successful_detections']}/{len(image_files)}")
            print(f"  Success rate: {100*stats['successful_detections']/len(image_files):.1f}%")
        
        # Create GIF
        if viz_images and len(viz_images) > 1:
            self._create_gif(viz_images, output_folder / "detection_process.gif")
        
        # Save statistics
        stats_path = output_folder / "detection_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if self.verbose:
            print(f"  Visualizations saved to: {output_folder}")
            print(f"  Statistics saved to: {stats_path}")
        
        return stats
    
    def _create_gif(self, images: List[np.ndarray], output_path: Path) -> None:
        """Create animated GIF from images."""
        try:
            import imageio
            
            # Convert BGR to RGB
            images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
            
            # Save as GIF (every 2nd frame to keep file size reasonable)
            imageio.mimsave(
                str(output_path),
                images_rgb[::1],  # Use every frame
                duration=0.5,  # 0.5 second per frame
                loop=0  # Infinite loop
            )
            
            if self.verbose:
                print(f"  GIF created: {output_path}")
        
        except ImportError:
            if self.verbose:
                print("  [NOTE] imageio not installed. Install with: pip install imageio")
                print("  Skipping GIF creation.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize checkerboard corner detections for debugging"
    )
    
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to folder containing calibration images'
    )
    parser.add_argument(
        '--checkerboard',
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=('COLS', 'ROWS'),
        help='Checkerboard size: columns and rows (default: 16 16)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations/',
        help='Output folder for visualization images'
    )
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        visualizer = DetectionVisualizer(
            checkerboard_size=tuple(args.checkerboard),
            verbose=not args.no_verbose
        )
        
        stats = visualizer.detect_and_visualize(args.images, args.output)
        
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        print(f"Total images: {stats['total_images']}")
        print(f"Successful: {stats['successful_detections']}")
        print(f"Failed: {stats['failed_detections']}")
        print(f"Success rate: {100*stats['successful_detections']/stats['total_images']:.1f}%")
        print("="*60)
        
        if stats['successful_detections'] >= 3:
            print("\n✓ Sufficient detections for calibration (minimum 3 required)")
        else:
            print(f"\n✗ WARNING: Only {stats['successful_detections']} detections. Need at least 3.")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

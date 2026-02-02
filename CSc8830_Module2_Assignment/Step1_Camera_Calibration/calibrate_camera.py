"""
CSc 8830 - Computer Vision Assignment: Step 1
Camera Calibration using Checkerboard Pattern

Description:
    This script performs camera calibration using OpenCV's calibration framework.
    It detects a checkerboard pattern in multiple images and computes:
    - Camera intrinsic matrix K (focal length and principal point)
    - Lens distortion coefficients (radial and tangential)
    - Reprojection error metrics

Theory:
    Camera calibration determines the intrinsic parameters K:
    
    K = [fx   0  cx]
        [ 0  fy  cy]
        [ 0   0   1]
    
    Where:
    - fx, fy: Focal length in pixels
    - cx, cy: Principal point (image center) in pixels
    
    This matrix is essential for perspective projection and 3D reconstruction.

Usage:
    python calibrate_camera.py --images calibration_data/calibration_images/ \
                               --checkerboard 8 6 \
                               --square_size 0.025 \
                               --output calibration_data/


"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional
import json


class CameraCalibrator:
    """
    Camera calibration using checkerboard pattern detection.
    
    Attributes:
        checkerboard_size: Tuple of (columns, rows) in checkerboard
        square_size: Physical size of each square in meters
        criteria: Refinement criteria for corner detection
    """
    
    def __init__(self, 
                 checkerboard_size: Tuple[int, int] = (8, 6),
                 square_size: float = 0.025,
                 verbose: bool = True):
        """
        Initialize the camera calibrator.
        
        Args:
            checkerboard_size: (cols, rows) of internal checkerboard corners
            square_size: Physical size of each square in meters
            verbose: Print progress information
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.verbose = verbose
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Storage for calibration results
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.img_size = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.reprojection_error = None
        
        if self.verbose:
            print(f"[INIT] Camera Calibrator initialized")
            print(f"       Checkerboard: {checkerboard_size[0]} x {checkerboard_size[1]}")
            print(f"       Square size: {square_size*1000:.1f} mm")
    
    def _generate_object_points(self) -> np.ndarray:
        """
        Generate 3D coordinates of checkerboard corners in world space.
        
        Returns:
            Array of shape (cols*rows, 3) with (x, y, z) coordinates
            Z is always 0 (checkerboard is planar)
        """
        cols, rows = self.checkerboard_size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def _detect_checkerboard(self, image: np.ndarray) -> Optional[Tuple[bool, np.ndarray]]:
        """
        Detect checkerboard corners in an image.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Tuple of (success: bool, corners: np.ndarray) or (False, None)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.checkerboard_size,
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corner positions to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(
                gray, 
                corners, 
                (11, 11), 
                (-1, -1), 
                self.criteria
            )
            return True, corners_refined
        else:
            return False, None
    
    def calibrate_from_folder(self, image_folder: str) -> bool:
        """
        Perform calibration using all images in a folder.
        
        Args:
            image_folder: Path to folder containing calibration images
        
        Returns:
            True if calibration successful, False otherwise
        """
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(ext))
            image_files.extend(image_folder.glob(ext.upper()))
        
        if not image_files:
            raise FileNotFoundError(f"No calibration images found in {image_folder}")
        
        image_files = sorted(image_files)
        
        if self.verbose:
            print(f"\n[CALIBRATION] Found {len(image_files)} images in {image_folder}")
        
        objp = self._generate_object_points()
        successful_detections = 0
        
        for idx, image_file in enumerate(image_files):
            if self.verbose:
                print(f"  [{idx+1}/{len(image_files)}] Processing: {image_file.name}", end=" ... ")
            
            image = cv2.imread(str(image_file))
            if image is None:
                if self.verbose:
                    print("ERROR: Could not read image")
                continue
            
            # Store image size
            if self.img_size is None:
                self.img_size = image.shape[1::-1]  # (width, height)
            
            # Detect checkerboard
            ret, corners = self._detect_checkerboard(image)
            
            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                successful_detections += 1
                if self.verbose:
                    print("✓ Detected")
            else:
                if self.verbose:
                    print("✗ Not detected")
        
        if self.verbose:
            print(f"\n[RESULT] Successfully detected checkerboard in {successful_detections}/{len(image_files)} images")
        
        if successful_detections < 3:
            raise ValueError("Need at least 3 successful detections for calibration")
        
        # Perform calibration
        ret, self.camera_matrix, self.distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.img_size,
            None,
            None
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed")
        
        # Calculate reprojection error
        self._calculate_reprojection_error(rvecs, tvecs)
        
        if self.verbose:
            print("\n[SUCCESS] Camera calibration completed successfully!")
        
        return True
    
    def _calculate_reprojection_error(self, rvecs: List, tvecs: List) -> None:
        """
        Calculate the reprojection error (how well the calibration works).
        
        Args:
            rvecs: Rotation vectors from calibration
            tvecs: Translation vectors from calibration
        """
        total_error = 0
        total_points = 0
        
        errors_per_image = []
        
        for i, (objpts, imgpts, rvec, tvec) in enumerate(
            zip(self.objpoints, self.imgpoints, rvecs, tvecs)
        ):
            # Project 3D points to image plane
            projected_points, _ = cv2.projectPoints(
                objpts, rvec, tvec, 
                self.camera_matrix, 
                self.distortion_coeffs
            )
            
            # Calculate error
            error = cv2.norm(imgpts, projected_points, cv2.NORM_L2) / len(projected_points)
            errors_per_image.append(error)
            total_error += error * len(projected_points)
            total_points += len(projected_points)
        
        self.reprojection_error = total_error / total_points
        self.errors_per_image = errors_per_image
        
        if self.verbose:
            print(f"[METRICS] Reprojection Error: {self.reprojection_error:.4f} pixels")
            print(f"          Mean Error per Image: {np.mean(errors_per_image):.4f} pixels")
            print(f"          Std Error: {np.std(errors_per_image):.4f} pixels")
            print(f"          Min Error: {np.min(errors_per_image):.4f} pixels")
            print(f"          Max Error: {np.max(errors_per_image):.4f} pixels")
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get the calibrated camera intrinsic matrix K."""
        if self.camera_matrix is None:
            raise RuntimeError("Calibration not performed yet")
        return self.camera_matrix
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """Get the distortion coefficients."""
        if self.distortion_coeffs is None:
            raise RuntimeError("Calibration not performed yet")
        return self.distortion_coeffs
    
    def save_calibration(self, output_folder: str) -> None:
        """
        Save calibration results to files.
        
        Args:
            output_folder: Path to output folder
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if self.camera_matrix is None:
            raise RuntimeError("No calibration to save")
        
        # Save camera matrix
        camera_matrix_file = output_folder / "camera_matrix.npy"
        np.save(camera_matrix_file, self.camera_matrix)
        
        # Save distortion coefficients
        distortion_file = output_folder / "distortion_coeffs.npy"
        np.save(distortion_file, self.distortion_coeffs)
        
        # Save calibration report as JSON
        report = {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.distortion_coeffs.flatten().tolist(),
            'reprojection_error': float(self.reprojection_error),
            'image_size': list(self.img_size),
            'num_images': len(self.objpoints),
            'checkerboard_size': list(self.checkerboard_size),
            'square_size_mm': float(self.square_size * 1000),
            'errors_per_image': [float(e) for e in self.errors_per_image]
        }
        
        report_file = output_folder / "calibration_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text report
        text_report_file = output_folder / "calibration_results.txt"
        with open(text_report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CAMERA CALIBRATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("CAMERA INTRINSIC MATRIX (K):\n")
            f.write("-" * 60 + "\n")
            f.write(str(self.camera_matrix) + "\n\n")
            
            f.write("FOCAL LENGTH:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  fx = {self.camera_matrix[0, 0]:.2f} pixels\n")
            f.write(f"  fy = {self.camera_matrix[1, 1]:.2f} pixels\n")
            f.write(f"  Average = {(self.camera_matrix[0, 0] + self.camera_matrix[1, 1])/2:.2f} pixels\n\n")
            
            f.write("PRINCIPAL POINT (Optical Center):\n")
            f.write("-" * 60 + "\n")
            f.write(f"  cx = {self.camera_matrix[0, 2]:.2f} pixels\n")
            f.write(f"  cy = {self.camera_matrix[1, 2]:.2f} pixels\n\n")
            
            f.write("DISTORTION COEFFICIENTS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  k1 (radial 1) = {self.distortion_coeffs[0, 0]:.6f}\n")
            f.write(f"  k2 (radial 2) = {self.distortion_coeffs[0, 1]:.6f}\n")
            f.write(f"  p1 (tangential 1) = {self.distortion_coeffs[0, 2]:.6f}\n")
            f.write(f"  p2 (tangential 2) = {self.distortion_coeffs[0, 3]:.6f}\n")
            f.write(f"  k3 (radial 3) = {self.distortion_coeffs[0, 4]:.6f}\n\n")
            
            f.write("CALIBRATION METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Total Reprojection Error: {self.reprojection_error:.4f} pixels\n")
            f.write(f"  Number of Images Used: {len(self.objpoints)}\n")
            f.write(f"  Image Size: {self.img_size[0]} x {self.img_size[1]} pixels\n")
            f.write(f"  Mean Error per Image: {np.mean(self.errors_per_image):.4f} pixels\n")
            f.write(f"  Std Error: {np.std(self.errors_per_image):.4f} pixels\n")
            f.write(f"  Min Error: {np.min(self.errors_per_image):.4f} pixels\n")
            f.write(f"  Max Error: {np.max(self.errors_per_image):.4f} pixels\n\n")
            
            f.write("QUALITY ASSESSMENT:\n")
            f.write("-" * 60 + "\n")
            if self.reprojection_error < 0.5:
                quality = "EXCELLENT"
            elif self.reprojection_error < 1.0:
                quality = "GOOD"
            elif self.reprojection_error < 2.0:
                quality = "ACCEPTABLE"
            else:
                quality = "POOR"
            f.write(f"  Calibration Quality: {quality}\n")
            f.write(f"  Recommendation: {'Ready for use' if self.reprojection_error < 1.5 else 'Recalibrate with better images'}\n")
        
        if self.verbose:
            print(f"\n[SAVE] Calibration saved to: {output_folder}")
            print(f"       - camera_matrix.npy")
            print(f"       - distortion_coeffs.npy")
            print(f"       - calibration_report.json")
            print(f"       - calibration_results.txt")
    
    def print_summary(self) -> None:
        """Print a summary of calibration results."""
        if self.camera_matrix is None:
            print("No calibration data available")
            return
        
        print("\n" + "=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"Focal Length (fx, fy): ({self.camera_matrix[0, 0]:.1f}, {self.camera_matrix[1, 1]:.1f}) pixels")
        print(f"Principal Point (cx, cy): ({self.camera_matrix[0, 2]:.1f}, {self.camera_matrix[1, 2]:.1f}) pixels")
        print(f"Image Size: {self.img_size[0]} x {self.img_size[1]} pixels")
        print(f"Reprojection Error: {self.reprojection_error:.4f} pixels")
        print(f"Number of Calibration Images: {len(self.objpoints)}")
        print("=" * 60 + "\n")


def main():
    """Main function to run camera calibration."""
    parser = argparse.ArgumentParser(
        description="Camera calibration using checkerboard pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate with default settings
  python calibrate_camera.py --images calibration_images/
  
  # Calibrate with custom checkerboard size
  python calibrate_camera.py --images calibration_images/ --checkerboard 9 7
  
  # Calibrate with custom square size
  python calibrate_camera.py --images calibration_images/ --square_size 0.03
        """
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
        default=[8, 6],
        metavar=('COLS', 'ROWS'),
        help='Checkerboard size: columns and rows (default: 8 6)'
    )
    parser.add_argument(
        '--square_size',
        type=float,
        default=0.025,
        help='Physical size of each square in meters (default: 0.025)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration_data/',
        help='Output folder for calibration results (default: calibration_data/)'
    )
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Create calibrator
        calibrator = CameraCalibrator(
            checkerboard_size=tuple(args.checkerboard),
            square_size=args.square_size,
            verbose=not args.no_verbose
        )
        
        # Run calibration
        calibrator.calibrate_from_folder(args.images)
        
        # Save results
        calibrator.save_calibration(args.output)
        
        # Print summary
        calibrator.print_summary()
        
        print("[SUCCESS] Camera calibration completed!")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

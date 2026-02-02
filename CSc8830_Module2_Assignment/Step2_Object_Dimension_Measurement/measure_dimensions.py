"""
Step 2: Object Dimension Measurement

This module implements perspective projection-based measurement of real-world
2D object dimensions from images using calibrated camera intrinsics.

The measurement pipeline:
1. Load calibration results (camera matrix K, distortion coefficients)
2. Load target image and undistort it
3. Detect object of interest (bounding box detection)
4. Measure object dimensions in pixels
5. Convert pixel measurements to real-world dimensions using perspective projection
6. Save results with confidence metrics

Mathematical Foundation:
    From perspective projection theory, an object's real-world height can be
    computed from its image height using the relationship:
    
    H_real = (h_pixel * distance * sensor_height) / (focal_length * image_height)
    
    Where:
    - h_pixel: Object height in pixels
    - distance: Distance from camera to object (meters)
    - sensor_height: Physical sensor height (mm)
    - focal_length: Camera focal length (pixels)
    - image_height: Image height (pixels)


"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
import warnings

from perspective_projection import (
    PerspectiveProjection,
    FieldOfView,
    MeasurementValidator
)


@dataclass
class MeasurementResult:
    """Container for measurement results."""
    image_path: str
    distance_m: float
    height_pixels: float
    width_pixels: float
    height_real_m: float
    width_real_m: float
    height_real_cm: float
    width_real_cm: float
    object_area_pixels: float
    object_area_m2: float
    confidence: float
    error_estimate_percent: float
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ObjectDetector:
    """
    Detects objects of interest in images.
    Provides methods for manual and automatic detection.
    """
    
    def __init__(self, image: np.ndarray, window_name: str = "Object Detection"):
        """
        Initialize detector with image.
        
        Args:
            image: Input image (BGR format)
            window_name: Window title for interactive detection
        """
        self.image = image
        self.window_name = window_name
        self.bbox = None
        self.points = []
    
    def detect_by_contour(
        self,
        lower_color: Tuple[int, int, int],
        upper_color: Tuple[int, int, int],
        min_area: int = 500
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect object by color range using contour detection.
        
        Args:
            lower_color: Lower HSV color bound (H, S, V)
            upper_color: Upper HSV color bound (H, S, V)
            min_area: Minimum contour area to consider
        
        Returns:
            Bounding box (x, y, w, h) or None if not found
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < min_area:
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        self.bbox = (x, y, w, h)
        
        return self.bbox
    
    def detect_by_edges(self, threshold1: int = 50, threshold2: int = 150) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect object using edge detection and contour analysis.
        
        Args:
            threshold1: Lower threshold for Canny edge detection
            threshold2: Upper threshold for Canny edge detection
        
        Returns:
            Bounding box (x, y, w, h) or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, threshold1, threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        self.bbox = (x, y, w, h)
        
        return self.bbox
    
    def detect_manually(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Allow user to manually draw bounding box on image.
        
        Usage: Click and drag to draw rectangle, press 'c' to confirm, 'r' to reset
        
        Returns:
            Bounding box (x, y, w, h) or None if cancelled
        """
        drawing = False
        start_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                # Show preview
                preview = self.image.copy()
                cv2.rectangle(preview, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, preview)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if start_point and (x, y) != start_point:
                    x1, y1 = start_point
                    x2, y2 = x, y
                    
                    # Normalize coordinates
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    
                    self.bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        print("Instructions:")
        print("  - Click and drag to draw bounding box")
        print("  - Press 'c' to confirm")
        print("  - Press 'r' to reset")
        print("  - Press 'q' to cancel")
        
        cv2.imshow(self.window_name, self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Confirm
                if self.bbox:
                    print(f"Bounding box confirmed: {self.bbox}")
                    cv2.destroyWindow(self.window_name)
                    return self.bbox
                else:
                    print("No bounding box drawn yet!")
            
            elif key == ord('r'):  # Reset
                self.bbox = None
                cv2.imshow(self.window_name, self.image)
                print("Reset")
            
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(self.window_name)
                return None
        
        cv2.destroyWindow(self.window_name)
        return None


class ObjectMeasurer:
    """
    Measures real-world dimensions of objects using calibrated camera parameters.
    """
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        sensor_height_mm: float = 5.76,
        num_calibration_images: int = 20
    ):
        """
        Initialize measurer with calibration data.
        
        Args:
            camera_matrix: 3x3 intrinsic matrix K from calibration
            distortion_coeffs: Distortion coefficients from calibration
            sensor_height_mm: Physical sensor height (default: 1/2.3" sensor)
            num_calibration_images: Number of images used in calibration
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.sensor_height_mm = sensor_height_mm
        self.num_calibration_images = num_calibration_images
        
        self.proj = PerspectiveProjection(camera_matrix, distortion_coeffs)
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort image using calibration parameters.
        
        Args:
            image: Input image (may be distorted)
        
        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]
        
        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion_coeffs,
            (w, h),
            1,
            (w, h)
        )
        
        # Undistort
        undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.distortion_coeffs,
            None,
            new_camera_matrix
        )
        
        return undistorted
    
    def measure_object(
        self,
        image: np.ndarray,
        distance_m: float,
        bbox: Tuple[int, int, int, int],
        undistort: bool = True
    ) -> MeasurementResult:
        """
        Measure object dimensions from image.
        
        Args:
            image: Input image (BGR)
            distance_m: Distance from camera to object (meters)
            bbox: Bounding box (x, y, width, height)
            undistort: Whether to undistort image first
        
        Returns:
            MeasurementResult with computed dimensions
        """
        if undistort:
            image = self.undistort_image(image)
        
        # Extract bounding box dimensions
        x, y, w, h = bbox
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Extract object region
        object_region = image[y:y+h, x:x+w]
        
        # Compute real-world dimensions using perspective projection
        height_real_m = PerspectiveProjection.compute_real_world_height(
            height_pixels=h,
            distance_meters=distance_m,
            focal_length_pixels=self.proj.fy,
            image_height_pixels=img_height,
            sensor_height_mm=self.sensor_height_mm
        )
        
        width_real_m = PerspectiveProjection.compute_real_world_width(
            width_pixels=w,
            distance_meters=distance_m,
            focal_length_pixels=self.proj.fx,
            image_width_pixels=img_width,
            sensor_height_mm=self.sensor_height_mm
        )
        
        # Compute areas
        object_area_pixels = w * h
        object_area_m2 = height_real_m * width_real_m
        
        # Compute confidence
        mean_reprojection_error = 0.5  # placeholder
        confidence = MeasurementValidator.compute_measurement_confidence(
            mean_reprojection_error,
            self.num_calibration_images
        )
        
        # Estimate measurement error
        error_dict = MeasurementValidator.estimate_measurement_error()
        error_estimate = error_dict['total_error_percent']
        
        # Create result
        result = MeasurementResult(
            image_path=str(image.shape),
            distance_m=distance_m,
            height_pixels=float(h),
            width_pixels=float(w),
            height_real_m=height_real_m,
            width_real_m=width_real_m,
            height_real_cm=height_real_m * 100,
            width_real_cm=width_real_m * 100,
            object_area_pixels=float(object_area_pixels),
            object_area_m2=object_area_m2,
            confidence=confidence,
            error_estimate_percent=error_estimate
        )
        
        return result
    
    def measure_from_file(
        self,
        image_path: str,
        distance_m: float,
        detection_method: str = 'manual',
        **kwargs
    ) -> MeasurementResult:
        """
        Measure object from image file.
        
        Args:
            image_path: Path to image file
            distance_m: Distance to object (meters)
            detection_method: 'manual', 'contour', or 'edges'
            **kwargs: Additional arguments for detection method
        
        Returns:
            MeasurementResult
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Detect object
        detector = ObjectDetector(image)
        
        if detection_method == 'manual':
            bbox = detector.detect_manually()
        elif detection_method == 'contour':
            lower_color = kwargs.get('lower_color', (20, 100, 100))
            upper_color = kwargs.get('upper_color', (30, 255, 255))
            bbox = detector.detect_by_contour(lower_color, upper_color)
        elif detection_method == 'edges':
            threshold1 = kwargs.get('threshold1', 50)
            threshold2 = kwargs.get('threshold2', 150)
            bbox = detector.detect_by_edges(threshold1, threshold2)
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")
        
        if bbox is None:
            raise RuntimeError(f"Failed to detect object in {image_path}")
        
        # Measure
        result = self.measure_object(image, distance_m, bbox)
        result.image_path = image_path
        
        return result


def main():
    """Command-line interface for object measurement."""
    parser = argparse.ArgumentParser(
        description="Measure real-world 2D object dimensions using perspective projection"
    )
    
    parser.add_argument(
        'image',
        help='Path to image file'
    )
    
    parser.add_argument(
        'distance',
        type=float,
        help='Distance from camera to object (meters)'
    )
    
    parser.add_argument(
        '--calibration',
        required=True,
        help='Path to calibration directory (containing camera_matrix.npy, distortion_coeffs.npy)'
    )
    
    parser.add_argument(
        '--detection',
        default='manual',
        choices=['manual', 'contour', 'edges'],
        help='Object detection method (default: manual)'
    )
    
    parser.add_argument(
        '--lower-hsv',
        type=int,
        nargs=3,
        default=[20, 100, 100],
        help='Lower HSV bounds for color detection (H S V)'
    )
    
    parser.add_argument(
        '--upper-hsv',
        type=int,
        nargs=3,
        default=[30, 255, 255],
        help='Upper HSV bounds for color detection (H S V)'
    )
    
    parser.add_argument(
        '--output',
        default='measurement_result.json',
        help='Output JSON file'
    )
    
    parser.add_argument(
        '--sensor-height-mm',
        type=float,
        default=5.76,
        help='Physical sensor height in mm'
    )
    
    args = parser.parse_args()
    
    # Load calibration data
    calib_path = Path(args.calibration)
    camera_matrix = np.load(calib_path / 'camera_matrix.npy')
    distortion_coeffs = np.load(calib_path / 'distortion_coeffs.npy')
    
    print(f"Loaded calibration from {args.calibration}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {distortion_coeffs}")
    
    # Create measurer
    measurer = ObjectMeasurer(
        camera_matrix,
        distortion_coeffs,
        sensor_height_mm=args.sensor_height_mm
    )
    
    # Measure
    print(f"\nMeasuring object in {args.image}...")
    print(f"Distance: {args.distance} meters")
    print(f"Detection method: {args.detection}")
    
    try:
        result = measurer.measure_from_file(
            args.image,
            args.distance,
            detection_method=args.detection,
            lower_color=tuple(args.lower_hsv),
            upper_color=tuple(args.upper_hsv)
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    # Print results
    print("\n" + "="*60)
    print("MEASUREMENT RESULTS")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Distance: {args.distance:.3f} m")
    print(f"\nDimensions:")
    print(f"  Width:  {result.width_real_cm:.2f} cm ({result.width_pixels:.0f} px)")
    print(f"  Height: {result.height_real_cm:.2f} cm ({result.height_pixels:.0f} px)")
    print(f"\nArea:")
    print(f"  Image: {result.object_area_pixels:.0f} pixels²")
    print(f"  Real:  {result.object_area_m2:.6f} m²")
    print(f"\nQuality:")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Error estimate: ±{result.error_estimate_percent:.1f}%")
    print("="*60)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())

"""
Perspective Projection Mathematics Module

This module implements the mathematical foundations for measuring real-world
object dimensions using perspective projection.

Theory:
    In perspective projection, a 3D point P_world is projected to a 2D image point p_image:
    
    p_image = K @ [R | t] @ P_world
    
    Where:
    - K: Camera intrinsic matrix
    - [R | t]: Extrinsic parameters (rotation R and translation t)
    
    For measuring object dimensions in a known plane:
    
    height_real = (height_pixel * distance * h_sensor) / (focal_length * h_image)
    
    Where:
    - height_pixel: Object height in pixels
    - distance: Distance from camera to object (meters)
    - h_sensor: Sensor height (derives from known focal length)
    - focal_length: Camera focal length in pixels
    - h_image: Image height in pixels

"""

import numpy as np
from typing import Tuple, Optional


class PerspectiveProjection:
    """
    Handles perspective projection calculations for 3D to 2D mapping.
    """
    
    def __init__(self, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray):
        """
        Initialize perspective projection with camera parameters.
        
        Args:
            camera_matrix: 3x3 intrinsic matrix K
            distortion_coeffs: Distortion coefficients
        """
        self.K = camera_matrix
        self.dist_coeffs = distortion_coeffs
        
        # Extract parameters
        self.fx = camera_matrix[0, 0]  # Focal length x (pixels)
        self.fy = camera_matrix[1, 1]  # Focal length y (pixels)
        self.cx = camera_matrix[0, 2]  # Principal point x (pixels)
        self.cy = camera_matrix[1, 2]  # Principal point y (pixels)
    
    def get_focal_length_mm(self, sensor_width_mm: float = 5.76) -> float:
        """
        Convert focal length from pixels to millimeters.
        
        Args:
            sensor_width_mm: Physical width of sensor in mm
            (typical smartphone: 5.76 mm for 1/2.3" sensor)
        
        Returns:
            Focal length in mm
        """
        # For 1/2.3" sensor with typical image size
        pixel_width = 4000  # typical for smartphone
        mm_per_pixel = sensor_width_mm / pixel_width
        focal_length_mm = self.fx * mm_per_pixel
        return focal_length_mm
    
    @staticmethod
    def compute_real_world_height(
        height_pixels: float,
        distance_meters: float,
        focal_length_pixels: float,
        image_height_pixels: float,
        sensor_height_mm: float = 5.76
    ) -> float:
        """
        Compute real-world object height using perspective projection.
        
        Derivation:
            From similar triangles in perspective projection:
            
            Object Height / Distance = Image Height (mm) / Focal Length (mm)
            
            Converting to pixels:
            Object Height = (height_pixels * distance * sensor_height_mm) / 
                           (focal_length_pixels * image_height_pixels)
            
            Where:
            - sensor_height_mm: Physical height of image sensor
            - image_height_pixels: Height of image in pixels
            
        Args:
            height_pixels: Object height in image (pixels)
            distance_meters: Distance from camera to object (meters)
            focal_length_pixels: Focal length in pixels (from calibration)
            image_height_pixels: Image height in pixels
            sensor_height_mm: Physical sensor height (mm)
        
        Returns:
            Real-world height in meters
        """
        # Convert sensor height to meters
        sensor_height_m = sensor_height_mm / 1000.0
        
        # Apply perspective projection formula
        height_real = (height_pixels * distance_meters * sensor_height_m) / \
                     (focal_length_pixels * image_height_pixels)
        
        return height_real
    
    @staticmethod
    def compute_real_world_width(
        width_pixels: float,
        distance_meters: float,
        focal_length_pixels: float,
        image_width_pixels: float,
        sensor_width_mm: float = 5.76
    ) -> float:
        """
        Compute real-world object width using perspective projection.
        
        Args:
            width_pixels: Object width in image (pixels)
            distance_meters: Distance from camera to object (meters)
            focal_length_pixels: Focal length in pixels
            image_width_pixels: Image width in pixels
            sensor_width_mm: Physical sensor width (mm)
        
        Returns:
            Real-world width in meters
        """
        sensor_width_m = sensor_width_mm / 1000.0
        
        width_real = (width_pixels * distance_meters * sensor_width_m) / \
                    (focal_length_pixels * image_width_pixels)
        
        return width_real
    
    @staticmethod
    def pixel_to_world_distance(
        pixel_distance: float,
        distance_to_object: float,
        focal_length_pixels: float,
        world_distance_unit: str = 'meters'
    ) -> float:
        """
        Convert a distance measured in image pixels to world coordinates.
        
        Args:
            pixel_distance: Distance measured in pixels
            distance_to_object: Distance from camera to object plane (meters)
            focal_length_pixels: Focal length in pixels
            world_distance_unit: 'meters' or 'cm' or 'mm'
        
        Returns:
            Distance in specified world unit
        """
        # Basic formula: world_distance = pixel_distance * depth / focal_length
        world_distance_m = (pixel_distance * distance_to_object) / focal_length_pixels
        
        if world_distance_unit == 'meters':
            return world_distance_m
        elif world_distance_unit == 'cm':
            return world_distance_m * 100
        elif world_distance_unit == 'mm':
            return world_distance_m * 1000
        else:
            return world_distance_m
    
    @staticmethod
    def world_to_pixel_distance(
        world_distance: float,
        distance_to_object: float,
        focal_length_pixels: float,
        world_unit: str = 'meters'
    ) -> float:
        """
        Convert a distance in world coordinates to image pixels.
        
        Args:
            world_distance: Distance in world units
            distance_to_object: Distance from camera to object (meters)
            focal_length_pixels: Focal length in pixels
            world_unit: 'meters', 'cm', or 'mm'
        
        Returns:
            Distance in pixels
        """
        # Convert to meters
        if world_unit == 'meters':
            world_distance_m = world_distance
        elif world_unit == 'cm':
            world_distance_m = world_distance / 100
        elif world_unit == 'mm':
            world_distance_m = world_distance / 1000
        else:
            world_distance_m = world_distance
        
        # Convert to pixels
        pixel_distance = (world_distance_m * focal_length_pixels) / distance_to_object
        
        return pixel_distance
    
    @staticmethod
    def estimate_distance_from_dimensions(
        known_real_height: float,
        measured_height_pixels: float,
        focal_length_pixels: float,
        image_height_pixels: float,
        sensor_height_mm: float = 5.76
    ) -> float:
        """
        Estimate distance to object from known dimension.
        
        Inverse of compute_real_world_height.
        If we know the real height and measure it in pixels, we can estimate distance.
        
        Args:
            known_real_height: Known real height of object (meters)
            measured_height_pixels: Measured height in image (pixels)
            focal_length_pixels: Focal length in pixels
            image_height_pixels: Image height in pixels
            sensor_height_mm: Physical sensor height (mm)
        
        Returns:
            Estimated distance in meters
        """
        sensor_height_m = sensor_height_mm / 1000.0
        
        # Rearrange: distance = (known_real_height * focal_length * image_height) / 
        #                       (measured_height_pixels * sensor_height)
        distance = (known_real_height * focal_length_pixels * image_height_pixels) / \
                  (measured_height_pixels * sensor_height_m)
        
        return distance


class FieldOfView:
    """
    Computes field of view angles from camera intrinsics.
    """
    
    def __init__(self, camera_matrix: np.ndarray, image_size: Tuple[int, int]):
        """
        Initialize with camera matrix and image size.
        
        Args:
            camera_matrix: 3x3 intrinsic matrix
            image_size: (width, height) in pixels
        """
        self.K = camera_matrix
        self.image_size = image_size
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
    
    def get_horizontal_fov(self) -> float:
        """
        Get horizontal field of view in degrees.
        
        Formula: FOV = 2 * arctan(width / (2 * fx))
        """
        width = self.image_size[0]
        fov_horizontal = 2 * np.arctan(width / (2 * self.fx))
        return np.degrees(fov_horizontal)
    
    def get_vertical_fov(self) -> float:
        """
        Get vertical field of view in degrees.
        
        Formula: FOV = 2 * arctan(height / (2 * fy))
        """
        height = self.image_size[1]
        fov_vertical = 2 * np.arctan(height / (2 * self.fy))
        return np.degrees(fov_vertical)
    
    def get_diagonal_fov(self) -> float:
        """Get diagonal field of view in degrees."""
        width, height = self.image_size
        diagonal = np.sqrt(width**2 + height**2)
        fov_diagonal = 2 * np.arctan(diagonal / (2 * self.fx))
        return np.degrees(fov_diagonal)


class MeasurementValidator:
    """
    Validates measurements and estimates uncertainty.
    """
    
    @staticmethod
    def estimate_measurement_error(
        distance_error_percent: float = 5.0,
        pixel_detection_error: float = 2.0,
        calibration_error_pixels: float = 0.5
    ) -> dict:
        """
        Estimate total measurement error from various sources.
        
        Error sources:
        1. Distance measurement error: typically 2-5%
        2. Pixel detection error: typically 1-3 pixels
        3. Calibration error: from reprojection error
        
        Args:
            distance_error_percent: Distance measurement error (%)
            pixel_detection_error: Edge detection error (pixels)
            calibration_error_pixels: Calibration reprojection error (pixels)
        
        Returns:
            Dict with error estimates
        """
        # Total error combines errors in quadrature (for independent errors)
        total_error_percent = np.sqrt(
            distance_error_percent**2 + 
            (pixel_detection_error**2 + calibration_error_pixels**2) * 10  # rough estimate
        )
        
        return {
            'distance_error_percent': distance_error_percent,
            'pixel_detection_error': pixel_detection_error,
            'calibration_error': calibration_error_pixels,
            'total_error_percent': total_error_percent,
            'typical_range': f"±{total_error_percent:.1f}%"
        }
    
    @staticmethod
    def compute_measurement_confidence(
        reprojection_error: float,
        num_calibration_images: int
    ) -> float:
        """
        Compute confidence level of measurements based on calibration quality.
        
        Args:
            reprojection_error: Mean reprojection error from calibration (pixels)
            num_calibration_images: Number of images used in calibration
        
        Returns:
            Confidence level (0-1), where 1 is maximum confidence
        """
        # Confidence decreases with higher error and fewer images
        base_confidence = 1.0 / (1.0 + reprojection_error)
        image_factor = np.sqrt(min(num_calibration_images, 50) / 20)  # norm to 20 images
        confidence = base_confidence * image_factor
        
        return min(confidence, 1.0)

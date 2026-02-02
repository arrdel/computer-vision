"""
Step 3: Validation Experiment

Validates measurement accuracy by comparing measured dimensions against ground truth
values. Processes multiple images at known distances and computes statistical error metrics.

Validation Pipeline:
1. Load ground truth data (known object dimensions)
2. Load calibration results from Step 1
3. For each test image:
   - Measure object dimensions using Step 2
   - Compare against ground truth
   - Compute measurement error
4. Aggregate statistics and generate validation report
5. Visualize results (error distribution, accuracy metrics)

Error Metrics:
- MAE (Mean Absolute Error): Average absolute difference
- RMSE (Root Mean Squared Error): Quadratic mean of errors
- MAPE (Mean Absolute Percentage Error): Error as percentage of ground truth
- Success Rate: Percentage of measurements within acceptable tolerance

"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Step2_Object_Dimension_Measurement.measure_dimensions import ObjectMeasurer, ObjectDetector
from Step2_Object_Dimension_Measurement.utils import (
    ResultManager, BatchProcessor, ImageProcessor, Visualization
)


@dataclass
class ValidationResult:
    """Container for validation results."""
    image_path: str
    distance_m: float
    ground_truth_height_cm: float
    ground_truth_width_cm: float
    measured_height_cm: float
    measured_width_cm: float
    height_error_cm: float
    height_error_percent: float
    width_error_cm: float
    width_error_percent: float
    area_error_percent: float
    success: bool  # Within acceptable tolerance
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ValidationExperiment:
    """Conducts validation experiments."""
    
    # Default tolerances
    DEFAULT_TOLERANCE_CM = 0.5  # ±0.5 cm
    DEFAULT_TOLERANCE_PERCENT = 10.0  # ±10%
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        num_calibration_images: int = 20,
        tolerance_cm: float = DEFAULT_TOLERANCE_CM,
        tolerance_percent: float = DEFAULT_TOLERANCE_PERCENT
    ):
        """
        Initialize validation experiment.
        
        Args:
            camera_matrix: 3x3 intrinsic matrix from calibration
            distortion_coeffs: Distortion coefficients from calibration
            num_calibration_images: Number of images used in calibration
            tolerance_cm: Absolute tolerance in cm
            tolerance_percent: Relative tolerance in percent
        """
        self.measurer = ObjectMeasurer(
            camera_matrix,
            distortion_coeffs,
            num_calibration_images=num_calibration_images
        )
        self.tolerance_cm = tolerance_cm
        self.tolerance_percent = tolerance_percent
        self.results = []
    
    def validate_single_measurement(
        self,
        image_path: str,
        distance_m: float,
        ground_truth_height_cm: float,
        ground_truth_width_cm: float,
        detection_method: str = 'manual',
        **detection_kwargs
    ) -> ValidationResult:
        """
        Validate a single measurement against ground truth.
        
        Args:
            image_path: Path to test image
            distance_m: Distance from camera to object
            ground_truth_height_cm: Known object height (cm)
            ground_truth_width_cm: Known object width (cm)
            detection_method: Object detection method
            **detection_kwargs: Additional detection parameters
        
        Returns:
            ValidationResult with error metrics
        """
        # Measure object
        measurement_result = self.measurer.measure_from_file(
            image_path,
            distance_m,
            detection_method=detection_method,
            **detection_kwargs
        )
        
        # Compute errors
        height_error_cm = measurement_result.height_real_cm - ground_truth_height_cm
        height_error_percent = (height_error_cm / ground_truth_height_cm) * 100
        
        width_error_cm = measurement_result.width_real_cm - ground_truth_width_cm
        width_error_percent = (width_error_cm / ground_truth_width_cm) * 100
        
        # Area error
        measured_area = measurement_result.height_real_cm * measurement_result.width_real_cm
        ground_truth_area = ground_truth_height_cm * ground_truth_width_cm
        area_error_percent = ((measured_area - ground_truth_area) / ground_truth_area) * 100
        
        # Check if within tolerance
        height_within = abs(height_error_cm) <= self.tolerance_cm or \
                       abs(height_error_percent) <= self.tolerance_percent
        width_within = abs(width_error_cm) <= self.tolerance_cm or \
                      abs(width_error_percent) <= self.tolerance_percent
        success = height_within and width_within
        
        # Create result
        result = ValidationResult(
            image_path=Path(image_path).name,
            distance_m=distance_m,
            ground_truth_height_cm=ground_truth_height_cm,
            ground_truth_width_cm=ground_truth_width_cm,
            measured_height_cm=measurement_result.height_real_cm,
            measured_width_cm=measurement_result.width_real_cm,
            height_error_cm=height_error_cm,
            height_error_percent=height_error_percent,
            width_error_cm=width_error_cm,
            width_error_percent=width_error_percent,
            area_error_percent=area_error_percent,
            success=success
        )
        
        self.results.append(result)
        return result
    
    def validate_batch_from_file(
        self,
        ground_truth_file: str,
        detection_method: str = 'manual',
        **detection_kwargs
    ) -> List[ValidationResult]:
        """
        Validate batch of measurements from ground truth file.
        
        Ground truth file format (JSON):
        ```json
        {
          "measurements": [
            {
              "image": "image1.jpg",
              "distance_m": 2.5,
              "height_cm": 10.0,
              "width_cm": 8.0
            },
            ...
          ]
        }
        ```
        
        Args:
            ground_truth_file: Path to ground truth JSON file
            detection_method: Object detection method
            **detection_kwargs: Additional detection parameters
        
        Returns:
            List of ValidationResults
        """
        # Load ground truth data
        with open(ground_truth_file, 'r') as f:
            gt_data = json.load(f)
        
        measurements = gt_data.get('measurements', [])
        print(f"Loaded {len(measurements)} ground truth measurements")
        
        # Process each measurement
        for i, m in enumerate(measurements, 1):
            print(f"\n[{i}/{len(measurements)}] Validating {m['image']}...")
            
            try:
                result = self.validate_single_measurement(
                    image_path=m['image'],
                    distance_m=m['distance_m'],
                    ground_truth_height_cm=m['height_cm'],
                    ground_truth_width_cm=m['width_cm'],
                    detection_method=detection_method,
                    **detection_kwargs
                )
                
                status = "✓" if result.success else "✗"
                print(f"  {status} H_error: {result.height_error_cm:+.2f} cm "
                      f"({result.height_error_percent:+.1f}%), "
                      f"W_error: {result.width_error_cm:+.2f} cm "
                      f"({result.width_error_percent:+.1f}%)")
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        return self.results
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistical metrics from all validation results.
        
        Returns:
            Dictionary with statistics
        """
        if not self.results:
            raise ValueError("No validation results to compute statistics")
        
        # Extract error values
        height_errors = [abs(r.height_error_cm) for r in self.results]
        width_errors = [abs(r.width_error_cm) for r in self.results]
        height_error_percents = [abs(r.height_error_percent) for r in self.results]
        width_error_percents = [abs(r.width_error_percent) for r in self.results]
        area_error_percents = [abs(r.area_error_percent) for r in self.results]
        success_count = sum(1 for r in self.results if r.success)
        
        stats = {
            'num_measurements': len(self.results),
            'success_count': success_count,
            'success_rate': success_count / len(self.results) * 100,
            'height_error': {
                'mae_cm': np.mean(height_errors),
                'rmse_cm': np.sqrt(np.mean(np.array(height_errors)**2)),
                'std_cm': np.std(height_errors),
                'min_cm': np.min(height_errors),
                'max_cm': np.max(height_errors),
                'mae_percent': np.mean(height_error_percents),
                'rmse_percent': np.sqrt(np.mean(np.array(height_error_percents)**2)),
            },
            'width_error': {
                'mae_cm': np.mean(width_errors),
                'rmse_cm': np.sqrt(np.mean(np.array(width_errors)**2)),
                'std_cm': np.std(width_errors),
                'min_cm': np.min(width_errors),
                'max_cm': np.max(width_errors),
                'mae_percent': np.mean(width_error_percents),
                'rmse_percent': np.sqrt(np.mean(np.array(width_error_percents)**2)),
            },
            'area_error': {
                'mae_percent': np.mean(area_error_percents),
                'rmse_percent': np.sqrt(np.mean(np.array(area_error_percents)**2)),
                'max_percent': np.max(area_error_percents),
            },
            'tolerance': {
                'tolerance_cm': self.tolerance_cm,
                'tolerance_percent': self.tolerance_percent,
            }
        }
        
        return stats
    
    def generate_report(self, output_path: str) -> bool:
        """
        Generate validation report.
        
        Args:
            output_path: Output file path
        
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            stats = self.compute_statistics()
            
            with open(output_path, 'w') as f:
                f.write("=" * 90 + "\n")
                f.write("VALIDATION EXPERIMENT REPORT\n")
                f.write("=" * 90 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Summary
                f.write("SUMMARY\n")
                f.write("-" * 90 + "\n")
                f.write(f"Total measurements: {stats['num_measurements']}\n")
                f.write(f"Successful measurements: {stats['success_count']}/{stats['num_measurements']}\n")
                f.write(f"Success rate: {stats['success_rate']:.1f}%\n")
                f.write(f"Tolerance: ±{stats['tolerance']['tolerance_cm']:.2f} cm "
                       f"or ±{stats['tolerance']['tolerance_percent']:.1f}%\n\n")
                
                # Height errors
                f.write("HEIGHT MEASUREMENT ERRORS\n")
                f.write("-" * 90 + "\n")
                h_err = stats['height_error']
                f.write(f"  Mean Absolute Error (MAE):  {h_err['mae_cm']:.3f} cm "
                       f"({h_err['mae_percent']:.2f}%)\n")
                f.write(f"  Root Mean Squared Error:    {h_err['rmse_cm']:.3f} cm "
                       f"({h_err['rmse_percent']:.2f}%)\n")
                f.write(f"  Standard Deviation:         {h_err['std_cm']:.3f} cm\n")
                f.write(f"  Min Error:                  {h_err['min_cm']:.3f} cm\n")
                f.write(f"  Max Error:                  {h_err['max_cm']:.3f} cm\n\n")
                
                # Width errors
                f.write("WIDTH MEASUREMENT ERRORS\n")
                f.write("-" * 90 + "\n")
                w_err = stats['width_error']
                f.write(f"  Mean Absolute Error (MAE):  {w_err['mae_cm']:.3f} cm "
                       f"({w_err['mae_percent']:.2f}%)\n")
                f.write(f"  Root Mean Squared Error:    {w_err['rmse_cm']:.3f} cm "
                       f"({w_err['rmse_percent']:.2f}%)\n")
                f.write(f"  Standard Deviation:         {w_err['std_cm']:.3f} cm\n")
                f.write(f"  Min Error:                  {w_err['min_cm']:.3f} cm\n")
                f.write(f"  Max Error:                  {w_err['max_cm']:.3f} cm\n\n")
                
                # Area errors
                f.write("AREA MEASUREMENT ERRORS\n")
                f.write("-" * 90 + "\n")
                a_err = stats['area_error']
                f.write(f"  Mean Absolute Error (MAE):  {a_err['mae_percent']:.2f}%\n")
                f.write(f"  Root Mean Squared Error:    {a_err['rmse_percent']:.2f}%\n")
                f.write(f"  Max Error:                  {a_err['max_percent']:.2f}%\n\n")
                
                # Detailed results
                f.write("DETAILED RESULTS\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Image':<20} {'GT H (cm)':<12} {'Meas H (cm)':<12} {'Error H':<12} "
                       f"{'GT W (cm)':<12} {'Meas W (cm)':<12} {'Error W':<12} {'Status':<8}\n")
                f.write("-" * 90 + "\n")
                
                for result in self.results:
                    status = "PASS" if result.success else "FAIL"
                    f.write(f"{result.image_path:<20} "
                           f"{result.ground_truth_height_cm:<12.2f} "
                           f"{result.measured_height_cm:<12.2f} "
                           f"{result.height_error_cm:+<11.3f} "
                           f"{result.ground_truth_width_cm:<12.2f} "
                           f"{result.measured_width_cm:<12.2f} "
                           f"{result.width_error_cm:+<11.3f} "
                           f"{status:<8}\n")
                
                f.write("\n" + "=" * 90 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 90 + "\n")
            
            print(f"Report saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error generating report: {e}")
            return False


def create_sample_ground_truth() -> str:
    """
    Create sample ground truth file for demonstration.
    
    Returns:
        Path to created ground truth file
    """
    gt_data = {
        "experiment": "Camera calibration validation",
        "object": "A4 paper (21cm x 29.7cm) or credit card (8.56cm x 5.398cm)",
        "measurements": [
            {
                "image": "test_image_1.jpg",
                "distance_m": 2.5,
                "height_cm": 10.0,
                "width_cm": 8.0
            },
            {
                "image": "test_image_2.jpg",
                "distance_m": 3.0,
                "height_cm": 10.0,
                "width_cm": 8.0
            },
            {
                "image": "test_image_3.jpg",
                "distance_m": 3.5,
                "height_cm": 10.0,
                "width_cm": 8.0
            }
        ]
    }
    
    output_path = Path("ground_truth.json")
    with open(output_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    
    print(f"Sample ground truth created at {output_path}")
    return str(output_path)


def main():
    """Command-line interface for validation experiment."""
    parser = argparse.ArgumentParser(
        description="Validate camera calibration and measurement accuracy"
    )
    
    parser.add_argument(
        '--calibration',
        required=True,
        help='Path to calibration directory'
    )
    
    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to ground truth JSON file'
    )
    
    parser.add_argument(
        '--detection',
        default='manual',
        choices=['manual', 'contour', 'edges'],
        help='Object detection method'
    )
    
    parser.add_argument(
        '--tolerance-cm',
        type=float,
        default=ValidationExperiment.DEFAULT_TOLERANCE_CM,
        help='Absolute tolerance in cm'
    )
    
    parser.add_argument(
        '--tolerance-percent',
        type=float,
        default=ValidationExperiment.DEFAULT_TOLERANCE_PERCENT,
        help='Relative tolerance in percent'
    )
    
    parser.add_argument(
        '--output',
        default='validation_report.txt',
        help='Output report file'
    )
    
    parser.add_argument(
        '--results-json',
        help='Output JSON file with detailed results'
    )
    
    parser.add_argument(
        '--create-sample-gt',
        action='store_true',
        help='Create sample ground truth file'
    )
    
    args = parser.parse_args()
    
    # Create sample ground truth if requested
    if args.create_sample_gt:
        create_sample_ground_truth()
        return 0
    
    # Load calibration
    calib_path = Path(args.calibration)
    camera_matrix = np.load(calib_path / 'camera_matrix.npy')
    distortion_coeffs = np.load(calib_path / 'distortion_coeffs.npy')
    
    print(f"Loaded calibration from {args.calibration}")
    
    # Create validation experiment
    validator = ValidationExperiment(
        camera_matrix,
        distortion_coeffs,
        tolerance_cm=args.tolerance_cm,
        tolerance_percent=args.tolerance_percent
    )
    
    print(f"Tolerance: ±{args.tolerance_cm} cm or ±{args.tolerance_percent}%\n")
    
    # Run validation
    print("Running validation experiment...")
    try:
        validator.validate_batch_from_file(
            args.ground_truth,
            detection_method=args.detection
        )
    except Exception as e:
        print(f"Error during validation: {e}")
        return 1
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = validator.compute_statistics()
    
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Success rate: {stats['success_rate']:.1f}% "
          f"({stats['success_count']}/{stats['num_measurements']})")
    print(f"\nHeight error:")
    print(f"  MAE: {stats['height_error']['mae_cm']:.3f} cm "
          f"({stats['height_error']['mae_percent']:.2f}%)")
    print(f"  RMSE: {stats['height_error']['rmse_cm']:.3f} cm "
          f"({stats['height_error']['rmse_percent']:.2f}%)")
    print(f"\nWidth error:")
    print(f"  MAE: {stats['width_error']['mae_cm']:.3f} cm "
          f"({stats['width_error']['mae_percent']:.2f}%)")
    print(f"  RMSE: {stats['width_error']['rmse_cm']:.3f} cm "
          f"({stats['width_error']['rmse_percent']:.2f}%)")
    print("=" * 70)
    
    # Generate report
    validator.generate_report(args.output)
    
    # Save detailed results if requested
    if args.results_json:
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'detailed_results': [r.to_dict() for r in validator.results]
        }
        
        output_path = Path(args.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Detailed results saved to {args.results_json}")
    
    return 0


if __name__ == '__main__':
    exit(main())

"""
Utility Functions for Object Dimension Measurement

Provides helper functions for image processing, visualization, and data management.

"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json
from datetime import datetime


class ImageProcessor:
    """Image processing utilities."""
    
    @staticmethod
    def load_image(image_path: str, color_format: str = 'BGR') -> Optional[np.ndarray]:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            color_format: 'BGR' (default) or 'RGB'
        
        Returns:
            Image array or None if load fails
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        if color_format == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str, color_format: str = 'BGR') -> bool:
        """
        Save image to file.
        
        Args:
            image: Image array
            output_path: Output file path
            color_format: 'BGR' (default) or 'RGB'
        
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if color_format == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        success = cv2.imwrite(str(output_path), image)
        
        if success:
            print(f"Image saved to {output_path}")
        else:
            print(f"Error: Failed to save image to {output_path}")
        
        return success
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Resize image to specified dimensions.
        
        Args:
            image: Input image
            width: Target width (None to maintain aspect ratio)
            height: Target height (None to maintain aspect ratio)
            scale: Scale factor (0.5 = 50%)
        
        Returns:
            Resized image
        """
        if scale is not None:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
        
        if width is None and height is None:
            return image
        
        if width is None:
            width = int(image.shape[1] * height / image.shape[0])
        elif height is None:
            height = int(image.shape[0] * width / image.shape[1])
        
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def draw_bounding_box(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        label: str = ""
    ) -> np.ndarray:
        """
        Draw bounding box on image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            color: BGR color tuple
            thickness: Line thickness
            label: Optional label text
        
        Returns:
            Image with bounding box
        """
        image = image.copy()
        x, y, w, h = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            
            # Background for text
            cv2.rectangle(
                image,
                (x, y - text_size[1] - 4),
                (x + text_size[0] + 4, y),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                image,
                label,
                (x + 2, y - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return image
    
    @staticmethod
    def draw_dimensions(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        width_cm: float,
        height_cm: float,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw dimensions on image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            width_cm: Width in cm
            height_cm: Height in cm
            color: BGR color tuple
        
        Returns:
            Image with dimensions
        """
        image = image.copy()
        x, y, w, h = bbox
        
        # Draw bounding box first
        image = ImageProcessor.draw_bounding_box(image, bbox, color, 2)
        
        # Width dimension
        mid_y = y + h // 2
        cv2.line(image, (x, mid_y), (x + w, mid_y), (255, 0, 0), 1)
        cv2.circle(image, (x, mid_y), 3, (255, 0, 0), -1)
        cv2.circle(image, (x + w, mid_y), 3, (255, 0, 0), -1)
        
        # Width label
        label_w = f"W: {width_cm:.2f} cm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label_w, font, 0.5, 1)[0]
        cv2.putText(
            image,
            label_w,
            (x + w // 2 - text_size[0] // 2, mid_y - 10),
            font,
            0.5,
            (255, 0, 0),
            1
        )
        
        # Height dimension
        mid_x = x + w // 2
        cv2.line(image, (mid_x, y), (mid_x, y + h), (0, 0, 255), 1)
        cv2.circle(image, (mid_x, y), 3, (0, 0, 255), -1)
        cv2.circle(image, (mid_x, y + h), 3, (0, 0, 255), -1)
        
        # Height label
        label_h = f"H: {height_cm:.2f} cm"
        cv2.putText(
            image,
            label_h,
            (mid_x + 10, y + h // 2),
            font,
            0.5,
            (0, 0, 255),
            1
        )
        
        return image


class ResultManager:
    """Manages measurement results and data storage."""
    
    @staticmethod
    def save_result_json(result: Dict[str, Any], output_path: str) -> bool:
        """
        Save measurement result to JSON file.
        
        Args:
            result: Result dictionary
            output_path: Output file path
        
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    @staticmethod
    def load_result_json(input_path: str) -> Optional[Dict[str, Any]]:
        """
        Load measurement result from JSON file.
        
        Args:
            input_path: Input file path
        
        Returns:
            Result dictionary or None if load fails
        """
        try:
            with open(input_path, 'r') as f:
                result = json.load(f)
            return result
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
    
    @staticmethod
    def save_batch_results(results: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Save batch of measurement results to JSON file.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
        
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(
                    {
                        'timestamp': datetime.now().isoformat(),
                        'num_results': len(results),
                        'results': results
                    },
                    f,
                    indent=2
                )
            print(f"Batch results saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving batch results: {e}")
            return False
    
    @staticmethod
    def generate_report(
        results: List[Dict[str, Any]],
        output_path: str
    ) -> bool:
        """
        Generate text report from multiple measurement results.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
        
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MEASUREMENT RESULTS REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Number of measurements: {len(results)}\n")
                f.write("\n")
                
                # Statistics
                if results:
                    heights = [r['height_real_cm'] for r in results]
                    widths = [r['width_real_cm'] for r in results]
                    confidences = [r['confidence'] for r in results]
                    errors = [r['error_estimate_percent'] for r in results]
                    
                    f.write("STATISTICS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Height: mean={np.mean(heights):.2f} cm, "
                           f"std={np.std(heights):.2f} cm, "
                           f"range=[{np.min(heights):.2f}, {np.max(heights):.2f}]\n")
                    f.write(f"Width:  mean={np.mean(widths):.2f} cm, "
                           f"std={np.std(widths):.2f} cm, "
                           f"range=[{np.min(widths):.2f}, {np.max(widths):.2f}]\n")
                    f.write(f"Confidence: mean={np.mean(confidences):.2%}, "
                           f"min={np.min(confidences):.2%}, "
                           f"max={np.max(confidences):.2%}\n")
                    f.write(f"Error:      mean={np.mean(errors):.2f}%, "
                           f"min={np.min(errors):.2f}%, "
                           f"max={np.max(errors):.2f}%\n\n")
                
                # Detailed results
                f.write("DETAILED RESULTS\n")
                f.write("-" * 80 + "\n")
                
                for i, result in enumerate(results, 1):
                    f.write(f"\n[{i}] {result['image_path']}\n")
                    f.write(f"    Distance: {result['distance_m']:.3f} m\n")
                    f.write(f"    Height: {result['height_real_cm']:.2f} cm "
                           f"({result['height_pixels']:.0f} px)\n")
                    f.write(f"    Width:  {result['width_real_cm']:.2f} cm "
                           f"({result['width_pixels']:.0f} px)\n")
                    f.write(f"    Area:   {result['object_area_m2']:.6f} m² "
                           f"({result['object_area_pixels']:.0f} px²)\n")
                    f.write(f"    Confidence: {result['confidence']:.2%}\n")
                    f.write(f"    Error estimate: ±{result['error_estimate_percent']:.1f}%\n")
            
            print(f"Report saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error generating report: {e}")
            return False


class BatchProcessor:
    """Process multiple images in batch."""
    
    @staticmethod
    def process_images_in_folder(
        folder_path: str,
        measurer,
        distance_m: float,
        detection_method: str = 'manual',
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> List[Dict[str, Any]]:
        """
        Process all images in folder.
        
        Args:
            folder_path: Path to folder containing images
            measurer: ObjectMeasurer instance
            distance_m: Distance to object
            detection_method: Detection method to use
            extensions: Image file extensions to process
        
        Returns:
            List of measurement results
        """
        folder_path = Path(folder_path)
        results = []
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        image_files = sorted(set(image_files))
        
        print(f"Found {len(image_files)} images in {folder_path}")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}...")
            
            try:
                result = measurer.measure_from_file(
                    str(image_path),
                    distance_m,
                    detection_method=detection_method
                )
                result_dict = result.to_dict()
                result_dict['image_path'] = image_path.name
                results.append(result_dict)
                print(f"  ✓ Success: H={result.height_real_cm:.2f} cm, "
                      f"W={result.width_real_cm:.2f} cm")
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\nProcessed {len(results)}/{len(image_files)} images successfully")
        
        return results


class Visualization:
    """Visualization utilities."""
    
    @staticmethod
    def plot_measurement_comparison(
        results: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ):
        """
        Plot comparison of measurements.
        
        Args:
            results: List of measurement results
            output_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if not results:
            print("No results to plot")
            return
        
        # Extract data
        labels = [r['image_path'].replace('.jpg', '').replace('.png', '') 
                 for r in results]
        heights = [r['height_real_cm'] for r in results]
        widths = [r['width_real_cm'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Height plot
        axes[0].bar(range(len(labels)), heights, color='steelblue')
        axes[0].set_xlabel('Image')
        axes[0].set_ylabel('Height (cm)')
        axes[0].set_title('Object Height Measurements')
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        
        # Width plot
        axes[1].bar(range(len(labels)), widths, color='darkorange')
        axes[1].set_xlabel('Image')
        axes[1].set_ylabel('Width (cm)')
        axes[1].set_title('Object Width Measurements')
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        
        # Confidence plot
        axes[2].bar(range(len(labels)), confidences, color='green', alpha=0.7)
        axes[2].set_xlabel('Image')
        axes[2].set_ylabel('Confidence')
        axes[2].set_title('Measurement Confidence')
        axes[2].set_ylim([0, 1])
        axes[2].set_xticks(range(len(labels)))
        axes[2].set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {output_path}")
        else:
            plt.show()

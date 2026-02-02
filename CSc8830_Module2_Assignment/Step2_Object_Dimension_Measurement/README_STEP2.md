## Step 2: Object Dimension Measurement

**Objective:** Use calibrated camera intrinsics to measure real-world 2D object dimensions from images using perspective projection.

**Theory:** See [Perspective Projection Mathematical Foundation](#perspective-projection-mathematical-foundation)

---

## Files

- **`measure_dimensions.py`**: Main measurement script with CLI
- **`perspective_projection.py`**: Mathematical module implementing projection formulas
- **`utils.py`**: Helper functions (processing, visualization, etc.)
- **`README_STEP2.md`**: Detailed procedure guide (this file)

---

## Quick Start

```bash
# 1. Prepare calibration data from Step 1
# (Should be in ../Step1_Camera_Calibration/output/)

# 2. Measure object with manual detection
python measure_dimensions.py \
  test_image.jpg \
  3.5 \
  --calibration ../Step1_Camera_Calibration/output/ \
  --detection manual \
  --output measurement_result.json

# 3. View results
cat measurement_result.json
```

**Input Requirements:**
- Image file (JPG, PNG, etc.)
- Distance from camera to object (in meters)
- Calibration directory with `camera_matrix.npy` and `distortion_coeffs.npy`

**Output:**
- JSON file with measurement results (dimensions, confidence, error estimates)

---

## Perspective Projection Mathematical Foundation

### 1. Basic Perspective Projection

In a pinhole camera model, a 3D point $\mathbf{P}_{\text{world}}$ is projected to a 2D image point $\mathbf{p}_{\text{image}}$ through the intrinsic matrix $K$:

$$\mathbf{p}_{\text{image}} = K \begin{bmatrix} R | \mathbf{t} \end{bmatrix} \mathbf{P}_{\text{world}}$$

Where:
- $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ (intrinsic matrix)
- $f_x, f_y$ = focal lengths (pixels)
- $c_x, c_y$ = principal point (image center)
- $[R | \mathbf{t}]$ = extrinsic parameters (rotation and translation)

### 2. Height Measurement Formula

For an object at distance $d$ from the camera, the real-world height can be derived from similar triangles:

$$\frac{H_{\text{real}}}{d} = \frac{h_{\text{sensor}}}{f_{\text{mm}}}$$

Converting to pixels and solving for real height:

$$H_{\text{real}} = \frac{h_{\text{pixel}} \cdot d \cdot h_{\text{sensor\_mm}}}{f_{\text{pixel}} \cdot h_{\text{image\_pixel}}}$$

Where:
- $h_{\text{pixel}}$ = object height in image (pixels)
- $d$ = distance from camera to object (meters)
- $h_{\text{sensor\_mm}}$ = physical sensor height (mm) ≈ 5.76 mm for 1/2.3" sensor
- $f_{\text{pixel}}$ = focal length from intrinsic matrix (pixels) = $f_y$
- $h_{\text{image\_pixel}}$ = image height in pixels

**Derivation:**

From pinhole camera geometry:
- Sensor height in mm: $h_s = 5.76$ mm (for 1/2.3" sensor)
- Focal length in mm: $f_s = f_{\text{pixel}} \times \frac{h_s}{h_{\text{image\_pixel}}}$
- Height: $H = \frac{h_{\text{pixel}}}{h_{\text{image\_pixel}}} \times \frac{f_s \times d}{f_s} = \frac{h_{\text{pixel}} \times d \times h_s}{f_{\text{pixel}} \times h_{\text{image\_pixel}}}$

### 3. Width Measurement Formula

Similarly for width:

$$W_{\text{real}} = \frac{w_{\text{pixel}} \cdot d \cdot w_{\text{sensor\_mm}}}{f_{\text{pixel}} \cdot w_{\text{image\_pixel}}}$$

Where $w_{\text{sensor\_mm}}$ depends on the sensor aspect ratio.

### 4. Area Calculation

Real-world area:

$$A_{\text{real}} = H_{\text{real}} \times W_{\text{real}}$$

### 5. Distance Estimation (Inverse Problem)

If we know an object's true height, we can estimate distance:

$$d_{\text{estimated}} = \frac{H_{\text{known}} \cdot f_{\text{pixel}} \cdot h_{\text{image\_pixel}}}{h_{\text{pixel}} \cdot h_{\text{sensor\_mm}}}$$

---

## Usage Guide

### Command-Line Interface

```bash
python measure_dimensions.py IMAGE DISTANCE [OPTIONS]
```

**Positional Arguments:**
- `IMAGE`: Path to image file
- `DISTANCE`: Distance from camera to object (meters)

**Required Options:**
- `--calibration CALIB_DIR`: Path to calibration output directory

**Optional Arguments:**
- `--detection {manual,contour,edges}`: Detection method (default: manual)
- `--lower-hsv H S V`: Lower HSV color bounds (default: 20 100 100)
- `--upper-hsv H S V`: Upper HSV color bounds (default: 30 255 255)
- `--output FILE`: Output JSON file (default: measurement_result.json)
- `--sensor-height-mm MM`: Physical sensor height in mm (default: 5.76)

### Detection Methods

#### 1. Manual Detection (Interactive)

```bash
python measure_dimensions.py test.jpg 2.5 --calibration calib/ --detection manual
```

**Instructions:**
- Click and drag to draw bounding box around object
- Press 'c' to confirm
- Press 'r' to reset
- Press 'q' to cancel

**Best for:** Single objects, precise control, verification

#### 2. Contour Detection (Color-based)

```bash
python measure_dimensions.py test.jpg 2.5 \
  --calibration calib/ \
  --detection contour \
  --lower-hsv 20 100 100 \
  --upper-hsv 30 255 255
```

**Works well for:** Objects with distinct colors

**HSV Color Ranges:**
- **Red**: (0, 100, 100) to (10, 255, 255) or (170, 100, 100) to (180, 255, 255)
- **Green**: (40, 100, 100) to (80, 255, 255)
- **Blue**: (100, 100, 100) to (130, 255, 255)
- **Yellow**: (20, 100, 100) to (40, 255, 255)

#### 3. Edge Detection

```bash
python measure_dimensions.py test.jpg 2.5 \
  --calibration calib/ \
  --detection edges \
  --threshold1 50 \
  --threshold2 150
```

**Best for:** Objects with clear edges and high contrast

---

## Python API

### Basic Usage

```python
import numpy as np
from measure_dimensions import ObjectMeasurer, MeasurementResult

# Load calibration
camera_matrix = np.load('calibration/camera_matrix.npy')
distortion_coeffs = np.load('calibration/distortion_coeffs.npy')

# Create measurer
measurer = ObjectMeasurer(camera_matrix, distortion_coeffs)

# Measure from file
result = measurer.measure_from_file(
    'test_image.jpg',
    distance_m=3.5,
    detection_method='manual'
)

# Access results
print(f"Width: {result.width_real_cm:.2f} cm")
print(f"Height: {result.height_real_cm:.2f} cm")
print(f"Confidence: {result.confidence:.2%}")
print(f"Error estimate: ±{result.error_estimate_percent:.1f}%")
```

### Advanced Usage

```python
import cv2
from measure_dimensions import ObjectMeasurer, ObjectDetector

# Load calibration and image
camera_matrix = np.load('calib/camera_matrix.npy')
distortion_coeffs = np.load('calib/distortion_coeffs.npy')
image = cv2.imread('test_image.jpg')

# Create measurer
measurer = ObjectMeasurer(camera_matrix, distortion_coeffs)

# Undistort image
undistorted = measurer.undistort_image(image)

# Detect object manually
detector = ObjectDetector(undistorted)
bbox = detector.detect_manually()  # Interactive

# Or detect by color
bbox = detector.detect_by_contour(
    lower_color=(20, 100, 100),
    upper_color=(30, 255, 255)
)

# Measure
result = measurer.measure_object(undistorted, distance_m=3.5, bbox=bbox)

# Access result data
result_dict = result.to_dict()
print(result_dict)
```

---

## Understanding Results

### Measurement Output

```json
{
  "image_path": "test_image.jpg",
  "distance_m": 3.5,
  "height_pixels": 125.0,
  "width_pixels": 87.0,
  "height_real_m": 0.0456,
  "width_real_m": 0.0318,
  "height_real_cm": 4.56,
  "width_real_cm": 3.18,
  "object_area_pixels": 10875.0,
  "object_area_m2": 0.00145,
  "confidence": 0.89,
  "error_estimate_percent": 8.5
}
```

**Field Descriptions:**
- `height_pixels`, `width_pixels`: Object dimensions in image (pixels)
- `height_real_m`, `width_real_m`: Measured real-world dimensions (meters)
- `height_real_cm`, `width_real_cm`: Measured real-world dimensions (cm)
- `object_area_pixels`: Object area in image (square pixels)
- `object_area_m2`: Real-world object area (square meters)
- `confidence`: Confidence level (0-1), based on calibration quality
- `error_estimate_percent`: Estimated measurement uncertainty (%)

### Confidence Score

Confidence is computed based on:
- **Calibration quality**: Lower reprojection error → higher confidence
- **Number of calibration images**: More images → more reliable calibration
- **Calibration formula**: $C = \frac{1}{1 + \text{reprojection\_error}} \times \sqrt{\frac{\text{num\_images}}{20}}$

**Interpretation:**
- **>0.85**: High confidence, reliable measurements
- **0.70-0.85**: Medium confidence, acceptable for most applications
- **<0.70**: Low confidence, use with caution

### Error Estimates

Total measurement error combines:
1. **Distance measurement error**: ±2-5% (depends on tape measure accuracy)
2. **Pixel detection error**: ±1-3 pixels (edge detection uncertainty)
3. **Calibration error**: From reprojection error (typically <1 pixel)

**Error propagation:**
$$\text{Total Error} \approx \sqrt{E_{\text{distance}}^2 + E_{\text{pixels}}^2 + E_{\text{calibration}}^2}$$

---

## Troubleshooting

### Problem: Object Not Detected

**Manual Detection:**
- Try clicking more carefully to outline object
- Ensure object has clear boundaries

**Contour Detection:**
- Check HSV color range with a color picker tool
- Try adjusting `--lower-hsv` and `--upper-hsv`
- Increase image contrast before detection

**Edge Detection:**
- Try different threshold values: `--threshold1 30 --threshold2 100`
- Ensure good lighting and high contrast

### Problem: Unrealistic Measurements

**Causes:**
1. Incorrect distance input
2. Poor calibration (reprojection error >1 pixel)
3. Object not perpendicular to camera
4. Significant lens distortion

**Solutions:**
- Re-calibrate with more images
- Measure distance more accurately
- Position object perpendicular to camera optical axis
- Use undistortion by default

### Problem: High Error Estimates

**Check:**
- Number of calibration images (aim for 20-50)
- Reprojection error from calibration results
- Checkerboard was fully visible in all calibration images
- Camera was not moved during calibration

---

## Advanced Topics

### Custom Sensor Parameters

For different camera phones, adjust `--sensor-height-mm`:

```bash
# iPhone 12 (1/1.6" sensor)
python measure_dimensions.py img.jpg 2.5 --calibration calib/ --sensor-height-mm 6.0

# Google Pixel 5 (1/1.9" sensor)  
python measure_dimensions.py img.jpg 2.5 --calibration calib/ --sensor-height-mm 5.3

# Typical smartphone (1/2.3" sensor)
python measure_dimensions.py img.jpg 2.5 --calibration calib/ --sensor-height-mm 5.76
```

### Field of View Analysis

```python
from perspective_projection import FieldOfView

fov = FieldOfView(camera_matrix, image_size=(4000, 3000))
print(f"Horizontal FOV: {fov.get_horizontal_fov():.1f}°")
print(f"Vertical FOV: {fov.get_vertical_fov():.1f}°")
print(f"Diagonal FOV: {fov.get_diagonal_fov():.1f}°")
```

### Measurement Validation

```python
from perspective_projection import MeasurementValidator

# Estimate error
error = MeasurementValidator.estimate_measurement_error(
    distance_error_percent=3.0,
    pixel_detection_error=2.0,
    calibration_error_pixels=0.5
)
print(f"Expected error: {error['total_error_percent']:.1f}%")

# Compute confidence
confidence = MeasurementValidator.compute_measurement_confidence(
    reprojection_error=0.45,
    num_calibration_images=25
)
print(f"Measurement confidence: {confidence:.2%}")
```

---

## References

1. **Zhang, Z.** (2000). "A flexible new technique for camera calibration." 
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 1330-1334.

2. **Hartley, R., & Zisserman, A.** (2003). *Multiple view geometry in computer vision.*
   Cambridge University Press.

3. **OpenCV Documentation:** 
   https://docs.opencv.org/4.5.2/d9/df8/tutorial_root.html

4. **Camera Calibration and 3D Reconstruction:**
   https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html

---

## File Structure

```
Step2_Object_Dimension_Measurement/
├── measure_dimensions.py          # Main measurement script
├── perspective_projection.py      # Mathematical module
├── utils.py                       # Helper functions (if added)
└── README_STEP2.md               # This file
```

---

## Next Steps

After completing measurements:
1. Document measurement procedures in measurement log
2. Compare measurements across multiple images
3. Analyze measurement consistency and error
4. **Continue to Step 3:** Validation experiment with ground truth comparisons

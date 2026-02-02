## Step 3: Validation Experiment

**Objective:** Validate measurement accuracy by comparing measured dimensions against ground truth values. Conduct systematic validation across multiple test images and distances.

**Theory:** See [Validation Methodology](#validation-methodology)

---

## Files

- **`validation_experiment.py`**: Main validation script with statistical analysis
- **`README_STEP3.md`**: Detailed procedure guide (this file)

---

## Quick Start

### Step 1: Create Ground Truth File

First, manually measure your test objects and create a ground truth file:

```bash
python validation_experiment.py --create-sample-gt
```

This creates `ground_truth.json`:

```json
{
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
    }
  ]
}
```

### Step 2: Run Validation

```bash
python validation_experiment.py \
  --calibration ../Step1_Camera_Calibration/output/ \
  --ground-truth ground_truth.json \
  --detection manual \
  --tolerance-cm 0.5 \
  --tolerance-percent 10.0 \
  --output validation_report.txt \
  --results-json validation_results.json
```

**Options:**
- `--tolerance-cm`: Absolute tolerance (default: 0.5 cm)
- `--tolerance-percent`: Relative tolerance (default: 10%)
- `--detection`: Detection method: `manual`, `contour`, or `edges`

### Step 3: Review Results

```bash
# View text report
cat validation_report.txt

# View detailed JSON results
cat validation_results.json
```

---

## Validation Methodology

### 1. Validation Pipeline

```
Ground Truth File
       ↓
Load Known Dimensions
       ↓
For Each Test Image:
  ├─ Load image and distance
  ├─ Measure using calibrated camera (Step 2)
  ├─ Compare against known dimensions
  └─ Compute error metrics
       ↓
Aggregate Statistics
       ↓
Generate Report & Visualizations
```

### 2. Error Metrics

#### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Where:
- $y_i$ = measured value
- $\hat{y}_i$ = ground truth value
- $n$ = number of measurements

**Interpretation:** Average magnitude of error in absolute units

#### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interpretation:** Penalizes larger errors more heavily than MAE

#### Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{\hat{y}_i} \right|$$

**Interpretation:** Error as percentage of ground truth value

#### Success Rate

$$\text{Success Rate} = \frac{\text{# measurements within tolerance}}{\text{total measurements}} \times 100\%$$

A measurement is considered successful if:
$$|y_i - \hat{y}_i| \leq \text{tolerance\_cm} \text{ OR } \left| \frac{y_i - \hat{y}_i}{\hat{y}_i} \times 100\% \right| \leq \text{tolerance\_percent}$$

### 3. Ground Truth File Format

```json
{
  "experiment": "Description of experiment",
  "object": "Description of object being measured",
  "date": "2026-02-02",
  "measurements": [
    {
      "image": "image_filename.jpg",
      "distance_m": 2.5,
      "height_cm": 10.0,
      "width_cm": 8.0,
      "notes": "Optional notes"
    }
  ]
}
```

**Requirements:**
- Image files must exist in current directory or searchable path
- Distance must be accurate (measure with tape measure)
- Height/width must be known dimensions (object spec or manual measurement)
- All distances in meters, dimensions in cm

---

## Python API

### Basic Validation

```python
import numpy as np
from validation_experiment import ValidationExperiment

# Load calibration
camera_matrix = np.load('calib/camera_matrix.npy')
distortion_coeffs = np.load('calib/distortion_coeffs.npy')

# Create validator
validator = ValidationExperiment(
    camera_matrix,
    distortion_coeffs,
    tolerance_cm=0.5,
    tolerance_percent=10.0
)

# Run validation from ground truth file
results = validator.validate_batch_from_file('ground_truth.json')

# Get statistics
stats = validator.compute_statistics()
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Height MAE: {stats['height_error']['mae_cm']:.3f} cm")
print(f"Width MAE: {stats['width_error']['mae_cm']:.3f} cm")

# Generate report
validator.generate_report('validation_report.txt')
```

### Single Measurement Validation

```python
result = validator.validate_single_measurement(
    image_path='test.jpg',
    distance_m=2.5,
    ground_truth_height_cm=10.0,
    ground_truth_width_cm=8.0,
    detection_method='manual'
)

print(f"Height error: {result.height_error_cm:+.3f} cm "
      f"({result.height_error_percent:+.2f}%)")
print(f"Width error: {result.width_error_cm:+.3f} cm "
      f"({result.width_error_percent:+.2f}%)")
print(f"Status: {'PASS' if result.success else 'FAIL'}")
```

---

## Understanding Results

### Sample Validation Report

```
==========================================================================================
VALIDATION EXPERIMENT REPORT
==========================================================================================
Generated: 2026-02-02T14:30:45.123456

SUMMARY
------------------------------------------------------------------------------------------
Total measurements: 12
Successful measurements: 11/12
Success rate: 91.7%
Tolerance: ±0.50 cm or ±10.0%

HEIGHT MEASUREMENT ERRORS
------------------------------------------------------------------------------------------
  Mean Absolute Error (MAE):  0.156 cm (1.56%)
  Root Mean Squared Error:    0.203 cm (2.03%)
  Standard Deviation:         0.127 cm
  Min Error:                  0.012 cm
  Max Error:                  0.487 cm

WIDTH MEASUREMENT ERRORS
------------------------------------------------------------------------------------------
  Mean Absolute Error (MAE):  0.089 cm (1.11%)
  Root Mean Squared Error:    0.134 cm (1.68%)
  Standard Deviation:         0.103 cm
  Min Error:                  0.003 cm
  Max Error:                  0.412 cm

AREA MEASUREMENT ERRORS
------------------------------------------------------------------------------------------
  Mean Absolute Error (MAE):  3.45%
  Root Mean Squared Error:    4.12%
  Max Error:                  9.87%

DETAILED RESULTS
------------------------------------------------------------------------------------------
Image               GT H (cm)    Meas H (cm)  Error H      GT W (cm)    Meas W (cm)  Error W      Status
------------------------------------------------------------------------------------------
test_01.jpg         10.00        10.12        +0.120       8.00         8.05         +0.050       PASS
test_02.jpg         10.00        9.88         -0.120       8.00         7.92         -0.080       PASS
...
```

### Interpreting Error Metrics

**Good Results:**
- Success rate > 90%
- MAE < 1% for both height and width
- RMSE < 2% for both dimensions
- Max error < 0.5 cm

**Acceptable Results:**
- Success rate 80-90%
- MAE 1-3% for both dimensions
- RMSE 2-5% for both dimensions
- Max error < 1.0 cm

**Poor Results (Recalibration Needed):**
- Success rate < 80%
- MAE > 3% for either dimension
- RMSE > 5% for either dimension
- Systematic bias in errors (all positive or all negative)

---

## Experimental Design

### Recommended Test Set

**Test 1: Fixed Object at Multiple Distances**
```
Object: Known dimensions (e.g., 10cm x 8cm)
Distances: 1.5m, 2.0m, 2.5m, 3.0m, 3.5m, 4.0m
Purpose: Test distance accuracy and perspective effects
```

**Test 2: Multiple Objects at Fixed Distance**
```
Distance: 2.5m (constant)
Objects: Vary height/width (5cm x 3cm, 10cm x 8cm, 15cm x 12cm)
Purpose: Test size measurement accuracy
```

**Test 3: Validation at Operating Conditions**
```
Distance: 2.5-3.5m (typical usage range)
Objects: Multiple sizes and orientations
Purpose: Simulate real-world usage
```

**Test 4: Challenging Conditions**
```
Lighting: Various lighting conditions (bright, dim, shadows)
Angles: Various object orientations
Purpose: Test robustness of detection algorithm
```

### Minimum Sample Size

- **Pilot validation:** 5-10 measurements
- **Thorough validation:** 15-25 measurements
- **Publication-quality:** 30-50 measurements

Recommendation: Start with 10 measurements across different distances and object sizes.

---

## Troubleshooting

### Problem: Low Success Rate

**Causes:**
1. Calibration error is high (reprojection error > 1 pixel)
2. Ground truth values are inaccurate
3. Distance measurements are inaccurate
4. Object detection is failing
5. Tolerance is too strict

**Solutions:**
- Recalibrate with more calibration images
- Verify ground truth measurements with multiple manual measurements
- Re-measure distances more carefully (use laser distance meter if available)
- Check detection method (try different method)
- Increase tolerance slightly (±0.75 cm or ±12%)

### Problem: Systematic Bias (All Errors Positive or Negative)

**Indicates:**
- Systematic error in calibration
- Systematic error in distance measurement
- Camera distortion not properly corrected

**Solutions:**
- Check if camera matrix and distortion coefficients were saved correctly
- Verify undistortion is working (check intermediate images)
- Re-measure distances (may be consistently off)
- Try different calibration images (checkerboard fully visible in all?)

### Problem: Large Outliers

**Indicates:**
- Poor detection on particular images
- Unusual lighting conditions
- Object at angle to camera (not perpendicular)

**Solutions:**
- Check detected bounding boxes (visualize with `ImageProcessor.draw_bounding_box`)
- Improve lighting conditions
- Ensure object is perpendicular to camera optical axis
- Use manual detection for problematic images

---

## Performance Benchmarks

### Expected Performance (Good Calibration)

| Metric | Typical Value | Good | Excellent |
|--------|--------------|------|-----------|
| Success Rate | 90% | >85% | >95% |
| Height MAE | 0.2 cm | <0.5 cm | <0.2 cm |
| Width MAE | 0.15 cm | <0.4 cm | <0.15 cm |
| Height RMSE | 0.3 cm | <0.6 cm | <0.3 cm |
| Width RMSE | 0.2 cm | <0.5 cm | <0.2 cm |

### Factors Affecting Performance

**Positive Impact:**
- More calibration images (20-50)
- Better calibration image coverage (all checkerboard features visible)
- Better focus on calibration images
- Larger object in image (more pixels)
- Better lighting conditions

**Negative Impact:**
- High calibration reprojection error
- Poor focus
- Small objects in image
- Objects at angles to camera
- Uneven lighting/shadows

---

## Advanced Topics

### Custom Error Thresholds

```python
# For strict tolerance (±3%)
validator = ValidationExperiment(
    camera_matrix,
    distortion_coeffs,
    tolerance_cm=0.2,
    tolerance_percent=3.0
)

# For loose tolerance (±15%)
validator = ValidationExperiment(
    camera_matrix,
    distortion_coeffs,
    tolerance_cm=1.0,
    tolerance_percent=15.0
)
```

### Batch Processing Multiple Ground Truth Files

```python
from pathlib import Path

gt_dir = Path('ground_truth_files')
all_results = []

for gt_file in gt_dir.glob('*.json'):
    print(f"\nValidating {gt_file.name}...")
    validator = ValidationExperiment(camera_matrix, distortion_coeffs)
    results = validator.validate_batch_from_file(str(gt_file))
    all_results.extend(results)

# Aggregate statistics
all_heights = [r.height_error_cm for r in all_results]
print(f"Overall height MAE: {np.mean(np.abs(all_heights)):.3f} cm")
```

---

## References

1. **Evaluating Model Performance:**
   - https://scikit-learn.org/stable/modules/model_evaluation.html

2. **Statistical Analysis:**
   - Mean Absolute Error: https://en.wikipedia.org/wiki/Mean_absolute_error
   - Root Mean Square Error: https://en.wikipedia.org/wiki/Root_mean_square_error

3. **Calibration Validation:**
   - Zhang, Z. (2000). "A flexible new technique for camera calibration"

4. **Computer Vision Literature:**
   - Hartley & Zisserman (2003). *Multiple view geometry in computer vision*

---

## File Structure

```
Step3_Validation_Experiment/
├── validation_experiment.py        # Main validation script
├── README_STEP3.md                # This file
└── ground_truth.json              # Ground truth measurements (example)
```

---

## Next Steps

After completing validation:
1. **Analyze results** - Understand error sources
2. **Document findings** - Record success rate, error metrics
3. **Optimize if needed** - Recalibrate if results are poor
4. **Integration** - Use calibration for your application
5. **Video documentation** - Record demo of measurement process

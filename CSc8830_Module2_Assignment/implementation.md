# Script Execution Guide

## Overview
This document provides complete instructions for executing each script in the CSc 8830 Module 2 assignment.

---

## Step 1: Camera Calibration

### Script: `calibrate_camera.py`

**Location:** `Step1_Camera_Calibration/calibrate_camera.py`

**Purpose:**
Performs camera calibration using OpenCV's calibration framework. Detects checkerboard patterns in multiple images and computes:
- Camera intrinsic matrix K (focal length and principal point)
- Lens distortion coefficients
- Reprojection error metrics

**Documentation:** See file header for theory and details

**Execution:**

```bash
cd Step1_Camera_Calibration

# Basic execution (uses default parameters)
python calibrate_camera.py

# With custom parameters
python calibrate_camera.py \
    --images clean_checkerboard_images/ \
    --checkerboard 16 16 \
    --square_size 0.025 \
    --output output/

# With verbose output
python calibrate_camera.py --verbose
```

**Arguments:**
- `--images`: Path to folder containing calibration images (default: `clean_checkerboard_images/`)
- `--checkerboard`: Checkerboard size as (cols rows) (default: 16 16)
- `--square_size`: Physical square size in meters (default: 0.025)
- `--output`: Output folder for calibration results (default: `output/`)
- `--verbose`: Print detailed progress information

**Outputs:**
- `output/camera_matrix.npy` - Camera intrinsic matrix K
- `output/distortion_coeffs.npy` - Lens distortion coefficients
- `output/calibration_info.json` - Calibration statistics and metadata

**Expected Results:**
- Reprojection error: ~0.009 pixels (excellent)
- Detection rate: 100% (26/26 images)

---

### Script: `visualize_detections.py`

**Location:** `Step1_Camera_Calibration/visualize_detections.py`

**Purpose:**
Creates visualization of checkerboard corner detections and generates an animated GIF showing the detection process.

**Documentation:** See file header for details

**Execution:**

```bash
cd Step1_Camera_Calibration

# Basic execution (creates GIF animation)
python visualize_detections.py

# With custom parameters
python visualize_detections.py \
    --images clean_checkerboard_images/ \
    --checkerboard 16 16 \
    --output calibration_viz/

# With detailed output
python visualize_detections.py --verbose
```

**Arguments:**
- `--images`: Path to calibration images folder (default: `clean_checkerboard_images/`)
- `--checkerboard`: Checkerboard size as (cols rows) (default: 16 16)
- `--output`: Output folder for visualizations (default: `calibration_viz/`)
- `--verbose`: Print detailed progress

**Outputs:**
- `calibration_viz/calibration_process.gif` - Animated GIF of detections
- `calibration_viz/detection_stats.json` - Detection statistics

**What to Look For:**
- GIF shows detected corners marked with circles
- All images should show successful corner detection
- Green circles indicate detected corners

---

## Step 2: Object Dimension Measurement

### Script: `batch_measure_corrected.py`

**Location:** `Step2_Object_Dimension_Measurement/batch_measure_corrected.py`

**Purpose:**
Measures real-world object dimensions from smartphone images using:
- Edge-based object detection
- Perspective projection formula
- iPhone 16 Pro Max calibration

**Key Formula (CORRECTED):**
```
H_real = (h_pixel * distance) / f_pixel
```

**Documentation:** See file header for formula explanation

**Execution:**

```bash
cd Step2_Object_Dimension_Measurement

# Basic execution (measures all images in 'images/' at 2.6 meters)
python batch_measure_corrected.py

# With custom distance
python batch_measure_corrected.py --distance 2.6

# With verbose output
python batch_measure_corrected.py --distance 2.6 --verbose

# Custom input/output paths
python batch_measure_corrected.py \
    --images images/ \
    --calibration iphone16_calibration/ \
    --distance 2.6 \
    --output measurements_step2/
```

**Arguments:**
- `--images`: Path to folder with test images (default: `images/`)
- `--calibration`: Path to calibration folder (default: `iphone16_calibration/`)
- `--distance`: Distance to objects in meters (default: 2.6)
- `--output`: Output folder for results (default: `measurements_step2/`)
- `--verbose`: Print detailed measurements

**Inputs Required:**
- Calibration files:
  - `iphone16_calibration/camera_matrix.npy`
  - `iphone16_calibration/distortion_coeffs.npy`
- Test images in `images/` folder (6 images)

**Outputs:**
- `measurements_step2/measurement_summary.json` - Summary of all measurements
- `measurements_step2/IMG_*_result.json` - Individual image results

**Expected Results:**
- Detection success: 100% (6/6 images)
- Average measurement: 382.26 cm × 289.40 cm
- All measurements realistic for 2.6m distance

---

## Step 3: Validation Experiment

**Note:** Step 3 measurements are pre-computed. No script execution needed.

**View Results:**

```bash
cd Step3_Validation_Experiment

# View ground truth values
cat measurements_step3/ground_truth.json

# View validation statistics
cat measurements_step3/validation_results.json

# View specific image results
cat measurements_step3/IMG_9586_result.json
```

**Pre-computed Results:**
- Ground truth values for 3 validation images
- Error metrics (MAE, RMSE, MAPE)
- Success rate: 66.7% (2/3 perfect measurements)

---

## Troubleshooting

### Step 1 - Calibration Issues

**Problem:** "No checkerboard detected in images"
- **Solution:** Ensure images are in `clean_checkerboard_images/` folder
- **Check:** Run with `--verbose` to see detection details

**Problem:** High reprojection error (>0.1 pixels)
- **Solution:** Ensure checkerboard images are clear and well-lit
- **Check:** Use `visualize_detections.py` to verify corner detection

### Step 2 - Measurement Issues

**Problem:** "No calibration files found"
- **Solution:** Ensure calibration matrices are in `iphone16_calibration/` folder
- **Check:** Files should be `camera_matrix.npy` and `distortion_coeffs.npy`

**Problem:** "Object not detected in image"
- **Solution:** Image may have poor contrast or lighting
- **Check:** Ensure object has clear edges and is well-separated from background

**Problem:** Unrealistic measurements
- **Solution:** Check distance parameter matches actual distance
- **Check:** Run with `--verbose` to see detected bounding boxes

---

## Summary

| Step | Script | Command | Output |
|------|--------|---------|--------|
| 1 | calibrate_camera.py | `python calibrate_camera.py` | Calibration matrices |
| 1 | visualize_detections.py | `python visualize_detections.py` | GIF animation, stats |
| 2 | batch_measure_corrected.py | `python batch_measure_corrected.py` | Measurement results |
| 3 | (pre-computed) | `cat measurements_step3/*.json` | Validation statistics |

---

**All scripts include comprehensive docstrings and comments. See individual files for technical details.**

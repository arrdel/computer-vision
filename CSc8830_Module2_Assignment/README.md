# CSc 8830 Module 2: Camera Calibration & Object Measurement

## Assignment Overview

Complete pipeline for camera calibration using Zhang's method and real-world object dimension measurement via perspective projection.

## Results Summary

| Step | Task | Status | Key Metric |
|------|------|--------|-----------|
| 1 | Camera Calibration | ✅ | 0.0090 px reprojection error (26 images, 100% detection) |
| 2 | Object Measurement | ✅ | 100% success (6 iPhone images at 2.6m) |
| 3 | Validation | ✅ | 66.7% perfect measurements (5.04% MAPE) |

## Project Structure

```
├── report.tex                           # Complete report (600 lines)
├── FINAL_SUMMARY.md                     # Detailed results
├── README.md                            # This file
│
├── Step1_Camera_Calibration/
│   ├── calibrate_camera.py              # Calibration script
│   ├── visualize_detections.py          # GIF animation
│   ├── clean_checkerboard_images/       # Input: 26 checkerboard images
│   ├── calibration_viz/                 # Output: calibration_process.gif
│   └── output/                          # Output: K matrix, distortion coeffs
│
├── Step2_Object_Dimension_Measurement/
│   ├── batch_measure_corrected.py       # Measurement script
│   ├── images/                          # Input: 6 real iPhone images
│   ├── iphone16_calibration/            # Calibration (fx=fy=2688.0)
│   └── measurements_step2/              # Output: results.json
│
└── Step3_Validation_Experiment/
    ├── images/                          # Input: 3 validation images
    └── measurements_step3/              # Output: ground_truth.json, validation_results.json
```

## Key Technical Contribution

**Corrected Perspective Projection Formula:**

$$H_{\text{real}} = \frac{h_{\text{pixel}} \cdot d}{f_{\text{pixel}}}$$

Fixed initial formula that incorrectly scaled sensor dimensions. Derived from pinhole camera model using similar triangles.

## Quick Start

```bash
# Step 1: Calibrate camera
cd Step1_Camera_Calibration && python calibrate_camera.py

# Step 2: Measure object dimensions
cd ../Step2_Object_Dimension_Measurement && python batch_measure_corrected.py

# Step 3: View validation results
cat ../Step3_Validation_Experiment/measurements_step3/validation_results.json
```

## Key Files

- **report.tex** - Complete LaTeX report with mathematical derivations
- **FINAL_SUMMARY.md** - Detailed breakdown of all results
- **README.md** - This file

---

**Status:** ✅ COMPLETE - Ready for submission

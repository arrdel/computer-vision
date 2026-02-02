# CSc 8830: Computer Vision - Module 2 Assignment
## Camera Calibration & Real-World Object Dimension Measurement

**Student:** [Your Name]  
**Course:** CSc 8830 - Computer Vision  
**Semester:** Spring 2026  
**Assignment:** Module 2 - Camera Calibration & Perspective Projection

---

## 📋 Project Overview

This project implements a complete pipeline for measuring real-world 2D dimensions of objects using smartphone camera calibration and perspective projection equations. The assignment consists of three sequential steps:

1. **Step 1:** Camera calibration using OpenCV with a checkerboard pattern
2. **Step 2:** Implementation of 2D object dimension measurement using perspective projection
3. **Step 3:** Validation experiment with ground truth measurement

---

## 🏗️ Project Structure

```
CSc8830_Module2_Assignment/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── Step1_Camera_Calibration/
│   ├── calibrate_camera.py           # Main calibration script
│   ├── checkerboard_pattern.pdf      # Printable checkerboard (8x6)
│   ├── calibration_data/
│   │   ├── calibration_images/       # Captured images for calibration
│   │   ├── camera_matrix.npy         # Calibrated camera matrix K
│   │   ├── distortion_coeffs.npy     # Distortion coefficients
│   │   └── calibration_results.txt   # Calibration statistics
│   └── README_STEP1.md               # Detailed Step 1 guide
│
├── Step2_Object_Dimension_Measurement/
│   ├── measure_dimensions.py         # Main measurement script
│   ├── perspective_projection.py     # Core perspective projection module
│   ├── utils.py                      # Helper functions
│   ├── test_images/                  # Test images for measurement
│   └── README_STEP2.md               # Detailed Step 2 guide
│
├── Step3_Validation_Experiment/
│   ├── validation_experiment.py      # Validation script
│   ├── experiment_images/            # Images captured for validation
│   ├── ground_truth_measurements.txt # Manual measurements
│   ├── experiment_results.txt        # Experiment output
│   └── README_STEP3.md               # Detailed Step 3 guide
│
└── Documentation/
    ├── math_derivations.md           # Mathematical formulations
    ├── calibration_theory.md         # Camera calibration theory
    └── perspective_projection_theory.md
```

---

## 🔧 Requirements

### Hardware
- Smartphone with camera (Android/iOS)
- Computer with Python 3.8+
- Printed checkerboard pattern (A4 size, 8×6)
- Objects to measure (at distance > 2 meters)

### Software Dependencies
```
opencv-python>=4.5.0
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
scikit-image>=0.17.0
Pillow>=8.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1: Camera Calibration
```bash
cd Step1_Camera_Calibration
python calibrate_camera.py --num_images 20 --output calibration_data/
```

### Step 2: Measure Object Dimensions
```bash
cd Step2_Object_Dimension_Measurement
python measure_dimensions.py --image path/to/image.jpg --known_height 1.7 --distance 3.0
```

### Step 3: Run Validation Experiment
```bash
cd Step3_Validation_Experiment
python validation_experiment.py --calibration_data ../Step1_Camera_Calibration/calibration_data/
```

---

## 📐 Mathematical Foundation

### Camera Intrinsic Matrix (Step 1)
The camera intrinsic matrix K contains focal length and principal point:

```
K = [fx   0  cx]
    [ 0  fy  cy]
    [ 0   0   1]
```

Where:
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (optical center)

### Perspective Projection (Step 2)
Given a 3D point in world coordinates and camera matrix:

```
p_image = K @ [R | t] @ P_world
```

For measuring 2D object dimensions:
```
height_real = (height_pixel * distance * sensor_height) / (focal_length * image_height)
```

### Measurement Error Analysis (Step 3)
```
error = |measured_value - ground_truth| / ground_truth × 100%
```

---

## 📖 Detailed Guides

Each step has a dedicated README:

- **[Step 1: Camera Calibration](./Step1_Camera_Calibration/README_STEP1.md)** - Detailed calibration procedure
- **[Step 2: Object Measurement](./Step2_Object_Dimension_Measurement/README_STEP2.md)** - Measurement pipeline
- **[Step 3: Validation](./Step3_Validation_Experiment/README_STEP3.md)** - Validation methodology

---

## 🎯 Key Features

✅ **Automatic Checkerboard Detection** - Detects corners with sub-pixel accuracy  
✅ **Lens Distortion Correction** - Accounts for radial and tangential distortion  
✅ **Real-World Dimension Calculation** - Uses perspective projection equations  
✅ **Error Metrics** - Compares measured vs ground truth  
✅ **Visualization Tools** - Plots calibration results and measurements  
✅ **Comprehensive Documentation** - Mathematical derivations included

---

## 📊 Expected Results

### Calibration Accuracy
- Reprojection Error: < 0.5 pixels (good calibration)
- Focal Length: 3000-4000 pixels (typical smartphone)

### Measurement Accuracy
- Error Range: 5-10% (depending on distance and image quality)
- Validation Test Case: < 8% error on ground truth measurements

---

## 🔬 Experimental Validation

### Experiment Setup
- **Object**: Fixed object (A4 paper, whiteboard, door, etc.)
- **Distance**: 3-5 meters (> 2 meters required)
- **Number of Images**: 10-15 at different angles
- **Ground Truth**: Manual measurement with measuring tape

### Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Percentage Error Distribution

---

## 📝 Usage Examples

### Example 1: Basic Camera Calibration
```python
from Step1_Camera_Calibration.calibrate_camera import CameraCalibrator

calibrator = CameraCalibrator(
    checkerboard_size=(8, 6),
    square_size=0.025,  # 25mm
    num_images=20
)

calibrator.calibrate(image_folder='calibration_data/calibration_images/')
calibrator.save_calibration('calibration_data/')
```

### Example 2: Measure Object Dimensions
```python
from Step2_Object_Dimension_Measurement.measure_dimensions import ObjectMeasurer

measurer = ObjectMeasurer(
    calibration_file='calibration_data/camera_matrix.npy',
    distortion_file='calibration_data/distortion_coeffs.npy'
)

height_real = measurer.measure_object_height(
    image_path='test_image.jpg',
    known_reference_height=1.7,  # meters
    distance_to_camera=3.5       # meters
)

print(f"Measured height: {height_real:.3f} meters")
```

### Example 3: Run Validation Experiment
```python
from Step3_Validation_Experiment.validation_experiment import ValidationExperiment

experiment = ValidationExperiment(
    calibration_data_path='calibration_data/',
    ground_truth_file='ground_truth_measurements.txt'
)

results = experiment.run_validation(
    image_folder='experiment_images/',
    distance=4.0
)

experiment.plot_results(results)
```

---

## 📚 References

1. **Camera Calibration**: Z. Zhang, "A Flexible New Technique for Camera Calibration," IEEE TPAMI, 2000
2. **Perspective Projection**: R. Hartley & A. Zisserman, "Multiple View Geometry," Cambridge University Press
3. **OpenCV Documentation**: https://docs.opencv.org/master/d9/d0c/group__calib3d.html

---

## ⚠️ Important Notes

1. **Checkerboard Quality**: Use a high-quality, printed checkerboard pattern (laminated preferred)
2. **Lighting Conditions**: Ensure uniform lighting during calibration and measurement
3. **Distance Measurement**: Measure distance from camera to object center using laser meter or measuring tape
4. **Multiple Trials**: Run validation multiple times for statistical significance
5. **Calibration Reusability**: Once calibrated, camera matrix can be reused for same camera/smartphone

---

## 🐛 Troubleshooting

### Issue: Checkerboard Not Detected
- Ensure printed pattern is high-contrast and flat
- Try different distances from checkerboard
- Improve lighting conditions

### Issue: Calibration Error Too High (> 1.0 pixels)
- Use more calibration images (aim for 20+)
- Ensure checkerboard covers entire image at various angles
- Check for lens distortion causing edge blur

### Issue: Measurement Results Inaccurate
- Verify distance measurement is accurate
- Ensure object is perpendicular to camera
- Check that calibration matrix is correctly loaded

---

## ✅ Assignment Checklist

- [ ] Step 1 Complete: Camera calibration with calibration images
- [ ] Step 2 Complete: Dimension measurement implementation
- [ ] Step 3 Complete: Validation experiment with ground truth
- [ ] Documentation: Mathematical derivations typed
- [ ] Code: Well-commented and documented
- [ ] Video: Screen recording or working demonstration
- [ ] README: Complete and clear
- [ ] GitHub: Repository created and link provided
- [ ] PDF: Submission document with all required elements

---

**Last Updated:** February 2, 2026  
**Status:** Ready for Submission

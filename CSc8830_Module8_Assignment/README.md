# CSc 8830 — Module 8: Stereo Camera Classroom Localisation

## Assignment

> Using a simple stereo camera setup, compute the locations (2D, parallel to floor) of each table and chair in the classroom. Submit a X-Y 2D plot marking tables in red and chairs as blue and your code file.

## Approach

### Stereo Geometry

A simple stereo camera setup consists of two horizontally aligned cameras separated by a known **baseline** $b$. For a point visible in both images at pixel columns $u_L$ (left) and $u_R$ (right), the **disparity** is:

$$d = u_L - u_R$$

The **depth** (distance from the camera) is then:

$$Z = \frac{f \cdot b}{d}$$

where $f$ is the focal length in pixels. The lateral position is:

$$X = \frac{(u - c_x) \cdot Z}{f}$$

This gives us the 2D floor-plan coordinates $(X, Y)$ where $X$ is lateral and $Y = Z$ is depth.

### Pipeline

1. **Object detection** — YOLOv8 (COCO-trained) detects `chair` and `dining table` classes in the left image.
2. **Stereo disparity** — Semi-Global Block Matching (SGBM) computes a dense disparity map from the rectified stereo pair.
3. **Depth estimation** — For each detected bounding box, the median disparity in the central region is converted to depth using the stereo formula above.
4. **2D localisation** — Depth + lateral offset gives $(X, Y)$ floor coordinates for every object.
5. **Plotting** — Tables plotted as **red squares**, chairs as **blue circles** on a 2D X-Y floor plan.

## Usage

```bash
conda activate computer_vision_env

# With real stereo images:
python stereo_localization.py --left images/left.jpg --right images/right.jpg

# Demo mode (synthetic classroom, no images needed):
python stereo_localization.py --demo

# Custom camera parameters:
python stereo_localization.py --left images/left.jpg --right images/right.jpg \
    --focal 700 --baseline 0.12
```

## Output

| File | Description |
|------|-------------|
| `output/detections_left.png` | Left image with annotated bounding boxes (red = table, blue = chair) |
| `output/disparity_map.png` | Stereo disparity map (colour-coded) |
| `output/classroom_layout_2d.png` | **2D X-Y floor-plan plot** (tables = red ■, chairs = blue ●) |
| `output/detections.csv` | Per-object: label, floor X, floor Y, depth, confidence |

## Requirements

```
opencv-python>=4.8
numpy
matplotlib
ultralytics   # for YOLOv8 detection (optional — heuristic fallback available)
```

Install: `pip install ultralytics` (auto-downloads YOLOv8 nano model on first run).

## Results

### 2D Classroom Layout

<p align="center">
  <img src="output/classroom_layout_2d.png" width="80%" alt="Classroom Layout — 2D Floor Plan"/>
</p>
<p align="center"><em>2D X-Y plot of detected tables (red ■) and chairs (blue ●). Camera is at the origin.</em></p>

### Detections on Left Image

<p align="center">
  <img src="output/detections_left.png" width="80%" alt="Detections on Left Image"/>
</p>

### Disparity Map

<p align="center">
  <img src="output/disparity_map.png" width="60%" alt="Stereo Disparity Map"/>
</p>

# Computer Vision Height Estimator

A Python application that estimates a person's height from images or live webcam feed using **MediaPipe Pose Landmarker**. No physical reference object (like a ruler or paper) is required — the system uses anthropometric body proportions to convert pixel measurements into real-world centimeters.

---

## How It Works

1. **Pose Detection** — MediaPipe's Pose Landmarker model detects 33 body landmarks (nose, eyes, shoulders, hips, knees, ankles, heels, toes, etc.) on the person in the image.

2. **Height Measurement (pixels)** — The vertical distance from the estimated top of the skull to the lowest foot point is calculated in pixels. The top of the head is approximated using the nose-to-eye offset since MediaPipe doesn't provide a skull-top landmark.

3. **Pixel-to-CM Conversion** — Two independent body proportion references are used to establish a pixel-to-centimeter scale:
   - **Head height** (top-of-skull to chin ≈ 22 cm average adult)
   - **Shoulder width** (biacromial distance ≈ 38 cm average adult)

   The final height estimate is the **average** of both methods, improving robustness across different body types and camera angles.

4. **Visualization** — The output image is annotated with:
   - Green skeleton lines and red landmark dots (full pose)
   - Blue vertical measurement line (head to feet)
   - Height text overlay (auto-scaled to image resolution)

---

## Features

| Feature | Description |
|---|---|
| **No reference object needed** | Uses body proportions instead of physical markers |
| **Three modes** | Single image, batch folder, or live webcam |
| **Pose skeleton overlay** | Draws all 33 landmarks and skeletal connections |
| **Auto-scaling text** | Height label scales with image resolution |
| **Auto model download** | Downloads the MediaPipe model on first run |

---

## Requirements

- Python 3.10+
- Dependencies:
  ```
  opencv-python
  mediapipe
  numpy
  ```

### Install

```bash
pip install opencv-python mediapipe numpy
```

---

## Usage

### Single Image

```bash
python test.py --image path/to/photo.jpg
```

### Batch Folder (press any key for next, 'q' to quit)

```bash
python test.py --folder path/to/image_folder
```

### Live Webcam (press 'q' to quit)

```bash
python test.py --webcam
```

### Default Mode

Edit the `MODE`, `IMAGE_PATH`, and `IMAGE_FOLDER` variables at the top of `test.py` to set defaults, then simply run:

```bash
python test.py
```

---

## Configuration

These variables can be adjusted at the top of `test.py`:

| Variable | Default | Description |
|---|---|---|
| `MODE` | `"image"` | `"image"`, `"folder"`, or `"webcam"` |
| `IMAGE_PATH` | `"test_images/image6.jpeg"` | Path to a single image |
| `IMAGE_FOLDER` | `"test_images"` | Folder of images for batch mode |
| `ESTIMATE_WITHOUT_REFERENCE` | `True` | Use body-proportion estimation |
| `MANUAL_PX_PER_CM` | `None` | Override with a known scale (px/cm) |

---

## Project Structure

```
Computer VIsion/
├── test.py                        # Main height estimator script
├── README.md                      # This file
├── pose_landmarker_lite.task      # MediaPipe model (auto-downloaded)
└── test_images/                   # Folder for batch processing
    ├── image1.jpg
    ├── image2.png
    └── ...
```

---

## Accuracy & Limitations

- **Accuracy** depends on image quality, pose visibility, and how closely the person's proportions match population averages.
- Works best when the person is **standing upright**, fully visible **head-to-toe**, and facing roughly toward the camera.
- The no-reference method relies on average adult body proportions (head height ≈ 22 cm, shoulder width ≈ 38 cm). Results may vary for children or individuals with non-average proportions.
- Camera lens distortion (e.g., wide-angle selfies) can affect accuracy.

---

## Technologies

- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — ML-based body landmark detection (33 keypoints)
- [OpenCV](https://opencv.org/) — Image I/O, drawing, and display
- [NumPy](https://numpy.org/) — Numerical operations

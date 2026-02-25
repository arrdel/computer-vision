# FaceMetrics

A computer vision application that analyses video footage to:

- **(A) Estimate eye blink rate** — blinks per second / per minute over a study session
- **(B) Estimate facial dimensions** — eyes, face (head↔chin, ear↔ear), nose, and mouth in millimetres

## Architecture

```
main.py                     ← Entry point (runs both tasks)
├── src/
│   ├── video_processor.py  ← Frame-by-frame video iterator
│   ├── landmark_extractor.py ← MediaPipe Face Mesh (478 landmarks)
│   ├── blink_detector.py   ← Task A: EAR-based blink detection
│   ├── face_measurer.py    ← Task B: IPD-calibrated measurements
│   └── utils.py            ← I/O, math, and plotting helpers
├── configs/
│   └── default.yaml        ← All tuneable parameters
├── data/videos/            ← Place your input video here
└── output/                 ← Generated logs, plots, and summary
```

## Quick Start

```bash
# 1. Activate environment
conda activate facemetrics

# 2. Place your video
cp /path/to/study_session.mp4 data/videos/study_session.mp4

# 3. Run
python main.py

# Or with a specific video:
python main.py --video data/videos/my_video.mp4
```

## Output

| File | Description |
|------|-------------|
| `output/blink_logs/blink_log.csv` | Per-frame EAR values |
| `output/measurements/measurements.csv` | Per-frame pixel measurements |
| `output/summary.json` | Full results (blinks, dimensions, rates) |
| `output/plots/ear_over_time.png` | EAR signal with blink events |
| `output/plots/face_dimensions.png` | Bar chart of dimensions (mm) |

## Methodology

### Task A — Blink Detection

Uses the **Eye Aspect Ratio (EAR)** algorithm (Soukupová & Čech, 2016):

$$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

A blink is counted when EAR drops below a threshold for ≥ N consecutive frames.

### Task B — Face Dimensions

1. Detect 478 face landmarks using **MediaPipe Face Mesh**
2. Calibrate pixel→mm using **interpupillary distance** (avg ~63 mm)
3. Measure bounding boxes / distances for eyes, face, nose, mouth
4. Report **median** across all frames for robustness

## Configuration

All parameters are in `configs/default.yaml`:

```yaml
blink:
  ear_threshold: 0.21        # Lower = less sensitive
  min_consec_frames: 2       # Higher = fewer false positives

measurement:
  reference_distance_mm: 63.0  # Known IPD for calibration
```

## Dependencies

- Python 3.10+
- OpenCV, MediaPipe, NumPy, SciPy, Matplotlib, PyYAML, tqdm

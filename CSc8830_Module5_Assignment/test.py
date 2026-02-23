"""
Computer Vision Height Estimator
================================
Estimates a person's height using MediaPipe Pose landmarks and optionally
an A4 reference object (29.7 cm tall).

Modes:
  MODE = "webcam"  – Live webcam
  MODE = "image"   – Single image file
  MODE = "folder"  – Process all images in a folder one by one
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import math
import urllib.request
import os

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MODE = "image"             # "webcam", "image", or "folder"
IMAGE_PATH = "test_images/image6.jpeg"
IMAGE_FOLDER = "test_images"  # folder of test images

REFERENCE_HEIGHT_CM = 29.7   # A4 portrait height

AVG_HEAD_HEIGHT_CM = 23.0
AVG_HEAD_TO_BODY_RATIO = 7.5
ESTIMATE_WITHOUT_REFERENCE = True
MANUAL_PX_PER_CM = None

# ──────────────────────────────────────────────
# MediaPipe Model Setup
# ──────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
)

# Landmark indices
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# ──────────────────────────────────────────────
# HEIGHT CALCULATION
# ──────────────────────────────────────────────

def _dist(lm_a, lm_b, frame_h, frame_w):
    """Euclidean distance between two landmarks in pixel space."""
    ax, ay = lm_a.x * frame_w, lm_a.y * frame_h
    bx, by = lm_b.x * frame_w, lm_b.y * frame_h
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _mid_y(lm_a, lm_b, frame_h):
    """Average y-position of two landmarks in pixels."""
    return (lm_a.y + lm_b.y) / 2.0 * frame_h


def get_person_height_px(landmarks, frame_h, frame_w):
    """Simple top-to-bottom bounding height in pixels (used for the blue line)."""
    nose_y = landmarks[NOSE].y * frame_h
    eye_y = min(landmarks[LEFT_EYE].y,
                landmarks[RIGHT_EYE].y) * frame_h

    face_offset = abs(nose_y - eye_y)
    top_y = nose_y - 2.5 * face_offset

    foot_points = [
        landmarks[LEFT_HEEL].y * frame_h,
        landmarks[RIGHT_HEEL].y * frame_h,
        landmarks[LEFT_FOOT_INDEX].y * frame_h,
        landmarks[RIGHT_FOOT_INDEX].y * frame_h,
        landmarks[LEFT_ANKLE].y * frame_h,
        landmarks[RIGHT_ANKLE].y * frame_h,
    ]

    bottom_y = max(foot_points)
    height_px = bottom_y - top_y

    nose_x = int(landmarks[NOSE].x * frame_w)

    return height_px, nose_x, int(top_y), int(bottom_y)


def get_segmented_height_px(landmarks, frame_h, frame_w):
    """
    More accurate height by summing body segment lengths along the
    skeletal chain:  top-of-head → nose → mid-shoulder → mid-hip →
    knee → ankle → foot-tip.
    Uses the average of left/right sides for symmetry.
    """
    lm = landmarks

    # 1) Top of head to nose (estimated)
    nose_y = lm[NOSE].y * frame_h
    eye_y = min(lm[LEFT_EYE].y, lm[RIGHT_EYE].y) * frame_h
    face_offset = abs(nose_y - eye_y)
    head_top_to_nose = 2.5 * face_offset  # skull above nose

    # 2) Nose to mid-shoulder
    mid_shoulder_y = _mid_y(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER], frame_h)
    nose_to_shoulder = abs(nose_y - mid_shoulder_y)

    # 3) Mid-shoulder to mid-hip
    shoulder_to_hip_L = _dist(lm[LEFT_SHOULDER], lm[LEFT_HIP], frame_h, frame_w)
    shoulder_to_hip_R = _dist(lm[RIGHT_SHOULDER], lm[RIGHT_HIP], frame_h, frame_w)
    shoulder_to_hip = (shoulder_to_hip_L + shoulder_to_hip_R) / 2.0

    # 4) Mid-hip to knee
    hip_to_knee_L = _dist(lm[LEFT_HIP], lm[LEFT_KNEE], frame_h, frame_w)
    hip_to_knee_R = _dist(lm[RIGHT_HIP], lm[RIGHT_KNEE], frame_h, frame_w)
    hip_to_knee = (hip_to_knee_L + hip_to_knee_R) / 2.0

    # 5) Knee to ankle
    knee_to_ankle_L = _dist(lm[LEFT_KNEE], lm[LEFT_ANKLE], frame_h, frame_w)
    knee_to_ankle_R = _dist(lm[RIGHT_KNEE], lm[RIGHT_ANKLE], frame_h, frame_w)
    knee_to_ankle = (knee_to_ankle_L + knee_to_ankle_R) / 2.0

    # 6) Ankle to foot (use the LONGER of ankle→heel or ankle→toe)
    ankle_to_heel_L = _dist(lm[LEFT_ANKLE], lm[LEFT_HEEL], frame_h, frame_w)
    ankle_to_heel_R = _dist(lm[RIGHT_ANKLE], lm[RIGHT_HEEL], frame_h, frame_w)
    ankle_to_toe_L = _dist(lm[LEFT_ANKLE], lm[LEFT_FOOT_INDEX], frame_h, frame_w)
    ankle_to_toe_R = _dist(lm[RIGHT_ANKLE], lm[RIGHT_FOOT_INDEX], frame_h, frame_w)
    ankle_to_foot_L = max(ankle_to_heel_L, ankle_to_toe_L)
    ankle_to_foot_R = max(ankle_to_heel_R, ankle_to_toe_R)
    ankle_to_foot = (ankle_to_foot_L + ankle_to_foot_R) / 2.0

    total_px = (head_top_to_nose
                + nose_to_shoulder
                + shoulder_to_hip
                + hip_to_knee
                + knee_to_ankle
                + ankle_to_foot)

    return total_px


def estimate_height_no_reference(landmarks, frame_h, frame_w):
    """
    Estimate height without a reference object using multiple body
    proportions averaged together for robustness.

    References used:
      - Head height (top-of-skull to chin) ≈ 22 cm (average adult)
      - Shoulder width (biacromial) ≈ 38 cm (average adult)
    The final estimate is the average of both methods.
    """
    AVG_HEAD_CM = 22.0       # average skull-to-chin height
    AVG_SHOULDER_CM = 38.0   # average biacromial shoulder width

    # Bounding box height in pixels
    height_px, _, _, _ = get_person_height_px(landmarks, frame_h, frame_w)
    if height_px <= 0:
        return None

    estimates = []

    # Method 1: Head height (top-of-skull to chin)
    nose_y = landmarks[NOSE].y * frame_h
    eye_y = min(landmarks[LEFT_EYE].y, landmarks[RIGHT_EYE].y) * frame_h
    face_offset = abs(nose_y - eye_y)
    top_y = nose_y - 2.5 * face_offset
    chin_y = nose_y + 2.0 * face_offset   # chin below nose
    head_px = chin_y - top_y

    if head_px > 0:
        px_per_cm = head_px / AVG_HEAD_CM
        estimates.append(height_px / px_per_cm)

    # Method 2: Shoulder width
    shoulder_px = _dist(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER],
                        frame_h, frame_w)
    if shoulder_px > 0:
        px_per_cm = shoulder_px / AVG_SHOULDER_CM
        estimates.append(height_px / px_per_cm)

    if not estimates:
        return None

    return sum(estimates) / len(estimates)


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# POSE DRAWING
# ──────────────────────────────────────────────

# MediaPipe Pose connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

def draw_landmarks(frame, landmarks):
    """Draw pose skeleton (connections + landmark dots) on the frame."""
    h, w, _ = frame.shape

    # Draw connections (lines)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x1 = int(landmarks[start_idx].x * w)
            y1 = int(landmarks[start_idx].y * h)
            x2 = int(landmarks[end_idx].x * w)
            y2 = int(landmarks[end_idx].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

    # Draw landmark points
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)   # red dot
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1) # white outline


# ──────────────────────────────────────────────
# FRAME PROCESSING
# ──────────────────────────────────────────────

def process_frame(frame, landmarker):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = landmarker.detect(mp_image)

    height_cm = None

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]

        # Draw skeleton and landmarks on the frame
        draw_landmarks(frame, landmarks)

        # Draw height measurement line
        height_px, nose_x, top_y, bottom_y = get_person_height_px(
            landmarks, h, w)
        cv2.line(frame, (nose_x, top_y), (nose_x, bottom_y), (255, 0, 0), 2)
        cv2.circle(frame, (nose_x, top_y), 6, (255, 0, 0), -1)
        cv2.circle(frame, (nose_x, bottom_y), 6, (255, 0, 0), -1)

        px_per_cm = MANUAL_PX_PER_CM

        if px_per_cm is not None:
            height_cm = height_px / px_per_cm
        elif ESTIMATE_WITHOUT_REFERENCE:
            height_cm = estimate_height_no_reference(
                landmarks, h, w)

        if height_cm is not None:
            # Scale text size to image resolution
            font_scale = max(1.0, h / 600.0)
            thickness = max(2, int(h / 400))
            margin = int(40 * font_scale)

            cv2.putText(frame,
                        f"Height: {height_cm:.1f} cm",
                        (margin, int(80 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 255),
                        thickness)

    else:
        font_scale = max(1.0, h / 600.0)
        thickness = max(2, int(h / 400))
        margin = int(40 * font_scale)

        cv2.putText(frame,
                    "No person detected",
                    (margin, int(80 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),
                    thickness)

    return frame, height_cm


# ──────────────────────────────────────────────
# MODES
# ──────────────────────────────────────────────

def run_image():
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print(f"ERROR: Cannot read image '{IMAGE_PATH}'")
        sys.exit(1)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        processed, height_cm = process_frame(frame, landmarker)

    if height_cm is not None:
        print(f"Estimated Height: {height_cm:.1f} cm")
    else:
        print("Could not estimate height.")

    cv2.imshow("Height Estimator", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder():
    """Process all images in a folder, one by one. Press any key for next."""
    SUPPORTED_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')

    if not os.path.isdir(IMAGE_FOLDER):
        print(f"ERROR: Folder '{IMAGE_FOLDER}' not found.")
        sys.exit(1)

    # Get sorted list of image files
    image_files = sorted([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(SUPPORTED_EXT)
    ])

    if not image_files:
        print(f"ERROR: No images found in '{IMAGE_FOLDER}'")
        sys.exit(1)

    print("=" * 50)
    print(f"  Processing {len(image_files)} images from '{IMAGE_FOLDER}'")
    print("  Press any key → next image  |  'q' → quit")
    print("=" * 50)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, filename in enumerate(image_files, 1):
            filepath = os.path.join(IMAGE_FOLDER, filename)
            frame = cv2.imread(filepath)

            if frame is None:
                print(f"[{i}/{len(image_files)}] Skipping '{filename}' (cannot read)")
                continue

            processed, height_cm = process_frame(frame, landmarker)

            # Add filename label to the image
            h_img, w_img = processed.shape[:2]
            font_scale = max(0.8, h_img / 800.0)
            thickness = max(1, int(h_img / 500))
            margin = int(40 * font_scale)

            cv2.putText(processed,
                        f"[{i}/{len(image_files)}] {filename}",
                        (margin, h_img - margin),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (200, 200, 200),
                        thickness)

            if height_cm is not None:
                print(f"[{i}/{len(image_files)}] {filename}: "
                      f"{height_cm:.1f} cm")
            else:
                print(f"[{i}/{len(image_files)}] {filename}: "
                      f"No person detected")

            cv2.imshow("Height Estimator", processed)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Quit early.")
                break

    cv2.destroyAllWindows()
    print("=" * 50)
    print("Done.")


def run_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed, _ = process_frame(frame, landmarker)
            cv2.imshow("Height Estimator", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "--image":
            MODE = "image"
            if len(sys.argv) > 2:
                IMAGE_PATH = sys.argv[2]
        elif sys.argv[1] == "--folder":
            MODE = "folder"
            if len(sys.argv) > 2:
                IMAGE_FOLDER = sys.argv[2]
        elif sys.argv[1] == "--webcam":
            MODE = "webcam"

    if MODE == "webcam":
        run_webcam()
    elif MODE == "folder":
        run_folder()
    else:
        run_image()
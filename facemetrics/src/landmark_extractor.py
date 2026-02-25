"""
Landmark Extractor — MediaPipe FaceLandmarker (Tasks API v0.10+).

Provides a consistent interface for extracting 478 face landmarks
(including iris) from video frames.
"""

import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────
#  MediaPipe Face Mesh landmark index map
#  (key landmarks used by blink detector & face measurer)
# ──────────────────────────────────────────────

# --- Eyes (6-point model for EAR computation) ---
# Each eye defined by [p1, p2, p3, p4, p5, p6] where:
#   p1-p4 = horizontal axis endpoints
#   p2-p6, p3-p5 = vertical axis pairs
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# --- Eye contour (full) for dimension measurement ---
LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# --- Iris (indices 468-477 in the 478-landmark model) ---
LEFT_IRIS  = [474, 475, 476, 477]   # center = average
RIGHT_IRIS = [469, 470, 471, 472]

# --- Face outline ---
FACE_TOP       = 10     # Top of forehead
FACE_BOTTOM    = 152    # Chin
FACE_LEFT_EAR  = 234    # Left ear (viewer's left = subject's right)
FACE_RIGHT_EAR = 454    # Right ear

# --- Pupils (approximated by iris center) ---
LEFT_PUPIL_IDX  = LEFT_IRIS
RIGHT_PUPIL_IDX = RIGHT_IRIS

# --- Nose ---
NOSE_TIP    = 1
NOSE_BRIDGE = 6
NOSE_LEFT   = 129
NOSE_RIGHT  = 358

# --- Mouth ---
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14


class LandmarkExtractor:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API).
    Call `process(frame_rgb)` to get pixel-coordinate landmarks.
    """

    MODEL_PATH = "data/face_landmarker_v2_with_blendshapes.task"

    def __init__(self, cfg: dict):
        mp_cfg = cfg.get("mediapipe", {})

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self.MODEL_PATH,
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=mp_cfg.get("max_num_faces", 1),
            min_face_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
            min_face_presence_confidence=0.5,
            min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_rgb: np.ndarray):
        """
        Run face landmarker on an RGB frame.

        Returns
        -------
        landmarks_px : list[tuple[float, float]] | None
            478 landmarks as (x_px, y_px) in pixel coords,
            or None if no face detected.
        """
        h, w = frame_rgb.shape[:2]

        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect — VIDEO mode requires monotonically increasing timestamps
        self._timestamp_ms += 33   # ~30 fps increment
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return None

        # First face — convert normalised coords to pixel coords
        face = result.face_landmarks[0]
        landmarks_px = [(lm.x * w, lm.y * h) for lm in face]
        return landmarks_px

    def close(self):
        self.landmarker.close()

    # ── Convenience extractors ──
    @staticmethod
    def get_eye_points(landmarks, indices):
        """Return list of (x, y) for given landmark indices."""
        return [landmarks[i] for i in indices]

    @staticmethod
    def get_point(landmarks, index):
        return landmarks[index]

    @staticmethod
    def get_center(landmarks, indices):
        """Average position of a set of landmarks."""
        pts = [landmarks[i] for i in indices]
        return (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        )

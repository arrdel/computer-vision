"""
Task B — Face Dimension Estimator

Estimates real-world dimensions (in mm) of:
  • Eyes      – width & height of each eye
  • Face      – forehead-to-chin (vertical) & ear-to-ear (horizontal)
  • Nose      – width & height
  • Mouth     – width & height

Calibration uses interpupillary distance (IPD) as a reference:
  scale_factor = known_IPD_mm / measured_IPD_pixels

All measurements are accumulated per-frame and then averaged
over the video for robust estimation.
"""

from __future__ import annotations
import numpy as np
from src.utils import euclidean
from src.landmark_extractor import (
    LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
    LEFT_IRIS, RIGHT_IRIS,
    FACE_TOP, FACE_BOTTOM, FACE_LEFT_EAR, FACE_RIGHT_EAR,
    NOSE_TIP, NOSE_BRIDGE, NOSE_LEFT, NOSE_RIGHT,
    MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM,
    LandmarkExtractor,
)


def _bounding_size(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Return (width_px, height_px) of the bounding box around a set of points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(xs) - min(xs), max(ys) - min(ys)


class FaceMeasurer:
    """
    Accumulates per-frame pixel measurements, converts to mm via
    IPD calibration, and produces averaged results.

    Attributes after `finalize()`:
        measurements_mm : dict  — averaged dimensions in mm
    """

    def __init__(self, cfg: dict):
        meas_cfg = cfg.get("measurement", {})
        self.ref_distance_mm = meas_cfg.get("reference_distance_mm", 63.0)

        # Per-frame accumulators  {name: [value_pixels, ...]}
        self._accum: dict[str, list[float]] = {
            "left_eye_width": [],
            "left_eye_height": [],
            "right_eye_width": [],
            "right_eye_height": [],
            "face_vertical (head to chin)": [],
            "face_horizontal (ear to ear)": [],
            "nose_width": [],
            "nose_height": [],
            "mouth_width": [],
            "mouth_height": [],
        }
        self._scale_factors: list[float] = []
        self.frames_measured = 0

    # ──────────────────────────────────────────
    #  Per-frame update
    # ──────────────────────────────────────────
    def update(self, landmarks: list[tuple[float, float]] | None):
        """Call once per frame with detected landmarks (or None)."""
        if landmarks is None:
            return

        # --- Calibration: interpupillary distance (pixel) ---
        left_pupil  = LandmarkExtractor.get_center(landmarks, LEFT_IRIS)
        right_pupil = LandmarkExtractor.get_center(landmarks, RIGHT_IRIS)
        ipd_px = euclidean(left_pupil, right_pupil)
        if ipd_px < 5.0:          # face too small or bad detection
            return

        scale = self.ref_distance_mm / ipd_px   # mm per pixel
        self._scale_factors.append(scale)

        # --- Eyes ---
        left_eye_pts  = LandmarkExtractor.get_eye_points(landmarks, LEFT_EYE_CONTOUR)
        right_eye_pts = LandmarkExtractor.get_eye_points(landmarks, RIGHT_EYE_CONTOUR)

        lw, lh = _bounding_size(left_eye_pts)
        rw, rh = _bounding_size(right_eye_pts)
        self._accum["left_eye_width"].append(lw)
        self._accum["left_eye_height"].append(lh)
        self._accum["right_eye_width"].append(rw)
        self._accum["right_eye_height"].append(rh)

        # --- Face ---
        top    = LandmarkExtractor.get_point(landmarks, FACE_TOP)
        bottom = LandmarkExtractor.get_point(landmarks, FACE_BOTTOM)
        left   = LandmarkExtractor.get_point(landmarks, FACE_LEFT_EAR)
        right  = LandmarkExtractor.get_point(landmarks, FACE_RIGHT_EAR)

        self._accum["face_vertical (head to chin)"].append(euclidean(top, bottom))
        self._accum["face_horizontal (ear to ear)"].append(euclidean(left, right))

        # --- Nose ---
        n_left  = LandmarkExtractor.get_point(landmarks, NOSE_LEFT)
        n_right = LandmarkExtractor.get_point(landmarks, NOSE_RIGHT)
        n_tip   = LandmarkExtractor.get_point(landmarks, NOSE_TIP)
        n_bridge = LandmarkExtractor.get_point(landmarks, NOSE_BRIDGE)

        self._accum["nose_width"].append(euclidean(n_left, n_right))
        self._accum["nose_height"].append(euclidean(n_bridge, n_tip))

        # --- Mouth ---
        m_left   = LandmarkExtractor.get_point(landmarks, MOUTH_LEFT)
        m_right  = LandmarkExtractor.get_point(landmarks, MOUTH_RIGHT)
        m_top    = LandmarkExtractor.get_point(landmarks, MOUTH_TOP)
        m_bottom = LandmarkExtractor.get_point(landmarks, MOUTH_BOTTOM)

        self._accum["mouth_width"].append(euclidean(m_left, m_right))
        self._accum["mouth_height"].append(euclidean(m_top, m_bottom))

        self.frames_measured += 1

    # ──────────────────────────────────────────
    #  Finalize
    # ──────────────────────────────────────────
    def finalize(self) -> dict:
        """
        Average all per-frame measurements, convert to mm.
        Returns a dict {dimension_name: value_mm}.
        """
        if not self._scale_factors:
            print("  ⚠ No valid frames for measurement!")
            return {}

        avg_scale = float(np.median(self._scale_factors))  # median is more robust

        self.measurements_mm = {}
        for name, values in self._accum.items():
            if values:
                avg_px = float(np.median(values))
                self.measurements_mm[name] = round(avg_px * avg_scale, 2)
            else:
                self.measurements_mm[name] = 0.0

        summary = {
            "measurements_mm": self.measurements_mm,
            "calibration": {
                "reference_ipd_mm": self.ref_distance_mm,
                "median_scale_mm_per_px": round(avg_scale, 6),
                "frames_measured": self.frames_measured,
            },
        }
        return summary

    def get_measurement_log_rows(self) -> list[dict]:
        """Per-frame pixel measurements for CSV export."""
        rows = []
        n = self.frames_measured
        if n == 0:
            return rows
        for i in range(n):
            row = {"frame_measured_idx": i}
            for name, values in self._accum.items():
                row[name + "_px"] = round(values[i], 2) if i < len(values) else ""
            row["scale_mm_per_px"] = round(self._scale_factors[i], 6) if i < len(self._scale_factors) else ""
            rows.append(row)
        return rows

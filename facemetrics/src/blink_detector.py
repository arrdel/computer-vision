"""
Task A — Eye Blink Detector

Uses the Eye Aspect Ratio (EAR) algorithm:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

When the eye is open, EAR is roughly constant (~0.25-0.30).
When the eye closes, EAR drops sharply toward 0.
A blink = EAR drops below threshold for ≥ N consecutive frames.

Reference: Soukupová & Čech, "Real-Time Eye Blink Detection using
           Facial Landmarks", 2016.
"""

from __future__ import annotations
import numpy as np
from src.utils import euclidean, smooth_signal
from src.landmark_extractor import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    LandmarkExtractor,
)


def compute_ear(eye_points: list[tuple[float, float]]) -> float:
    """
    Compute Eye Aspect Ratio for 6 landmark points.

    Points layout:
        p1 ---- p4      (horizontal)
          p2  p6
          p3  p5         (vertical pairs: p2-p6, p3-p5)
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    # Vertical distances
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    # Horizontal distance
    h = euclidean(p1, p4)
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


class BlinkDetector:
    """
    Processes per-frame landmarks and detects blinks over the full video.

    Attributes after `finalize()`:
        blink_count       : total number of blinks
        blink_frames      : list of frame indices where each blink started
        blink_timestamps  : list of timestamps (seconds)
        ear_values        : per-frame EAR (smoothed)
        blinks_per_second : overall blink rate
        blink_rate_per_min: blink rate in blinks/min (more intuitive)
        per_minute_rates  : list of blink counts per 1-min window
    """

    def __init__(self, cfg: dict):
        blink_cfg = cfg.get("blink", {})
        self.threshold = blink_cfg.get("ear_threshold", 0.21)
        self.min_consec = blink_cfg.get("min_consec_frames", 2)
        self.smooth_win = blink_cfg.get("smoothing_window", 5)

        # Accumulators
        self._raw_ears: list[float] = []
        self._below_count = 0
        self._blink_started_frame = -1

        self.blink_count = 0
        self.blink_frames: list[int] = []     # frame index of each blink start
        self.frames_processed = 0

    # ──────────────────────────────────────────
    #  Per-frame update
    # ──────────────────────────────────────────
    def update(self, frame_idx: int, landmarks: list[tuple[float, float]] | None):
        """Call once per frame with the detected landmarks (or None if no face)."""
        if landmarks is None:
            self._raw_ears.append(np.nan)
            self._below_count = 0
            self.frames_processed += 1
            return

        # Compute EAR for both eyes and average
        left_pts  = LandmarkExtractor.get_eye_points(landmarks, LEFT_EYE_IDX)
        right_pts = LandmarkExtractor.get_eye_points(landmarks, RIGHT_EYE_IDX)
        ear_left  = compute_ear(left_pts)
        ear_right = compute_ear(right_pts)
        ear = (ear_left + ear_right) / 2.0

        self._raw_ears.append(ear)

        # State machine: detect blink
        if ear < self.threshold:
            if self._below_count == 0:
                self._blink_started_frame = frame_idx
            self._below_count += 1
        else:
            if self._below_count >= self.min_consec:
                # Valid blink ended
                self.blink_count += 1
                self.blink_frames.append(self._blink_started_frame)
            self._below_count = 0

        self.frames_processed += 1

    # ──────────────────────────────────────────
    #  Finalize and compute summary stats
    # ──────────────────────────────────────────
    def finalize(self, fps: float) -> dict:
        """
        Call after all frames are processed.
        Returns a summary dict.
        """
        # Handle edge: blink still open at end of video
        if self._below_count >= self.min_consec:
            self.blink_count += 1
            self.blink_frames.append(self._blink_started_frame)

        duration_s = self.frames_processed / fps if fps else 0
        duration_min = duration_s / 60.0

        # Smooth EAR for plotting
        raw = np.array(self._raw_ears, dtype=float)
        # Interpolate NaN frames (face not detected)
        nan_mask = np.isnan(raw)
        if nan_mask.any() and not nan_mask.all():
            raw[nan_mask] = np.interp(
                np.flatnonzero(nan_mask),
                np.flatnonzero(~nan_mask),
                raw[~nan_mask],
            )
        self.ear_values = smooth_signal(raw.tolist(), self.smooth_win)

        # Blink timestamps
        self.blink_timestamps = [f / fps for f in self.blink_frames]

        # Overall rates
        self.blinks_per_second = self.blink_count / duration_s if duration_s > 0 else 0
        self.blink_rate_per_min = self.blink_count / duration_min if duration_min > 0 else 0

        # Per-minute breakdown
        self.per_minute_rates = self._compute_per_minute_rates(fps, duration_s)

        # Face-detected ratio
        detected = int(np.sum(~np.isnan(self._raw_ears)))

        summary = {
            "total_blinks": self.blink_count,
            "duration_seconds": round(duration_s, 2),
            "duration_minutes": round(duration_min, 2),
            "blinks_per_second": round(self.blinks_per_second, 4),
            "blinks_per_minute": round(self.blink_rate_per_min, 2),
            "ear_threshold": self.threshold,
            "min_consec_frames": self.min_consec,
            "frames_processed": self.frames_processed,
            "face_detected_frames": detected,
            "face_detection_rate": round(detected / self.frames_processed * 100, 1)
                                   if self.frames_processed else 0,
            "per_minute_rates": self.per_minute_rates,
        }
        return summary

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────
    def _compute_per_minute_rates(self, fps: float, duration_s: float) -> list[dict]:
        """Compute blink count for each 1-minute window."""
        if not fps or duration_s <= 0:
            return []

        window_sec = 60.0
        rates = []
        window_start = 0.0

        while window_start < duration_s:
            window_end = min(window_start + window_sec, duration_s)
            count = sum(
                1 for t in self.blink_timestamps
                if window_start <= t < window_end
            )
            minute_idx = int(window_start // 60)
            rates.append({
                "minute": minute_idx,
                "start_s": round(window_start, 1),
                "end_s": round(window_end, 1),
                "blink_count": count,
            })
            window_start += window_sec

        return rates

    def get_ear_log_rows(self, fps: float) -> list[dict]:
        """Return per-frame EAR data as list of dicts (for CSV export)."""
        rows = []
        for i, ear_val in enumerate(self._raw_ears):
            rows.append({
                "frame": i,
                "time_s": round(i / fps, 4) if fps else 0,
                "ear_raw": round(ear_val, 6) if not np.isnan(ear_val) else "",
                "ear_smoothed": round(float(self.ear_values[i]), 6)
                               if i < len(self.ear_values) else "",
                "below_threshold": ear_val < self.threshold
                                   if not np.isnan(ear_val) else "",
            })
        return rows

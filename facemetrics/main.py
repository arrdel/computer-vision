#!/usr/bin/env python3
"""
conda activate facemetrics && python batch_run.py --video-dir data/videos --output output/batch_report

FaceMetrics — main entry point.

Runs both Task A (blink detection) and Task B (face dimension estimation)
on a video file, then outputs logs, plots, and a summary JSON.

Usage:
    python main.py                              # uses configs/default.yaml
    python main.py --config configs/custom.yaml
    python main.py --video data/videos/my.mp4   # override video path
"""

import argparse
import json
import os
import time
import yaml
from tqdm import tqdm

from src.video_processor import VideoProcessor
from src.landmark_extractor import LandmarkExtractor
from src.blink_detector import BlinkDetector
from src.face_measurer import FaceMeasurer
from src.utils import (
    save_csv,
    save_json,
    ensure_dirs,
    plot_ear_over_time,
    plot_measurements_summary,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run(cfg: dict):
    """Run the full pipeline: video → landmarks → blink + measurements → output."""

    video_path = cfg["video"]["path"]
    resize_w = cfg["video"].get("resize_width")
    output_cfg = cfg["output"]

    # Ensure output dirs exist
    plots_dir = output_cfg.get("plots_dir", "output/plots")
    ensure_dirs(
        output_cfg["blink_log"],
        output_cfg["measurement_log"],
        output_cfg["summary"],
        plots_dir,
    )

    # ─── Initialise components ───
    print("\n╔══════════════════════════════════════════╗")
    print("║         FaceMetrics Pipeline             ║")
    print("╚══════════════════════════════════════════╝\n")

    print("[1/5] Loading video...")
    video = VideoProcessor(video_path, resize_width=resize_w)
    fps = cfg["video"].get("fps") or video.fps

    print("[2/5] Initialising landmark extractor...")
    extractor = LandmarkExtractor(cfg)

    print("[3/5] Initialising blink detector (Task A) & face measurer (Task B)...")
    blink_det = BlinkDetector(cfg)
    face_meas = FaceMeasurer(cfg)

    # ─── Process frames ───
    print(f"\n[4/5] Processing {video.total_frames} frames...\n")
    t0 = time.time()

    for frame_idx, frame_rgb in tqdm(video, total=video.total_frames, unit="frame"):
        landmarks = extractor.process(frame_rgb)

        # Task A: blink detection
        blink_det.update(frame_idx, landmarks)

        # Task B: face measurements
        face_meas.update(landmarks)

    elapsed = time.time() - t0
    video.release()
    extractor.close()

    print(f"\n  Processing complete in {elapsed:.1f}s "
          f"({video.total_frames / elapsed:.0f} fps throughput)")

    # ─── Finalize & output ───
    print("\n[5/5] Generating results...\n")

    # Task A results
    blink_summary = blink_det.finalize(fps)
    print("  ── Task A: Blink Detection ──")
    print(f"  Total blinks      : {blink_summary['total_blinks']}")
    print(f"  Duration           : {blink_summary['duration_minutes']:.1f} min")
    print(f"  Blinks/second      : {blink_summary['blinks_per_second']:.4f}")
    print(f"  Blinks/minute      : {blink_summary['blinks_per_minute']:.2f}")
    print(f"  Face detection rate: {blink_summary['face_detection_rate']:.1f}%")

    # Task B results
    meas_summary = face_meas.finalize()
    print("\n  ── Task B: Face Dimensions ──")
    if meas_summary.get("measurements_mm"):
        for name, val in meas_summary["measurements_mm"].items():
            print(f"  {name:35s}: {val:7.2f} mm")
    else:
        print("  ⚠ No measurements available (face not detected)")

    # ─── Save outputs ───
    print("\n  ── Saving outputs ──")

    # Blink log CSV
    if cfg["blink"].get("log_ear", True):
        ear_rows = blink_det.get_ear_log_rows(fps)
        save_csv(ear_rows, output_cfg["blink_log"])

    # Measurement log CSV
    meas_rows = face_meas.get_measurement_log_rows()
    save_csv(meas_rows, output_cfg["measurement_log"])

    # Combined summary JSON
    full_summary = {
        "task_a_blink_detection": blink_summary,
        "task_b_face_dimensions": meas_summary,
        "video": {
            "path": video_path,
            "fps": fps,
            "total_frames": video.total_frames,
            "duration_seconds": round(video.duration_seconds, 2),
        },
        "processing_time_seconds": round(elapsed, 2),
    }
    save_json(full_summary, output_cfg["summary"])

    # Plots
    ear_plot_path = os.path.join(plots_dir, "ear_over_time.png")
    plot_ear_over_time(
        blink_det.ear_values.tolist(),
        blink_det.blink_frames,
        blink_det.threshold,
        fps,
        ear_plot_path,
    )

    if meas_summary.get("measurements_mm"):
        meas_plot_path = os.path.join(plots_dir, "face_dimensions.png")
        plot_measurements_summary(meas_summary["measurements_mm"], meas_plot_path)

    print("\n  ✅  All done!\n")
    return full_summary


def main():
    parser = argparse.ArgumentParser(description="FaceMetrics — Blink & Face Dimension Estimator")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--video", type=str, default=None,
                        help="Override video path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.video:
        cfg["video"]["path"] = args.video

    run(cfg)


if __name__ == "__main__":
    main()

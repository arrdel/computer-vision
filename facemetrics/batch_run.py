#!/usr/bin/env python3
"""
FaceMetrics — Batch Runner

Processes ALL videos in a directory through Task A (blink detection) and
Task B (face dimension estimation), then generates:

  • Per-video: EAR plots, dimension charts, CSV logs, JSON summaries
  • Comparative: overlay plots, heatmaps, radar charts, grouped bars
  • HTML dashboard: single self-contained report with everything

Usage:
    python batch_run.py
    python batch_run.py --video-dir data/videos --config configs/default.yaml
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import yaml
from tqdm import tqdm

from src.video_processor import VideoProcessor
from src.landmark_extractor import LandmarkExtractor
from src.blink_detector import BlinkDetector
from src.face_measurer import FaceMeasurer
from src.utils import save_csv, save_json, ensure_dirs
from src.report_generator import (
    plot_ear_detailed,
    plot_dimensions_detailed,
    plot_blink_rate_comparison,
    plot_ear_overlay,
    plot_ear_heatmap,
    plot_dimensions_comparison,
    plot_dimensions_radar,
    plot_detection_quality,
    plot_summary_dashboard,
    generate_html_report,
)


BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗ █████╗  ██████╗███████╗                            ║
║   ██╔════╝██╔══██╗██╔════╝██╔════╝                            ║
║   █████╗  ███████║██║     █████╗                              ║
║   ██╔══╝  ██╔══██║██║     ██╔══╝                              ║
║   ██║     ██║  ██║╚██████╗███████╗                            ║
║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝                            ║
║   ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗███████╗    ║
║   ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝██╔════╝    ║
║   ██╔████╔██║█████╗     ██║   ██████╔╝██║██║     ███████╗    ║
║   ██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║     ╚════██║    ║
║   ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗███████║    ║
║   ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝    ║
║                                                               ║
║          B A T C H   A N A L Y S I S   E N G I N E            ║
╚═══════════════════════════════════════════════════════════════╝
"""


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def discover_videos(video_dir: str) -> list[str]:
    """Find all .mp4 files in the directory (non-recursive), sorted."""
    patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(video_dir, p)))
    files = sorted(set(files))
    return files


def process_single_video(video_path: str, cfg: dict, output_base: str) -> dict:
    """
    Process one video through the full pipeline.
    Returns a result dict with all data needed for reporting.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_base, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    # Override video path in config
    local_cfg = copy.deepcopy(cfg)
    local_cfg["video"]["path"] = video_path

    resize_w = local_cfg["video"].get("resize_width")

    # ─── Initialize ───
    video = VideoProcessor(video_path, resize_width=resize_w)
    fps = local_cfg["video"].get("fps") or video.fps
    extractor = LandmarkExtractor(local_cfg)
    blink_det = BlinkDetector(local_cfg)
    face_meas = FaceMeasurer(local_cfg)

    # ─── Process frames ───
    t0 = time.time()
    for frame_idx, frame_rgb in tqdm(video, total=video.total_frames,
                                      unit="frame", desc=f"  {video_name}",
                                      leave=True, ncols=90):
        landmarks = extractor.process(frame_rgb)
        blink_det.update(frame_idx, landmarks)
        face_meas.update(landmarks)

    elapsed = time.time() - t0
    throughput = video.total_frames / elapsed if elapsed > 0 else 0

    video.release()
    extractor.close()

    # ─── Finalize ───
    blink_summary = blink_det.finalize(fps)
    meas_summary = face_meas.finalize()

    # ─── Save per-video outputs ───
    # CSV: blink log
    if local_cfg["blink"].get("log_ear", True):
        ear_rows = blink_det.get_ear_log_rows(fps)
        save_csv(ear_rows, os.path.join(video_out_dir, "blink_log.csv"))

    # CSV: measurement log
    meas_rows = face_meas.get_measurement_log_rows()
    save_csv(meas_rows, os.path.join(video_out_dir, "measurements.csv"))

    # JSON summary
    full_summary = {
        "video_name": video_name,
        "video_path": video_path,
        "task_a_blink_detection": blink_summary,
        "task_b_face_dimensions": meas_summary,
        "video_info": {
            "path": video_path,
            "fps": fps,
            "total_frames": video.total_frames,
            "duration_seconds": round(video.duration_seconds, 2),
            "resolution": f"{video.width}×{video.height}",
        },
        "processing_time_seconds": round(elapsed, 2),
        "throughput_fps": round(throughput, 1),
    }
    save_json(full_summary, os.path.join(video_out_dir, "summary.json"))

    # Build result dict for cross-video analysis
    result = {
        "video_name": video_name,
        "video_path": video_path,
        "blink_summary": blink_summary,
        "meas_summary": meas_summary,
        "ear_values": blink_det.ear_values.tolist(),
        "blink_frames": blink_det.blink_frames,
        "threshold": blink_det.threshold,
        "fps": fps,
        "video_info": {
            "fps": fps,
            "total_frames": video.total_frames,
            "duration": round(video.duration_seconds, 2),
            "resolution": f"{video.width}×{video.height}",
        },
        "processing_time": round(elapsed, 2),
        "throughput_fps": round(throughput, 1),
        "output_dir": video_out_dir,
    }
    return result


def generate_all_reports(all_results: list[dict], output_base: str):
    """Generate ALL plots and the HTML report."""

    plots_dir = os.path.join(output_base, "_comparative")
    os.makedirs(plots_dir, exist_ok=True)

    plot_images = {}  # name → base64 string

    # ──────────────────────────────────────────
    #  PER-VIDEO DETAILED PLOTS
    # ──────────────────────────────────────────
    print("\n  ── Generating per-video plots ──")
    for r in all_results:
        vname = r["video_name"]
        vdir = r["output_dir"]

        # EAR detailed
        ear_path = os.path.join(vdir, "ear_detailed.png")
        b64 = plot_ear_detailed(
            r["ear_values"], r["blink_frames"], r["threshold"],
            r["fps"], vname, ear_path,
        )
        plot_images[f"ear_detail_{vname}"] = b64
        print(f"    ✓ {vname}: EAR plot")

        # Dimensions detailed
        meas_mm = r.get("meas_summary", {}).get("measurements_mm", {})
        if meas_mm:
            dim_path = os.path.join(vdir, "dimensions_detailed.png")
            b64 = plot_dimensions_detailed(meas_mm, vname, dim_path)
            plot_images[f"dim_detail_{vname}"] = b64
            print(f"    ✓ {vname}: Dimensions plot")

    # ──────────────────────────────────────────
    #  CROSS-VIDEO COMPARATIVE PLOTS
    # ──────────────────────────────────────────
    print("\n  ── Generating comparative plots ──")

    b64 = plot_blink_rate_comparison(
        all_results, os.path.join(plots_dir, "blink_comparison.png"))
    plot_images["blink_comparison"] = b64
    print("    ✓ Blink rate comparison")

    b64 = plot_ear_overlay(
        all_results, os.path.join(plots_dir, "ear_overlay.png"))
    plot_images["ear_overlay"] = b64
    print("    ✓ EAR overlay")

    b64 = plot_ear_heatmap(
        all_results, os.path.join(plots_dir, "ear_heatmap.png"))
    plot_images["ear_heatmap"] = b64
    print("    ✓ EAR heatmap")

    b64 = plot_dimensions_comparison(
        all_results, os.path.join(plots_dir, "dim_comparison.png"))
    if b64:
        plot_images["dim_comparison"] = b64
        print("    ✓ Dimensions comparison")

    b64 = plot_dimensions_radar(
        all_results, os.path.join(plots_dir, "dim_radar.png"))
    if b64:
        plot_images["dim_radar"] = b64
        print("    ✓ Dimensions radar")

    b64 = plot_detection_quality(
        all_results, os.path.join(plots_dir, "quality.png"))
    plot_images["quality"] = b64
    print("    ✓ Detection quality")

    b64 = plot_summary_dashboard(
        all_results, os.path.join(plots_dir, "dashboard.png"))
    plot_images["dashboard"] = b64
    print("    ✓ Summary dashboard")

    # ──────────────────────────────────────────
    #  HTML REPORT
    # ──────────────────────────────────────────
    print("\n  ── Generating HTML report ──")
    html_path = os.path.join(output_base, "facemetrics_report.html")
    generate_html_report(all_results, plot_images, html_path)

    # ──────────────────────────────────────────
    #  COMBINED JSON
    # ──────────────────────────────────────────
    combined_json = {
        "batch_summary": {
            "videos_processed": len(all_results),
            "total_frames": sum(r["blink_summary"]["frames_processed"] for r in all_results),
            "total_duration_seconds": round(
                sum(r["blink_summary"]["duration_seconds"] for r in all_results), 2),
            "total_blinks": sum(r["blink_summary"]["total_blinks"] for r in all_results),
            "avg_blinks_per_minute": round(
                np.mean([r["blink_summary"]["blinks_per_minute"] for r in all_results]), 2),
            "avg_face_detection_rate": round(
                np.mean([r["blink_summary"].get("face_detection_rate", 0) for r in all_results]), 1),
            "total_processing_seconds": round(
                sum(r.get("processing_time", 0) for r in all_results), 2),
        },
        "per_video": [
            {
                "video_name": r["video_name"],
                "blink_summary": r["blink_summary"],
                "meas_summary": r.get("meas_summary", {}),
                "video_info": r["video_info"],
                "processing_time": r.get("processing_time", 0),
                "throughput_fps": r.get("throughput_fps", 0),
            }
            for r in all_results
        ],
    }
    combined_path = os.path.join(output_base, "batch_summary.json")
    save_json(combined_json, combined_path)

    return html_path


def main():
    parser = argparse.ArgumentParser(
        description="FaceMetrics — Batch Analysis across multiple videos")
    parser.add_argument("--video-dir", type=str, default="data/videos",
                        help="Directory containing video files")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--output", type=str, default="output/batch_report",
                        help="Output directory for batch results")
    args = parser.parse_args()

    print(BANNER)

    # ─── Load config ───
    cfg = load_config(args.config)

    # ─── Discover videos ───
    videos = discover_videos(args.video_dir)
    if not videos:
        print(f"  ❌  No video files found in {args.video_dir}")
        sys.exit(1)

    print(f"  Found {len(videos)} video(s) in {args.video_dir}:\n")
    for i, v in enumerate(videos, 1):
        print(f"    {i}. {os.path.basename(v)}")
    print()

    # ─── Process each video ───
    output_base = args.output
    os.makedirs(output_base, exist_ok=True)
    all_results = []
    total_t0 = time.time()

    for i, vpath in enumerate(videos, 1):
        vname = os.path.basename(vpath)
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(videos)}]  Processing: {vname}")
        print(f"{'─' * 60}")

        try:
            result = process_single_video(vpath, cfg, output_base)
            all_results.append(result)

            bs = result["blink_summary"]
            print(f"\n    ✅  {vname}:")
            print(f"        Blinks: {bs['total_blinks']}  "
                  f"({bs['blinks_per_minute']:.1f}/min)  "
                  f"| Detection: {bs.get('face_detection_rate', 0):.0f}%  "
                  f"| Time: {result['processing_time']:.1f}s")
        except Exception as e:
            print(f"\n    ❌  Error processing {vname}: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_t0

    if not all_results:
        print("\n  ❌  No videos were successfully processed.")
        sys.exit(1)

    # ─── Print batch summary ───
    print(f"\n{'═' * 60}")
    print(f"  BATCH PROCESSING COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Videos processed  : {len(all_results)} / {len(videos)}")
    print(f"  Total time        : {total_elapsed:.1f}s")
    total_frames = sum(r["blink_summary"]["frames_processed"] for r in all_results)
    print(f"  Total frames      : {total_frames:,}")
    print(f"  Avg throughput    : {total_frames / total_elapsed:.0f} fps")
    print(f"  Total blinks      : {sum(r['blink_summary']['total_blinks'] for r in all_results)}")
    print(f"  Avg blinks/min    : {np.mean([r['blink_summary']['blinks_per_minute'] for r in all_results]):.1f}")

    # ─── Generate reports ───
    print(f"\n{'─' * 60}")
    print(f"  GENERATING REPORTS & VISUALIZATIONS")
    print(f"{'─' * 60}")

    html_path = generate_all_reports(all_results, output_base)

    # ─── Final output listing ───
    print(f"\n{'═' * 60}")
    print(f"  📁  OUTPUT FILES")
    print(f"{'═' * 60}")
    for root, dirs, files in os.walk(output_base):
        level = root.replace(output_base, "").count(os.sep)
        indent = "  " + "│  " * level
        subdir = os.path.basename(root)
        if level == 0:
            print(f"  {output_base}/")
        else:
            print(f"{indent}📂 {subdir}/")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.0f}KB"
            else:
                size_str = f"{size}B"
            print(f"{indent}│  📄 {f}  ({size_str})")

    abs_html = os.path.abspath(html_path)
    print(f"\n  🌐  Open the HTML report:")
    print(f"      file://{abs_html}\n")
    print(f"  ✅  All done!\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Module 7 -- run_all.py
=====================
Master script to run the entire pipeline end-to-end.

Usage:
    python run_all.py --video1 videos/video1.mp4 --video2 videos/video2.mp4
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run complete Module 7 pipeline")
    parser.add_argument("--video1", required=True, help="Path to first video")
    parser.add_argument("--video2", required=True, help="Path to second video")
    parser.add_argument("--sfm_images", default=None,
                        help="Directory with images for SfM (optional; "
                             "if not provided, frames are extracted from video1)")
    parser.add_argument("--sfm_video", default=None,
                        help="Video to extract SfM frames from (defaults to video1)")
    parser.add_argument("--K", nargs=4, type=float, default=None,
                        metavar=("fx", "fy", "cx", "cy"),
                        help="Camera intrinsics for SfM (optional)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("  Module 7 -- Complete Pipeline")
    print("=" * 70)

    # ── Part 1: Optical Flow ──
    print("\n" + "-" * 70)
    print("  PART 1: Optical Flow & Motion Tracking")
    print("-" * 70)
    cmd = (f'{sys.executable} "{os.path.join(base_dir, "optical_flow.py")}" '
           f'--video1 "{args.video1}" --video2 "{args.video2}"')
    print(f"  Running: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print("[ERROR] Optical flow step failed")
        sys.exit(1)

    # ── Part 2: Structure from Motion ──
    print("\n" + "-" * 70)
    print("  PART 2: Structure from Motion")
    print("-" * 70)

    sfm_cmd_parts = [f'{sys.executable} "{os.path.join(base_dir, "structure_from_motion.py")}"']

    # Determine SfM image source
    sfm_images_dir = args.sfm_images or os.path.join(base_dir, "sfm_images")
    has_images = False
    if os.path.isdir(sfm_images_dir):
        import glob
        imgs = glob.glob(os.path.join(sfm_images_dir, "*.jpg")) + \
               glob.glob(os.path.join(sfm_images_dir, "*.png")) + \
               glob.glob(os.path.join(sfm_images_dir, "*.jpeg"))
        has_images = len(imgs) >= 2

    if has_images:
        sfm_cmd_parts.append(f'--images "{sfm_images_dir}"')
    else:
        # Use video1 to extract frames
        sfm_video = args.sfm_video or args.video1
        sfm_cmd_parts.append(f'--images "{sfm_images_dir}"')
        sfm_cmd_parts.append(f'--video "{sfm_video}"')

    if args.K:
        sfm_cmd_parts.append(f"--K {' '.join(map(str, args.K))}")

    cmd = " ".join(sfm_cmd_parts)
    print(f"  Running: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print("[ERROR] SfM step failed")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  All steps complete!")
    print("    Optical flow results: output/optical_flow/")
    print("    SfM results:          output/sfm/")
    print("=" * 70)


if __name__ == "__main__":
    main()

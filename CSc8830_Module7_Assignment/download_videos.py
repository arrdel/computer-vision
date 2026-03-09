#!/usr/bin/env python3
"""
Module 7 — Video downloader helper
===================================
Downloads YouTube videos using yt-dlp and trims them to 30-second clips.

Usage:
    python download_videos.py --url1 <URL1> --url2 <URL2>
    python download_videos.py  # uses default sample videos
"""

import argparse
import os
import subprocess
import sys


def check_yt_dlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_and_trim(url, output_path, start="0:00", duration="30", label="video"):
    """Download a YouTube video and trim to a clip."""
    raw_path = output_path.replace(".mp4", "_raw.mp4")

    print(f"\n[{label}] Downloading: {url}")
    cmd_dl = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", raw_path,
        "--no-playlist",
        url,
    ]
    subprocess.run(cmd_dl, check=True)

    if not os.path.exists(raw_path):
        print(f"[ERROR] Download failed for {label}")
        return False

    print(f"[{label}] Trimming {duration}s starting at {start} …")
    cmd_trim = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", raw_path,
        "-t", duration,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        output_path,
    ]
    subprocess.run(cmd_trim, check=True)

    # Remove raw file
    if os.path.exists(raw_path) and os.path.exists(output_path):
        os.remove(raw_path)

    print(f"[{label}] Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and trim YouTube videos for optical flow analysis")
    parser.add_argument("--url1", type=str, default=None,
                        help="YouTube URL for video 1 (motion scene)")
    parser.add_argument("--url2", type=str, default=None,
                        help="YouTube URL for video 2 (motion scene)")
    parser.add_argument("--start1", type=str, default="0:10",
                        help="Start time for video 1 trim (default: 0:10)")
    parser.add_argument("--start2", type=str, default="0:10",
                        help="Start time for video 2 trim (default: 0:10)")
    parser.add_argument("--duration", type=str, default="30",
                        help="Duration of each clip in seconds (default: 30)")
    parser.add_argument("--outdir", type=str, default="videos",
                        help="Output directory (default: videos)")
    args = parser.parse_args()

    if not check_yt_dlp():
        print("[ERROR] yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)
    if not check_ffmpeg():
        print("[ERROR] ffmpeg not found. Install with: sudo apt install ffmpeg")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if args.url1 is None or args.url2 is None:
        print("=" * 60)
        print("  No URLs provided. Please supply --url1 and --url2.")
        print()
        print("  RECOMMENDED VIDEOS (see README for suggestions):")
        print("  Video 1: A video with lateral/translational motion")
        print("           e.g., traffic dashcam, sports sideline")
        print("  Video 2: A video with rotational/complex motion")
        print("           e.g., drone footage, dancing, gymnastics")
        print()
        print("  Example:")
        print("    python download_videos.py \\")
        print('      --url1 "https://www.youtube.com/watch?v=VIDEO_ID_1" \\')
        print('      --url2 "https://www.youtube.com/watch?v=VIDEO_ID_2"')
        print("=" * 60)
        sys.exit(0)

    success1 = download_and_trim(
        args.url1, os.path.join(args.outdir, "video1.mp4"),
        start=args.start1, duration=args.duration, label="Video 1")

    success2 = download_and_trim(
        args.url2, os.path.join(args.outdir, "video2.mp4"),
        start=args.start2, duration=args.duration, label="Video 2")

    if success1 and success2:
        print("\n✓ Both videos downloaded and trimmed successfully!")
        print(f"  → {args.outdir}/video1.mp4")
        print(f"  → {args.outdir}/video2.mp4")
        print("\nNext step:")
        print(f"  python optical_flow.py --video1 {args.outdir}/video1.mp4 --video2 {args.outdir}/video2.mp4")


if __name__ == "__main__":
    main()

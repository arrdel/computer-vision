#!/usr/bin/env python3
"""
CSc 8830 — Module 8 Assignment
Stereo-Camera Classroom Object Localisation

Given a pair of stereo images (left / right) of a classroom, this script:
  1. Detects tables and chairs using a YOLOv8 object detector.
  2. Computes a disparity map (SGBM) from the rectified stereo pair.
  3. Converts each detection's centre disparity → depth → (X, Y) floor
     coordinates using the stereo geometry.
  4. Produces a 2D X-Y plot with tables in RED and chairs in BLUE.

Usage
-----
  # With real stereo images:
  python stereo_localization.py --left images/left.jpg --right images/right.jpg

  # With synthetic / demo data (no images required):
  python stereo_localization.py --demo

  # Override camera parameters:
  python stereo_localization.py --left images/left.jpg --right images/right.jpg \
        --focal 700 --baseline 0.12 --cx 640 --cy 360

Output
------
  output/detections_left.png       — annotated left image with bounding boxes
  output/disparity_map.png         — stereo disparity map
  output/classroom_layout_2d.png   — 2D X-Y floor-plan plot (tables=red, chairs=blue)
  output/detections.csv            — per-object numeric results
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# COCO class IDs that map to "table" and "chair"
# (YOLOv8 uses COCO 80-class labels)
CHAIR_LABELS  = {"chair"}
TABLE_LABELS  = {"dining table", "table"}   # COCO calls them "dining table"

# Colours (BGR for OpenCV, RGB for matplotlib)
RED_BGR  = (0, 0, 255)
BLUE_BGR = (255, 0, 0)
RED_RGB  = "#e74c3c"
BLUE_RGB = "#2980b9"


# ---------------------------------------------------------------------------
# Stereo camera parameters (defaults — override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_FOCAL_PX   = 700.0    # focal length in pixels
DEFAULT_BASELINE_M = 0.12     # stereo baseline in metres
DEFAULT_CX         = None     # principal point x (defaults to W/2)
DEFAULT_CY         = None     # principal point y (defaults to H/2)


# ---------------------------------------------------------------------------
# Helper: compute disparity with SGBM
# ---------------------------------------------------------------------------
def compute_disparity(left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
    """Semi-Global Block Matching disparity map (float, in pixels)."""
    min_disp = 0
    num_disp = 128  # must be divisible by 16
    block    = 5

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8  * 3 * block**2,
        P2=32 * 3 * block**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disp


# ---------------------------------------------------------------------------
# Helper: disparity → 3-D world coordinates
# ---------------------------------------------------------------------------
def pixel_to_world(u: float, v: float, disp: float,
                   focal: float, baseline: float,
                   cx: float, cy: float):
    """
    Convert pixel (u, v) + disparity → (X_world, Y_world, Z_world) in metres.

    Coordinate frame (camera-centred, looking along +Z):
      X = right,  Y = down,  Z = forward

    We then map to a floor-plan:
      floor_X = X_world   (lateral)
      floor_Y = Z_world   (depth / forward)
    """
    if disp <= 0.5:
        return None  # invalid disparity
    Z = (focal * baseline) / disp        # depth in metres
    X = (u - cx) * Z / focal             # lateral position
    Y = (v - cy) * Z / focal             # vertical (not used for floor plan)
    return X, Y, Z


# ---------------------------------------------------------------------------
# YOLO detection wrapper (ultralytics)
# ---------------------------------------------------------------------------
def detect_objects_yolo(image_bgr: np.ndarray):
    """
    Run YOLOv8 on *image_bgr* and return a list of dicts:
      [{"label": str, "bbox": (x1,y1,x2,y2), "conf": float}, ...]
    Only chairs and tables are kept.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not installed — falling back to demo mode.")
        print("    Install with: pip install ultralytics")
        return None

    model = YOLO("yolov8n.pt")  # auto-downloads ~6 MB nano model
    results = model(image_bgr, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id].lower()
        if label in CHAIR_LABELS or label in TABLE_LABELS:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                "label": "table" if label in TABLE_LABELS else "chair",
                "bbox":  (float(x1), float(y1), float(x2), float(y2)),
                "conf":  float(box.conf[0]),
            })
    return detections


# ---------------------------------------------------------------------------
# Fallback: simple colour / contour heuristic (no YOLO)
# ---------------------------------------------------------------------------
def detect_objects_heuristic(image_bgr: np.ndarray):
    """
    Very simple fallback when YOLO is unavailable.
    Uses edge detection + contour area to pick large rectangular blobs.
    Larger blobs → tables; smaller blobs → chairs.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image_bgr.shape[:2]
    min_area = (h * w) * 0.005   # at least 0.5 % of image
    table_thresh = (h * w) * 0.04  # > 4 % → table

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        label = "table" if area > table_thresh else "chair"
        detections.append({
            "label": label,
            "bbox":  (float(x), float(y), float(x + bw), float(y + bh)),
            "conf":  0.5,
        })
    return detections


# ---------------------------------------------------------------------------
# Generate synthetic demo data (no real images needed)
# ---------------------------------------------------------------------------
def generate_demo_data():
    """
    Create a synthetic classroom layout with known ground-truth positions
    and return the 2-D floor coordinates directly.
    Also generates a synthetic stereo pair for illustration.
    """
    np.random.seed(42)

    # ---- Ground-truth floor positions (metres) ----
    # Tables arranged in rows
    tables = []
    for row in range(3):
        for col in range(4):
            x = -1.5 + col * 1.0 + np.random.uniform(-0.05, 0.05)
            y =  2.0 + row * 2.0 + np.random.uniform(-0.05, 0.05)
            tables.append((x, y))

    # Chairs: one in front of each table, slightly offset
    chairs = []
    for tx, ty in tables:
        cx = tx + np.random.uniform(-0.15, 0.15)
        cy = ty - 0.6 + np.random.uniform(-0.05, 0.05)
        chairs.append((cx, cy))
    # Extra chairs along the sides
    for i in range(4):
        chairs.append((-2.5, 2.0 + i * 1.5 + np.random.uniform(-0.1, 0.1)))
        chairs.append(( 2.5, 2.0 + i * 1.5 + np.random.uniform(-0.1, 0.1)))

    # ---- Render a very simple synthetic left image ----
    W, H = 1280, 720
    focal    = DEFAULT_FOCAL_PX
    baseline = DEFAULT_BASELINE_M
    cx, cy   = W / 2, H / 2

    left  = np.full((H, W, 3), 220, dtype=np.uint8)  # light grey floor
    right = left.copy()

    def world_to_pixel(X, Y_depth, f, b, cx_, cy_, is_right=False):
        """Project floor point (X, Y_depth) to pixel. Y_depth = Z (forward)."""
        Z = Y_depth
        u = f * X / Z + cx_
        v = f * 0.3 / Z + cy_   # assume camera ~0.3 m above objects for vertical
        if is_right:
            u -= f * b / Z      # disparity shift
        return int(u), int(v)

    for tx, ty in tables:
        lu, lv = world_to_pixel(tx, ty, focal, baseline, cx, cy)
        ru, rv = world_to_pixel(tx, ty, focal, baseline, cx, cy, is_right=True)
        sz = max(10, int(focal * 0.4 / ty))
        cv2.rectangle(left,  (lu - sz, lv - sz//2), (lu + sz, lv + sz//2), (0, 0, 180), -1)
        cv2.rectangle(right, (ru - sz, rv - sz//2), (ru + sz, rv + sz//2), (0, 0, 180), -1)

    for cx_, cy_ in chairs:
        lu, lv = world_to_pixel(cx_, cy_, focal, baseline, cx, cy)
        ru, rv = world_to_pixel(cx_, cy_, focal, baseline, cx, cy, is_right=True)
        sz = max(6, int(focal * 0.2 / cy_))
        cv2.circle(left,  (lu, lv), sz, (180, 0, 0), -1)
        cv2.circle(right, (ru, rv), sz, (180, 0, 0), -1)

    return left, right, tables, chairs, focal, baseline, cx, cy


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_stereo_localization(left_img, right_img, focal, baseline, cx, cy, outdir):
    """Full pipeline: detect → disparity → locate → plot."""

    os.makedirs(outdir, exist_ok=True)
    H, W = left_img.shape[:2]
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    # ---- 1. Detect objects ----
    print("[1/4] Detecting tables and chairs …")
    detections = detect_objects_yolo(left_img)
    if detections is None:
        print("      → Using heuristic fallback detector.")
        detections = detect_objects_heuristic(left_img)

    if not detections:
        print("[!] No tables or chairs detected.  Check your input images.")
        sys.exit(1)

    n_tables = sum(1 for d in detections if d["label"] == "table")
    n_chairs = sum(1 for d in detections if d["label"] == "chair")
    print(f"      Found {n_tables} table(s), {n_chairs} chair(s).")

    # ---- 2. Disparity map ----
    print("[2/4] Computing stereo disparity (SGBM) …")
    gray_l = cv2.cvtColor(left_img,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disp = compute_disparity(gray_l, gray_r)

    # Save disparity visualisation
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(outdir, "disparity_map.png"), disp_color)

    # ---- 3. Localise each detection ----
    print("[3/4] Computing 3-D positions from disparity …")
    results = []  # (label, floor_x, floor_y, depth, conf)
    annotated = left_img.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        uc = (x1 + x2) / 2.0
        vc = (y1 + y2) / 2.0

        # Median disparity in the central region of the bounding box
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        roi = disp[int(y1)+margin_y : int(y2)-margin_y,
                    int(x1)+margin_x : int(x2)-margin_x]
        valid = roi[roi > 0.5]
        if valid.size == 0:
            continue
        med_disp = float(np.median(valid))

        pos = pixel_to_world(uc, vc, med_disp, focal, baseline, cx, cy)
        if pos is None:
            continue
        X, _Y_vert, Z = pos

        floor_x = X      # lateral (metres)
        floor_y = Z      # depth   (metres)
        results.append((det["label"], floor_x, floor_y, Z, det["conf"]))

        # Draw on annotated image
        color = RED_BGR if det["label"] == "table" else BLUE_BGR
        cv2.rectangle(annotated,
                      (int(x1), int(y1)), (int(x2), int(y2)),
                      color, 2)
        txt = f'{det["label"]} {Z:.1f}m'
        cv2.putText(annotated, txt, (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(os.path.join(outdir, "detections_left.png"), annotated)

    if not results:
        print("[!] Could not compute positions for any detection.")
        sys.exit(1)

    # Save CSV
    csv_path = os.path.join(outdir, "detections.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "floor_x_m", "floor_y_m", "depth_m", "confidence"])
        for row in results:
            writer.writerow(row)
    print(f"      Saved {len(results)} detections to {csv_path}")

    # ---- 4. Plot 2-D floor plan ----
    print("[4/4] Plotting 2-D classroom layout …")
    plot_floor_plan(results, outdir)

    print(f"\n✓ All outputs saved to {outdir}/")
    return results


# ---------------------------------------------------------------------------
# 2-D floor-plan plot
# ---------------------------------------------------------------------------
def plot_floor_plan(results, outdir, tables_gt=None, chairs_gt=None):
    """
    X-Y scatter plot: tables = RED, chairs = BLUE.
    X = lateral position, Y = depth (distance from camera).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    table_x = [r[1] for r in results if r[0] == "table"]
    table_y = [r[2] for r in results if r[0] == "table"]
    chair_x = [r[1] for r in results if r[0] == "chair"]
    chair_y = [r[2] for r in results if r[0] == "chair"]

    ax.scatter(table_x, table_y, c=RED_RGB,  s=200, marker="s",
               edgecolors="black", linewidths=0.8, zorder=5,
               label=f"Table ({len(table_x)})")
    ax.scatter(chair_x, chair_y, c=BLUE_RGB, s=120, marker="o",
               edgecolors="black", linewidths=0.8, zorder=5,
               label=f"Chair ({len(chair_x)})")

    # If ground-truth is available (demo mode), draw faint markers
    if tables_gt:
        gtx = [t[0] for t in tables_gt]
        gty = [t[1] for t in tables_gt]
        ax.scatter(gtx, gty, c="none", edgecolors=RED_RGB, s=250,
                   marker="s", linewidths=1.5, linestyle="--", alpha=0.35,
                   label="Table (GT)")
    if chairs_gt:
        gcx = [c[0] for c in chairs_gt]
        gcy = [c[1] for c in chairs_gt]
        ax.scatter(gcx, gcy, c="none", edgecolors=BLUE_RGB, s=160,
                   marker="o", linewidths=1.5, linestyle="--", alpha=0.35,
                   label="Chair (GT)")

    # Camera position indicator
    ax.plot(0, 0, "k^", markersize=14, zorder=10)
    ax.annotate("Camera", (0, 0), textcoords="offset points",
                xytext=(10, -15), fontsize=10, fontweight="bold")

    ax.set_xlabel("X — Lateral Position (metres)", fontsize=13)
    ax.set_ylabel("Y — Depth / Forward (metres)",  fontsize=13)
    ax.set_title("Classroom Layout — Stereo Localisation\n"
                 "(Tables = Red ■,  Chairs = Blue ●)", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # mirror so left in image = left on plot

    fig.tight_layout()
    out_path = os.path.join(outdir, "classroom_layout_2d.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"      → {out_path}")


# ---------------------------------------------------------------------------
# Demo mode: uses synthetic data, no real images needed
# ---------------------------------------------------------------------------
def run_demo(outdir):
    """Run the full pipeline on synthetic classroom data."""
    print("=" * 60)
    print("  DEMO MODE — Synthetic Classroom")
    print("=" * 60)
    os.makedirs(outdir, exist_ok=True)

    left, right, tables_gt, chairs_gt, focal, baseline, cx, cy = generate_demo_data()

    cv2.imwrite(os.path.join(outdir, "demo_left.png"),  left)
    cv2.imwrite(os.path.join(outdir, "demo_right.png"), right)

    # Build "results" directly from GT (simulating perfect detection)
    results = []
    for tx, ty in tables_gt:
        results.append(("table", tx, ty, ty, 1.0))
    for cx_, cy_ in chairs_gt:
        results.append(("chair", cx_, cy_, cy_, 1.0))

    # Save CSV
    csv_path = os.path.join(outdir, "detections.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "floor_x_m", "floor_y_m", "depth_m", "confidence"])
        for row in results:
            writer.writerow(row)

    # Plot
    plot_floor_plan(results, outdir, tables_gt, chairs_gt)

    # Also run disparity on the synthetic pair for illustration
    gray_l = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disp = compute_disparity(gray_l, gray_r)
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(outdir, "disparity_map.png"), disp_color)

    # Save annotated image
    annotated = left.copy()
    for tx, ty in tables_gt:
        Z = ty
        u = int(focal * tx / Z + left.shape[1]/2)
        v = int(focal * 0.3 / Z + left.shape[0]/2)
        sz = max(10, int(focal * 0.4 / Z))
        cv2.rectangle(annotated, (u-sz, v-sz//2), (u+sz, v+sz//2), RED_BGR, 2)
        cv2.putText(annotated, f"table {Z:.1f}m", (u-sz, v-sz//2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED_BGR, 1)
    for cx_, cy_ in chairs_gt:
        Z = cy_
        u = int(focal * cx_ / Z + left.shape[1]/2)
        v = int(focal * 0.3 / Z + left.shape[0]/2)
        sz = max(6, int(focal * 0.2 / Z))
        cv2.circle(annotated, (u, v), sz, BLUE_BGR, 2)
        cv2.putText(annotated, f"chair {Z:.1f}m", (u+sz+2, v+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, BLUE_BGR, 1)
    cv2.imwrite(os.path.join(outdir, "detections_left.png"), annotated)

    print(f"\n✓ Demo outputs saved to {outdir}/")
    print(f"  Tables: {len(tables_gt)},  Chairs: {len(chairs_gt)}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stereo camera classroom localisation — Module 8")
    parser.add_argument("--left",     type=str, help="Path to LEFT stereo image")
    parser.add_argument("--right",    type=str, help="Path to RIGHT stereo image")
    parser.add_argument("--demo",     action="store_true",
                        help="Run with synthetic classroom data (no images needed)")
    parser.add_argument("--focal",    type=float, default=DEFAULT_FOCAL_PX,
                        help=f"Focal length in pixels (default {DEFAULT_FOCAL_PX})")
    parser.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_M,
                        help=f"Stereo baseline in metres (default {DEFAULT_BASELINE_M})")
    parser.add_argument("--cx",       type=float, default=None,
                        help="Principal point X (default: image_width/2)")
    parser.add_argument("--cy",       type=float, default=None,
                        help="Principal point Y (default: image_height/2)")
    parser.add_argument("--outdir",   type=str, default="output",
                        help="Output directory (default: output)")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.outdir)
        return

    if not args.left or not args.right:
        print("Error: provide --left and --right images, or use --demo.")
        parser.print_help()
        sys.exit(1)

    left_img  = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    if left_img is None:
        print(f"Error: cannot read left image: {args.left}")
        sys.exit(1)
    if right_img is None:
        print(f"Error: cannot read right image: {args.right}")
        sys.exit(1)

    print(f"Left:  {args.left}  ({left_img.shape[1]}x{left_img.shape[0]})")
    print(f"Right: {args.right} ({right_img.shape[1]}x{right_img.shape[0]})")

    run_stereo_localization(
        left_img, right_img,
        focal=args.focal,
        baseline=args.baseline,
        cx=args.cx,
        cy=args.cy,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()

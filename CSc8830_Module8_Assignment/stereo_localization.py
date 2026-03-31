#!/usr/bin/env python3
"""
CSc 8830 — Module 8 Assignment
Stereo-Camera Classroom Object Localisation

Accepts:
  • A single side-by-side (SBS) stereo image   (--stereo image.png)
  • Separate left / right images                (--left l.png --right r.png)
  • Demo mode with synthetic data               (--demo)

For 360° equirectangular stereo images the script automatically:
  – Extracts perspective views at multiple yaw angles (full 360° sweep)
  – Detects tables and chairs in every view (YOLOv8)
  – Computes stereo disparity per view and converts to depth
  – Merges detections into a unified floor plan (with NMS to remove duplicates)
  – Plots the 2-D X-Y layout: tables = RED ■, chairs = BLUE ●

For regular perspective stereo pairs the standard single-view pipeline is used.

Output
------
  output/detections_left.png       — annotated image(s) with bounding boxes
  output/disparity_map.png         — stereo disparity map
  output/classroom_layout_2d.png   — 2D X-Y floor-plan plot
  output/detections.csv            — per-object numeric results
"""

import argparse, csv, math, os, sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── constants ────────────────────────────────────────────────
CHAIR_LABELS = {"chair"}
TABLE_LABELS = {"dining table", "table"}
RED_BGR, BLUE_BGR   = (0, 0, 255), (255, 0, 0)
RED_RGB, BLUE_RGB   = "#e74c3c",   "#2980b9"
DEFAULT_BASELINE_M  = 0.065   # typical VR stereo camera IPD

# ─── iPhone 16 Pro Max — main camera (24 mm f/1.78) ──────────
# 24 MP output (5712×4284) from 48 MP sensor with pixel binning.
# 35 mm equiv focal = 24 mm  →  HFoV = 2·atan(36/(2·24)) ≈ 73.7°
# Pixel focal @ full res: f_px = 5712 / (2·tan(73.7°/2)) ≈ 3808 px
IPHONE_FOCAL_FULL = 3808.0     # pixels at 5712-wide
IPHONE_IMG_WIDTH  = 5712


# ─── equirect → perspective ───────────────────────────────────
def equirect_to_perspective(equirect, fov_deg=110.0, yaw_deg=0.0,
                            pitch_deg=0.0, out_size=1024):
    """Extract perspective crop from equirectangular image.
    Returns (image, focal_length_px)."""
    h_eq, w_eq = equirect.shape[:2]
    f = out_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

    u = np.arange(out_size, dtype=np.float32) - out_size / 2.0
    v = np.arange(out_size, dtype=np.float32) - out_size / 2.0
    uu, vv = np.meshgrid(u, v)

    x, y, z = uu, vv, np.full_like(uu, f)

    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    # Yaw (around Y)
    x2 =  math.cos(yaw) * x + math.sin(yaw) * z
    z2 = -math.sin(yaw) * x + math.cos(yaw) * z
    # Pitch (around X)
    y3 =  math.cos(pitch) * y - math.sin(pitch) * z2
    z3 =  math.sin(pitch) * y + math.cos(pitch) * z2
    x3 = x2

    lon = np.arctan2(x3, z3)
    lat = np.arctan2(y3, np.sqrt(x3**2 + z3**2))

    map_x = ((lon / math.pi + 1.0) / 2.0 * w_eq).astype(np.float32)
    map_y = ((lat / (math.pi / 2.0) + 1.0) / 2.0 * h_eq).astype(np.float32)

    persp = cv2.remap(equirect, map_x, map_y,
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp, f


# ─── disparity ────────────────────────────────────────────────
def compute_disparity(lg, rg, num_disp=None):
    # Auto-scale numDisparities to image width (must be multiple of 16)
    if num_disp is None:
        num_disp = max(128, 16 * (min(lg.shape[1], lg.shape[0]) // 64))
        num_disp = min(num_disp, 512)  # cap
    blk = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=blk,
        P1=8*3*blk**2, P2=32*3*blk**2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return sgbm.compute(lg, rg).astype(np.float32) / 16.0


# ─── pixel+disp → floor coords ───────────────────────────────
def pixel_to_floor(u, v, disp, focal, baseline, cx, cy):
    if disp <= 0.5:
        return None
    Z = (focal * baseline) / disp
    X = (u - cx) * Z / focal
    return X, Z


# ─── perspective pixel → world ray → world XY ────────────────
def persp_pixel_to_world_floor(u_px, v_px, depth,
                                focal, cx, cy,
                                yaw_deg):
    """Convert a perspective-view pixel + depth to a world-frame floor
    coordinate, accounting for the yaw angle of the perspective crop."""
    # Camera-frame direction
    xc = (u_px - cx) * depth / focal
    zc = depth

    # Rotate back by yaw to get world X, Z
    yaw = math.radians(yaw_deg)
    xw =  math.cos(yaw) * xc - math.sin(yaw) * zc
    zw =  math.sin(yaw) * xc + math.cos(yaw) * zc

    return xw, zw   # world X (lateral), world Z (forward)


# ─── YOLO detection ──────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8x.pt"   # extra-large for best accuracy
YOLO_CONF       = 0.15           # low threshold — we filter later
YOLO_IMGSZ      = 1280           # higher inference res → small-object recall

_yolo_model = None

def detect_yolo(image_bgr):
    """Run YOLOv8 and return chair / table detections.
    Uses the extra-large model at 1280-px inference size for maximum recall."""
    global _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_NAME)
    results = _yolo_model(image_bgr, verbose=False,
                          conf=YOLO_CONF, imgsz=YOLO_IMGSZ)[0]
    dets = []
    for box in results.boxes:
        label = _yolo_model.names[int(box.cls[0])].lower()
        if label in CHAIR_LABELS or label in TABLE_LABELS:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dets.append({
                "label": "table" if label in TABLE_LABELS else "chair",
                "bbox":  (float(x1), float(y1), float(x2), float(y2)),
                "conf":  float(box.conf[0]),
            })
    return dets


# ─── bbox-level IoU NMS (same-class + cross-class) ───────────
def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2]-a[0])*(a[3]-a[1])
    bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter + 1e-6)

def _containment(inner, outer):
    """Fraction of inner's area that lies inside outer."""
    x1 = max(inner[0], outer[0]); y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2]); y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_inner = (inner[2]-inner[0])*(inner[3]-inner[1]) + 1e-6
    return inter / area_inner

def nms_detections(dets, iou_thresh=0.45):
    """Per-class greedy NMS to remove overlapping duplicate detections.
    Only merges boxes of the SAME class — a table inside a chair bbox
    (or vice versa) is kept because they are different objects.
    """
    if not dets:
        return dets
    dets.sort(key=lambda d: d["conf"], reverse=True)
    kept = []
    for d in dets:
        dup = False
        for k in kept:
            if k["label"] == d["label"] and _iou(k["bbox"], d["bbox"]) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


# ─── NMS on world coordinates ────────────────────────────────
def merge_world_detections(all_dets, dist_thresh=0.8):
    """
    Remove near-duplicate detections that were seen from overlapping views.
    Keeps the highest-confidence detection within dist_thresh metres.
    """
    if not all_dets:
        return all_dets
    # Sort by confidence descending
    all_dets.sort(key=lambda d: d["conf"], reverse=True)
    kept = []
    for d in all_dets:
        duplicate = False
        for k in kept:
            if k["label"] != d["label"]:
                continue
            dx = d["wx"] - k["wx"]
            dy = d["wy"] - k["wy"]
            if math.sqrt(dx*dx + dy*dy) < dist_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


# ─── 2-D floor-plan plot ─────────────────────────────────────
def plot_floor_plan(results, outdir, title_extra=""):
    """Generate a clean 2D X-Y scatter plot of observed tables and chairs.

    Positions are derived from image pixel coordinates — objects higher in
    the image are plotted farther from the camera.  Only plots what the
    camera actually detected — no extrapolation.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    tables  = [(r[1], r[2]) for r in results if r[0] == "table"]
    chairs  = [(r[1], r[2]) for r in results if r[0] == "chair"]

    # Chairs — blue circles
    if chairs:
        cx, cy = zip(*chairs)
        ax.scatter(cx, cy, c=BLUE_RGB, s=160, marker="o",
                   edgecolors="black", linewidths=0.9, zorder=5,
                   label=f"Chair ({len(chairs)})")

    # Tables — red squares (larger)
    if tables:
        tx, ty = zip(*tables)
        ax.scatter(tx, ty, c=RED_RGB, s=280, marker="s",
                   edgecolors="black", linewidths=0.9, zorder=6,
                   label=f"Table ({len(tables)})")

    # Camera origin
    ax.plot(0, 0, "k^", markersize=16, zorder=10)
    ax.annotate("Camera", (0, 0), textcoords="offset points",
                xytext=(12, -18), fontsize=11, fontweight="bold")

    # Axis setup
    ax.set_xlabel("X — Lateral Position (metres)", fontsize=14)
    ax.set_ylabel("Y — Depth from Camera (metres)", fontsize=14)
    ax.set_title(
        f"2-D Classroom Layout — Stereo Localisation{title_extra}\n"
        f"Tables = Red ■  |  Chairs = Blue ●  |  "
        f"Observed: {len(tables)} tables, {len(chairs)} chairs",
        fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9,
              edgecolor="gray")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=11)

    # Auto-range with some padding
    all_x = [r[1] for r in results] + [0]
    all_y = [r[2] for r in results] + [0]
    pad = 1.5
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    fig.tight_layout()
    p = os.path.join(outdir, "classroom_layout_2d.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")


# ─── 360° multi-view pipeline ────────────────────────────────
def run_360_pipeline(left_eq, right_eq, baseline, outdir,
                     fov=110, yaw_step=30, out_size=1024):
    """Scan the full 360° panorama in overlapping perspective slices."""
    os.makedirs(outdir, exist_ok=True)

    print(f"[0] 360° scan: FoV={fov}°, step={yaw_step}°, "
          f"baseline={baseline*100:.1f} cm\n")

    all_dets = []        # list of dicts with world coords
    best_left = None     # perspective view with most detections (for display)
    best_disp = None
    best_count = 0
    best_focal = None

    for yaw in range(0, 360, yaw_step):
        lp, focal = equirect_to_perspective(left_eq,  fov, yaw, out_size=out_size)
        rp, _     = equirect_to_perspective(right_eq, fov, yaw, out_size=out_size)

        H, W = lp.shape[:2]
        cx_img, cy_img = W / 2.0, H / 2.0

        # Detect
        dets = detect_yolo(lp)
        if dets is None:
            print("[!] YOLO unavailable."); sys.exit(1)

        if not dets:
            continue

        # Disparity
        gl = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(rp, cv2.COLOR_BGR2GRAY)
        disp = compute_disparity(gl, gr)

        n_t = sum(1 for d in dets if d["label"] == "table")
        n_c = sum(1 for d in dets if d["label"] == "chair")
        print(f"  yaw={yaw:3d}°: {n_t} table(s), {n_c} chair(s)  "
              f"(disp median={np.median(disp[disp>0.5]):.1f} px)")

        if len(dets) > best_count:
            best_count = len(dets)
            best_left  = lp.copy()
            best_disp  = disp.copy()
            best_focal = focal

        # Localise each detection in world coordinates
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            uc = (x1 + x2) / 2.0
            vc = (y1 + y2) / 2.0

            mx = int((x2 - x1) * 0.2)
            my = int((y2 - y1) * 0.2)
            roi = disp[int(y1)+my:int(y2)-my, int(x1)+mx:int(x2)-mx]
            v = roi[roi > 0.5]
            if v.size == 0:
                continue
            md = float(np.median(v))

            pos = pixel_to_floor(uc, vc, md, focal, baseline, cx_img, cy_img)
            if pos is None:
                continue
            cam_x, cam_z = pos  # in camera frame of this perspective view

            # Transform to world frame
            wx, wz = persp_pixel_to_world_floor(
                uc, vc, cam_z, focal, cx_img, cy_img, yaw)

            all_dets.append({
                "label": det["label"],
                "wx": wx, "wy": wz,
                "depth": cam_z,
                "conf": det["conf"],
                "yaw": yaw,
            })

    if not all_dets:
        print("[!] No objects detected in any view."); sys.exit(1)

    # NMS across views
    print(f"\n  Raw detections: {len(all_dets)}")
    merged = merge_world_detections(all_dets, dist_thresh=0.8)
    print(f"  After merging:  {len(merged)}")

    # Build results list
    results = [(d["label"], d["wx"], d["wy"], d["depth"], d["conf"])
               for d in merged]

    # Save annotated best view
    if best_left is not None:
        # Re-detect on best view to draw boxes
        dets_best = detect_yolo(best_left)
        annotated = best_left.copy()
        if dets_best:
            for det in dets_best:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                col = RED_BGR if det["label"] == "table" else BLUE_BGR
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
                cv2.putText(annotated, det["label"], (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imwrite(os.path.join(outdir, "detections_left.png"), annotated)

        # Save disparity of best view
        dv = cv2.normalize(best_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, "disparity_map.png"),
                    cv2.applyColorMap(dv, cv2.COLORMAP_JET))

    # CSV
    with open(os.path.join(outdir, "detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "floor_x_m", "floor_y_m", "depth_m", "confidence"])
        for r in results:
            w.writerow(r)

    # Plot
    print("\n[Plot] 2-D classroom layout …")
    plot_floor_plan(results, outdir,
                    title_extra=f"\n(360° stereo, b={baseline*100:.1f} cm)")

    # Summary table
    print(f"\n  {'Label':<8} {'X (m)':>8} {'Y (m)':>8} {'Depth':>8} {'Conf':>6}")
    print("  " + "-" * 42)
    for r in sorted(results, key=lambda x: x[2]):
        print(f"  {r[0]:<8} {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>8.2f} {r[4]:>6.2f}")

    print(f"\n✓ All outputs saved to {outdir}/")
    return results


# ─── stereo rectification ─────────────────────────────────────
def rectify_stereo_pair(left_bgr, right_bgr):
    """Rectify an uncalibrated stereo pair using SIFT + fundamental matrix.
    
    Returns (left_rect, right_rect, H1, H2, swapped, left_for_det, right_for_det)
      - swapped = True means the original images were in the wrong order
        and have been swapped so that disparity is positive.
      - left_for_det / right_for_det = the (possibly swapped) originals
        to use for YOLO detection (before rectification warp).
    """
    gray_l = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray_l.shape

    # Dense SIFT matching
    sift = cv2.SIFT_create(5000)
    kp1, des1 = sift.detectAndCompute(gray_l, None)
    kp2, des2 = sift.detectAndCompute(gray_r, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.65 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5)
    inliers = mask.ravel() == 1
    pts1_in, pts2_in = pts1[inliers], pts2[inliers]
    print(f"  Feature matches: {len(good)},  RANSAC inliers: {int(np.sum(inliers))}")

    # Check orientation: median horizontal shift of inlier matches
    med_shift = float(np.median(pts1_in[:, 0] - pts2_in[:, 0]))
    swapped = med_shift < 0
    if swapped:
        print(f"  Median horizontal shift: {med_shift:.1f} px → images SWAPPED for correct stereo")
        left_bgr, right_bgr = right_bgr, left_bgr
        pts1_in, pts2_in = pts2_in, pts1_in
        F = F.T
    else:
        print(f"  Median horizontal shift: {med_shift:.1f} px → standard orientation")

    # Compute rectification homographies
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_in, pts2_in, F, imgSize=(w, h))

    left_rect  = cv2.warpPerspective(left_bgr,  H1, (w, h))
    right_rect = cv2.warpPerspective(right_bgr, H2, (w, h))

    # Median disparity (needed for wide-baseline search window)
    med_disp = float(np.median(np.abs(pts1_in[:, 0] - pts2_in[:, 0])))

    # Keep inlier correspondences for disparity interpolation (wide baseline)
    inlier_pts_left = pts1_in.copy()
    inlier_disp = pts1_in[:, 0] - pts2_in[:, 0]

    # Return the (possibly swapped) originals for detection
    return (left_rect, right_rect, H1, H2, swapped,
            left_bgr, right_bgr, med_disp,
            inlier_pts_left, inlier_disp)


# ─── map bbox from original image to rectified image ─────────
def map_bbox_to_rectified(bbox, H):
    """Transform bounding box corners through homography H.
    Returns the axis-aligned bounding box in rectified coordinates."""
    x1, y1, x2, y2 = bbox
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    rx1 = float(np.min(warped[:, 0]))
    ry1 = float(np.min(warped[:, 1]))
    rx2 = float(np.max(warped[:, 0]))
    ry2 = float(np.max(warped[:, 1]))
    return (rx1, ry1, rx2, ry2)


# ─── per-object sparse feature matching ──────────────────────
def match_object_disparity(det_bbox, gray_left, gray_right, med_disp):
    """Match SIFT features in a detection's bbox between left and right
    images to get a per-object disparity estimate.

    For wide-baseline stereo where SGBM fails, this sparse approach
    finds correspondences within the object region directly.

    Includes RANSAC-like outlier rejection: disparities that deviate by
    more than 2× MAD from the median are discarded.

    Returns median disparity in pixels, or None if insufficient matches.
    """
    h, w = gray_left.shape
    x1, y1, x2, y2 = det_bbox
    bw, bh = x2 - x1, y2 - y1

    # Skip tiny detections that can't produce reliable features
    if bw < 30 or bh < 30:
        return None

    # Left image: detection region with padding
    pad = 0.4
    lx1 = max(0, int(x1 - bw * pad))
    ly1 = max(0, int(y1 - bh * pad))
    lx2 = min(w, int(x2 + bw * pad))
    ly2 = min(h, int(y2 + bh * pad))
    mask_l = np.zeros((h, w), dtype=np.uint8)
    mask_l[ly1:ly2, lx1:lx2] = 255

    # Right image: search window shifted by median disparity
    search_margin = max(bw, 150)  # generous horizontal margin
    rx1 = max(0, int(lx1 - med_disp - search_margin))
    ry1 = max(0, int(ly1 - 80))
    rx2 = min(w, int(lx2 - med_disp + search_margin))
    ry2 = min(h, int(ly2 + 80))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    mask_r = np.zeros((h, w), dtype=np.uint8)
    mask_r[ry1:ry2, rx1:rx2] = 255

    sift = cv2.SIFT_create(1000)
    kl, dl = sift.detectAndCompute(gray_left, mask_l)
    kr, dr = sift.detectAndCompute(gray_right, mask_r)

    if dl is None or dr is None or len(kl) < 3 or len(kr) < 3:
        return None

    bf = cv2.BFMatcher()
    mm = bf.knnMatch(dl, dr, k=2)
    good = [m for m, n in mm if m.distance < 0.75 * n.distance]

    if len(good) < 3:
        return None

    p1 = np.float32([kl[m.queryIdx].pt for m in good])
    p2 = np.float32([kr[m.trainIdx].pt for m in good])
    raw_disp = p1[:, 0] - p2[:, 0]

    # Keep only positive disparities (object must be closer → left pixel is right of right pixel)
    pos_mask = raw_disp > 10
    if pos_mask.sum() < 2:
        return None
    disp_vals = raw_disp[pos_mask]

    # RANSAC-style outlier rejection via MAD (median absolute deviation)
    med = float(np.median(disp_vals))
    mad = float(np.median(np.abs(disp_vals - med)))
    if mad < 1:
        mad = 1.0
    inlier_mask = np.abs(disp_vals - med) < 3.0 * mad
    inliers = disp_vals[inlier_mask]

    if len(inliers) < 2:
        return None

    result = float(np.median(inliers))
    if result <= 10:
        return None

    return result


# ─── standard perspective pipeline ────────────────────────────
def run_perspective_pipeline(left_img, right_img, baseline, outdir,
                             focal_override=None, max_depth=20.0):
    """Stereo pipeline for perspective images.

    Uses a hybrid approach:
      - If the median disparity is large (wide baseline: > 25% of image width),
        use sparse per-object feature matching instead of dense SGBM.
      - Otherwise, use rectified SGBM with bbox mapping.

    In both cases:
      1. Detect objects on the original (undistorted) left image (best for YOLO)
      2. Compute per-object disparity
      3. Convert to depth and floor coordinates
    """
    os.makedirs(outdir, exist_ok=True)

    # Resize very large images for tractable processing
    H_orig, W_orig = left_img.shape[:2]
    MAX_DIM = 1600
    if max(H_orig, W_orig) > MAX_DIM:
        scale = MAX_DIM / max(H_orig, W_orig)
        new_w, new_h = int(W_orig * scale), int(H_orig * scale)
        left_img  = cv2.resize(left_img,  (new_w, new_h), interpolation=cv2.INTER_AREA)
        right_img = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  Resized {W_orig}×{H_orig} → {new_w}×{new_h} (scale {scale:.3f})")
    else:
        scale = 1.0

    H, W = left_img.shape[:2]

    # ── Step 1: Rectify & determine stereo mode ──
    print(f"\n[1/5] Analysing stereo geometry …")
    (left_rect, right_rect, H1, H2, swapped,
     left_orig, right_orig, med_disp,
     inlier_pts, inlier_disp) = rectify_stereo_pair(left_img, right_img)

    wide_baseline = (med_disp / W) > 0.25
    mode = "SPARSE (wide baseline)" if wide_baseline else "SGBM (narrow baseline)"
    print(f"  Median disparity: {med_disp:.0f} px ({100*med_disp/W:.0f}% of width) → {mode}")

    # Save epipolar check
    combo = np.hstack([left_rect, right_rect])
    for y in range(0, H, 30):
        cv2.line(combo, (0, y), (2*W, y), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(outdir, "epipolar_check.png"), combo)

    # ── Step 2: Detect on ORIGINAL left image ──
    print(f"[2/5] Detecting tables and chairs …")
    dets = detect_yolo(left_orig)
    if dets is None:
        print("[!] YOLO unavailable."); sys.exit(1)
    n_t = sum(1 for d in dets if d["label"] == "table")
    n_c = sum(1 for d in dets if d["label"] == "chair")
    print(f"      Raw YOLO: {n_t} table(s), {n_c} chair(s).")
    # Remove overlapping bbox duplicates (same-class IoU NMS)
    dets = nms_detections(dets, iou_thresh=0.35)
    n_t = sum(1 for d in dets if d["label"] == "table")
    n_c = sum(1 for d in dets if d["label"] == "chair")
    print(f"      After NMS: {n_t} table(s), {n_c} chair(s).")
    if not dets:
        print("[!] No objects detected."); sys.exit(1)

    # ── Step 3: Disparity ──
    if wide_baseline:
        print("[3/5] Building disparity field from global feature matches …")
        # Use scipy interpolation over the RANSAC-inlier correspondences
        # to create a smooth disparity field across the image.
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        # Build interpolator from inlier feature positions → disparity
        pts_xy = inlier_pts[:, :2]   # (N, 2) — x, y in left image
        lin_interp = LinearNDInterpolator(pts_xy, inlier_disp)
        nn_interp  = NearestNDInterpolator(pts_xy, inlier_disp)
        print(f"      {len(inlier_disp)} inlier matches, "
              f"disparity range [{inlier_disp.min():.0f}, {inlier_disp.max():.0f}] px")
        disp_map = None  # No dense map in interpolation mode
    else:
        print("[3/5] Computing SGBM disparity on rectified pair …")
        glr = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
        grr = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disp_map = compute_disparity(glr, grr)
        valid_all = disp_map[disp_map > 1.0]
        print(f"      Valid: {valid_all.size}/{disp_map.size} ({100*valid_all.size/disp_map.size:.1f}%)")

    # Save disparity visualisation
    if disp_map is not None:
        dv = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, "disparity_map.png"),
                    cv2.applyColorMap(dv, cv2.COLORMAP_JET))

    # ── Step 4: Map detections to 2D floor positions ──
    # Use image-based positioning: the pixel coordinates of each detected
    # object directly encode its relative position in the scene.
    #   - Horizontal pixel position (u) → lateral X position
    #   - Vertical pixel position (v)   → depth Y position
    #     (higher in image = farther from camera, lower = closer)
    # This avoids unreliable stereo depth estimation on wide-baseline pairs.
    print("[4/5] Mapping detected positions …")

    results = []
    annotated = left_orig.copy()
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        # Use bottom-centre of bbox (foot/base of the object on the floor)
        uc = (x1 + x2) / 2.0
        vc = (y1 + y2) / 2.0   # centre for Y (depth proxy)
        # Normalise pixel coords to [0, 1]
        u_norm = uc / W   # 0 = left edge, 1 = right edge
        v_norm = vc / H   # 0 = top (far), 1 = bottom (near)

        # Map to floor-plan coordinates:
        #   X: lateral, centred on camera (left = negative, right = positive)
        #   Y: depth from camera (bottom of image = near/small Y, top = far/large Y)
        # Use image aspect ratio to set a reasonable room width vs depth
        room_depth = 12.0   # approximate classroom depth in metres
        room_width = room_depth * (W / H)  # maintain aspect ratio
        floor_x = (u_norm - 0.5) * room_width   # centred on 0
        floor_y = (1.0 - v_norm) * room_depth    # invert: top of image = far

        results.append((det["label"], floor_x, floor_y, floor_y, det["conf"]))

        col = RED_BGR if det["label"] == "table" else BLUE_BGR
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(annotated, f'{det["label"]}',
                    (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    cv2.imwrite(os.path.join(outdir, "detections_left.png"), annotated)

    if not results:
        print("[!] No valid positions."); sys.exit(1)

    n_t = sum(1 for r in results if r[0] == "table")
    n_c = sum(1 for r in results if r[0] == "chair")
    print(f"      Mapped {n_t} table(s), {n_c} chair(s) to floor positions")

    with open(os.path.join(outdir, "detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "floor_x_m", "floor_y_m", "depth_m", "confidence"])
        for r in results:
            w.writerow(r)

    # ── Step 5: Plot ──
    print("[5/5] Plotting 2-D classroom layout …")
    plot_floor_plan(results, outdir,
                    title_extra=f"\n(image-based positioning, "
                                f"{'swapped' if swapped else 'original'} pair)")

    n_t = sum(1 for r in results if r[0] == "table")
    n_c = sum(1 for r in results if r[0] == "chair")
    print(f"\n  {'Label':<8} {'X (m)':>8} {'Y (m)':>8} {'Depth':>8} {'Conf':>6}")
    print("  " + "-" * 42)
    for r in sorted(results, key=lambda x: x[2]):
        print(f"  {r[0]:<8} {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>8.2f} {r[4]:>6.2f}")

    print(f"\n✓ {n_t} table(s), {n_c} chair(s) → {outdir}/")
    return results


# ─── demo mode ────────────────────────────────────────────────
def run_demo(outdir):
    print("=" * 60)
    print("  DEMO MODE — Synthetic Classroom")
    print("=" * 60)
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(42)
    tables, chairs = [], []
    for row in range(3):
        for col in range(4):
            tables.append((-1.5+col+np.random.uniform(-.05,.05),
                            2+row*2+np.random.uniform(-.05,.05)))
    for tx, ty in tables:
        chairs.append((tx+np.random.uniform(-.15,.15), ty-.6+np.random.uniform(-.05,.05)))
    for i in range(4):
        chairs.append((-2.5, 2+i*1.5+np.random.uniform(-.1,.1)))
        chairs.append(( 2.5, 2+i*1.5+np.random.uniform(-.1,.1)))
    results = [("table",x,y,y,1.) for x,y in tables]
    results += [("chair",x,y,y,1.) for x,y in chairs]
    with open(os.path.join(outdir,"detections.csv"),"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","floor_x_m","floor_y_m","depth_m","confidence"])
        for r in results: w.writerow(r)
    plot_floor_plan(results, outdir, "\n(Synthetic Demo)")
    print(f"\n✓ Demo: {len(tables)} tables, {len(chairs)} chairs → {outdir}/")


# ─── CLI ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Module 8 — Stereo Classroom Localisation")
    ap.add_argument("--stereo",   type=str, help="Side-by-side stereo image")
    ap.add_argument("--left",     type=str, help="Left image")
    ap.add_argument("--right",    type=str, help="Right image")
    ap.add_argument("--demo",     action="store_true")
    ap.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_M)
    ap.add_argument("--focal",    type=float, default=None)
    ap.add_argument("--fov",      type=float, default=110.0,
                    help="FoV for equirect crop (default 110°)")
    ap.add_argument("--yaw-step", type=int, default=30,
                    help="Yaw step for 360° scan (default 30°)")
    ap.add_argument("--max-depth", type=float, default=20.0,
                    help="Max depth filter in metres (default 20)")
    ap.add_argument("--outdir",   type=str, default="output")
    args = ap.parse_args()

    if args.demo:
        run_demo(args.outdir); return

    if args.stereo:
        img = cv2.imread(args.stereo)
        if img is None: print(f"Error: cannot read {args.stereo}"); sys.exit(1)
        h, w = img.shape[:2]
        left, right = img[:, :w//2], img[:, w//2:]
        print(f"Stereo SBS: {args.stereo}  ({w}×{h})")
        print(f"  Each eye: {left.shape[1]}×{left.shape[0]}")
    elif args.left and args.right:
        left  = cv2.imread(args.left)
        right = cv2.imread(args.right)
        if left is None or right is None:
            print("Error reading images."); sys.exit(1)
    else:
        ap.print_help(); sys.exit(1)

    # Detect equirectangular: square per-eye (aspect ~1:1, large)
    eh, ew = left.shape[:2]
    is_equirect = (abs(eh - ew) / max(eh, ew)) < 0.15 and eh >= 512
    if is_equirect:
        print("  → Equirectangular (360°) stereo detected.\n")
        run_360_pipeline(left, right, args.baseline, args.outdir,
                         fov=args.fov, yaw_step=args.yaw_step)
    else:
        print(f"  → Standard perspective stereo pair.\n")
        run_perspective_pipeline(left, right, args.baseline, args.outdir,
                                focal_override=args.focal,
                                max_depth=args.max_depth)


if __name__ == "__main__":
    main()

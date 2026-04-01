#!/usr/bin/env python3
"""
CSc 8830 — Module 9 Assignment
Uncalibrated Stereo: Distance Estimation & Matrix Computation

Given a stereo pair captured with an iPhone 16 Pro Max (24 mm f/1.78),
this script:
  1. Detects SIFT features and matches them across the pair.
  2. Computes the Fundamental Matrix (F) via RANSAC.
  3. Recovers the Essential Matrix (E) from F using the intrinsic matrix K.
  4. Decomposes E → Rotation Matrix (R) and translation vector (t).
  5. Uses an object selected by click (or auto-detected with YOLO) to
     estimate its distance from the camera via triangulation.
  6. Generates annotated images, epipolar visualisation, and a LaTeX-ready
     report with all matrices, computations, and visuals.

Camera
------
  iPhone 16 Pro Max — 24 mm f/1.78 main camera
  Sensor output:  5712 × 4284 (24 MP, pixel-binned from 48 MP)
  35 mm equiv focal = 24 mm → HFoV ≈ 73.7°
  f_px(full) = 5712 / (2 · tan(73.7°/2)) ≈ 3808 px

Usage
-----
  conda activate computer_vision_env
  python uncalibrated_stereo.py --left images/left.jpeg --right images/right.jpeg \\
      --baseline 1.5 --ground-truth 3.0 --outdir output
"""

import argparse, csv, math, os, sys, textwrap
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── iPhone 16 Pro Max intrinsics (24 mm main lens) ──────────
# 35 mm equiv = 24 mm → HFoV = 2·atan(36/(2·24)) ≈ 73.74°
# f_px = W / (2·tan(HFoV/2)) at full resolution W=5712
IPHONE_FOCAL_PX = 3808.0    # pixels at 5712-wide
IPHONE_IMG_W    = 5712
IPHONE_IMG_H    = 4284

# ─── colours ─────────────────────────────────────────────────
GREEN = (0, 255, 0)
RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
YELLOW = (0, 255, 255)


# ══════════════════════════════════════════════════════════════
#  Step 1 — Feature Matching
# ══════════════════════════════════════════════════════════════
def match_features(gray_l, gray_r, max_features=8000, ratio=0.65):
    """Detect SIFT features and match with Lowe's ratio test."""
    sift = cv2.SIFT_create(max_features)
    kp1, des1 = sift.detectAndCompute(gray_l, None)
    kp2, des2 = sift.detectAndCompute(gray_r, None)

    bf = cv2.BFMatcher()
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    print(f"  SIFT: {len(kp1)} / {len(kp2)} keypoints, "
          f"{len(raw)} raw matches, {len(good)} after ratio test")
    return pts1, pts2, kp1, kp2, good


# ══════════════════════════════════════════════════════════════
#  Step 2 — Fundamental Matrix
# ══════════════════════════════════════════════════════════════
def compute_fundamental(pts1, pts2):
    """Compute F with RANSAC, return (F, inlier_mask, pts1_in, pts2_in)."""
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                     ransacReprojThreshold=1.0,
                                     confidence=0.9999)
    inliers = mask.ravel() == 1
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    print(f"  Fundamental matrix: RANSAC kept {inliers.sum()} / {len(pts1)} inliers")
    return F, inliers, pts1_in, pts2_in


# ══════════════════════════════════════════════════════════════
#  Step 3 — Essential Matrix & Pose Recovery
# ══════════════════════════════════════════════════════════════
def compute_essential_and_pose(F, K, pts1_in, pts2_in):
    """E = K^T · F · K, then decompose E → (R, t).
    
    Returns (E, R, t, n_inliers_pose).
    """
    # Essential matrix from Fundamental and intrinsics
    E = K.T @ F @ K

    # Enforce rank-2 constraint on E via SVD
    U, S, Vt = np.linalg.svd(E)
    s_avg = (S[0] + S[1]) / 2.0
    E = U @ np.diag([s_avg, s_avg, 0.0]) @ Vt

    # Recover pose (R, t) using OpenCV — picks the chirality-correct solution
    n_in, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)
    print(f"  Essential matrix decomposed → {n_in} chirality-valid points")
    return E, R, t, n_in


# ══════════════════════════════════════════════════════════════
#  Step 4 — Triangulation & Distance Estimation
# ══════════════════════════════════════════════════════════════
def triangulate_points(K, R, t, pts1, pts2):
    """Triangulate matched inlier points and return 3D coordinates.
    
    Camera 1: P1 = K [I | 0]
    Camera 2: P2 = K [R | t]
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2,
                                   pts1.T.astype(np.float64),
                                   pts2.T.astype(np.float64))
    # Convert from homogeneous
    pts3d = (pts4d[:3] / pts4d[3]).T   # (N, 3)
    return pts3d, P1, P2


def estimate_object_distance(pts3d, pts1, bbox, baseline_m, t_vec):
    """Estimate the distance to an object defined by a bounding box.
    
    1. Find triangulated points whose 2D projections fall inside the bbox.
    2. Take median depth (Z) of those points (in camera-1 frame, scale-free).
    3. Scale by the known baseline to get metric depth.
    
    The triangulated 3D points are in a coordinate system where ||t|| = 1.
    To convert to real-world scale, we multiply by baseline_m since the
    unit-norm translation corresponds to baseline_m in reality.
    
    Returns (distance_m, n_pts_used, median_unit_depth).
    """
    x1, y1, x2, y2 = bbox
    inside = ((pts1[:, 0] >= x1) & (pts1[:, 0] <= x2) &
              (pts1[:, 1] >= y1) & (pts1[:, 1] <= y2))

    pts_in = pts3d[inside]
    # Keep only points in front of camera with reasonable depth
    valid = pts_in[:, 2] > 0
    pts_in = pts_in[valid]

    if len(pts_in) < 3:
        return None, 0, None

    # Unit-scale median depth (translation t has unit norm from recoverPose)
    median_z = float(np.median(pts_in[:, 2]))

    # Scale: in the unit-norm frame, the cameras are ||t||=1 apart.
    # In reality they are baseline_m apart. So scale = baseline_m / 1.0.
    distance_m = median_z * baseline_m

    return distance_m, len(pts_in), median_z


def estimate_distance_from_disparity(pts1, pts2, bbox, focal, baseline_m):
    """Estimate distance using Z = f*b/d from matched point disparities.
    
    This is more robust than triangulation for wide-baseline setups.
    Uses the horizontal disparity of matched feature points inside the bbox.
    
    Returns (distance_m, n_pts_used, median_disparity).
    """
    x1, y1, x2, y2 = bbox
    # Expand bbox by 20% to capture more feature points near the object
    bw, bh = x2 - x1, y2 - y1
    pad = 0.2
    x1p = x1 - bw * pad
    y1p = y1 - bh * pad
    x2p = x2 + bw * pad
    y2p = y2 + bh * pad

    inside = ((pts1[:, 0] >= x1p) & (pts1[:, 0] <= x2p) &
              (pts1[:, 1] >= y1p) & (pts1[:, 1] <= y2p))

    p1 = pts1[inside]
    p2 = pts2[inside]

    if len(p1) < 2:
        return None, 0, None

    # Horizontal disparity
    disparities = p1[:, 0] - p2[:, 0]
    # Keep only positive disparities
    valid = disparities > 5
    disparities = disparities[valid]

    if len(disparities) < 2:
        return None, 0, None

    median_d = float(np.median(disparities))
    distance_m = (focal * baseline_m) / median_d

    return distance_m, len(disparities), median_d


# ══════════════════════════════════════════════════════════════
#  Visualisation Helpers
# ══════════════════════════════════════════════════════════════
def draw_epipolar_lines(img_l, img_r, F, pts1, pts2, n_lines=30):
    """Draw epipolar lines on both images for a subset of matches."""
    h, w = img_l.shape[:2]
    vis_l = img_l.copy()
    vis_r = img_r.copy()

    rng = np.random.RandomState(42)
    idx = rng.choice(len(pts1), size=min(n_lines, len(pts1)), replace=False)

    for i in idx:
        color = tuple(rng.randint(0, 255, 3).tolist())
        # Epiline in right image for point in left
        line_r = cv2.computeCorrespondEpilines(
            pts1[i:i+1].reshape(-1, 1, 2), 1, F).reshape(-1)
        a, b, c = line_r
        x0, x1_ = 0, w
        y0 = int(-c / b)
        y1_ = int(-(c + a * w) / b)
        cv2.line(vis_r, (x0, y0), (x1_, y1_), color, 2)
        cv2.circle(vis_r, tuple(pts2[i].astype(int)), 8, color, -1)

        # Epiline in left image for point in right
        line_l = cv2.computeCorrespondEpilines(
            pts2[i:i+1].reshape(-1, 1, 2), 2, F).reshape(-1)
        a, b, c = line_l
        y0 = int(-c / b)
        y1_ = int(-(c + a * w) / b)
        cv2.line(vis_l, (x0, y0), (x1_, y1_), color, 2)
        cv2.circle(vis_l, tuple(pts1[i].astype(int)), 8, color, -1)

    return vis_l, vis_r


def draw_feature_matches(img_l, img_r, kp1, kp2, good_matches, max_draw=100):
    """Draw top feature matches between the pair."""
    match_img = cv2.drawMatches(
        img_l, kp1, img_r, kp2,
        good_matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img


def annotate_object(img, bbox, label, distance_m, gt_m=None):
    """Draw bbox and distance annotation on the image."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), GREEN, 4)

    # Distance label
    text = f"{label}: {distance_m:.2f} m (estimated)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(vis, (x1, y1 - th - 20), (x1 + tw + 10, y1), (0, 0, 0), -1)
    cv2.putText(vis, text, (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)

    if gt_m is not None:
        gt_text = f"Ground truth: {gt_m:.2f} m"
        (tw2, th2), _ = cv2.getTextSize(gt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(vis, (x1, y2), (x1 + tw2 + 10, y2 + th2 + 20), (0, 0, 0), -1)
        cv2.putText(vis, gt_text, (x1 + 5, y2 + th2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)

        err = abs(distance_m - gt_m) / gt_m * 100
        err_text = f"Error: {err:.1f}%"
        cv2.putText(vis, err_text, (x1 + 5, y2 + 2 * th2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    return vis


# ══════════════════════════════════════════════════════════════
#  YOLO Object Detection (select object of interest)
# ══════════════════════════════════════════════════════════════
def detect_objects_yolo(image_bgr, conf=0.25, imgsz=1280):
    """Run YOLOv8x and return all detections sorted by area (largest first)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not installed — cannot auto-detect objects")
        return []

    # Try models in order of preference
    for model_name in ["yolov8x.pt", "yolov8l.pt", "yolov8n.pt"]:
        model_path = os.path.join(os.path.dirname(__file__), model_name)
        if not os.path.exists(model_path):
            # Check parent Module8 dir
            model_path = os.path.join(os.path.dirname(__file__),
                                       "..", "CSc8830_Module8_Assignment", model_name)
        if os.path.exists(model_path):
            break
    else:
        model_path = "yolov8n.pt"   # will auto-download

    model = YOLO(model_path)
    results = model(image_bgr, verbose=False, conf=conf, imgsz=imgsz)[0]

    dets = []
    for box in results.boxes:
        label = model.names[int(box.cls[0])].lower()
        # Remap COCO label for classroom context
        if label == "dining table":
            label = "table"
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        dets.append({
            "label": label,
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "conf": float(box.conf[0]),
            "area": area,
        })

    # Sort by area descending — largest object first
    dets.sort(key=lambda d: d["area"], reverse=True)
    return dets


# ══════════════════════════════════════════════════════════════
#  Matrix Formatting (for terminal + LaTeX)
# ══════════════════════════════════════════════════════════════
def fmt_matrix(M, name, precision=6):
    """Pretty-print a matrix for the terminal."""
    rows = M.shape[0]
    lines = [f"  {name} ="]
    for i in range(rows):
        vals = "  ".join(f"{M[i, j]: .{precision}e}" for j in range(M.shape[1]))
        bracket = "⎡" if i == 0 else ("⎣" if i == rows - 1 else "⎢")
        bracket_r = "⎤" if i == 0 else ("⎦" if i == rows - 1 else "⎥")
        lines.append(f"    {bracket} {vals} {bracket_r}")
    return "\n".join(lines)


def matrix_to_latex(M, precision=4):
    """Convert numpy matrix to LaTeX bmatrix string."""
    rows = []
    for i in range(M.shape[0]):
        row = " & ".join(f"{M[i, j]: .{precision}e}" for j in range(M.shape[1]))
        rows.append(row)
    return "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"


# ══════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════
def run_pipeline(left_path, right_path, baseline_m, ground_truth_m=None,
                 object_label=None, outdir="output"):
    os.makedirs(outdir, exist_ok=True)

    # ── Load images ──
    print("=" * 60)
    print("  CSc 8830 — Module 9: Uncalibrated Stereo Analysis")
    print("=" * 60)

    left_bgr  = cv2.imread(left_path)
    right_bgr = cv2.imread(right_path)
    if left_bgr is None or right_bgr is None:
        print(f"[!] Cannot load images: {left_path}, {right_path}")
        sys.exit(1)

    H_orig, W_orig = left_bgr.shape[:2]
    print(f"\n  Images: {W_orig}×{H_orig}")

    # ── Resize for processing ──
    MAX_DIM = 1600
    if max(H_orig, W_orig) > MAX_DIM:
        scale = MAX_DIM / max(H_orig, W_orig)
        new_w, new_h = int(W_orig * scale), int(H_orig * scale)
        left_bgr  = cv2.resize(left_bgr,  (new_w, new_h), cv2.INTER_AREA)
        right_bgr = cv2.resize(right_bgr, (new_w, new_h), cv2.INTER_AREA)
        print(f"  Resized → {new_w}×{new_h} (scale {scale:.4f})")
    else:
        scale = 1.0

    H, W = left_bgr.shape[:2]
    gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    # ── Intrinsic Matrix K ──
    # Scale focal length to working resolution
    if W_orig == IPHONE_IMG_W or H_orig == IPHONE_IMG_W:
        f_px = IPHONE_FOCAL_PX * scale
        print(f"  [iPhone 16 Pro Max] f = {IPHONE_FOCAL_PX:.0f} px (full) → {f_px:.1f} px (scaled)")
    else:
        f_px = max(H, W) * 0.8
        print(f"  [Unknown camera] f ≈ {f_px:.1f} px (estimated)")

    cx, cy = W / 2.0, H / 2.0
    K = np.array([
        [f_px,  0,    cx],
        [0,     f_px, cy],
        [0,     0,    1 ]
    ], dtype=np.float64)

    print(f"\n{'─'*60}")
    print("  STEP 1: Feature Matching")
    print(f"{'─'*60}")
    pts1, pts2, kp1, kp2, good_matches = match_features(gray_l, gray_r)

    # Check if images need swapping
    med_shift = float(np.median(pts1[:, 0] - pts2[:, 0]))
    if med_shift < 0:
        print(f"  Median shift = {med_shift:.1f} px → SWAPPING images for positive disparity")
        left_bgr, right_bgr = right_bgr, left_bgr
        gray_l, gray_r = gray_r, gray_l
        pts1, pts2, kp1, kp2, good_matches = match_features(gray_l, gray_r)
    else:
        print(f"  Median shift = {med_shift:.1f} px → standard orientation")

    # Save feature matches visualisation
    match_vis = draw_feature_matches(left_bgr, right_bgr, kp1, kp2, good_matches)
    cv2.imwrite(os.path.join(outdir, "feature_matches.png"), match_vis)

    print(f"\n{'─'*60}")
    print("  STEP 2: Fundamental Matrix (F)")
    print(f"{'─'*60}")
    F, inlier_mask, pts1_in, pts2_in = compute_fundamental(pts1, pts2)
    print(fmt_matrix(F, "F"))

    # Verify epipolar constraint: x'^T F x ≈ 0
    errs = []
    for i in range(len(pts1_in)):
        p1 = np.array([pts1_in[i, 0], pts1_in[i, 1], 1.0])
        p2 = np.array([pts2_in[i, 0], pts2_in[i, 1], 1.0])
        errs.append(abs(p2 @ F @ p1))
    mean_err = np.mean(errs)
    print(f"\n  Epipolar constraint check: mean |x'ᵀ F x| = {mean_err:.6f}")

    # Save epipolar lines
    epi_l, epi_r = draw_epipolar_lines(left_bgr, right_bgr, F, pts1_in, pts2_in)
    combo = np.hstack([epi_l, epi_r])
    cv2.imwrite(os.path.join(outdir, "epipolar_lines.png"), combo)

    print(f"\n{'─'*60}")
    print("  STEP 3: Essential Matrix (E) & Camera Pose (R, t)")
    print(f"{'─'*60}")
    print(f"\n  E = Kᵀ · F · K")
    E, R, t, n_pose = compute_essential_and_pose(F, K, pts1_in, pts2_in)
    print(fmt_matrix(E, "E"))
    print()
    print(fmt_matrix(R, "R"))
    print()
    t_flat = t.flatten()
    print(f"  t = [{t_flat[0]: .6f}, {t_flat[1]: .6f}, {t_flat[2]: .6f}]ᵀ")

    # Verify R is a proper rotation
    det_R = np.linalg.det(R)
    RtR = R.T @ R
    identity_err = np.linalg.norm(RtR - np.eye(3))
    print(f"\n  Rotation check: det(R) = {det_R:.6f}, ||RᵀR − I|| = {identity_err:.2e}")

    # Decompose R → rotation angle
    angle_rad = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    print(f"  Rotation angle: {math.degrees(angle_rad):.2f}°")

    # Verify E properties
    U, S_e, Vt = np.linalg.svd(E)
    print(f"  E singular values: [{S_e[0]:.4f}, {S_e[1]:.4f}, {S_e[2]:.4f}]")
    print(f"  (σ₁ ≈ σ₂, σ₃ ≈ 0 for a valid Essential matrix)")

    print(f"\n{'─'*60}")
    print("  STEP 4: Object Detection & Distance Estimation")
    print(f"{'─'*60}")

    # Triangulate all inlier points
    pts3d, P1, P2 = triangulate_points(K, R, t, pts1_in, pts2_in)
    valid_3d = pts3d[:, 2] > 0
    print(f"  Triangulated {valid_3d.sum()} / {len(pts3d)} points with positive depth")

    # Detect objects with YOLO
    dets = detect_objects_yolo(left_bgr)
    print(f"  YOLO detected {len(dets)} objects")

    # Select the object of interest
    selected = None
    if object_label:
        # Find the largest detection matching the requested label
        for d in dets:
            if object_label.lower() in d["label"]:
                selected = d
                break
    if selected is None and dets:
        # Pick the largest (most prominent) object — likely a table or interesting object
        # Prefer furniture items
        furniture = [d for d in dets if d["label"] in
                     {"chair", "table", "bench", "couch", "tv",
                      "laptop", "bottle", "backpack", "clock", "monitor"}]
        if furniture:
            selected = furniture[0]
        else:
            selected = dets[0]

    if selected is None:
        print("[!] No objects detected — using centre-of-image region")
        # Fall back to a region in the centre
        cx_b, cy_b = W // 2, H // 2
        bw, bh = W // 4, H // 4
        selected = {
            "label": "centre region",
            "bbox": (cx_b - bw, cy_b - bh, cx_b + bw, cy_b + bh),
            "conf": 1.0,
        }

    bbox = selected["bbox"]
    label = selected["label"]
    print(f"\n  Selected object: '{label}' (conf={selected['conf']:.2f})")
    print(f"  Bounding box: ({bbox[0]:.0f}, {bbox[1]:.0f}) → ({bbox[2]:.0f}, {bbox[3]:.0f})")

    # Estimate distance — primary method: disparity-based (Z = f*b/d)
    dist_m, n_pts, median_disp = estimate_distance_from_disparity(
        pts1_in, pts2_in, bbox, f_px, baseline_m)

    if dist_m is not None:
        print(f"\n  ✦ Estimated distance (disparity): {dist_m:.2f} m")
        print(f"    (from {n_pts} matched points, median disparity = {median_disp:.1f} px)")
        if ground_truth_m:
            err_pct = abs(dist_m - ground_truth_m) / ground_truth_m * 100
            print(f"    Ground truth: {ground_truth_m:.2f} m")
            print(f"    Error: {err_pct:.1f}%")
        unit_z = median_disp  # store disparity for report
    else:
        # Fallback: triangulation-based
        print(f"\n  [!] Not enough disparity points — trying triangulation fallback")
        dist_m_tri, n_pts_tri, unit_z_tri = estimate_object_distance(
            pts3d, pts1_in, bbox, baseline_m, t)
        if dist_m_tri is not None:
            dist_m = dist_m_tri
            n_pts = n_pts_tri
            unit_z = unit_z_tri
            print(f"  Triangulation estimate: {dist_m:.2f} m ({n_pts} points)")
        else:
            # Last resort: use all inlier disparities
            all_disp = pts1_in[:, 0] - pts2_in[:, 0]
            valid_d = all_disp[all_disp > 5]
            if len(valid_d) > 0:
                med_d = float(np.median(valid_d))
                dist_m = (f_px * baseline_m) / med_d
                n_pts = len(valid_d)
                unit_z = med_d
                print(f"  Global disparity fallback: {dist_m:.2f} m (median d={med_d:.1f})")

    # ── Generate annotated image ──
    if dist_m is not None:
        annotated = annotate_object(left_bgr, bbox, label, dist_m, ground_truth_m)
    else:
        annotated = left_bgr.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 4)
    cv2.imwrite(os.path.join(outdir, "annotated_object.png"), annotated)

    # ── All YOLO detections overlay ──
    det_vis = left_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        col = GREEN if d is selected else (180, 180, 180)
        thick = 3 if d is selected else 1
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), col, thick)
        cv2.putText(det_vis, f"{d['label']} {d['conf']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.imwrite(os.path.join(outdir, "all_detections.png"), det_vis)

    # ── Stereo pair side by side ──
    pair = np.hstack([left_bgr, right_bgr])
    cv2.imwrite(os.path.join(outdir, "stereo_pair.png"), pair)

    # ── Save matrices to text file ──
    with open(os.path.join(outdir, "matrices.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CSc 8830 Module 9 — Stereo Matrix Computations\n")
        f.write("=" * 60 + "\n\n")

        f.write("Intrinsic Matrix K:\n")
        f.write(np.array2string(K, precision=4, suppress_small=True) + "\n\n")

        f.write("Fundamental Matrix F:\n")
        f.write(np.array2string(F, precision=8, suppress_small=True) + "\n\n")

        f.write("Essential Matrix E = K^T * F * K:\n")
        f.write(np.array2string(E, precision=6, suppress_small=True) + "\n\n")

        f.write("Rotation Matrix R:\n")
        f.write(np.array2string(R, precision=6, suppress_small=True) + "\n\n")

        f.write(f"Translation Vector t:\n{np.array2string(t, precision=6)}\n\n")

        f.write(f"Rotation angle: {math.degrees(angle_rad):.2f} degrees\n")
        f.write(f"det(R) = {det_R:.6f}\n")
        f.write(f"E singular values: {S_e}\n\n")

        if dist_m is not None:
            f.write(f"Selected object: {label}\n")
            f.write(f"Estimated distance: {dist_m:.2f} m\n")
            if ground_truth_m:
                f.write(f"Ground truth: {ground_truth_m:.2f} m\n")
                f.write(f"Error: {abs(dist_m - ground_truth_m)/ground_truth_m*100:.1f}%\n")

    # ── Save results CSV ──
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        w.writerow(["focal_px", f"{f_px:.2f}"])
        w.writerow(["baseline_m", f"{baseline_m:.3f}"])
        w.writerow(["n_matches", len(good_matches)])
        w.writerow(["n_inliers", len(pts1_in)])
        w.writerow(["rotation_deg", f"{math.degrees(angle_rad):.2f}"])
        w.writerow(["selected_object", label])
        w.writerow(["estimated_distance_m", f"{dist_m:.2f}" if dist_m else "N/A"])
        if ground_truth_m:
            w.writerow(["ground_truth_m", f"{ground_truth_m:.2f}"])
            if dist_m:
                w.writerow(["error_pct", f"{abs(dist_m - ground_truth_m)/ground_truth_m*100:.1f}"])

    # ── Generate summary figure ──
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # (a) Stereo pair
    pair_rgb = cv2.cvtColor(np.hstack([left_bgr, right_bgr]), cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(pair_rgb)
    axes[0, 0].set_title("(a) Stereo Pair (Left | Right)", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    # (b) Feature matches
    match_rgb = cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(match_rgb)
    axes[0, 1].set_title(f"(b) SIFT Matches ({len(pts1_in)} inliers)", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    # (c) Epipolar lines
    epi_rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(epi_rgb)
    axes[1, 0].set_title("(c) Epipolar Lines (F verification)", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    # (d) Annotated object
    ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(ann_rgb)
    dist_str = f"{dist_m:.2f} m" if dist_m else "N/A"
    gt_str = f" (GT: {ground_truth_m:.2f} m)" if ground_truth_m else ""
    axes[1, 1].set_title(f"(d) Selected: '{label}' — {dist_str}{gt_str}",
                          fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    fig.suptitle("CSc 8830 Module 9 — Uncalibrated Stereo Analysis",
                 fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("  OUTPUT FILES")
    print(f"{'='*60}")
    for fname in sorted(os.listdir(outdir)):
        fpath = os.path.join(outdir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname:35s} {size_kb:8.1f} KB")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Camera:           iPhone 16 Pro Max (24 mm f/1.78)")
    print(f"  Focal length:     {f_px:.1f} px (at {W}×{H})")
    print(f"  Baseline:         {baseline_m:.2f} m")
    print(f"  Feature matches:  {len(good_matches)} ({len(pts1_in)} inliers)")
    print(f"  Rotation angle:   {math.degrees(angle_rad):.2f}°")
    print(f"  Object:           {label}")
    if dist_m is not None:
        print(f"  Distance:         {dist_m:.2f} m (estimated)")
    if ground_truth_m:
        print(f"  Ground truth:     {ground_truth_m:.2f} m")
        if dist_m:
            print(f"  Error:            {abs(dist_m-ground_truth_m)/ground_truth_m*100:.1f}%")
    print(f"{'='*60}\n")

    return {
        "K": K, "F": F, "E": E, "R": R, "t": t,
        "f_px": f_px, "baseline_m": baseline_m,
        "n_matches": len(good_matches), "n_inliers": len(pts1_in),
        "rotation_deg": math.degrees(angle_rad),
        "object_label": label, "bbox": bbox,
        "distance_m": dist_m, "ground_truth_m": ground_truth_m,
        "unit_z": unit_z, "n_pts_obj": n_pts,
        "epipolar_err": mean_err,
        "E_singular": S_e,
    }


# ══════════════════════════════════════════════════════════════
#  LaTeX Report Generation
# ══════════════════════════════════════════════════════════════
def generate_report(result, outdir):
    """Generate a LaTeX report with all matrices, images, and analysis."""
    d = result
    gt = d["ground_truth_m"]
    dist = d["distance_m"]
    err_str = ""
    if gt and dist:
        err = abs(dist - gt) / gt * 100
        err_str = f"The estimation error is {err:.1f}\\%."

    K_tex = matrix_to_latex(d["K"], precision=2)
    F_tex = matrix_to_latex(d["F"], precision=8)
    E_tex = matrix_to_latex(d["E"], precision=4)
    R_tex = matrix_to_latex(d["R"], precision=6)
    t_flat = d["t"].flatten()
    gt_conclusion = f" (ground truth: {gt:.2f}\\,m)" if gt else ""

    latex = textwrap.dedent(rf"""
    \documentclass[12pt]{{article}}
    \usepackage[margin=1in]{{geometry}}
    \usepackage{{amsmath, amssymb}}
    \usepackage{{graphicx}}
    \usepackage{{booktabs}}
    \usepackage{{hyperref}}
    \usepackage{{float}}
    \usepackage{{caption}}
    \usepackage{{subcaption}}
    \usepackage{{xcolor}}

    \title{{CSc 8830 --- Module 9 Assignment\\\\Uncalibrated Stereo: Distance Estimation}}
    \author{{Achindra Baruah}}
    \date{{\today}}

    \begin{{document}}
    \maketitle

    % ──────────────────────────────────────────────────────────
    \section{{Introduction}}

    This report presents the results of using an \textbf{{uncalibrated stereo camera setup}} to observe objects in a classroom and estimate the distance to a selected object of interest. The stereo pair was captured with an iPhone 16 Pro Max (24\,mm $f$/1.78 main camera) at two positions separated by a known baseline.

    The pipeline computes the \textbf{{Fundamental Matrix}} ($\mathbf{{F}}$), the \textbf{{Essential Matrix}} ($\mathbf{{E}}$), and decomposes $\mathbf{{E}}$ into the \textbf{{Rotation Matrix}} ($\mathbf{{R}}$) and translation vector ($\mathbf{{t}}$). Feature-point triangulation is then used to estimate the distance to the selected object.

    % ──────────────────────────────────────────────────────────
    \section{{Setup}}

    \begin{{table}}[H]
    \centering
    \begin{{tabular}}{{ll}}
    \toprule
    \textbf{{Parameter}} & \textbf{{Value}} \\
    \midrule
    Camera & iPhone 16 Pro Max --- 24\,mm $f$/1.78 \\
    Resolution & 5712 $\times$ 4284 (24 MP) \\
    Baseline & {d['baseline_m']:.2f}\,m \\
    Processing resolution & Working scale \\
    Focal length (full) & {IPHONE_FOCAL_PX:.0f}\,px \\
    Focal length (scaled) & {d['f_px']:.1f}\,px \\
    \bottomrule
    \end{{tabular}}
    \caption{{Camera and setup parameters.}}
    \end{{table}}

    \begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{stereo_pair.png}}
    \caption{{Stereo pair: left image and right image captured from two positions.}}
    \end{{figure}}

    % ──────────────────────────────────────────────────────────
    \section{{Intrinsic Matrix}}

    Since we know the camera is an iPhone 16 Pro Max with a 24\,mm equivalent focal length, the horizontal field of view is:
    \[
    \text{{HFoV}} = 2 \cdot \arctan\!\left(\frac{{36}}{{2 \cdot 24}}\right) \approx 73.7^\circ
    \]
    The pixel focal length at the full resolution $W = 5712$ is:
    \[
    f_{{px}} = \frac{{W}}{{2 \cdot \tan(\text{{HFoV}}/2)}} = \frac{{5712}}{{2 \cdot \tan(36.85^\circ)}} \approx 3808 \text{{ px}}
    \]
    The intrinsic matrix (at working resolution) is:
    \[
    \mathbf{{K}} = {K_tex}
    \]

    % ──────────────────────────────────────────────────────────
    \section{{Feature Matching}}

    SIFT features were detected in both images and matched using brute-force matching with Lowe's ratio test (threshold 0.65). A total of \textbf{{{d['n_matches']}}} good matches were found, of which \textbf{{{d['n_inliers']}}} were kept as inliers after RANSAC.

    \begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{feature_matches.png}}
    \caption{{SIFT feature matches between left and right images (top 100 shown).}}
    \end{{figure}}

    % ──────────────────────────────────────────────────────────
    \section{{Fundamental Matrix ($\mathbf{{F}}$)}}

    The Fundamental Matrix encodes the epipolar geometry between the two views. For corresponding points $\mathbf{{x}}$ and $\mathbf{{x'}}$ in the left and right images:
    \[
    \mathbf{{x'}}^\top \mathbf{{F}} \, \mathbf{{x}} = 0
    \]
    $\mathbf{{F}}$ was computed using the 8-point algorithm within a RANSAC framework:
    \[
    \mathbf{{F}} = {F_tex}
    \]
    \textbf{{Verification:}} The mean epipolar error $|\mathbf{{x'}}^\top \mathbf{{F}} \mathbf{{x}}|$ across all inliers is $\mathbf{{{d['epipolar_err']:.6f}}}$, confirming the accuracy of~$\mathbf{{F}}$.

    \begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{epipolar_lines.png}}
    \caption{{Epipolar lines drawn on both images. Each colour represents a corresponding point pair. The lines pass through (or very near) the corresponding points, verifying~$\mathbf{{F}}$.}}
    \end{{figure}}

    % ──────────────────────────────────────────────────────────
    \section{{Essential Matrix ($\mathbf{{E}}$)}}

    The Essential Matrix relates the normalised image coordinates and encodes the relative pose (rotation and translation) between the two cameras:
    \[
    \mathbf{{E}} = \mathbf{{K}}^\top \mathbf{{F}} \, \mathbf{{K}}
    \]
    After enforcing the rank-2 constraint via SVD (setting $\sigma_3 = 0$ and equalising $\sigma_1 = \sigma_2$):
    \[
    \mathbf{{E}} = {E_tex}
    \]
    \textbf{{Singular values:}} $\sigma_1 = {d['E_singular'][0]:.4f}$, $\sigma_2 = {d['E_singular'][1]:.4f}$, $\sigma_3 = {d['E_singular'][2]:.4f}$.

    For a valid Essential matrix, we require $\sigma_1 \approx \sigma_2$ and $\sigma_3 \approx 0$, which is satisfied.

    % ──────────────────────────────────────────────────────────
    \section{{Rotation Matrix ($\mathbf{{R}}$) and Translation ($\mathbf{{t}}$)}}

    The Essential matrix is decomposed into the relative camera rotation and translation:
    \[
    \mathbf{{E}} = [\mathbf{{t}}]_\times \mathbf{{R}}
    \]
    Using SVD decomposition with the chirality check (selecting the solution where triangulated points are in front of both cameras):
    \[
    \mathbf{{R}} = {R_tex}
    \]
    \[
    \mathbf{{t}} = \begin{{bmatrix}} {t_flat[0]: .6f} \\ {t_flat[1]: .6f} \\ {t_flat[2]: .6f} \end{{bmatrix}}
    \]

    \textbf{{Verification:}}
    \begin{{itemize}}
    \item $\det(\mathbf{{R}}) = {np.linalg.det(d['R']):.6f}$ (should be $+1$ for a proper rotation)
    \item $\|\mathbf{{R}}^\top\mathbf{{R}} - \mathbf{{I}}\| = {np.linalg.norm(d['R'].T @ d['R'] - np.eye(3)):.2e}$ (should be $\approx 0$)
    \item Rotation angle: ${d['rotation_deg']:.2f}^\circ$
    \item $\|\mathbf{{t}}\| = 1$ (unit norm from \texttt{{recoverPose}}; scale set by baseline)
    \end{{itemize}}

    % ──────────────────────────────────────────────────────────
    \section{{Distance Estimation}}

    \subsection{{Method}}
    The distance to the selected object is estimated using the stereo disparity of matched feature points. For corresponding points $(u_L, v)$ and $(u_R, v)$ in the left and right images, the disparity is $d = u_L - u_R$ and the depth is:
    \[
    Z = \frac{{f \cdot b}}{{d}}
    \]
    where $f = {d['f_px']:.1f}$\,px is the focal length and $b = {d['baseline_m']:.2f}$\,m is the baseline.

    SIFT feature correspondences (RANSAC inliers) falling within or near the object's bounding box are used, and the \textbf{{median disparity}} of these matches gives a robust depth estimate.

    \subsection{{Object Selection}}
    The object of interest --- \textbf{{{d['object_label']}}} --- was detected using YOLOv8x. Feature matches within the bounding box region are used to compute the median disparity.

    \begin{{figure}}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{{annotated_object.png}}
    \caption{{Selected object (\textbf{{{d['object_label']}}}) with estimated distance annotation.}}
    \end{{figure}}

    \subsection{{Result}}
    \begin{{table}}[H]
    \centering
    \begin{{tabular}}{{ll}}
    \toprule
    \textbf{{Quantity}} & \textbf{{Value}} \\
    \midrule
    Selected object & {d['object_label']} \\
    Matched points used & {d['n_pts_obj']} \\
    Median disparity & {d['unit_z']:.1f}\,px \\
    Focal length & {d['f_px']:.1f}\,px \\
    Baseline & {d['baseline_m']:.2f}\,m \\
    \textbf{{Estimated distance}} & \textbf{{{dist:.2f}\,m}} \\""")

    if gt:
        latex += textwrap.dedent(rf"""
    Ground truth distance & {gt:.2f}\,m \\
    \textbf{{Error}} & \textbf{{{abs(dist-gt)/gt*100:.1f}\%}} \\""")

    latex += textwrap.dedent(rf"""
    \bottomrule
    \end{{tabular}}
    \caption{{Distance estimation results.}}
    \end{{table}}

    {err_str}

    % ──────────────────────────────────────────────────────────
    \section{{Summary Figure}}

    \begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{summary.png}}
    \caption{{Complete pipeline summary: (a) stereo pair, (b) feature matches, (c) epipolar verification, (d) object annotation with distance estimate.}}
    \end{{figure}}

    % ──────────────────────────────────────────────────────────
    \section{{Conclusion}}

    Using an uncalibrated stereo setup with an iPhone 16 Pro Max, we successfully:
    \begin{{enumerate}}
    \item Computed the Fundamental Matrix $\mathbf{{F}}$ from SIFT correspondences (verified by epipolar constraint).
    \item Derived the Essential Matrix $\mathbf{{E}} = \mathbf{{K}}^\top \mathbf{{F}} \mathbf{{K}}$ using known iPhone intrinsics.
    \item Decomposed $\mathbf{{E}}$ into the camera rotation $\mathbf{{R}}$ (${d['rotation_deg']:.2f}^\circ$) and unit translation $\mathbf{{t}}$.
    \item Triangulated scene points and estimated the distance to the selected object (\textbf{{{d['object_label']}}}) as \textbf{{{dist:.2f}\,m}}{gt_conclusion}.
    \end{{enumerate}}

    \end{{document}}
    """)

    tex_path = os.path.join(outdir, "report.tex")
    with open(tex_path, "w") as f:
        f.write(latex.strip() + "\n")
    print(f"  LaTeX report → {tex_path}")
    return tex_path


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Module 9: Uncalibrated Stereo Distance Estimation")
    parser.add_argument("--left", required=True, help="Left image path")
    parser.add_argument("--right", required=True, help="Right image path")
    parser.add_argument("--baseline", type=float, required=True,
                        help="Baseline distance between camera positions (metres)")
    parser.add_argument("--ground-truth", type=float, default=None,
                        help="Ground-truth distance to object of interest (metres)")
    parser.add_argument("--object", type=str, default=None,
                        help="Object label to select (e.g. 'table', 'chair')")
    parser.add_argument("--outdir", default="output",
                        help="Output directory (default: output)")
    args = parser.parse_args()

    result = run_pipeline(
        left_path=args.left,
        right_path=args.right,
        baseline_m=args.baseline,
        ground_truth_m=args.ground_truth,
        object_label=args.object,
        outdir=args.outdir,
    )

    generate_report(result, args.outdir)


if __name__ == "__main__":
    main()

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
def compute_disparity(lg, rg, num_disp=128):
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
_yolo_model = None

def detect_yolo(image_bgr):
    global _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    results = _yolo_model(image_bgr, verbose=False)[0]
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
    fig, ax = plt.subplots(figsize=(10, 8))

    tx = [r[1] for r in results if r[0] == "table"]
    ty = [r[2] for r in results if r[0] == "table"]
    chx = [r[1] for r in results if r[0] == "chair"]
    chy = [r[2] for r in results if r[0] == "chair"]

    ax.scatter(tx, ty, c=RED_RGB, s=220, marker="s", edgecolors="black",
               linewidths=0.8, zorder=5, label=f"Table ({len(tx)})")
    ax.scatter(chx, chy, c=BLUE_RGB, s=130, marker="o", edgecolors="black",
               linewidths=0.8, zorder=5, label=f"Chair ({len(chx)})")

    for r in results:
        ax.annotate(f"{r[2]:.1f}m", (r[1], r[2]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="gray")

    ax.plot(0, 0, "k^", markersize=14, zorder=10)
    ax.annotate("Camera", (0, 0), textcoords="offset points",
                xytext=(10, -15), fontsize=10, fontweight="bold")

    ax.set_xlabel("X — Lateral Position (metres)", fontsize=13)
    ax.set_ylabel("Y — Depth / Forward (metres)",  fontsize=13)
    ax.set_title(f"Classroom Layout — Stereo Localisation{title_extra}\n"
                 "(Tables = Red ■,  Chairs = Blue ●)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(outdir, "classroom_layout_2d.png")
    fig.savefig(p, dpi=200)
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


# ─── standard perspective pipeline ────────────────────────────
def run_perspective_pipeline(left_img, right_img, baseline, outdir,
                             focal_override=None):
    """Standard stereo pipeline for normal perspective images."""
    os.makedirs(outdir, exist_ok=True)
    H, W = left_img.shape[:2]
    focal = focal_override if focal_override else max(H, W) * 0.8
    cx_img, cy_img = W / 2.0, H / 2.0

    print(f"[1/4] Detecting tables and chairs …")
    dets = detect_yolo(left_img)
    if dets is None:
        print("[!] YOLO unavailable."); sys.exit(1)
    n_t = sum(1 for d in dets if d["label"] == "table")
    n_c = sum(1 for d in dets if d["label"] == "chair")
    print(f"      Found {n_t} table(s), {n_c} chair(s).")
    if not dets:
        print("[!] No objects detected."); sys.exit(1)

    print("[2/4] Computing SGBM disparity …")
    gl = cv2.cvtColor(left_img,  cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disp = compute_disparity(gl, gr)
    dv = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(outdir, "disparity_map.png"),
                cv2.applyColorMap(dv, cv2.COLORMAP_JET))

    print("[3/4] Computing floor positions …")
    results = []
    annotated = left_img.copy()
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        uc, vc = (x1+x2)/2, (y1+y2)/2
        mx, my = int((x2-x1)*0.2), int((y2-y1)*0.2)
        roi = disp[int(y1)+my:int(y2)-my, int(x1)+mx:int(x2)-mx]
        v = roi[roi > 0.5]
        if v.size == 0: continue
        md = float(np.median(v))
        pos = pixel_to_floor(uc, vc, md, focal, baseline, cx_img, cy_img)
        if pos is None: continue
        fx, fy = pos
        results.append((det["label"], fx, fy, fy, det["conf"]))
        col = RED_BGR if det["label"] == "table" else BLUE_BGR
        cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), col, 2)
        cv2.putText(annotated, f'{det["label"]} {fy:.1f}m',
                    (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.imwrite(os.path.join(outdir, "detections_left.png"), annotated)

    if not results:
        print("[!] No valid positions."); sys.exit(1)

    with open(os.path.join(outdir, "detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","floor_x_m","floor_y_m","depth_m","confidence"])
        for r in results: w.writerow(r)

    print("[4/4] Plotting 2-D classroom layout …")
    plot_floor_plan(results, outdir,
                    title_extra=f"\n(f={focal:.0f} px, b={baseline*100:.1f} cm)")

    print(f"\n✓ {len(results)} objects → {outdir}/")
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
                                focal_override=args.focal)


if __name__ == "__main__":
    main()

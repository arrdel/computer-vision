#!/usr/bin/env python3
"""
Module 7 — Part 1: Optical Flow Computation, Visualization & Motion Tracking
=============================================================================

This script:
1. Reads two input videos and processes 30-second clips.
2. Computes dense optical flow (Farnebäck) and sparse optical flow (Lucas-Kanade).
3. Saves visualisation videos (colour-coded flow field + tracked feature points).
4. Analyses what information can be inferred from optical flow (with evidence).
5. Demonstrates the Lucas-Kanade tracking equations from first principles.
6. Derives and demonstrates bilinear interpolation with numerical examples.
7. Picks two consecutive frames from each video and validates the theoretical
   tracking result against actual pixel locations.

Usage:
    python optical_flow.py --video1 videos/video1.mp4 --video2 videos/video2.mp4
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Dense Optical Flow (Farnebäck) colour visualisation
# ──────────────────────────────────────────────────────────────────────────────
def flow_to_colour(flow):
    """Convert a 2-channel (dx, dy) flow field to an HSV->BGR colour image."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2        # Hue   = direction
    hsv[..., 1] = 255                            # Saturation = max
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(gray, flow, step=20, scale=3):
    """Draw sparse quiver arrows on a gray image for flow visualisation."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            cv2.arrowedLine(vis, (x, y),
                            (int(x + fx * scale), int(y + fy * scale)),
                            (0, 255, 0), 1, tipLength=0.3)
    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Draw sparse (Lucas-Kanade) tracks
# ──────────────────────────────────────────────────────────────────────────────
def draw_tracks(frame, good_new, good_old, mask, colour=(0, 255, 0)):
    """Draw tracked feature-point trails on *frame* using accumulated *mask*."""
    for new, old in zip(good_new, good_old):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), colour, 2)
        frame = cv2.circle(frame, (a, b), 4, (0, 0, 255), -1)
    return cv2.add(frame, mask), mask


# ──────────────────────────────────────────────────────────────────────────────
# Part A — Compute & save optical-flow videos
# ──────────────────────────────────────────────────────────────────────────────
def compute_optical_flow_video(video_path: str, out_dir: str, label: str,
                               start_sec: float = 0, duration: float = 30):
    """Compute dense + sparse optical flow and write visualisation videos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame = min(int((start_sec + duration) * fps), total_frames)

    # Scale down for speed if the video is very large
    scale = 1.0
    if w > 960:
        scale = 960 / w
        w = 960
        h = int(h * scale)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    os.makedirs(out_dir, exist_ok=True)

    dense_out    = os.path.join(out_dir, f"{label}_dense_flow.mp4")
    sparse_out   = os.path.join(out_dir, f"{label}_sparse_flow.mp4")
    arrows_out   = os.path.join(out_dir, f"{label}_arrows_flow.mp4")
    combined_out = os.path.join(out_dir, f"{label}_combined_flow.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dense_writer    = cv2.VideoWriter(dense_out, fourcc, fps, (w, h))
    sparse_writer   = cv2.VideoWriter(sparse_out, fourcc, fps, (w, h))
    arrows_writer   = cv2.VideoWriter(arrows_out, fourcc, fps, (w, h))
    combined_writer = cv2.VideoWriter(combined_out, fourcc, fps, (w * 2, h))

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        return None, None
    if scale != 1.0:
        frame = cv2.resize(frame, (w, h))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # LK parameters
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=7, blockSize=7)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    track_mask = np.zeros_like(frame)

    frame_idx = start_frame + 1
    pair_frames = {}
    all_mags = []
    all_mean_angles = []

    print(f"[{label}] Processing frames {start_frame} -> {end_frame} "
          f"({duration}s @ {fps:.1f} fps, {w}x{h}) ...")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (w, h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dense (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        dense_vis = flow_to_colour(flow)
        dense_writer.write(dense_vis)

        # Arrow visualisation
        arrow_vis = draw_flow_arrows(prev_gray, flow, step=24, scale=3)
        arrows_writer.write(arrow_vis)

        # Sparse (Lucas-Kanade)
        sparse_vis = frame.copy()
        if prev_pts is not None and len(prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params)
            if next_pts is not None:
                good_new = next_pts[status.flatten() == 1]
                good_old = prev_pts[status.flatten() == 1]
                sparse_vis, track_mask = draw_tracks(
                    sparse_vis, good_new, good_old, track_mask)
                prev_pts = good_new.reshape(-1, 1, 2)
            else:
                prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
                track_mask = np.zeros_like(frame)
        else:
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
            track_mask = np.zeros_like(frame)
        sparse_writer.write(sparse_vis)

        # Combined side-by-side
        combined = np.hstack([dense_vis, sparse_vis])
        combined_writer.write(combined)

        # Statistics
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        all_mags.append(mag.mean())
        all_mean_angles.append(ang.mean())

        # Save a consecutive pair at the middle of the clip for validation
        mid_frame = (start_frame + end_frame) // 2
        if frame_idx == mid_frame:
            pair_frames["prev_gray"] = prev_gray.copy()
            pair_frames["prev_bgr"] = frame.copy()
        if frame_idx == mid_frame + 1:
            pair_frames["curr_gray"] = curr_gray.copy()
            pair_frames["curr_bgr"] = frame.copy()
            pair_frames["flow"] = flow.copy()
            pair_frames["frame_idx_prev"] = mid_frame
            pair_frames["frame_idx_curr"] = mid_frame + 1

        prev_gray = curr_gray
        frame_idx += 1

        # Re-detect features periodically
        if frame_idx % int(fps * 2) == 0:
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
            track_mask = np.zeros_like(frame)

    cap.release()
    for wr in [dense_writer, sparse_writer, arrows_writer, combined_writer]:
        wr.release()

    # Plot flow magnitude over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    frames_range = range(len(all_mags))
    ax1.plot(frames_range, all_mags, linewidth=0.8, color="steelblue")
    ax1.set_ylabel("Mean flow magnitude (px)")
    ax1.set_title(f"{label} -- Optical flow statistics over time")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(np.mean(all_mags), color="red", ls="--", lw=0.8,
                label=f"avg = {np.mean(all_mags):.2f} px")
    ax1.legend()

    ax2.plot(frames_range, np.degrees(all_mean_angles), linewidth=0.8, color="darkorange")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Mean flow direction (deg)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_flow_magnitude.png"), dpi=150)
    plt.close(fig)

    print(f"[{label}] Saved dense    -> {dense_out}")
    print(f"[{label}] Saved sparse   -> {sparse_out}")
    print(f"[{label}] Saved arrows   -> {arrows_out}")
    print(f"[{label}] Saved combined -> {combined_out}")

    return pair_frames, all_mags


# ──────────────────────────────────────────────────────────────────────────────
# Part B — Flow analysis: what information can be inferred?
# ──────────────────────────────────────────────────────────────────────────────
def analyse_flow_inference(pair_frames, out_dir, label):
    """
    Analyse a flow field to demonstrate what information can be inferred:
      1. Motion direction & speed of objects
      2. Moving vs static regions (foreground / background segmentation)
      3. Divergence  (expansion / contraction -- looming)
      4. Curl  (rotational motion)
      5. Flow histogram  (dominant motion statistics)
    """
    if not pair_frames or "flow" not in pair_frames:
        return

    flow   = pair_frames["flow"]
    prev_g = pair_frames["prev_gray"]
    curr_g = pair_frames["curr_gray"]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 6-panel analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].imshow(prev_g, cmap="gray")
    axes[0, 0].set_title("(a) Original frame t", fontsize=11)

    im1 = axes[0, 1].imshow(mag, cmap="hot")
    axes[0, 1].set_title("(b) Flow magnitude = speed of motion", fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(ang, cmap="hsv")
    axes[0, 2].set_title("(c) Flow direction (angle)", fontsize=11)
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Motion mask
    motion_thresh = mag.mean() + 1.5 * mag.std()
    motion_mask = (mag > motion_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    axes[1, 0].imshow(motion_mask, cmap="gray")
    axes[1, 0].set_title(f"(d) Motion mask (thresh={motion_thresh:.2f} px)", fontsize=11)

    # Divergence
    du_dx = np.gradient(flow[..., 0], axis=1)
    dv_dy = np.gradient(flow[..., 1], axis=0)
    divergence = du_dx + dv_dy
    im3 = axes[1, 1].imshow(np.clip(divergence, -2, 2), cmap="RdBu_r")
    axes[1, 1].set_title("(e) Divergence (red=expansion, blue=contraction)", fontsize=11)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    # Curl
    dv_dx = np.gradient(flow[..., 1], axis=1)
    du_dy = np.gradient(flow[..., 0], axis=0)
    curl = dv_dx - du_dy
    im4 = axes[1, 2].imshow(np.clip(curl, -2, 2), cmap="PiYG")
    axes[1, 2].set_title("(f) Curl (green=CW, pink=CCW rotation)", fontsize=11)
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{label} -- Information inferred from optical flow", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_flow_analysis.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Flow histogram
    fig2, (ax_m, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    ax_m.hist(mag.flatten(), bins=100, range=(0, min(mag.max(), 20)),
              color="steelblue", edgecolor="none", alpha=0.8)
    ax_m.set_xlabel("Flow magnitude (px)")
    ax_m.set_ylabel("Pixel count")
    ax_m.set_title("Flow magnitude histogram")
    ax_m.axvline(mag.mean(), color="red", ls="--", label=f"mean={mag.mean():.2f}")
    ax_m.legend()

    ax_a.hist(np.degrees(ang.flatten()), bins=72, range=(0, 360),
              color="darkorange", edgecolor="none", alpha=0.8)
    ax_a.set_xlabel("Flow direction (deg)")
    ax_a.set_ylabel("Pixel count")
    ax_a.set_title("Flow direction histogram")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{label}_flow_histograms.png"), dpi=150)
    plt.close(fig2)

    # Overlay moving objects on original
    overlay = cv2.cvtColor(prev_g, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_moving = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            num_moving += 1
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            roi_flow = flow[y:y+bh, x:x+bw]
            avg_dx = roi_flow[..., 0].mean()
            avg_dy = roi_flow[..., 1].mean()
            speed = np.sqrt(avg_dx**2 + avg_dy**2)
            direction = np.degrees(np.arctan2(avg_dy, avg_dx))
            cv2.putText(overlay, f"{speed:.1f}px {direction:.0f}deg",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(out_dir, f"{label}_moving_objects.png"), overlay)

    # Write inference summary
    summary = textwrap.dedent(f"""\
    Optical Flow Inference Summary -- {label}
    ================================================

    1. MOTION SPEED & DIRECTION
       Mean magnitude: {mag.mean():.3f} px/frame
       Max  magnitude: {mag.max():.3f} px/frame
       Dominant direction: {np.degrees(ang[mag > mag.mean()].mean()):.1f} deg
       -> Pixels with high magnitude indicate fast-moving objects/regions.
       -> The HSV colour map encodes direction (hue) and speed (brightness).

    2. FOREGROUND / BACKGROUND SEGMENTATION
       Threshold used: {motion_thresh:.2f} px
       Moving pixels: {(motion_mask > 0).sum()} / {motion_mask.size} ({100*(motion_mask>0).sum()/motion_mask.size:.1f}%)
       Static pixels: {(motion_mask == 0).sum()} / {motion_mask.size}
       -> The motion mask separates moving foreground from static background.
       -> Detected {num_moving} moving object regions.

    3. DIVERGENCE (expansion / contraction)
       Mean divergence: {divergence.mean():.4f}
       Positive -> objects approaching camera (expansion / looming)
       Negative -> objects receding (contraction)

    4. CURL (rotation)
       Mean curl: {curl.mean():.4f}
       Non-zero curl indicates rotational motion in the scene.

    5. TIME-TO-CONTACT
       For forward camera motion, the Focus of Expansion (FOE) is the point
       where flow vectors radiate outward. Time to contact ~ distance / speed.
    """)
    print(summary)
    with open(os.path.join(out_dir, f"{label}_flow_inference.txt"), "w") as f:
        f.write(summary)

    print(f"[{label}] Flow analysis saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Part C — Manual Lucas-Kanade from first principles (for validation)
# ──────────────────────────────────────────────────────────────────────────────
def manual_lucas_kanade(prev_gray, curr_gray, points, win_size=21):
    """
    Implement single-scale Lucas-Kanade tracker from scratch to demonstrate
    the theoretical equations.

    For each point, we solve:
        A^T A [u; v] = A^T b
    where A_i = [Ix_i, Iy_i] and b_i = -It_i for pixels in the window.
    """
    half_w = win_size // 2
    h, w = prev_gray.shape
    prev_f = prev_gray.astype(np.float64)
    curr_f = curr_gray.astype(np.float64)

    # Compute gradients
    Ix = cv2.Sobel(prev_f, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(prev_f, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    It = curr_f - prev_f

    predicted = []
    details = []

    for pt in points:
        px, py = int(round(pt[0, 0])), int(round(pt[0, 1]))

        if (px - half_w < 0 or px + half_w >= w or
            py - half_w < 0 or py + half_w >= h):
            predicted.append(pt.copy())
            details.append(None)
            continue

        ix_win = Ix[py - half_w:py + half_w + 1, px - half_w:px + half_w + 1].flatten()
        iy_win = Iy[py - half_w:py + half_w + 1, px - half_w:px + half_w + 1].flatten()
        it_win = It[py - half_w:py + half_w + 1, px - half_w:px + half_w + 1].flatten()

        A = np.column_stack([ix_win, iy_win])
        b = -it_win

        ATA = A.T @ A
        ATb = A.T @ b

        eigvals = np.linalg.eigvalsh(ATA)
        if eigvals[0] < 1e-4:
            predicted.append(pt.copy())
            details.append({"ATA": ATA, "ATb": ATb, "eigvals": eigvals,
                            "u": 0, "v": 0, "solvable": False})
            continue

        uv = np.linalg.solve(ATA, ATb)
        u, v = uv[0], uv[1]

        new_pt = pt.copy()
        new_pt[0, 0] += u
        new_pt[0, 1] += v
        predicted.append(new_pt)
        details.append({"ATA": ATA, "ATb": ATb, "eigvals": eigvals,
                        "u": u, "v": v, "solvable": True})

    return np.array(predicted), details


# ──────────────────────────────────────────────────────────────────────────────
# Part D — Bilinear Interpolation demonstration
# ──────────────────────────────────────────────────────────────────────────────
def demonstrate_bilinear_interpolation(gray_img, x, y, out_dir, label):
    """
    Demonstrate bilinear interpolation at sub-pixel location (x, y).
    Show full step-by-step derivation and compare with OpenCV remap.
    """
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    a = x - x0
    b = y - y0

    I00 = float(gray_img[y0, x0])
    I10 = float(gray_img[y0, x1])
    I01 = float(gray_img[y1, x0])
    I11 = float(gray_img[y1, x1])

    # Step 1: interpolate along x
    f_top    = (1 - a) * I00 + a * I10
    f_bottom = (1 - a) * I01 + a * I11

    # Step 2: interpolate along y
    I_interp = (1 - b) * f_top + b * f_bottom

    # Expanded form check
    I_expanded = ((1-a)*(1-b)*I00 + a*(1-b)*I10 +
                  (1-a)*b*I01 + a*b*I11)

    # OpenCV verification
    map_x = np.array([[x]], dtype=np.float32)
    map_y = np.array([[y]], dtype=np.float32)
    cv_result = float(cv2.remap(gray_img, map_x, map_y, cv2.INTER_LINEAR)[0, 0])

    report = textwrap.dedent(f"""\
    ================================================================
      BILINEAR INTERPOLATION -- {label}
    ================================================================

      Sub-pixel location: ({x:.4f}, {y:.4f})

      Four corner pixels:
        I({x0:3d}, {y0:3d}) = {I00:6.1f}    I({x1:3d}, {y0:3d}) = {I10:6.1f}
        I({x0:3d}, {y1:3d}) = {I01:6.1f}    I({x1:3d}, {y1:3d}) = {I11:6.1f}

      Fractional offsets:
        a = x - floor(x) = {x:.4f} - {x0} = {a:.4f}
        b = y - floor(y) = {y:.4f} - {y0} = {b:.4f}

      Step 1 -- Interpolate along x:
        f(x, y0) = (1-a)*I(x0,y0) + a*I(x1,y0)
                 = {1-a:.4f} * {I00:.1f} + {a:.4f} * {I10:.1f}
                 = {f_top:.4f}

        f(x, y1) = (1-a)*I(x0,y1) + a*I(x1,y1)
                 = {1-a:.4f} * {I01:.1f} + {a:.4f} * {I11:.1f}
                 = {f_bottom:.4f}

      Step 2 -- Interpolate along y:
        I(x, y) = (1-b)*f(x,y0) + b*f(x,y1)
                = {1-b:.4f} * {f_top:.4f} + {b:.4f} * {f_bottom:.4f}
                = {I_interp:.4f}

      Expanded formula verification:
        (1-a)(1-b)*I00 + a(1-b)*I10 + (1-a)b*I01 + ab*I11
        = {(1-a)*(1-b):.4f}*{I00:.1f} + {a*(1-b):.4f}*{I10:.1f}
          + {(1-a)*b:.4f}*{I01:.1f} + {a*b:.4f}*{I11:.1f}
        = {I_expanded:.4f}

      OpenCV remap result: {cv_result:.4f}
      Difference:          {abs(I_interp - cv_result):.8f}
    ================================================================
    """)
    print(report)

    # Save zoomed figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pad = 4
    r0, r1 = max(y0 - pad, 0), min(y1 + pad + 1, gray_img.shape[0])
    c0, c1 = max(x0 - pad, 0), min(x1 + pad + 1, gray_img.shape[1])
    patch = gray_img[r0:r1, c0:c1]
    ax1.imshow(patch, cmap="gray", interpolation="nearest",
               extent=[c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5])
    ax1.plot(x, y, "r+", markersize=18, markeredgewidth=2.5,
             label=f"Query ({x:.2f}, {y:.2f})")
    for px, py, val in [(x0, y0, I00), (x1, y0, I10), (x0, y1, I01), (x1, y1, I11)]:
        ax1.plot(px, py, "cs", markersize=10)
        ax1.annotate(f"{val:.0f}", (px, py), textcoords="offset points",
                     xytext=(6, 6), fontsize=10, color="cyan", fontweight="bold")
    ax1.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "c--", lw=1.5, alpha=0.7)
    ax1.set_title(f"Pixel neighbourhood\nResult = {I_interp:.2f}")
    ax1.legend(fontsize=9)

    # Weight diagram
    weights = np.array([[(1-a)*(1-b), a*(1-b)], [(1-a)*b, a*b]])
    im = ax2.imshow(weights, cmap="YlOrRd", interpolation="nearest",
                    extent=[-0.5, 1.5, 1.5, -0.5])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"w={weights[i,j]:.3f}\nI={[I00,I10,I01,I11][i*2+j]:.0f}",
                     ha="center", va="center", fontsize=11, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"x0={x0}", f"x1={x1}"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([f"y0={y0}", f"y1={y1}"])
    ax2.set_title("Interpolation weights")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    fig.suptitle(f"Bilinear Interpolation Demonstration -- {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_bilinear_interp.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, f"{label}_bilinear_interp.txt"), "w") as f:
        f.write(report)

    return I_interp, cv_result


# ──────────────────────────────────────────────────────────────────────────────
# Part E — Validate tracking: LK theory vs actual pixel locations
# ──────────────────────────────────────────────────────────────────────────────
def validate_tracking(pair_frames, out_dir, label):
    """
    For two consecutive frames:
    1. Detect features in frame t.
    2. Compute predicted location in frame t+1 using:
       a) Manual single-scale LK (theoretical derivation).
       b) OpenCV pyramidal LK.
    3. Find ground truth via template matching (NCC).
    4. Also compare with dense flow prediction.
    5. Validate theory against all methods.
    """
    if not pair_frames or "prev_gray" not in pair_frames:
        print(f"[{label}] No frame pair available -- skipping validation.")
        return

    prev_g = pair_frames["prev_gray"]
    curr_g = pair_frames["curr_gray"]
    flow   = pair_frames["flow"]

    feature_params = dict(maxCorners=30, qualityLevel=0.3, minDistance=20, blockSize=7)
    pts = cv2.goodFeaturesToTrack(prev_g, mask=None, **feature_params)
    if pts is None or len(pts) == 0:
        print(f"[{label}] No features detected -- skipping validation.")
        return

    # Keep only interior points
    h, w = prev_g.shape
    margin = 25
    valid = []
    for pt in pts:
        px, py = pt[0]
        if margin < px < w - margin and margin < py < h - margin:
            valid.append(pt)
    pts = np.array(valid[:20])
    if len(pts) == 0:
        print(f"[{label}] No interior features -- skipping.")
        return

    # Method 1: Manual single-scale LK
    pts_manual, lk_details = manual_lucas_kanade(prev_g, curr_g, pts, win_size=21)

    # Method 2: OpenCV pyramidal LK
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts_opencv, status_cv, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts, None, **lk_params)

    # Method 3: Dense flow prediction
    pts_dense = np.zeros_like(pts)
    for i, pt in enumerate(pts):
        px, py = int(round(pt[0, 0])), int(round(pt[0, 1]))
        px = np.clip(px, 0, flow.shape[1] - 1)
        py = np.clip(py, 0, flow.shape[0] - 1)
        dx, dy = flow[py, px]
        pts_dense[i, 0, 0] = pt[0, 0] + dx
        pts_dense[i, 0, 1] = pt[0, 1] + dy

    # Method 4: Template matching ground truth
    pts_template = np.zeros_like(pts)
    template_size = 21
    half_t = template_size // 2
    search_size = 51
    half_s = search_size // 2
    for i, pt in enumerate(pts):
        px, py = int(round(pt[0, 0])), int(round(pt[0, 1]))
        t_r0 = max(py - half_t, 0)
        t_r1 = min(py + half_t + 1, h)
        t_c0 = max(px - half_t, 0)
        t_c1 = min(px + half_t + 1, w)
        template = prev_g[t_r0:t_r1, t_c0:t_c1]

        s_r0 = max(py - half_s, 0)
        s_r1 = min(py + half_s + 1, h)
        s_c0 = max(px - half_s, 0)
        s_c1 = min(px + half_s + 1, w)
        search = curr_g[s_r0:s_r1, s_c0:s_c1]

        if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
            pts_template[i] = pts_dense[i]
            continue

        res = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        best_x = s_c0 + max_loc[0] + half_t
        best_y = s_r0 + max_loc[1] + half_t
        pts_template[i, 0, 0] = best_x
        pts_template[i, 0, 1] = best_y

    # Compute errors
    err_manual_vs_template = np.sqrt(np.sum((pts_manual - pts_template) ** 2, axis=(1, 2)))
    err_opencv_vs_template = np.sqrt(np.sum((pts_opencv - pts_template) ** 2, axis=(1, 2)))
    err_manual_vs_opencv   = np.sqrt(np.sum((pts_manual - pts_opencv) ** 2, axis=(1, 2)))
    err_dense_vs_template  = np.sqrt(np.sum((pts_dense - pts_template) ** 2, axis=(1, 2)))
    disp_opencv = np.sqrt(np.sum((pts_opencv - pts) ** 2, axis=(1, 2)))

    # Build report
    lines = [
        f"\n{'='*100}",
        f"  TRACKING VALIDATION -- {label}",
        f"  Frames: {pair_frames.get('frame_idx_prev','?')} -> {pair_frames.get('frame_idx_curr','?')}",
        f"{'='*100}",
        f"  Number of features tracked: {len(pts)}",
        f"",
        f"  Error comparison (pixels):",
        f"    Manual LK vs Template Match:  mean={err_manual_vs_template.mean():.3f}  max={err_manual_vs_template.max():.3f}",
        f"    OpenCV LK vs Template Match:  mean={err_opencv_vs_template.mean():.3f}  max={err_opencv_vs_template.max():.3f}",
        f"    Manual LK vs OpenCV LK:       mean={err_manual_vs_opencv.mean():.3f}  max={err_manual_vs_opencv.max():.3f}",
        f"    Dense Flow vs Template Match:  mean={err_dense_vs_template.mean():.3f}  max={err_dense_vs_template.max():.3f}",
        f"    Mean displacement (OpenCV LK): {disp_opencv.mean():.3f} px",
        f"",
        f"  {'Pt':>3}  {'Original (x,y)':>18}  {'Manual LK':>18}  {'OpenCV LK':>18}  {'Template GT':>18}  {'Err(M-T)':>9}  {'Err(O-T)':>9}",
        f"  {'-'*105}",
    ]
    for i in range(len(pts)):
        ox, oy = pts[i, 0]
        mx, my = pts_manual[i, 0]
        cx, cy = pts_opencv[i, 0]
        tx, ty = pts_template[i, 0]
        lines.append(
            f"  {i:3d}  ({ox:7.2f},{oy:7.2f})  ({mx:7.2f},{my:7.2f})  "
            f"({cx:7.2f},{cy:7.2f})  ({tx:7.2f},{ty:7.2f})  "
            f"{err_manual_vs_template[i]:8.3f}  {err_opencv_vs_template[i]:8.3f}"
        )

    # Detailed LK computation for first 3 points
    lines.append(f"\n{'─'*100}")
    lines.append("  DETAILED LUCAS-KANADE COMPUTATION (first 3 points)")
    lines.append(f"{'─'*100}")
    for i in range(min(3, len(pts))):
        d = lk_details[i]
        if d is None:
            continue
        ox, oy = pts[i, 0]
        lines.append(f"\n  Point {i}: ({ox:.2f}, {oy:.2f})")
        lines.append(f"    Structure tensor A^T A = [{d['ATA'][0,0]:12.2f}  {d['ATA'][0,1]:12.2f}]")
        lines.append(f"                            [{d['ATA'][1,0]:12.2f}  {d['ATA'][1,1]:12.2f}]")
        lines.append(f"    A^T b = [{d['ATb'][0]:12.2f}, {d['ATb'][1]:12.2f}]")
        lines.append(f"    Eigenvalues: lambda1={d['eigvals'][0]:.2f}, lambda2={d['eigvals'][1]:.2f}")
        lines.append(f"    -> Both eigenvalues {'are' if d['eigvals'][0] > 1 else 'are NOT'} large => {'good' if d['eigvals'][0] > 1 else 'poor'} corner for tracking")
        lines.append(f"    Solution: (A^T A)^-1 A^T b => u={d['u']:.4f}, v={d['v']:.4f}")
        lines.append(f"    Predicted: ({ox:.2f} + {d['u']:.4f}, {oy:.2f} + {d['v']:.4f}) = ({ox + d['u']:.4f}, {oy + d['v']:.4f})")
        lines.append(f"    Template GT:  ({pts_template[i, 0, 0]:.2f}, {pts_template[i, 0, 1]:.2f})")
        lines.append(f"    Error: {err_manual_vs_template[i]:.4f} px")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, f"{label}_tracking_validation.txt"), "w") as f:
        f.write(report)

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(prev_g, cmap="gray")
    for i, pt in enumerate(pts):
        axes[0].plot(pt[0, 0], pt[0, 1], "r.", markersize=8)
        axes[0].annotate(str(i), (pt[0, 0], pt[0, 1]), fontsize=7,
                         color="yellow", xytext=(3, 3), textcoords="offset points")
    axes[0].set_title(f"Frame t -- {len(pts)} detected features")

    axes[1].imshow(curr_g, cmap="gray")
    for i in range(len(pts)):
        mx, my = pts_manual[i, 0]
        cx, cy = pts_opencv[i, 0]
        tx, ty = pts_template[i, 0]
        axes[1].plot(mx, my, "r^", markersize=6, label="Manual LK" if i == 0 else "")
        axes[1].plot(cx, cy, "gs", markersize=5, label="OpenCV LK" if i == 0 else "")
        axes[1].plot(tx, ty, "bx", markersize=6, label="Template GT" if i == 0 else "")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title("Frame t+1 -- Predicted positions")

    axes[2].imshow(curr_g, cmap="gray")
    for i in range(len(pts)):
        ox, oy = pts[i, 0]
        cx, cy = pts_opencv[i, 0]
        axes[2].annotate("", xy=(cx, cy), xytext=(ox, oy),
                         arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
        axes[2].plot(ox, oy, "r.", markersize=6)
    axes[2].set_title("Motion vectors (OpenCV LK)")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{label} -- Tracking Validation: Theory vs Actual", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_tracking_validation.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Error histogram
    fig2, ax = plt.subplots(figsize=(8, 4))
    bins_arr = np.linspace(0, max(err_manual_vs_template.max(), err_opencv_vs_template.max(), 5), 20)
    ax.hist(err_manual_vs_template, bins=bins_arr, alpha=0.6, label="Manual LK vs Template",
            color="red", edgecolor="darkred")
    ax.hist(err_opencv_vs_template, bins=bins_arr, alpha=0.6, label="OpenCV LK vs Template",
            color="green", edgecolor="darkgreen")
    ax.axvline(err_manual_vs_template.mean(), color="red", ls="--",
               label=f"Manual mean={err_manual_vs_template.mean():.2f}")
    ax.axvline(err_opencv_vs_template.mean(), color="green", ls="--",
               label=f"OpenCV mean={err_opencv_vs_template.mean():.2f}")
    ax.set_xlabel("Tracking error (px)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} -- Tracking error distribution")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"{label}_tracking_error_hist.png"), dpi=150)
    plt.close(fig2)

    # Save frame pair
    cv2.imwrite(os.path.join(out_dir, f"{label}_frame_t.png"), prev_g)
    cv2.imwrite(os.path.join(out_dir, f"{label}_frame_t1.png"), curr_g)

    return err_manual_vs_template, err_opencv_vs_template


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Optical Flow & Motion Tracking (Module 7 -- Part 1)")
    parser.add_argument("--video1", required=True, help="Path to first video")
    parser.add_argument("--video2", required=True, help="Path to second video")
    parser.add_argument("--outdir", default="output/optical_flow",
                        help="Output directory")
    parser.add_argument("--start", type=float, default=0,
                        help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=30,
                        help="Clip duration in seconds")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for vid_path, label in [(args.video1, "video1"), (args.video2, "video2")]:
        print(f"\n{'='*70}")
        print(f"  Processing {label}: {vid_path}")
        print(f"{'='*70}")

        pair_frames, mags = compute_optical_flow_video(
            vid_path, args.outdir, label,
            start_sec=args.start, duration=args.duration)

        if pair_frames:
            # Flow analysis -- what can be inferred
            analyse_flow_inference(pair_frames, args.outdir, label)

            # Bilinear interpolation demo
            h, w = pair_frames["prev_gray"].shape
            feature_params = dict(maxCorners=5, qualityLevel=0.3,
                                  minDistance=30, blockSize=7)
            some_pts = cv2.goodFeaturesToTrack(
                pair_frames["prev_gray"], mask=None, **feature_params)
            if some_pts is not None and len(some_pts) > 0:
                bx = some_pts[0, 0, 0] + 0.37
                by = some_pts[0, 0, 1] + 0.63
            else:
                bx, by = w / 2 + 0.37, h / 2 + 0.63
            demonstrate_bilinear_interpolation(
                pair_frames["prev_gray"], bx, by, args.outdir, label)

            # Tracking validation
            validate_tracking(pair_frames, args.outdir, label)

    print(f"\n{'='*70}")
    print(f"  All optical flow processing complete.")
    print(f"  Results in: {os.path.abspath(args.outdir)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

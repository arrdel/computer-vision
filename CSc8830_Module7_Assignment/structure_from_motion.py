#!/usr/bin/env python3
"""
Module 7 -- Part 2: Structure from Motion (SfM)
===============================================

This script:
1. Loads 4 images of a flat/planar object taken from different viewpoints.
   If no images are provided, it can extract 4 frames from a video at
   different time offsets to simulate different viewpoints.
2. Detects and matches features (SIFT) across all image pairs.
3. Estimates fundamental/essential matrices and recovers camera poses.
4. Triangulates 3D points from matched features.
5. Reconstructs the boundary of the planar object and visualises the result.
6. Saves all mathematical workouts, camera parameters, and positions.

Usage:
    python structure_from_motion.py --images sfm_images/
    python structure_from_motion.py --video videos/video1.mp4  # extract frames
"""

import argparse
import glob
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from pathlib import Path
import textwrap


# ──────────────────────────────────────────────────────────────────────────────
# Camera intrinsics
# ──────────────────────────────────────────────────────────────────────────────
def build_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)


def default_K_from_image(w, h):
    """Rough intrinsic estimate: focal ~ max(w,h), principal point at centre."""
    f = max(w, h) * 1.2
    return build_K(f, f, w / 2.0, h / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# Extract frames from video as pseudo-viewpoints
# ──────────────────────────────────────────────────────────────────────────────
def extract_frames_from_video(video_path, out_dir, n_frames=4):
    """Extract n_frames evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Pick frames spread across the video to get different viewpoints
    # Use frames from the first 15 seconds with spacing
    max_frame = min(total, int(15 * fps))
    indices = np.linspace(int(fps * 1), max_frame - 1, n_frames, dtype=int)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(out_dir, f"view{i}.png")
            cv2.imwrite(path, frame)
            paths.append(path)
            print(f"  Extracted frame {idx} (t={idx/fps:.1f}s) -> {path}")
    cap.release()
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Feature detection & matching
# ──────────────────────────────────────────────────────────────────────────────
def detect_and_match(img1, img2, ratio_thresh=0.75):
    """SIFT detection + BFMatcher with Lowe's ratio test."""
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return np.array([]), np.array([]), [], kp1, kp2

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    pts1, pts2 = [], []
    for m_n in raw_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.array(pts1), np.array(pts2), good, kp1, kp2


# ──────────────────────────────────────────────────────────────────────────────
# Pose recovery
# ──────────────────────────────────────────────────────────────────────────────
def recover_pose_from_pair(pts1, pts2, K):
    """Estimate essential matrix -> R, t between two views."""
    if len(pts1) < 8:
        return None, None, None, None
    E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                     prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None, None
    _, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, K, mask=mask_e)
    return R, t, E, mask_p


# ──────────────────────────────────────────────────────────────────────────────
# Triangulation
# ──────────────────────────────────────────────────────────────────────────────
def triangulate(pts1, pts2, K, R1, t1, R2, t2):
    """Triangulate 3D points from two projection matrices."""
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])
    pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


# ──────────────────────────────────────────────────────────────────────────────
# Incremental SfM
# ──────────────────────────────────────────────────────────────────────────────
def run_sfm(images, K, out_dir):
    """
    Incremental Structure from Motion.
    View 0 is the reference (R=I, t=0).
    """
    n = len(images)
    assert n >= 2, "Need at least 2 images"

    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    h, w = grays[0].shape[:2]

    # Pairwise matching
    matches_dict = {}
    print("\n[SfM] Pairwise feature matching ...")
    for i, j in combinations(range(n), 2):
        pts_i, pts_j, good, kp_i, kp_j = detect_and_match(grays[i], grays[j])
        matches_dict[(i, j)] = (pts_i, pts_j, good, kp_i, kp_j)
        print(f"  Views {i} <-> {j}: {len(good)} matches")

        # Save match visualisation
        if len(pts_i) > 0:
            vis = np.hstack([images[i], images[j]])
            for pi, pj in zip(pts_i[:80], pts_j[:80]):
                pt1 = (int(pi[0]), int(pi[1]))
                pt2 = (int(pj[0]) + images[i].shape[1], int(pj[1]))
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(vis, pt1, pt2, color, 1)
                cv2.circle(vis, pt1, 4, color, -1)
                cv2.circle(vis, pt2, 4, color, -1)
            cv2.imwrite(os.path.join(out_dir, f"matches_{i}_{j}.png"), vis)

    # Initialise with views 0 & 1
    print("\n[SfM] Initialising with views 0 & 1 ...")
    pts0, pts1_pts, _, _, _ = matches_dict[(0, 1)]
    if len(pts0) < 8:
        print("[ERROR] Not enough matches between views 0 and 1")
        return None, None, None

    R01, t01, E01, mask01 = recover_pose_from_pair(pts0, pts1_pts, K)
    if R01 is None:
        print("[ERROR] Cannot estimate pose between views 0 and 1")
        return None, None, None

    cam_R = [np.eye(3), R01]
    cam_t = [np.zeros((3, 1)), t01]

    # Triangulate initial points
    inlier_mask = mask01.flatten() > 0
    pts0_in = pts0[inlier_mask]
    pts1_in = pts1_pts[inlier_mask]
    if len(pts0_in) == 0:
        print("[ERROR] No inlier matches for triangulation")
        return None, None, None

    points_3d = triangulate(pts0_in, pts1_in, K,
                            cam_R[0], cam_t[0], cam_R[1], cam_t[1])
    print(f"  Initial triangulation: {len(points_3d)} points")

    # Add remaining views
    for v in range(2, n):
        print(f"\n[SfM] Adding view {v} ...")
        best_count = 0
        best_R, best_t = None, None

        for ref in range(v):
            key = (ref, v) if (ref, v) in matches_dict else None
            if key is None:
                continue
            pts_ref, pts_v, _, _, _ = matches_dict[key]
            if len(pts_ref) < 8:
                continue

            R_rv, t_rv, _, mask_rv = recover_pose_from_pair(pts_ref, pts_v, K)
            if R_rv is None:
                continue

            R_v = R_rv @ cam_R[ref]
            t_v = R_rv @ cam_t[ref] + t_rv

            inlier_count = int(mask_rv.sum()) if mask_rv is not None else 0
            if inlier_count > best_count:
                best_count = inlier_count
                best_R, best_t = R_v, t_v

        if best_R is not None:
            cam_R.append(best_R)
            cam_t.append(best_t)
            print(f"  View {v} registered ({best_count} inliers)")
        else:
            key = (0, v)
            if key in matches_dict:
                p0, pv, _, _, _ = matches_dict[key]
                if len(p0) >= 8:
                    Rv, tv, _, _ = recover_pose_from_pair(p0, pv, K)
                    if Rv is not None:
                        cam_R.append(Rv)
                        cam_t.append(tv)
                        print(f"  View {v} registered (fallback via view 0)")
                    else:
                        cam_R.append(np.eye(3))
                        cam_t.append(np.zeros((3, 1)))
                        print(f"  [WARN] View {v} -- pose failed, using identity")
                else:
                    cam_R.append(np.eye(3))
                    cam_t.append(np.zeros((3, 1)))
            else:
                cam_R.append(np.eye(3))
                cam_t.append(np.zeros((3, 1)))

        # Triangulate new points
        key = (0, v)
        if key in matches_dict:
            p0v, pvv, _, _, _ = matches_dict[key]
            if len(p0v) > 0:
                new_pts = triangulate(p0v, pvv, K,
                                      cam_R[0], cam_t[0], cam_R[v], cam_t[v])
                points_3d = np.vstack([points_3d, new_pts])
                print(f"  +{len(new_pts)} triangulated points (total: {len(points_3d)})")

    # Filter outliers
    if len(points_3d) > 10:
        median = np.median(points_3d, axis=0)
        dists = np.linalg.norm(points_3d - median, axis=1)
        thresh = np.percentile(dists, 90)
        inlier_3d = points_3d[dists < thresh]
        # Also remove points behind cameras (negative Z)
        inlier_3d = inlier_3d[inlier_3d[:, 2] > 0]
    else:
        inlier_3d = points_3d
    print(f"\n[SfM] After outlier removal: {len(inlier_3d)}/{len(points_3d)} points")

    # Visualise
    visualise_sfm(inlier_3d, cam_R, cam_t, images, out_dir)

    # Save numerical results & math workouts
    save_math_workouts(inlier_3d, cam_R, cam_t, K, matches_dict, out_dir)

    print(f"\n  SfM results saved to {os.path.abspath(out_dir)}")
    return inlier_3d, cam_R, cam_t


# ──────────────────────────────────────────────────────────────────────────────
# Save mathematical workouts
# ──────────────────────────────────────────────────────────────────────────────
def save_math_workouts(pts3d, cam_R, cam_t, K, matches_dict, out_dir):
    """Save detailed mathematical workouts and camera parameters."""

    with open(os.path.join(out_dir, "camera_poses.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  STRUCTURE FROM MOTION -- CAMERA PARAMETERS & POSES\n")
        f.write("=" * 80 + "\n\n")

        f.write("Camera Intrinsic Matrix K:\n")
        f.write(np.array2string(K, precision=4, suppress_small=True) + "\n\n")

        for i in range(len(cam_R)):
            f.write(f"{'─'*60}\n")
            f.write(f"View {i}:\n")
            f.write(f"  Rotation R_{i} =\n")
            f.write(f"    {np.array2string(cam_R[i], precision=6, suppress_small=True)}\n")
            f.write(f"  Translation t_{i} = {cam_t[i].flatten()}\n")

            # Camera centre: C = -R^T @ t
            C = (-cam_R[i].T @ cam_t[i]).flatten()
            f.write(f"  Camera centre (world) C_{i} = -R^T @ t = {C}\n")

            # Projection matrix
            P = K @ np.hstack([cam_R[i], cam_t[i]])
            f.write(f"  Projection matrix P_{i} = K [R|t] =\n")
            f.write(f"    {np.array2string(P, precision=4, suppress_small=True)}\n")

            # Rodrigues (axis-angle)
            rvec, _ = cv2.Rodrigues(cam_R[i])
            angle = np.linalg.norm(rvec)
            if angle > 1e-6:
                axis = rvec.flatten() / angle
                f.write(f"  Rotation: {np.degrees(angle):.2f} deg around axis [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]\n")
            f.write("\n")

        # Epipolar geometry for pair (0, 1)
        f.write("\n" + "=" * 80 + "\n")
        f.write("  EPIPOLAR GEOMETRY WORKOUT (Views 0 <-> 1)\n")
        f.write("=" * 80 + "\n\n")

        R01 = cam_R[1]
        t01 = cam_t[1]
        # Essential matrix: E = [t]_x R
        tx = np.array([[0, -t01[2,0], t01[1,0]],
                        [t01[2,0], 0, -t01[0,0]],
                        [-t01[1,0], t01[0,0], 0]])
        E_computed = tx @ R01
        f.write("  Skew-symmetric [t]_x:\n")
        f.write(f"    {np.array2string(tx, precision=6)}\n\n")
        f.write("  Essential matrix E = [t]_x @ R:\n")
        f.write(f"    {np.array2string(E_computed, precision=6)}\n\n")

        # Fundamental matrix: F = K^{-T} E K^{-1}
        K_inv = np.linalg.inv(K)
        F_computed = K_inv.T @ E_computed @ K_inv
        f.write("  Fundamental matrix F = K^{-T} E K^{-1}:\n")
        f.write(f"    {np.array2string(F_computed, precision=8)}\n\n")

        # Verify epipolar constraint on a few points
        if (0, 1) in matches_dict:
            pts0, pts1, _, _, _ = matches_dict[(0, 1)]
            if len(pts0) > 0:
                f.write("  Epipolar constraint verification (x2^T F x1 should be ~0):\n")
                for k in range(min(5, len(pts0))):
                    x1 = np.array([pts0[k][0], pts0[k][1], 1.0])
                    x2 = np.array([pts1[k][0], pts1[k][1], 1.0])
                    val = x2 @ F_computed @ x1
                    f.write(f"    Point {k}: x1=({pts0[k][0]:.1f}, {pts0[k][1]:.1f}), "
                            f"x2=({pts1[k][0]:.1f}, {pts1[k][1]:.1f}), "
                            f"x2^T F x1 = {val:.6f}\n")
                f.write("\n")

        # Triangulation workout for one point
        f.write("=" * 80 + "\n")
        f.write("  TRIANGULATION WORKOUT (DLT Method)\n")
        f.write("=" * 80 + "\n\n")
        f.write("  Given projection matrices P1, P2 and correspondence x1 <-> x2,\n")
        f.write("  we solve A X = 0 where:\n")
        f.write("    A = [x1[0]*P1[2] - P1[0];\n")
        f.write("         x1[1]*P1[2] - P1[1];\n")
        f.write("         x2[0]*P2[2] - P2[0];\n")
        f.write("         x2[1]*P2[2] - P2[1]]\n\n")

        if (0, 1) in matches_dict:
            pts0, pts1, _, _, _ = matches_dict[(0, 1)]
            if len(pts0) > 0:
                P1 = K @ np.hstack([cam_R[0], cam_t[0]])
                P2 = K @ np.hstack([cam_R[1], cam_t[1]])

                x1 = pts0[0]
                x2 = pts1[0]

                A_dlt = np.array([
                    x1[0] * P1[2] - P1[0],
                    x1[1] * P1[2] - P1[1],
                    x2[0] * P2[2] - P2[0],
                    x2[1] * P2[2] - P2[1]
                ])

                f.write(f"  Example for point: x1=({x1[0]:.2f}, {x1[1]:.2f}), x2=({x2[0]:.2f}, {x2[1]:.2f})\n\n")
                f.write(f"  DLT matrix A (4x4):\n")
                f.write(f"    {np.array2string(A_dlt, precision=4, suppress_small=True)}\n\n")

                # SVD
                U, S, Vt = np.linalg.svd(A_dlt)
                X_homo = Vt[-1]
                X_3d = X_homo[:3] / X_homo[3]
                f.write(f"  SVD solution (last row of V^T): {X_homo}\n")
                f.write(f"  3D point (after dehomogenisation): ({X_3d[0]:.4f}, {X_3d[1]:.4f}, {X_3d[2]:.4f})\n\n")

                # Verify by reprojection
                X_h = np.append(X_3d, 1.0)
                reproj1 = P1 @ X_h
                reproj1 = reproj1[:2] / reproj1[2]
                reproj2 = P2 @ X_h
                reproj2 = reproj2[:2] / reproj2[2]
                f.write(f"  Reprojection verification:\n")
                f.write(f"    View 0: original=({x1[0]:.2f}, {x1[1]:.2f}), reprojected=({reproj1[0]:.2f}, {reproj1[1]:.2f}), error={np.linalg.norm(reproj1 - x1):.4f} px\n")
                f.write(f"    View 1: original=({x2[0]:.2f}, {x2[1]:.2f}), reprojected=({reproj2[0]:.2f}, {reproj2[1]:.2f}), error={np.linalg.norm(reproj2 - x2):.4f} px\n")

    # Save 3D points
    np.savetxt(os.path.join(out_dir, "points_3d.txt"), pts3d, fmt="%.6f",
               header="X Y Z")

    print(f"[SfM] Mathematical workouts saved to {out_dir}/camera_poses.txt")


# ──────────────────────────────────────────────────────────────────────────────
# 3D Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def visualise_sfm(pts3d, cam_R_list, cam_t_list, images, out_dir):
    """Plot 3D point cloud + camera positions."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
               s=1, c="steelblue", alpha=0.5, label="3D Points")

    colours = ["red", "green", "orange", "purple"]
    for i, (R, t) in enumerate(zip(cam_R_list, cam_t_list)):
        C = (-R.T @ t).flatten()
        ax.scatter(*C, s=100, c=colours[i % len(colours)], marker="^",
                   edgecolors="black", label=f"Camera {i}")
        direction = R.T @ np.array([[0], [0], [1]])
        ax.quiver(C[0], C[1], C[2],
                  direction[0, 0], direction[1, 0], direction[2, 0],
                  length=0.3, color=colours[i % len(colours)], arrow_length_ratio=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Structure from Motion -- 3D Reconstruction")
    ax.legend(fontsize=8)

    for elev, azim, suffix in [(30, -60, "view1"), (60, -120, "view2"), (10, -30, "view3")]:
        ax.view_init(elev=elev, azim=azim)
        fig.savefig(os.path.join(out_dir, f"sfm_3d_{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Top-down 2D projection with boundary
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(pts3d[:, 0], pts3d[:, 1], s=2, c="steelblue", alpha=0.6)

    try:
        from scipy.spatial import ConvexHull
        if len(pts3d) >= 3:
            hull = ConvexHull(pts3d[:, :2])
            for simplex in hull.simplices:
                ax2.plot(pts3d[simplex, 0], pts3d[simplex, 1], "r-", linewidth=1.5)
            # Fill the hull
            hull_pts = pts3d[hull.vertices, :2]
            from matplotlib.patches import Polygon
            poly = Polygon(hull_pts, fill=True, alpha=0.15, color="red",
                           label="Convex hull boundary")
            ax2.add_patch(poly)
            ax2.legend()
            ax2.set_title("Top-down view with convex-hull boundary estimate")
    except Exception as e:
        ax2.set_title(f"Top-down view (X-Y projection)")

    # Plot camera positions
    for i, (R, t) in enumerate(zip(cam_R_list, cam_t_list)):
        C = (-R.T @ t).flatten()
        ax2.scatter(C[0], C[1], s=80, c=colours[i % len(colours)], marker="^",
                    edgecolors="black", zorder=5, label=f"Cam {i}" if i == 0 else "")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "sfm_boundary_topdown.png"), dpi=150)
    plt.close(fig2)

    # Input images grid
    n_imgs = len(images)
    fig3, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
    if n_imgs == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        C = (-cam_R_list[i].T @ cam_t_list[i]).flatten()
        ax.set_title(f"View {i}\nC=({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f})")
        ax.axis("off")
    fig3.suptitle("Input images from different viewpoints", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "sfm_input_views.png"), dpi=150)
    plt.close(fig3)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Structure from Motion (Module 7 -- Part 2)")
    parser.add_argument("--images", default=None,
                        help="Directory containing images of the object")
    parser.add_argument("--video", default=None,
                        help="Video to extract frames from (if no images dir)")
    parser.add_argument("--K", nargs=4, type=float, default=None,
                        metavar=("fx", "fy", "cx", "cy"),
                        help="Camera intrinsic parameters")
    parser.add_argument("--outdir", default="output/sfm",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine image source
    img_paths = []
    if args.images and os.path.isdir(args.images):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for ext in exts:
            img_paths.extend(sorted(glob.glob(os.path.join(args.images, ext))))
        img_paths = sorted(set(img_paths))

    if len(img_paths) < 2 and args.video:
        print(f"[SfM] Extracting frames from video: {args.video}")
        sfm_img_dir = args.images if args.images else "sfm_images"
        img_paths = extract_frames_from_video(args.video, sfm_img_dir, n_frames=4)

    if len(img_paths) < 2:
        print("[ERROR] Need at least 2 images. Provide --images dir or --video.")
        sys.exit(1)

    # Load images
    images = []
    for p in img_paths[:4]:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Cannot read {p}, skipping")
            continue
        target_w = 800
        scale_f = target_w / img.shape[1]
        img = cv2.resize(img, (target_w, int(img.shape[0] * scale_f)))
        images.append(img)
        print(f"  Loaded: {p}  ({img.shape[1]}x{img.shape[0]})")

    if len(images) < 2:
        print("[ERROR] Not enough valid images.")
        sys.exit(1)

    # Build K
    h, w = images[0].shape[:2]
    if args.K:
        K = build_K(*args.K)
        print(f"\n  Using provided K: fx={args.K[0]}, fy={args.K[1]}, "
              f"cx={args.K[2]}, cy={args.K[3]}")
    else:
        K = default_K_from_image(w, h)
        print(f"\n  Using estimated K (focal={max(w,h)*1.2:.0f}, "
              f"cx={w/2:.0f}, cy={h/2:.0f})")

    run_sfm(images, K, args.outdir)


if __name__ == "__main__":
    main()

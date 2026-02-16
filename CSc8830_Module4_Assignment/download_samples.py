#!/usr/bin/env python3
"""
================================================================================
Download sample thermal infrared images of animals for testing.

Downloads a set of thermal/infrared images from publicly available sources
for use with the thermal animal segmentation pipeline.

HOW TO RUN:
    python download_samples.py

OUTPUT:
    Downloads images to the images/ directory.
================================================================================
"""

import os
import urllib.request
import ssl

# Create a default SSL context that doesn't verify certificates
# (needed for some image hosting sites)
ssl._create_default_https_context = ssl._create_unverified_context


def download_image(url, filename, output_dir="images"):
    """Download an image from URL to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"  Already exists: {filepath}")
        return filepath

    try:
        print(f"  Downloading: {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return None


def create_synthetic_thermal(output_dir="images"):
    """
    Create synthetic thermal-like images for testing when
    downloads are not available.

    Generates images that simulate thermal imaging camera output
    with bright (warm) animal-like shapes against a cool background.
    """
    import numpy as np
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    print("\n  Creating synthetic thermal test images...")

    # Image 1: Single animal (deer-like shape)
    img1 = np.random.randint(20, 60, (480, 640), dtype=np.uint8)
    # Add some background variation
    img1 = cv2.GaussianBlur(img1, (21, 21), 0)
    # Draw warm animal body (ellipse)
    cv2.ellipse(img1, (320, 280), (100, 60), 0, 0, 360, 200, -1)
    # Head
    cv2.circle(img1, (440, 260), 30, 210, -1)
    # Legs
    cv2.rectangle(img1, (260, 340), (280, 400), 180, -1)
    cv2.rectangle(img1, (350, 340), (370, 400), 180, -1)
    # Apply blur for realism
    img1 = cv2.GaussianBlur(img1, (9, 9), 0)
    # Apply colormap to make it look thermal
    img1_color = cv2.applyColorMap(img1, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(output_dir, "synthetic_thermal_1.png"), img1_color)
    print(f"    Created: synthetic_thermal_1.png (single animal)")

    # Image 2: Multiple animals
    img2 = np.random.randint(15, 50, (480, 640), dtype=np.uint8)
    img2 = cv2.GaussianBlur(img2, (21, 21), 0)
    # Animal 1
    cv2.ellipse(img2, (200, 250), (80, 50), 10, 0, 360, 190, -1)
    cv2.circle(img2, (290, 230), 25, 200, -1)
    # Animal 2
    cv2.ellipse(img2, (480, 300), (70, 45), -5, 0, 360, 195, -1)
    cv2.circle(img2, (400, 285), 22, 205, -1)
    # Animal 3 (smaller, farther away)
    cv2.ellipse(img2, (350, 150), (40, 25), 0, 0, 360, 170, -1)
    img2 = cv2.GaussianBlur(img2, (9, 9), 0)
    img2_color = cv2.applyColorMap(img2, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(output_dir, "synthetic_thermal_2.png"), img2_color)
    print(f"    Created: synthetic_thermal_2.png (multiple animals)")

    # Image 3: Grayscale thermal (raw thermal style)
    img3 = np.random.randint(30, 70, (480, 640), dtype=np.uint8)
    img3 = cv2.GaussianBlur(img3, (15, 15), 0)
    # Large animal
    cv2.ellipse(img3, (300, 260), (120, 70), 5, 0, 360, 220, -1)
    cv2.circle(img3, (440, 230), 35, 230, -1)
    # Tail
    pts = np.array([[180, 260], [140, 240], [120, 260]], np.int32)
    cv2.fillPoly(img3, [pts], 180)
    img3 = cv2.GaussianBlur(img3, (11, 11), 0)
    # Save as grayscale (typical raw thermal output)
    cv2.imwrite(os.path.join(output_dir, "synthetic_thermal_3.png"), img3)
    print(f"    Created: synthetic_thermal_3.png (grayscale thermal)")

    # Image 4: Challenging case (animal near warm background)
    img4 = np.random.randint(60, 100, (480, 640), dtype=np.uint8)
    # Warm ground in lower half
    img4[300:, :] = np.random.randint(100, 140, (180, 640), dtype=np.uint8)
    img4 = cv2.GaussianBlur(img4, (21, 21), 0)
    # Animal (slightly warmer than background)
    cv2.ellipse(img4, (320, 220), (90, 55), 0, 0, 360, 210, -1)
    cv2.circle(img4, (420, 200), 28, 220, -1)
    img4 = cv2.GaussianBlur(img4, (9, 9), 0)
    img4_color = cv2.applyColorMap(img4, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "synthetic_thermal_4.png"), img4_color)
    print(f"    Created: synthetic_thermal_4.png (challenging case)")

    print(f"\n  Created 4 synthetic thermal test images in {output_dir}/")


def main():
    """Main download function."""
    print("=" * 60)
    print("  Thermal Image Download / Generation")
    print("=" * 60)

    # Roboflow thermal dataset sample URLs
    # These are publicly available thermal infrared images of animals
    thermal_urls = [
        # Thermal images from open datasets
        ("https://datasets.roboflow.com/thermal-dogs-and-people/thermal-dogs-and-people-1/images/frame_000542.jpg",
         "thermal_animal_01.jpg"),
        ("https://datasets.roboflow.com/thermal-dogs-and-people/thermal-dogs-and-people-1/images/frame_000610.jpg",
         "thermal_animal_02.jpg"),
        ("https://datasets.roboflow.com/thermal-dogs-and-people/thermal-dogs-and-people-1/images/frame_000890.jpg",
         "thermal_animal_03.jpg"),
    ]

    print("\n  Attempting to download thermal images...")
    downloaded = 0
    for url, filename in thermal_urls:
        result = download_image(url, filename)
        if result:
            downloaded += 1

    if downloaded == 0:
        print("\n  No images downloaded (URLs may be unavailable).")

    # Always create synthetic images as fallback/additional test cases
    try:
        create_synthetic_thermal()
    except Exception as e:
        print(f"  Error creating synthetic images: {e}")

    print("\n  Done! Place your own thermal images in the images/ directory.")
    print("  Thermal infrared datasets available at:")
    print("    https://blog.roboflow.com/thermal-infrared-dataset-computer-vision/")
    print("    https://universe.roboflow.com/search?q=thermal%20animal")
    print("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline:
1. Run SAM on a subset of images -> obtain multiple masks.
2. Build a consensus mask (pixel included in at least X% of masks).
3. Filter out small segments (min_area).
4. (Optional) Restrict with an ROI (polygon).
5. Save the final mask.
6. Apply the final mask to all images.

Adjust:
- image_dir
- filename pattern (pattern)
- SAM prompts (points, labels)
- number of example images (num_examples)
- consensus fraction (consensus_fraction)
- min_area
- ROI polygon (pts)
"""

# Required libraries
from ultralytics import SAM  # Segment Anything Model
import cv2                   # Image processing
import numpy as np           # Arrays and computations
import glob                  # File matching with patterns
import os                    # Paths and directories

# ----------------------------------------------------
# 0. Basic settings
# ----------------------------------------------------

# Directory with images
image_dir = "data/images"

# Filename pattern for images
pattern = os.path.join(image_dir, "*.jpg")

# Number of images used to build the consensus mask
num_examples = 10  # first 10 images

# Fraction of masks a pixel must appear in to be kept
consensus_fraction = 0.5  # at least half

# Minimum area (in pixels) for a segment to be kept
min_area = 500  # at least 500 pixels


# ----------------------------------------------------
# 1. Collect image paths
# ----------------------------------------------------

# Sorted list of filenames
image_paths = sorted(glob.glob(pattern))

# Select the first num_examples images
example_paths = image_paths[:min(num_examples, len(image_paths))]


# ----------------------------------------------------
# 2. Run SAM on example images and collect masks
# ----------------------------------------------------

# Load SAM model
model = SAM("sam_b.pt")  # base variant

# TODO: Define prompt points (x, y) for your own images
# Example:
points = [[[100, 100],   # + keep (e.g., vegetation)
           [500, 500]]]  # - exclude (e.g., background)

# Point labels: 1 = positive (keep), 0 = negative (exclude)
labels = [[1, 0]]

# List of masks
masks_list = []

# Loop through example images
for img_path in example_paths:
    # Run SAM and get results (masks)
    results = model(img_path, points=points, labels=labels)

    # Take the first mask from the SAM output
    m = results[0].masks.data[0].cpu().numpy()  # move to CPU and convert to numpy
    m = m.astype(np.uint8)                      # convert to 0/1 uint8
    masks_list.append(m)

# Stack masks into a 3D array: (N, H, W)
masks_arr = np.stack(masks_list, axis=0)
num_masks, H, W = masks_arr.shape  # N = number of example images


# ----------------------------------------------------
# 3. Build consensus mask
# ----------------------------------------------------

votes = masks_arr.sum(axis=0)  # how many times each pixel is included
threshold = int(np.ceil(consensus_fraction * num_masks))
print(f"Consensus threshold: at least {threshold} of {num_masks} masks.")

consensus_mask = (votes >= threshold).astype(np.uint8)  # 0/1


# ----------------------------------------------------
# 4. Filter out small segments (area filtering)
# ----------------------------------------------------

num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
    consensus_mask, connectivity=8
)

filtered_mask = np.zeros_like(consensus_mask, dtype=np.uint8)

for label in range(1, num_labels):  # 0 is background
    area = stats[label, cv2.CC_STAT_AREA]
    if area >= min_area:
        filtered_mask[labels_cc == label] = 1

print(f"Total segments: {num_labels - 1}")
print("Pixels kept after area filtering:", filtered_mask.sum(), "(count of 1s, not area).")


# ----------------------------------------------------
# 5. ROI polygon (optional)
# ----------------------------------------------------

# Create ROI mask with the same size
roi_mask = np.zeros_like(filtered_mask, dtype=np.uint8)

# TODO: Define polygon vertices (x, y) for the area you want to keep
# Example:
pts = np.array([
        [0, 0],
        [400, 400],
        [400, 0],
        [0, 0]
    ], np.int32)

cv2.fillPoly(roi_mask, [pts], 1)  # 1 = inside polygon

combined_mask = (filtered_mask & roi_mask).astype(np.uint8)

# Final mask as 0/255 uint8
final_mask_uint8 = (combined_mask * 255).astype(np.uint8)


# ----------------------------------------------------
# 6. Save final mask
# ----------------------------------------------------

final_mask_npy_path = os.path.join(image_dir, "final_mask.npy")
final_mask_png_path = os.path.join(image_dir, "final_mask.png")

np.save(final_mask_npy_path, final_mask_uint8)
cv2.imwrite(final_mask_png_path, final_mask_uint8)

print("Saved final mask to:")
print("  ", final_mask_npy_path)
print("  ", final_mask_png_path)


# ----------------------------------------------------
# 7. Apply final mask to all images
# ----------------------------------------------------

out_dir = os.path.join(image_dir, "segmented")
os.makedirs(out_dir, exist_ok=True)

for img_path in image_paths:
    print("Masking image:", img_path)
    img = cv2.imread(img_path)
    if img is None:
        print("  ⚠ Could not read image, skipping.")
        continue

    if img.shape[:2] != final_mask_uint8.shape[:2]:
        print("  ⚠ Size mismatch, skipping.")
        print("    Image:", img.shape[:2], "Mask:", final_mask_uint8.shape[:2])
        continue

    vegetation_only = cv2.bitwise_and(img, img, mask=final_mask_uint8)

    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_final_vegetation.png")
    cv2.imwrite(out_path, vegetation_only)
    print("  Saved:", out_path)

print("✔ Done! Consensus + area filter + ROI applied to all images.")

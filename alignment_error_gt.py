import cv2
import numpy as np

from src.matcher import FeatureMatcher
from src.homography import compute_homography

summer_path = "data/summer/images-00900.png"
winter_path = "data/winter/images-00900.png"

matcher = FeatureMatcher()

mkpts0, mkpts1 = matcher.match_images(
    summer_path,
    winter_path
)

(
    H,
    inliers,
    total_matches,
    num_inliers,
    inlier_ratio
) = compute_homography(
    mkpts0,
    mkpts1
)

# Only inlier correspondences

src_pts = mkpts0[inliers]
dst_pts = mkpts1[inliers]

# Project source points

projected = cv2.perspectiveTransform(
    src_pts.reshape(-1, 1, 2),
    H
).reshape(-1, 2)

# Euclidean distance

errors = np.linalg.norm(
    projected - dst_pts,
    axis=1
)

print("\n===== REPROJECTION ERROR =====")

print(
    f"Mean Error: {np.mean(errors):.3f} px"
)

print(
    f"Median Error: {np.median(errors):.3f} px"
)

print(
    f"Max Error: {np.max(errors):.3f} px"
)
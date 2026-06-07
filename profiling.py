import time
import cv2

from src.matcher import FeatureMatcher
from src.homography import compute_homography

summer_path = "data/summer/images-00900.png"
winter_path = "data/winter/images-00900.png"

# ---------------------------------
# Data Loading
# ---------------------------------

t0 = time.perf_counter()

img1 = cv2.imread(summer_path)
img2 = cv2.imread(winter_path)

load_time = (time.perf_counter() - t0) * 1000

# ---------------------------------
# Feature Extraction + Matching
# ---------------------------------

matcher = FeatureMatcher()

t1 = time.perf_counter()

mkpts0, mkpts1 = matcher.match_images(
    summer_path,
    winter_path
)

matching_time = (time.perf_counter() - t1) * 1000

# ---------------------------------
# Matrix Estimation
# ---------------------------------

t2 = time.perf_counter()

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

matrix_time = (time.perf_counter() - t2) * 1000

total_time = (
    load_time
    + matching_time
    + matrix_time
)

print("\n===== PROFILING =====")

print(f"Data Loading      : {load_time:.2f} ms")
print(f"Feature+Matching  : {matching_time:.2f} ms")
print(f"Matrix Estimation : {matrix_time:.2f} ms")
print(f"Total             : {total_time:.2f} ms")

print(f"Inlier Ratio      : {inlier_ratio:.4f}")
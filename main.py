import cv2

from src.matcher import FeatureMatcher
from src.homography import (
    compute_homography,
    warp_image
)
from src.visualization import (
    draw_matches,
    draw_overlay
)


SUMMER_IMG = (
    "data/summer/images-00905.png"
)

WINTER_IMG = (
    "data/winter/images-00905.png"
)


matcher = FeatureMatcher()

mkpts0, mkpts1 = matcher.match_images(
    SUMMER_IMG,
    WINTER_IMG
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

print("\n===== RESULTS =====")

print(
    f"Total Matches: {total_matches}"
)

print(
    f"Inliers: {num_inliers}"
)

print(
    f"Inlier Ratio: {inlier_ratio:.4f}"
)

print("\nHomography Matrix:")

print(H)

summer_img = cv2.imread(
    SUMMER_IMG
)

winter_img = cv2.imread(
    WINTER_IMG
)

draw_matches(
    cv2.cvtColor(
        summer_img,
        cv2.COLOR_BGR2RGB
    ),
    cv2.cvtColor(
        winter_img,
        cv2.COLOR_BGR2RGB
    ),
    mkpts0[inliers],
    mkpts1[inliers],
    save_path=(
        "outputs/matches/"
        "match_00905.png"
    )
)

warped = warp_image(
    summer_img,
    H,
    winter_img.shape
)

draw_overlay(
    warped,
    winter_img,
    save_path=(
        "outputs/warped/"
        "overlay_00905.png"
    )
)
import cv2
import numpy as np


def compute_homography(
        points1,
        points2,
        ransac_thresh=5.0
):

    H, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        ransac_thresh
    )

    inliers = mask.ravel().astype(bool)

    total_matches = len(points1)

    num_inliers = np.sum(inliers)

    inlier_ratio = (
        num_inliers / total_matches
    )

    return (
        H,
        inliers,
        total_matches,
        num_inliers,
        inlier_ratio
    )


def warp_image(
        image,
        H,
        target_shape
):

    warped = cv2.warpPerspective(
        image,
        H,
        (
            target_shape[1],
            target_shape[0]
        )
    )

    return warped
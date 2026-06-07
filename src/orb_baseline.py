import cv2
import numpy as np


class ORBBaseline:

    def __init__(self, nfeatures=2000):

        self.orb = cv2.ORB_create(
            nfeatures=nfeatures
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False
        )

    def match_images(
            self,
            image_path1,
            image_path2
    ):

        img1 = cv2.imread(
            image_path1,
            cv2.IMREAD_GRAYSCALE
        )

        img2 = cv2.imread(
            image_path2,
            cv2.IMREAD_GRAYSCALE
        )

        kp1, des1 = self.orb.detectAndCompute(
            img1,
            None
        )

        kp2, des2 = self.orb.detectAndCompute(
            img2,
            None
        )

        if des1 is None or des2 is None:
            raise ValueError(
                "ORB could not compute descriptors"
            )

        # KNN Matching
        knn_matches = self.matcher.knnMatch(
            des1,
            des2,
            k=2
        )

        good_matches = []

        # Lowe Ratio Test
        for m, n in knn_matches:

            if m.distance < 0.75 * n.distance:

                good_matches.append(m)

        if len(good_matches) < 4:
            raise ValueError(
                "Not enough good matches"
            )

        pts1 = np.float32([
            kp1[m.queryIdx].pt
            for m in good_matches
        ])

        pts2 = np.float32([
            kp2[m.trainIdx].pt
            for m in good_matches
        ])

        return pts1, pts2
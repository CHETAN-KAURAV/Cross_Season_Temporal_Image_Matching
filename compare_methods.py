import os
import pandas as pd

from src.matcher import FeatureMatcher
from src.orb_baseline import ORBBaseline
from src.homography import compute_homography


SUMMER_DIR = "data/summer"
WINTER_DIR = "data/winter"

lightglue = FeatureMatcher()

orb = ORBBaseline()

summer_images = sorted(
    os.listdir(SUMMER_DIR)
)

winter_images = sorted(
    os.listdir(WINTER_DIR)
)

results = []

for summer_name, winter_name in zip(
        summer_images,
        winter_images):

    summer_path = os.path.join(
        SUMMER_DIR,
        summer_name
    )

    winter_path = os.path.join(
        WINTER_DIR,
        winter_name
    )

    try:

        # -------------------
        # ORB
        # -------------------

        pts1, pts2 = orb.match_images(
            summer_path,
            winter_path
        )

        (
            H,
            inliers,
            total_matches,
            num_inliers,
            orb_ratio
        ) = compute_homography(
            pts1,
            pts2
        )

        # -------------------
        # LightGlue
        # -------------------

        mkpts0, mkpts1 = (
            lightglue.match_images(
                summer_path,
                winter_path
            )
        )

        (
            H,
            inliers,
            total_matches,
            num_inliers,
            lg_ratio
        ) = compute_homography(
            mkpts0,
            mkpts1
        )

        results.append({

            "image":
                summer_name,

            "orb_ratio":
                round(orb_ratio, 4),

            "lightglue_ratio":
                round(lg_ratio, 4)
        })

        print(
            f"{summer_name}"
            f" | ORB: {orb_ratio:.3f}"
            f" | LG: {lg_ratio:.3f}"
        )

    except Exception as e:

        print(
            f"Failed: {summer_name}"
        )

        print(e)

df = pd.DataFrame(results)

df.to_csv(
    "outputs/profiling/comparison.csv",
    index=False
)

print("\n")

print(df.describe())
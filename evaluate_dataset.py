import os
import time
import pandas as pd

from src.matcher import FeatureMatcher
from src.homography import compute_homography


SUMMER_DIR = "data/summer"
WINTER_DIR = "data/winter"

matcher = FeatureMatcher()

summer_images = sorted(os.listdir(SUMMER_DIR))
winter_images = sorted(os.listdir(WINTER_DIR))

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

    start = time.perf_counter()

    try:

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

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        results.append({

            "image": summer_name,

            "total_matches":
                total_matches,

            "inliers":
                num_inliers,

            "inlier_ratio":
                round(
                    inlier_ratio,
                    4
                ),

            "latency_ms":
                round(
                    latency_ms,
                    2
                )
        })

        print(
            f"{summer_name}"
            f" | Inlier Ratio:"
            f" {inlier_ratio:.3f}"
        )

    except Exception as e:

        print(
            f"Failed:"
            f" {summer_name}"
        )

        print(e)

df = pd.DataFrame(results)

os.makedirs(
    "outputs/profiling",
    exist_ok=True
)

df.to_csv(

    "outputs/profiling/results.csv",

    index=False
)

print("\nDone!")

print(df.describe())
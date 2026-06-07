import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_matches(
        img0,
        img1,
        points0,
        points1,
        save_path=None
):

    h = max(
        img0.shape[0],
        img1.shape[0]
    )

    w = (
        img0.shape[1]
        + img1.shape[1]
    )

    canvas = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    canvas[
        :img0.shape[0],
        :img0.shape[1]
    ] = img0

    canvas[
        :img1.shape[0],
        img0.shape[1]:
    ] = img1

    for p0, p1 in zip(points0, points1):

        x0, y0 = int(p0[0]), int(p0[1])

        x1 = int(
            p1[0] + img0.shape[1]
        )

        y1 = int(p1[1])

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            1
        )

    if save_path:
        cv2.imwrite(
            save_path,
            cv2.cvtColor(
                canvas,
                cv2.COLOR_RGB2BGR
            )
        )

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas)
    plt.axis("off")
    plt.show()


def draw_overlay(
        warped,
        target,
        save_path=None
):

    warped_rgb = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2RGB
    )

    target_rgb = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        warped_rgb,
        0.5,
        target_rgb,
        0.5,
        0
    )

    if save_path:
        cv2.imwrite(
            save_path,
            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )
        )

    plt.figure(figsize=(10, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

    return overlay
import torch
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image


class FeatureMatcher:

    def __init__(self, device="cpu"):
        self.device = device

        self.extractor = SuperPoint(
            max_num_keypoints=1024
        ).eval().to(device)

        self.matcher = LightGlue(
            features="superpoint"
        ).eval().to(device)

    def match_images(self, image_path1, image_path2):

        img0 = load_image(image_path1).to(self.device)
        img1 = load_image(image_path2).to(self.device)

        feats0 = self.extractor.extract(img0)
        feats1 = self.extractor.extract(img1)

        matches01 = self.matcher(
            {
                "image0": feats0,
                "image1": feats1
            }
        )

        matches = matches01["matches"][0]

        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()

        mkpts0 = kpts0[
            matches[:, 0].cpu().numpy()
        ]

        mkpts1 = kpts1[
            matches[:, 1].cpu().numpy()
        ]

        return mkpts0, mkpts1
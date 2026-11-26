"""Appearance features for the cross-camera scores: DINOv2 ViT-B/14 (Apache-2.0), through torch.hub.
One vector per object crop (the class token) and one per 14 x 14 pixel patch of a frame. Features are used in place
of raw colour because the same surface legitimately changes colour with the viewing angle and the light.
Runs in the eval env. The first use downloads the code and weights into $TORCH_HOME/hub.
"""
import cv2
import numpy as np
import torch

NAME, VERSION = "dinov2_vitb14", 1
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class Reader:
    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False).cuda().eval()

    def _prep(self, ims):
        x = (np.stack(ims).astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x).permute(0, 3, 1, 2).cuda()

    def crops(self, ims, batch=64):
        """ims: list of uint8 [224, 224, 3]. Returns unit vectors [n, D]."""
        out = []
        with torch.inference_mode():
            for i in range(0, len(ims), batch):
                f = self.model.forward_features(self._prep(ims[i:i + batch]))["x_norm_clstoken"]
                out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def dense(self, frame):
        """One frame [H, W, 3] -> a feature map tensor [1, D, h, w] on the GPU (one cell per 14 px patch)."""
        H, W = frame.shape[:2]
        h, w = H // 14, W // 14
        im = cv2.resize(frame, (w * 14, h * 14), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            f = self.model.forward_features(self._prep([im]))["x_norm_patchtokens"]
        return f.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()

    @staticmethod
    def sample(fmap, u, v, size):
        """Features at pixel coordinates u, v [N] of a frame of size (W, H): bilinear in the patch grid. Returns [N, D]."""
        W, H = size
        g = torch.from_numpy(np.stack([(u + 0.5) / W * 2 - 1, (v + 0.5) / H * 2 - 1], -1).astype(np.float32)).cuda()[None, :, None]
        with torch.inference_mode():
            s = torch.nn.functional.grid_sample(fmap, g, mode="bilinear", align_corners=False)
        return s[0, :, :, 0].T.cpu().numpy()

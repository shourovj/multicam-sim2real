"""Point-track reader: CoTracker3 (offline). Each simulator surface point is handed to the tracker at the first
frame it is visible (in its shot), at its simulator position, and followed through the generated clip.
Runs in the eval env. Checkpoint: $S2R_ROOT/checkpoints/cotracker/scaled_offline.pth (setup/README.md).
"""
import os

import numpy as np
import torch

NAME, VERSION = "cotracker3", 1


class Reader:
    def __init__(self):
        from cotracker.predictor import CoTrackerPredictor
        ckpt = os.path.join(os.environ["S2R_ROOT"], "checkpoints", "cotracker", "scaled_offline.pth")
        assert os.path.exists(ckpt), f"missing {ckpt} (setup/README.md: eval checkpoints)"
        self.model = CoTrackerPredictor(checkpoint=ckpt, offline=True, window_len=60).cuda()

    def run(self, frames, view):
        """Returns points [N, T, 2] and visible [N, T], in the order of the view's track points (all shots joined)."""
        q = [np.stack([tr["query_t"], tr["points"][np.arange(len(tr["points"])), tr["query_t"], 0],
                       tr["points"][np.arange(len(tr["points"])), tr["query_t"], 1]], 1) for tr in view.tracks if len(tr["points"])]
        if not q:
            return np.zeros((0, len(frames), 2), np.float32), np.zeros((0, len(frames)), bool)
        q = np.concatenate(q).astype(np.float32)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()
        with torch.inference_mode():
            pts, vis = self.model(video, queries=torch.from_numpy(q)[None].cuda())
        return pts[0].permute(1, 0, 2).float().cpu().numpy(), vis[0].permute(1, 0).cpu().numpy().astype(bool)

"""Mask reader: SAM 2.1, each object started once from its simulator box and followed through the clip.

The box, with up to three clicks well inside the rendered mask, is taken at the frame where that mask is largest (per
shot, for a cut), so the start is the object's clearest view, never the frame it enters the picture. From there SAM
follows it forward and backward.
This asks "is there one coherent thing where the simulator puts this object, all the way through?". It does not ask
whether that thing is recognisable as its class; a reader prompted by class name (SAM 3) would add that.
Runs in the eval env. Checkpoint: $S2R_ROOT/checkpoints/sam2/sam2.1_hiera_large.pt (setup/README.md).
"""
import os
import tempfile

import cv2
import numpy as np
import torch

NAME, VERSION = "sam2", 2
MIN_PX = 100          # an object is started in a shot only if its rendered mask reaches this size there


class Reader:
    def __init__(self):
        from sam2.build_sam import build_sam2_video_predictor
        ckpt = os.path.join(os.environ["S2R_ROOT"], "checkpoints", "sam2", "sam2.1_hiera_large.pt")
        assert os.path.exists(ckpt), f"missing {ckpt} (setup/README.md: eval checkpoints)"
        self.pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device="cuda")

    def run(self, frames, view):
        """frames [T, H, W, 3]; view: p2s/4_score/common/reference.View. Returns {instance id: bool [T, H, W]}, filled only on
        the frames of the shots where the object was started."""
        T, H, W = view.inst.shape
        with tempfile.TemporaryDirectory(dir=os.environ.get("S2R_CACHE")) as jd:
            for t in range(T):
                cv2.imwrite(os.path.join(jd, f"{t:05d}.jpg"), cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            starts = {}                                   # SAM object id -> (instance id, first, last, start frame)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.pred.init_state(video_path=jd)
                for s, (a, b, cam) in enumerate(view.shots):
                    for o in view.objects:
                        if cam not in o["cameras"]:
                            continue
                        px = (view.inst[a:b + 1] == o["instance_id"]).reshape(b - a + 1, -1).sum(1)
                        if px.max() < MIN_PX:
                            continue
                        f = a + int(px.argmax())
                        m = view.inst[f] == o["instance_id"]
                        ys, xs = np.nonzero(m)
                        sid = 1000 * s + o["instance_id"]
                        # the box, plus clicks well inside the mask: a box alone can lock onto whatever stands in front
                        dist = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
                        clicks = []
                        for _ in range(3):
                            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
                            if dist[y, x] < 1.5:
                                break
                            clicks.append([x, y])
                            cv2.circle(dist, (int(x), int(y)), int(max(dist[y, x], 4)), 0, -1)
                        kw = dict(points=np.array(clicks, np.float32), labels=np.ones(len(clicks), np.int32)) if clicks else {}
                        self.pred.add_new_points_or_box(state, frame_idx=f, obj_id=sid, box=np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32), **kw)
                        starts[sid] = (o["instance_id"], a, b, f)
                out = {i: np.zeros((T, H, W), bool) for i in {v[0] for v in starts.values()}}
                if starts:
                    for reverse in (False, True):
                        start = max(v[3] for v in starts.values()) if reverse else min(v[3] for v in starts.values())
                        for t, ids, logits in self.pred.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                            for n, sid in enumerate(ids):
                                i, a, b, f = starts[int(sid)]
                                if a <= t <= b and ((t < f) if reverse else (t >= f)):
                                    out[i][t] = (logits[n, 0] > 0).cpu().numpy()
                self.pred.reset_state(state)
        return out

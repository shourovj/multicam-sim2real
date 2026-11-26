"""The control videos moved to p2s/2_render/control_videos.py (both routes share them). This keeps
`from world import controls` working for the older code."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "p2s", "2_render"))
from control_videos import *  # noqa: E402,F401,F403
from control_videos import CONTROLS, PALETTE, build_all, read_mp4, sheet, write_mp4  # noqa: E402,F401

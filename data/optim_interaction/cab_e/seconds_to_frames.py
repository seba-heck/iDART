"""
Code snipped to convert the json parameter duration in seconds to frames
in Blender (v.4.4.3). Its compatible for all fps (24, 30, 60) types.

Use this code in Blender python interface.
--------------------------------------------------------------------------------
Author: David Blickenstorfer (assisted by GPT-4o)
Created : 21/05/2025
"""

import json

config_path = "/full/path/to/config.json"

with open("config.json") as f:
    config = json.load(f)

fps = bpy.context.scene.render.fps  # e.g., 30 or 60
duration_secs = config["duration_in_seconds"]
duration_frames = int(duration_secs * fps)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = duration_frames

# Now you animate from init to goal over duration_frames frames

import bpy
import json
import os


# Filter all empties in the scene
empties = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']

if len(empties) < 2:
    print("You need at least TWO empties in the scene.")
else:
    # Sort empties by their order in bpy.data.objects (closest proxy to creation order)
    empties_sorted = sorted(empties, key=lambda o: bpy.data.objects.find(o.name))

    # Take last two empties
    start_obj = empties_sorted[-2]
    goal_obj = empties_sorted[-1]

    start_pos = list(start_obj.location)
    goal_pos = list(goal_obj.location)

    #default_orientation = [0, 0, 0, 1]

    config = {
        "scene_id": bpy.path.basename(bpy.data.filepath) or "cab benches",
        "start": {
            "name": start_obj.name,
            "position": start_pos,
            #"orientation": default_orientation
        },
        "goal": {
            "name": goal_obj.name,
            "position": goal_pos,
            #"orientation": default_orientation
        },
        "navigation_type": "point_to_point",
        "comment": "C"
        
    }

    output_dir = bpy.path.abspath("//") or os.path.expanduser("//wsl.localhost/Ubuntu/home/eifachliza/DH_proj/iDART/config_files")
    output_path = os.path.join(output_dir, "cab_benches_start_goal_config_10.json")

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Start and goal config saved to: {output_path}")

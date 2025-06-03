from pathfinder import navmesh_baker as nmb
import trimesh
import pickle
import pathfinder as pf

inp_path = "./data/scenes/cab_g_benches/cab_g_benches.obj"
out_path = "./data/scenes/cab_g_benches/cab_g_benches.nav"

def make_navmesh(inp_path,out_path):
    # read scene file
    scene_mesh = trimesh.load(inp_path, force='mesh') # open obj-file as mesh

    # Validate and repair the mesh
    if not scene_mesh.is_watertight:
        print("[WARNING] Mesh is not watertight. Attempting to repair...")
        scene_mesh.fix_normals()
        scene_mesh.remove_degenerate_faces()
        scene_mesh.remove_duplicate_faces()
        scene_mesh.remove_infinite_values()

    # create baker object
    baker = nmb.NavmeshBaker()

    # add geometry, for example a simple plane
    # the first array contains vertex positions, the second array contains polygons of the geometry
    baker.add_geometry(scene_mesh.vertices, scene_mesh.faces)

    # bake navigation mesh
    baker.bake(cell_size=0.2)

    # obtain polygonal description of the mesh
    vertices, polygons = baker.get_polygonization()

    print(vertices)
    print(polygons)

#     # Save the navigation mesh as a pickle file
#     navmesh_data = {
#         "vertices": vertices,
#         "polygons": polygons,
#     }

#     with open(out_path, "wb") as f:
#         pickle.dump(navmesh_data, f)


    # Validate the navigation mesh
    if len(vertices) == 0 or len(polygons) == 0:
        raise ValueError("[ERROR] Loaded navigation mesh is empty. Please check the baking process.")

    # with open(out_path, "rb") as f:
    #     navmesh_data = pickle.load(f)

# read_from_text(file_path)

# pathfinder = pf.PathFinder(vertices,polygons)
# # pathfinder = pf.PathFinder(navmesh_data['vertices'], navmesh_data['polygons'])

# path = pathfinder.search_path((0,0,0), (1,1,1))

# print(path)

make_navmesh(inp_path,out_path)
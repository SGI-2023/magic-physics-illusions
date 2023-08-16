import trimesh
import utils

# Load a mesh
mesh = trimesh.load_mesh('example_meshes/dumbell.obj')

# utils.plot_com(mesh,add_convex_hull=True)
utils.plot_stability_heatmap(mesh)


import trimesh
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# Load a mesh
mesh = trimesh.load_mesh('example_meshes/spot_low_resolution.obj')
# Get the center of mass
com = mesh.center_mass


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.vertices[:, 0],
                mesh.vertices[:,1],
                triangles=mesh.faces,
                Z=mesh.vertices[:,2],
                alpha=0.5) 
ax.axis("equal")
ax.scatter(com[0],com[1],com[2],color="r")





# # Get the convex hull
# convex_hull = mesh.convex_hull
# # You can also visualize the convex hull
# convex_hull.show()


plt.show()
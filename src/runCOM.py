import trimesh
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# Load a mesh
mesh = trimesh.load_mesh('example_meshes/spot_low_resolution.obj')
# Get the center of mass
com = mesh.center_mass


fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(mesh.vertices[:, 0],
                mesh.vertices[:,1],
                triangles=mesh.faces,
                Z=mesh.vertices[:,2],
                alpha=0.5) 
ax1.axis("equal")
ax1.title.set_text('COM of mesh')
ax1.scatter(com[0],com[1],com[2],color="r")


# Get the convex hull
convex_hull = mesh.convex_hull
com_ch = convex_hull.center_mass
# You can also visualize the convex hull

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_trisurf(convex_hull.vertices[:, 0],
                convex_hull.vertices[:,1],
                triangles=convex_hull.faces,
                Z=convex_hull.vertices[:,2],
                alpha=0.5) 
ax2.axis("equal")
ax2.title.set_text('COM of convex hull of mesh')
ax2.scatter(com_ch[0],com_ch[1],com_ch[2],color="r")





plt.show()
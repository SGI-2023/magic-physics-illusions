import trimesh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # used for 3D visualization 
import utils

# Load a mesh
mesh = trimesh.load_mesh('example_meshes/Bird.stl')

# utils.plot_com(mesh,add_convex_hull=True)
# utils.plot_stability_heatmap(mesh,add_convex_hull=True)


# Dot product
dp = utils.compute_stability_forall_vertices(mesh)
# Adjust the dot product values for enhanced visualization
p10, p90 = np.percentile(dp, [10, 90])
scaled_dp = (dp - p10) / (p90 - p10)
scaled_dp = np.clip(scaled_dp, 0, 1)

# Compute necessary data for plot
faces = [mesh.vertices[face] for face in mesh.faces]
colors = plt.cm.RdYlGn(dp)#! Had scaled before
face_colors = colors[mesh.faces].mean(axis=1)


fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
poly3d = Poly3DCollection(faces, facecolors=face_colors)
ax1.add_collection3d(poly3d)

# Set axis limits based on mesh extents
ax1.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
ax1.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
ax1.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
ax1.title.set_text('Stability')
ax1.axis("off")
sm = cm.ScalarMappable(cmap=plt.cm.RdYlGn)
cb = fig.colorbar(sm, ax=ax1, orientation="vertical", shrink=0.6)


#! I'm not sure if this is computed correctly, I don't see any difference wrt the previous plot
# Here add_convex_hull = True so we make a second plot
# Get the necessary information for plot
dp_ch = utils.compute_stability_forall_vertices(mesh,use_com_of_ch=True)
p10_ch, p90_ch = np.percentile(dp_ch, [10, 90])
scaled_dp_ch = (dp_ch - p10_ch) / (p90_ch - p10_ch)
scaled_dp_ch = np.clip(scaled_dp_ch, 0, 1)
colors_ch = plt.cm.RdYlGn(dp_ch)#! had scaled before
face_colors_ch = colors_ch[mesh.faces].mean(axis=1)

# Make the plot
ax2 = fig.add_subplot(132, projection='3d')
poly3d_ch = Poly3DCollection(faces, facecolors=face_colors_ch)
ax2.add_collection3d(poly3d_ch)

# Set axis limits based on mesh extents
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())
ax2.title.set_text('Stability as perceived by people')
ax2.axis("off")
sm = cm.ScalarMappable(cmap=plt.cm.RdYlGn)
cb = fig.colorbar(sm, ax=ax2, orientation="vertical", shrink=0.6)



# plot 3

diff = dp - dp_ch
p10_ch, p90_ch = np.percentile(dp_ch, [10, 90])
scaled_dp_ch = (dp_ch - p10_ch) / (p90_ch - p10_ch)
scaled_dp_ch = np.clip(scaled_dp_ch, 0, 1)
colors_ch = plt.cm.RdYlGn(diff)#! had scaled before
face_colors_ch = colors_ch[mesh.faces].mean(axis=1)

# Make the plot
ax3 = fig.add_subplot(133, projection='3d')
poly3d_ch = Poly3DCollection(faces, facecolors=face_colors_ch)
ax3.add_collection3d(poly3d_ch)

# Set axis limits based on mesh extents
ax3.set_xlim(ax1.get_xlim())
ax3.set_ylim(ax1.get_ylim())
ax3.set_zlim(ax1.get_zlim())
ax3.title.set_text('Difference')
ax3.axis("off")
sm = cm.ScalarMappable(cmap=plt.cm.RdYlGn)
cb = fig.colorbar(sm, ax=ax3, orientation="vertical", shrink=0.6)

plt.savefig("results/Bird.png")


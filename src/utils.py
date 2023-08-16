import trimesh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # used for 3D visualization 

def plot_com(mesh, add_convex_hull: bool = False):
    """
    Plots the center of mass of the given mesh.

    Parameters:
            mesh: Mesh
            add_convex_hull: If set to true, it does the same thing for the convex hull
    """

    com = mesh.center_mass
    if add_convex_hull:
        plot_layout = 120
    else:
        plot_layout = 110

    fig = plt.figure()
    ax1 = fig.add_subplot(plot_layout+1, projection='3d')
    ax1.plot_trisurf(mesh.vertices[:, 0],
                    mesh.vertices[:,1],
                    triangles=mesh.faces,
                    Z=mesh.vertices[:,2],
                    alpha=0.5) 
    ax1.axis("equal")
    ax1.title.set_text('COM of mesh')
    ax1.scatter(com[0],com[1],com[2],color="r")

    if not add_convex_hull:
        plt.show()
        return
    
    # Here add_convex_hull = True so we make two plots
    # Get the convex hull
    convex_hull = mesh.convex_hull
    com_ch = convex_hull.center_mass
    # Visualize the convex hull
    ax2 = fig.add_subplot(plot_layout+2, projection='3d')
    ax2.plot_trisurf(convex_hull.vertices[:, 0],
                    convex_hull.vertices[:,1],
                    triangles=convex_hull.faces,
                    Z=convex_hull.vertices[:,2],
                    alpha=0.5) 
    ax2.axis("equal")
    ax2.title.set_text('COM of convex hull of mesh')
    ax2.scatter(com_ch[0],com_ch[1],com_ch[2],color="r")
    plt.show()

def compute_stability_forall_vertices(mesh,use_com_of_ch: bool = False):
    """
    For each vertex p of the mesh, it computes the vector v from p to the com and computes the dot product with the vertex normal.

    Parameters:
            mesh: mesh
            use_com_of_ch: If set to true, it uses the com of the convex hull instead
    """


    # Get the centers of mass
    if use_com_of_ch:
        COM = mesh.convex_hull.center_mass
    else:
        COM = mesh.center_mass
    # Get the unit vertex normals
    vertex_normals = mesh.vertex_normals
    # Normalized vectores to COM
    vertex_to_COM = COM - mesh.vertices
    norm = np.linalg.norm(vertex_to_COM, axis=1)[:, None]
    vertex_to_COM = vertex_to_COM/norm

    # Dot products
    return np.einsum('ij,ij->i', vertex_normals, vertex_to_COM)


def plot_stability_heatmap(mesh,add_convex_hull: bool = False):
    """
    Plots the mesh with a heatmap representing the stability of each point

    Parameters:
            mesh: Mesh
            add_convex_hull: If set to true, it adds a second plot for the convex hull
    """

    # Dot product
    dp = compute_stability_forall_vertices(mesh)
    # Adjust the dot product values for enhanced visualization
    p10, p90 = np.percentile(dp, [10, 90])
    scaled_dp = (dp - p10) / (p90 - p10)
    scaled_dp = np.clip(scaled_dp, 0, 1)

    # Compute necessary data for plot
    faces = [mesh.vertices[face] for face in mesh.faces]
    colors = plt.cm.RdYlGn(dp)#! Had scaled before
    face_colors = colors[mesh.faces].mean(axis=1)

    # Make the plot
    if add_convex_hull:
        plot_layout = 120 # Two subplots
    else:
        plot_layout = 110 # One plot

    fig = plt.figure()
    ax1 = fig.add_subplot(plot_layout+1, projection='3d')
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

    if not add_convex_hull:
        plt.show()
        return
    
    #! I'm not sure if this is computed correctly, I don't see any difference wrt the previous plot
    # Here add_convex_hull = True so we make a second plot
    # Get the necessary information for plot
    dp_ch = compute_stability_forall_vertices(mesh,use_com_of_ch=True)
    p10_ch, p90_ch = np.percentile(dp_ch, [10, 90])
    scaled_dp_ch = (dp_ch - p10_ch) / (p90_ch - p10_ch)
    scaled_dp_ch = np.clip(scaled_dp_ch, 0, 1)
    colors_ch = plt.cm.RdYlGn(dp_ch)#! had scaled before
    face_colors_ch = colors_ch[mesh.faces].mean(axis=1)

    # Make the plot
    ax2 = fig.add_subplot(plot_layout+2, projection='3d')
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

    plt.show()







    



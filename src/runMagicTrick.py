import trimesh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # used for 3D visualization 
import utils
import os, shutil, json


w=2; wch = 1
def make_plot(mesh,mesh_name="",plot_path: str = None):
    # Makes a plot of a particular mesh and finds the "best" vertex to do the balancing trick
    dp = utils.compute_stability_forall_vertices(mesh)
    dp_ch = utils.compute_stability_forall_vertices(mesh,use_com_of_ch=True)
    idx = np.nanargmax(w*dp-wch*dp_ch)

    com = mesh.center_mass
    com_ch = mesh.convex_hull.center_mass
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_trisurf(mesh.vertices[:, 0],
                    mesh.vertices[:,1],
                    triangles=mesh.faces,
                    Z=mesh.vertices[:,2],
                    alpha=0.1) 
    ax1.axis("equal")
    ax1.axis("off")
    ax1.title.set_text(f'Mesh {mesh_name} point {idx}')
    vertex = mesh.vertices[idx,:]
    # Pto_plot = np.array([com,com_ch,vertex])
    ax1.scatter(*com,marker="o",label="com",color="b")
    ax1.scatter(*com_ch,marker="s",label="com_ch",color="k")
    ax1.scatter(*vertex,marker="*",label="vertex",color="r")
    ax1.legend()

    if plot_path == None:
        plt.show()
    else:
        plt.savefig(plot_path)

# def remesh_dataset(dataset_path,new_dir):
#     # Remesh every object in a folder
#     for filename in os.listdir(dataset_path):
#         if filename == ".DS_Store":
#             continue

#         mesh_path = os.path.join(dataset_path,filename)
#         ms = pymeshlab.MeshSet()
#         ms.load_new_mesh(mesh_path)
#         ms.meshing_isotropic_explicit_remeshing(adaptive=True,targetlen=pymeshlab.Percentage(0.1))
#         new_mesh_path = os.path.join(new_dir,filename)
#         ms.save_current_mesh(new_mesh_path)

def write_mesh_info(mesh,text_file,mesh_name="",diff_bound=0.3):
    # Add line to a txt file with information for that particular mesh
    dp = utils.compute_stability_forall_vertices(mesh)
    dp_ch = utils.compute_stability_forall_vertices(mesh,use_com_of_ch=True)
    # w=2; wch = 1
    v_idx = np.nanargmax(w*dp-wch*dp_ch)
    dp_r = np.round(dp[v_idx],decimals=2) 
    dp_ch_r = np.round(dp_ch[v_idx],decimals=2)
    diff = np.round(dp_r-dp_ch_r,decimals=2)
    if diff < diff_bound:
        return
    with open(text_file,"a") as f:
        f.write(f"Mesh {mesh_name} -- at v_idx={v_idx} we have dp={dp_r}, dp_ch={dp_ch_r} with diff {diff}\n")

def rotate_mesh(mesh):
    # Find special vertex
    dp = utils.compute_stability_forall_vertices(mesh)
    dp_ch = utils.compute_stability_forall_vertices(mesh,use_com_of_ch=True)
    w=2; wch = 1
    v_idx = np.nanargmax(w*dp-wch*dp_ch)
    vertex = mesh.vertices[v_idx]
    # Find vectors that you want to rotate
    com = mesh.center_mass
    v1 = com - vertex
    v1 = v1/np.linalg.norm(v1)
    v2 = np.array([0,-1,0])
    # Find rotation that takes v1 into v2
    w = np.cross(v1,v2)
    M = np.array([
        [0,-w[2],w[1]],
        [w[2],0,-w[0]],
        [-w[1],w[0],0]
    ])
    c = np.dot(v1,v2)
    R = np.eye(3) + M + M@M/(1+c) #Rotation that takes v1 to v2
    # Transform the vertices appropriately
    mesh.vertices = vertex + (mesh.vertices - vertex) @ np.transpose(R)

# def get_shapenet_dirs(name: str):
#     """Given a common name, finds the list of directories associated with it"""

#     result = []
#     with open("ShapeNetCore.v2/taxonomy.json","r") as f:
#         data = json.load(f)
#         for synset in data:
#             if name in synset["name"]:
#                 path = f"ShapeNetCore.v2/{synset['synsetId']}"

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

            
################################################################################################################

# mesh_path = "ShapeNetCore.v2/02691156/64dbeac5250de2bfc5d50480246d2c40" # Airplane
# mesh_path = "ShapeNetCore.v2/03001627/1c685bc2a93f87a2504721639e19f609" #Chair1
# mesh_path = "ShapeNetCore.v2/03001627/212e266cab46f1dd5b903ba10d2ec446" #Chair2
# mesh_path = "/Users/unaicaja/Desktop/airplane.obj" # Remeshed airplane
# mesh_path = "ShapeNetCore.v2/03046257/929afe0b3ad23608c7b37fe5d21f2385"  # Clock
mesh_path = "ShapeNetCore.v2/03636649/a21744a28956c93cc5a208307819a3a1" #Lamp1
# mesh_path = "ShapeNetCore.v2/03636649/763ceeaa4ce2ce1d6afaa22adc38077c" #Lamp2
# mesh_path = "ShapeNetCore.v2/03636649/8425fcfab0cd9e601f0d826a92e0299f" #Lamp3


mesh_path = os.path.join(mesh_path,"models/model_normalized.obj")
mesh = trimesh.load_mesh(mesh_path)
mesh = as_mesh(mesh)
# write_mesh_info(mesh,"ShapeNet_results/summary.txt","Remeshed airplane",diff_bound=0)
make_plot(mesh)


################################################################################################################
# # SEARCHING A SYNSET IN SHAPENET
# name = "bed"
# f = open("ShapeNetCore.v2/taxonomy.json","r")
# data = json.load(f)
# # Build a map with keys being all directories to study and values being
# dirs_to_study = []
# for synset in data:
#     if not name in synset["name"]:
#         continue
    
#     id = synset['synsetId']
#     path = f"ShapeNetCore.v2/{id}"
#     dirs_to_study.append(path)
# f.close()

# count = 0
# for path in dirs_to_study:
#     if not os.path.isdir(path):
#         print(f"CAREFUL: DIRECTORY {path} DOES NOT EXIST")
#         continue

#     for id in os.listdir(path):
#         dir = os.path.join(path,id)
#         if not os.path.isdir(dir):
#             continue
        
#         mesh_path = os.path.join(dir,"models/model_normalized.obj")
#         scene_or_mesh = trimesh.load_mesh(mesh_path)
#         mesh = as_mesh(scene_or_mesh)

#         if mesh is None:
#             print(f"{dir} seems to have no geometry")
#             continue

#         print(f"Is {dir} watertight: {mesh.is_watertight}")
#         if not mesh.is_watertight:
#             continue
        
#         text_file = "ShapeNet_results/summary.txt"
#         write_mesh_info(mesh=mesh,mesh_name=dir,text_file=text_file)
        

################################################################################################################
# TRY TO MAKE COLORFUL PLOT
# dp = utils.compute_stability_forall_vertices(mesh)
# colors = plt.cm.RdYlGn(dp)
# mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh,vertex_colors=colors)
# x=mesh.show(
#     smooth=True
# )

# mesh_path = f"fixed_hands/hand_29.obj"
# mesh = trimesh.load_mesh(mesh_path)
# rotate_mesh(mesh)
# new_mesh_path = f"meshes_to_render/hand_29.obj"
# mesh.export(new_mesh_path)
# trimesh.export(mesh,new_mesh_path)

################################################################################################################
# LOOKING FOR INTERESTING POINTS IN DATASET

# dataset_path = "/Users/mac/Documents/GitHub/magic-physics-illusions/shrek_dataset/test_normal/test_normal"
# dataset_path = "meshes_to_examine"
# text_file = "results/shrek1.txt"
# mesh_names = os.listdir(dataset_path)
# differences = np.zeros(len(mesh_names))
# for i,filename in enumerate(mesh_names):
#     if filename == ".DS_Store":
#         continue

#     mesh_path = os.path.join(dataset_path,filename)
#     mesh = trimesh.load_mesh(mesh_path)
#     dp = utils.compute_stability_forall_vertices(mesh)
#     dp_ch = utils.compute_stability_forall_vertices(mesh,use_com_of_ch=True)
#     v_idx = np.argmax(dp-dp_ch)
#     differences[i] = dp[v_idx] - dp_ch[v_idx]

# for i in range(len(mesh_names)):
#     if differences[i] <= 0.3 or dp[i] <= 0.5:
#         continue

#     mesh_path = os.path.join(dataset_path,mesh_names[i])
#     mesh = trimesh.load_mesh(mesh_path)
#     make_plot(mesh,i,mesh_names[i])#,plot_path=f"results/{mesh_names[i]}_idx{i}.png"







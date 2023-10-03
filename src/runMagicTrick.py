import bpy, bmesh, mathutils
import math
import trimesh
import numpy as np, scipy as sp

mesh_name = "Grimm"
armature_name = "Armature"

# From stackoverflow (for printing into console here)
def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")    

#################################################

# Get the original and deformed meshes
mesh = bpy.data.objects[mesh_name]  # (basically just needed for local to global coord matrix)
def get_deformed_mesh():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return mesh.evaluated_get(depsgraph).to_mesh()
#################################################
# Define energy (objective) function
def energy():
    deformed_mesh = get_deformed_mesh()
    local_nose = (deformed_mesh.vertices[48].co + deformed_mesh.vertices[66].co) / 2.0
    world_nose = mesh.matrix_world @ local_nose
    return (target.location - world_nose).length
#################################################
# Depth first search through bones
def dfs_bone(bone,v_idx,mean=0.01,dev=0.005,reg_coef = 0.01):
    '''Coordinate descent for each of the bone angles'''
    
    step_size = np.random.normal(loc=mean,scale=dev)
    # Save the initial angles for regularization purposes
    try:
        bone["initial_angles"] = bone["initial_angles"]
    except:
        bone["initial_angles"] = [bone.rotation_euler[i] for i in range(3)]
    
    bone.rotation_mode = 'XYZ'
    for i in range(3):
        # Compute energies for each update choice
        old_angle = bone.rotation_euler[i]
        old_energy = objective_function(v_idx)
        # Make update and copute new energies
        bone.rotation_euler[i] += step_size
        plus_angle = bone.rotation_euler[i]
        plus_reg = (bone["initial_angles"][i] - plus_angle)**2*reg_coef
        plus_energy = objective_function(v_idx) + plus_reg
        bone.rotation_euler[i] -= 2 * step_size
        minus_angle = bone.rotation_euler[i]
        minus_reg = (bone["initial_angles"][i] - minus_angle)**2*reg_coef
        minus_energy = objective_function(v_idx) + minus_reg
        # Select energy minimizing choice
        if plus_energy < min(old_energy, minus_energy):
            bone.rotation_euler[i] = plus_angle
        elif minus_energy < max(old_energy, plus_energy):
            bone.rotation_euler[i] = minus_angle
        else:
            bone.rotation_euler[i] = old_angle
            
#        if bone.name == 'top_fin':
#            print(('old angle & energy:', old_angle, old_energy))
#            print(('new angle & energy:', bone.rotation_euler[i], energy()))
    # Recurse
    for child in bone.children:
        dfs_bone(child,v_idx)
#################################################

def approx_grad_descent(bone,v_idx,step,reg_coef = 0.01):
    '''Does one gradient descent step on each angle for the bone'''
    
    # Save the initial angles for regularization purposes
    try:
        bone["initial_angles"] = bone["initial_angles"]
    except:
        bone["initial_angles"] = [bone.rotation_euler[i] for i in range(3)]
    
    bone.rotation_mode = 'XYZ'
    for i in range(3):
        # Compute energies at two points
        bone.rotation_euler[i] += step
        plus_angle = bone.rotation_euler[i]
        plus_reg = (bone["initial_angles"][i] - plus_angle)**2*reg_coef
        plus_energy = objective_function(v_idx) + plus_reg
        # Repeat
        bone.rotation_euler[i] -= 2 * step
        minus_angle = bone.rotation_euler[i]
        minus_reg = (bone["initial_angles"][i] - minus_angle)**2*reg_coef
        minus_energy = objective_function(v_idx) + minus_reg
        # Set the angle back to its original value
        bone.rotation_euler[i] += step
        # Approximate the derivative at that point
        deriv = (plus_energy - minus_energy)/(2*step)
        #Update the angle
        bone.rotation_euler[i] += step*deriv   
    # Recurse
    for child in bone.children:
        dfs_bone(child,v_idx)

#################################################
def get_trimesh_object():
    # Convert mesh to bmesh
    dmesh = get_deformed_mesh()
    bm = bmesh.new()
    bm.from_mesh(dmesh)
    # Get the coordinates of all vertices
    M = mesh.matrix_world
    V = np.array([M @ v.co for v in bm.verts])
    # Get face list
    F = [
    [v.index for v in f.verts]
    for f in bm.faces
    ]
    # Get the face normals
    N = [np.array(f.normal) for f in bm.faces]
    
    return trimesh.Trimesh(vertices=V,faces=F,face_normals=N)

#################################################

def compute_dot_products(v_idx):
    tmesh = get_trimesh_object()
    COM = tmesh.center_mass
    COM_CH = tmesh.convex_hull.center_mass
    # Get the vertex normal
    vertex_normal = tmesh.vertex_normals[v_idx]
    vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)
    # Vectors going to com and com_ch
    vertex_to_com = COM - tmesh.vertices[v_idx]
    vertex_to_com = vertex_to_com/np.linalg.norm(vertex_to_com)
    vertex_to_com_ch = COM_CH - tmesh.vertices[v_idx]
    vertex_to_com_ch = vertex_to_com_ch/np.linalg.norm(vertex_to_com_ch)
    # Compute the dot produtcs
    dp = np.dot(vertex_to_com,vertex_normal)
    dp_ch = np.dot(vertex_to_com_ch,vertex_normal)
    return dp, dp_ch
#################################################

def compute_all_dot_products():
    tmesh = get_trimesh_object()
    COM = tmesh.center_mass
    COM_CH = tmesh.convex_hull.center_mass
    # Get the vertex normal
    vertex_normals = tmesh.vertex_normals
    norm = np.linalg.norm(vertex_normals, axis=1)[:, None]
    vertex_normals = vertex_normals/norm
    # Vectors going to com and com_ch
    vertex_to_com = COM - tmesh.vertices
    norm = np.linalg.norm(vertex_to_com, axis=1)[:, None]
    vertex_to_com = vertex_to_com/norm
    vertex_to_com_ch = COM_CH - tmesh.vertices
    norm = np.linalg.norm(vertex_to_com_ch, axis=1)[:, None]
    vertex_to_com_ch = vertex_to_com_ch/norm
    # Compute the dot produtcs
    # Dot products
    dp = np.einsum('ij,ij->i', vertex_normals,vertex_to_com)
    dp_ch = np.einsum('ij,ij->i', vertex_normals,vertex_to_com_ch)
    return dp, dp_ch

#################################################

def find_best_index(w1=2,w2=1):
    dp, dp_ch = compute_all_dot_products()
    return np.argmax(w1*dp - w2*dp_ch)

#################################################
def objective_function(v_idx,w1 = 2, w2 = 1):  
    dp, dp_ch = compute_dot_products(v_idx)
    # Return the weighted difference
    return -(w1*dp - w2*dp_ch)

#################################################
def show_vertex(v_idx,ball_size=0.003):
    """Creates an icosahedron at the location of a point"""
    # Get the mesh
    dmesh = get_deformed_mesh()
    bm = bmesh.new()
    bm.from_mesh(dmesh)
    # Get the coordinates of the vertex
    bm.verts.ensure_lookup_table()
    point = mesh.matrix_world @ bm.verts[v_idx].co
    # Draw the icosahedron
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1,
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=point,
                                        scale=(ball_size,ball_size,ball_size))

#################################################
def show_com(ball_size=0.003):
    """Creates an icosahedron at the location of a point"""
    # Get the mesh
    tmesh = get_trimesh_object()
    COM = tmesh.center_mass
    # Draw the icosahedron
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1,
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=COM,
                                        scale=(ball_size,ball_size,ball_size))

#################################################
def run_optimization(epochs,step,reg_coef,v_idx=None,delta=0.0001):
    # Get the armature & target
    armature = bpy.data.objects.get(armature_name)
    # Set mode to pose + get bone (assuming armature is selected)
    bpy.ops.object.mode_set(mode='POSE')
    # Select vertex for the optimization
    if v_idx == None:
        v_idx = find_best_index()
    print("The chosen index is {idx}".format(idx=v_idx))
    dp_best,dp_ch_best = compute_dot_products(v_idx)
    print("With values dp={a}, dp_ch={b}".format(a=dp_best,b=dp_ch_best))
    # Print initial values of dot products
    dp, dp_ch = compute_dot_products(v_idx)
    print("Initial values dp:{a}, dp_ch:{b}".format(a=dp,b=dp_ch))
    # Run the optimzation loop
    for i in range(epochs):
        old_energy = objective_function(v_idx)
        for bone in armature.pose.bones:
            if not bone.parent:
                approx_grad_descent(bone,v_idx,step,reg_coef)
        # Termination criteria
        if objective_function(v_idx) - old_energy > delta:
            print('stopped after %i epochs' % i)
            break
    # Print final values of dot products
    dp, dp_ch = compute_dot_products(v_idx)
    print("Final values dp:{a}, dp_ch:{b}".format(a=dp,b=dp_ch))
    # Update the scene to reflect the changes
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    return v_idx

#################################################
def find_rotation_matrix(v1,v2,p):
    """
    Computes the matrix of a rotation with origin at p.
    The transforation maps v1 to v2 and is stored as a 4x4 matrix.
    v1 and v2 are assumed to be unit vectors.
    """
    
    w = np.cross(v1,v2)
    M = np.array([
        [0,-w[2],w[1]],
        [w[2],0,-w[0]],
        [-w[1],w[0],0]
    ])
    c = np.dot(v1,v2)
    R = np.eye(3) + M + M@M/(1+c) #Rotation that takes v1 to v2
    b = p - R@p
    T = np.ones((4,4))
    T[0:3,0:3] = R
    T[3,0:3] = b
    T[0:3,3] = b
    
    return mathutils.Matrix(T)

    
    

#################################################

epochs = 20;step = 0.2;reg_coef = 1
v_idx = 8457
v_idx = run_optimization(epochs,step,reg_coef,v_idx)
show_vertex(v_idx)
show_com()

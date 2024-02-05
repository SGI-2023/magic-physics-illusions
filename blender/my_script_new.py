import bpy, bmesh, mathutils
import math, mathutils
import numpy as np, scipy as sp
import matplotlib.pyplot as plt
import os, pickle
import time
import copy
import trimesh

mesh_name = "_t_obj_AnonymousMesh0"
armature_name = "Armature"
my_dir = "/Users/adityaabhyankar/Desktop/Programming/Magic-Physics-Illusions/magic-physics-illusions/blender"
#target = bpy.data.objects["target"]

# From stackoverflow (for printing into console here)
# NOW GIVING ERROR, FOR SOME REASON
#def print(data):
#    for window in bpy.context.window_manager.windows:
#        screen = window.screen
#        for area in screen.areas:
#            if area.type == 'CONSOLE':
#                override = {'window': window, 'screen': screen, 'area': area}
#                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")    

#################################################

# Get the original and deformed meshes
mesh = bpy.data.objects[mesh_name]  # (basically just needed for local to global coord matrix)
def get_deformed_mesh():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return mesh.evaluated_get(depsgraph).to_mesh()
#################################################
# Simple target-based energy (objective) function for testing purposes
def energy():
    deformed_mesh = get_deformed_mesh()
    local_nose = (deformed_mesh.vertices[48].co + deformed_mesh.vertices[66].co) / 2.0
    world_nose = mesh.matrix_world @ local_nose
    return (target.location - world_nose).length
#################################################
# Old optimzation algorithm (CCD) for modifying armature via Depth First Search
def dfs_bone(bone,v_idx,mean=0.01,dev=0.005,reg_coef = 0.01):
    '''Coordinate descent for each of the bone angles'''
    step_size = np.random.normal(loc=mean,scale=dev)
    # Save the initial angles for regularization purposes
    try:
        bone["initial_angles"] = bone["initial_angles"]  # do nothing if already exists
    except:
        bone["initial_angles"] = [bone.rotation_euler[i] for i in range(3)]  # recursive base case
    
    # Actual optimization. For this bone, iterate through the 3 Euler Angles.
    bone.rotation_mode = 'XYZ'
    for i in range(3):
        # Compute old energies
        old_angle = bone.rotation_euler[i]
        old_energy = energy() # objective_function(v_idx)  # TODO: Change to actual objective function!
        # Make update and compute new energies
        bone.rotation_euler[i] += step_size
        plus_angle = bone.rotation_euler[i]
        plus_reg = (bone["initial_angles"][i] - plus_angle)**2*reg_coef
        plus_energy = energy() # objective_function(v_idx) + plus_reg   # TODO: CHange to actual obj function!
        bone.rotation_euler[i] -= 2 * step_size
        minus_angle = bone.rotation_euler[i]
        minus_reg = (bone["initial_angles"][i] - minus_angle)**2*reg_coef
        minus_energy = energy() # objective_function(v_idx) + minus_reg  # TODO: Change to actual obj function!
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
# DEPRECATED
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
def get_com():
    tmesh = get_trimesh_object()
    return tmesh.center_mass
#    # Select the object to duplicate
#    bpy.context.view_layer.objects.active = mesh
#    bpy.ops.object.mode_set(mode='OBJECT')
#    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
#    mesh.select_set(True)  # Select the object you want to duplicate
#    bpy.context.view_layer.objects.active = mesh  # make it "active"
#    bpy.ops.object.duplicate()  # duplicate it
#    duplicated_obj = bpy.context.active_object  # get the duplicated object, which is the active one
#    # Apply armature modifier to duplicate
#    bpy.ops.object.modifier_apply(modifier=armature_name)
#    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
#    # Get COM of duplicate
#    COM = copy.copy(duplicated_obj.location)
#    world_rotation = mesh.matrix_world.to_3x3()#.normalized() # ADDED BY UNAI DEBUGGING
#    COM = world_rotation @ COM # ADDED BY UNAI DEBUGGING
#    # Delete duplicate
#    bpy.ops.object.delete()
#    return np.array(COM)

#################################################
def get_comch():
    tmesh = get_trimesh_object()
    return tmesh.convex_hull.center_mass
#    # Select the object to duplicate
#    bpy.context.view_layer.objects.active = mesh
#    bpy.ops.object.mode_set(mode='OBJECT')
#    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
#    mesh.select_set(True)  # Select the object you want to duplicate
#    bpy.context.view_layer.objects.active = mesh  # make it "active"
#    bpy.ops.object.duplicate()  # duplicate it
#    duplicated_obj = bpy.context.active_object
#    # Apply armature modifier
#    bpy.ops.object.modifier_apply(modifier=armature_name)
#    # Get the Convex Hull
#    bpy.ops.object.mode_set(mode='EDIT')
#    bpy.ops.mesh.convex_hull()
#    bpy.ops.object.mode_set(mode='OBJECT')
#    # Get COM of Convex Hull
#    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
#    COM = copy.copy(duplicated_obj.location)
#    world_rotation = mesh.matrix_world.to_3x3()# # ADDED BY UNAI DEBUGGING
#    COM = world_rotation @ COM # ADDED BY UNAI DEBUGGING
#    # Delete duplicate (i.e. the convex hull mesh)
#    bpy.ops.object.delete()
#    return np.array(COM)
    

#################################################

# OLD METHOD
#def compute_dot_products(v_idx):
#    tmesh = get_trimesh_object()
#    COM = tmesh.center_mass
#    COM_CH = tmesh.convex_hull.center_mass
#    # Get the vertex normal
#    vertex_normal = tmesh.vertex_normals[v_idx]
#    vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)
#    # Vectors going to com and com_ch
#    vertex_to_com = COM - tmesh.vertices[v_idx]
#    vertex_to_com = vertex_to_com/np.linalg.norm(vertex_to_com)
#    vertex_to_com_ch = COM_CH - tmesh.vertices[v_idx]
#    vertex_to_com_ch = vertex_to_com_ch/np.linalg.norm(vertex_to_com_ch)
#    # Compute the dot produtcs
#    dp = np.dot(vertex_to_com,vertex_normal)
#    dp_ch = np.dot(vertex_to_com_ch,vertex_normal)
#    return dp, dp_ch

# NEW METHOD
def compute_dps_and_distances(v_idx):
    tmesh = get_trimesh_object()
    COM = tmesh.center_mass
    COM_CH = tmesh.convex_hull.center_mass

    # Distance from vertex to centers of mass
    v_idx = v_idx
    vertex = tmesh.vertices[v_idx]
    dist_com = np.linalg.norm(vertex-COM)
    dist_com_ch = np.linalg.norm(vertex-COM_CH)

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

    return dp, dp_ch, dist_com, dist_com_ch

# #################################################



#    bm = bmesh.new()
#    bm.from_mesh(get_deformed_mesh())
#    bm.verts.ensure_lookup_table()
#    COM = get_com()
#    COM_CH = get_comch()
#    # Get the vertex normal
#    world_rotation = mesh.matrix_world.to_3x3()#.normalized()
#    vertex_normal = world_rotation @ bm.verts[v_idx].normal
#    vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)
#    # Vectors going to com and com_ch
#    vertex_to_com = COM - (mesh.matrix_world @ bm.verts[v_idx].co)
#    vertex_to_com = vertex_to_com/np.linalg.norm(vertex_to_com)
#    vertex_to_com_ch = COM_CH - (mesh.matrix_world @ bm.verts[v_idx].co)
#    vertex_to_com_ch = vertex_to_com_ch/np.linalg.norm(vertex_to_com_ch)
#    # Compute the dot produtcs
#    dp = np.dot(vertex_to_com,vertex_normal)
#    dp_ch = np.dot(vertex_to_com_ch,vertex_normal)
#    return dp, dp_ch
#################################################

def compute_all_dot_products(): # VERTEX NORMALS NOT UNITARY.
#    tmesh = get_trimesh_object()
    bm = bmesh.new()
    bm.from_mesh(get_deformed_mesh())
    bm.verts.ensure_lookup_table()
    COM = get_com()
    COM_CH = get_comch()
    # Get the vertex normal
    world_rotation = mesh.matrix_world.to_3x3()#.normalized()
    vertex_normals = [world_rotation @ vert.normal for vert in bm.verts]
    # Vectors going to com and com_ch
    vertices = [mesh.matrix_world @ np.array(vertex.co) for vertex in bm.verts]
    vertex_to_com = COM - vertices
    norm = np.linalg.norm(vertex_to_com, axis=1)[:, None]
    vertex_to_com = vertex_to_com/norm
    vertex_to_com_ch = COM_CH - vertices
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
    v_idx = np.argmax(w1*dp - w2*dp_ch)
    return v_idx, dp[v_idx], dp_ch[v_idx]

#################################################
#def objective_function(v_idx,w1 = 2, w2 = 1,barrier=None):
#    dp, dp_ch = compute_dot_products(v_idx)
#    # Return the weighted difference
#    # More dp alignment = less energy. More dp_ch alignment = more energy.
#    # Best possible case: dp=1, dp_ch=-1, so value=-(w1 - w2).
#    
#    # Once we have a nice value of dp, say dp = 0.98 we add a term to
#    # the objective function which will force dp >= barrier (for example 0.95)
#    if barrier == None:
#        return w2*dp_ch - w1*dp
#    if dp - barrier <= 0:
#        raise Exception("Constraint violated in barrier method")
#    return w2*dp_ch - w1*np.log(dp - barrier)

def objective_function(v_idx,w1 = 2, w2 = 1,barrier=None):
#    dp, dp_ch = compute_dot_products(v_idx)
    # Return the weighted difference
    # More dp alignment = less energy. More dp_ch alignment = more energy.
    # Best possible case: dp=1, dp_ch=-1, so value=-(w1 - w2).
    
    
#    w1 = 0.#100.
#    w2 = 0.#30.
#    w3 = 0.
#    dist = np.linalg.norm(get_com() - get_comch())
#    dist = np.power(dist - w3, 2.) if dist < w3 else 0.
#    return (w1 * np.power(dp - 1., 2.)) + (w2 * np.power(dp_ch + 1., 2.))# + (dist / w3)
    
    dp, dp_ch, dist_com, dist_com_ch = compute_dps_and_distances(v_idx)
    return w1 * (dp-1)**2 * dist_com**2 + w2 * (dp_ch + 1)**2 * dist_com_ch**2

#################################################
def show_vertex(v_idx,ball_size=0.003):
    """Creates an icosahedron at the location of a vertex of the mesh"""
    # Get the mesh
    dmesh = get_deformed_mesh()
    bm = bmesh.new()
    bm.from_mesh(dmesh)
    # Get the WORLD coordinates of the vertex
    bm.verts.ensure_lookup_table()
    point = mesh.matrix_world @ bm.verts[v_idx].co
    # Draw the icosahedron
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1,
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=point,
                                        scale=(ball_size,ball_size,ball_size))
    new_icosphere = bpy.context.active_object
    new_icosphere.name = 'VERTEX'

#################################################
def show_com(ball_size=0.003):
    """Creates an icosahedron at the location of the COM of the mesh"""
    # Get the mesh
#    tmesh = get_trimesh_object()
    COM = get_com()
    COM = mathutils.Vector(COM)
    # Draw the icosahedron
#    COM = mesh.matrix_world @ COM
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1,
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=COM,
                                        scale=(ball_size,ball_size,ball_size))
                                        
    new_icosphere = bpy.context.active_object
    new_icosphere.name = 'COM'   


#################################################
def show_comch(ball_size=0.003):
    """Creates an icosahedron at the location of the COM of the mesh"""
    # Get the mesh
#    tmesh = get_trimesh_object()
    COM = get_comch()
    COM = mathutils.Vector(COM)
    # Draw the icosahedron
#    COM = mesh.matrix_world @ COM
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1,
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=COM,
                                        scale=(ball_size,ball_size,ball_size))
                                        
    new_icosphere = bpy.context.active_object
    new_icosphere.name = 'COM_CH'                      
    
                
#################################################
# Old optimization routine for testing
def run_target_optimization(epochs,step,reg_coef,v_idx=None,delta=0.0001):
    # Get the armature & target
    armature = bpy.data.objects.get(armature_name)
    # Set mode to pose + get bone (assuming armature is selected)
    bpy.ops.object.mode_set(mode='POSE')
    
    # Run the optimzation loop
    for i in range(epochs):
        old_energy = energy()
        for bone in armature.pose.bones:
            if not bone.parent:
                approx_grad_descent(bone,v_idx,step,reg_coef)
        # Termination criteria
        if abs(energy() - old_energy) < delta:
            break
    print('stopped after %i epochs' % i)
    
#################################################
# Method that reads the original angles for the armature from a text file and
# returns the angles as a dictionary
def get_bone_angles():
    # Set the armature as the active object
    armature = bpy.data.objects.get(armature_name)
    bpy.context.view_layer.objects.active = armature
    # First check if the text file exists
    angle_file_name = os.path.join(my_dir,armature_name+"_angles.pkl")
    if os.path.isfile(angle_file_name):
        # If the file exists, we just read it and return it
        print('Used angle file')
        with open(angle_file_name, 'rb') as file:
            bone_angles = pickle.load(file)
            return bone_angles
    # If there is no angle file, we make one
    # Get the armature
    armature = bpy.data.objects.get(armature_name)
    # Set mode to pose + get bone (assuming armature is selected)
    bpy.ops.object.mode_set(mode='POSE')
    # Dictionary in which we will sabe the angles
    bone_angles = {}
    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone_angles[bone.name] = bone.rotation_euler[:]
    # Save the dictionary
    with open(angle_file_name, 'wb') as file:
        pickle.dump(bone_angles, file)
    return bone_angles

#################################################
# Saves the current bone angles to a file and restores the 
# angles to the previous position
def save_bone_angles(filepath):
    # Get the armature
    armature = bpy.data.objects.get(armature_name)
    # Dictionary in which we will sabe the angles
    bone_angles = {}
    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone_angles[bone.name] = [0]*3
        for i in range(3):
            bone_angles[bone.name][i] = bone.rotation_euler[i]
    # Save the dictionary
    with open(filepath, 'wb') as file:
        pickle.dump(bone_angles, file)
        
#################################################
# Sets the angles of the armature to whatever they where at the beginning
def reset_bone_angles():
    # Get the armature
    initial_angles = get_bone_angles()
    armature = bpy.data.objects.get(armature_name)
    # Set mode to pose + get bone (assuming armature is selected)
    bpy.ops.object.mode_set(mode='POSE')
    # Initial angles
    # Dictionary in which we will sabe the angles
    bone_angles = {}
    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'
        for i in range(3):
            a = bone.rotation_euler[i]
            bone.rotation_euler[i] += (initial_angles[bone.name][i] - a)

#################################################
# sets the bone angles equal to the ones stored in a file
def set_bone_angles(filepath):
    if not os.path.isfile(filepath):
        raise Exception("No angles file found")
        
    with open(filepath, 'rb') as file:
        bone_angles = pickle.load(file)
        # Get the armature
        armature = bpy.data.objects.get(armature_name)
        # Set mode to pose + get bone (assuming armature is selected)
        bpy.ops.object.mode_set(mode='POSE')
        for bone in armature.pose.bones:
            bone.rotation_mode = 'XYZ'
            for i in range(3):
                a = bone.rotation_euler[i]
                bone.rotation_euler[i] += (bone_angles[bone.name][i] - a)
        
#################################################
# New optimization algo (gradient descent instead of fixed step-size)
def approx_grad_descent(bone,v_idx,step,bone_angles,reg_coef = 0.01,w1=2,w2=1,barrier=None):
    '''Does one gradient descent step on each angle for the bone'''
    
    # Read the initial angles
    initial_angles = bone_angles[bone.name]
    # Actual optimization
    bone.rotation_mode = 'XYZ'
    for i in range(3):
        # Compute energies at two points
        bone.rotation_euler[i] += step
        plus_angle = bone.rotation_euler[i]
        plus_reg = (initial_angles[i] - plus_angle)**2*reg_coef
        plus_energy = objective_function(v_idx,w1=w1,w2=w2,barrier=barrier) + plus_reg  # NOTE: Change to "energy()" for testing target task!
        # Repeat
        bone.rotation_euler[i] -= 2 * step
        minus_angle = bone.rotation_euler[i]
        minus_reg = (initial_angles[i] - minus_angle)**2*reg_coef
        minus_energy = objective_function(v_idx,w1=w1,w2=w2,barrier=barrier) + minus_reg  # NOTE: Change to "energy()" for testing target task!
        # Set the angle back to its original value
        bone.rotation_euler[i] += step
        # Approximate the derivative at that point
        deriv = (plus_energy - minus_energy)/(2*step)
        #Update the angle
        bone.rotation_euler[i] -= step*deriv 
        
    # Recurse
    for child in bone.children:
        approx_grad_descent(child,v_idx,step,bone_angles,reg_coef,w1=w1,w2=w2,barrier=barrier)

#################################################
def run_optimization(epochs,step,reg_coef,v_idx=None,delta=0.0001,w1=2,w2=1,barrier=None):
    
    dp_history = []
    # Get the armature & target
    armature = bpy.data.objects.get(armature_name)
    # Set mode to pose + get bone (assuming armature is selected)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Select vertex for the optimization
    if v_idx == None:
        v_idx,dp_old,dp_ch_old = find_best_index()
    else:
        dp_old,dp_ch_old, _, _ = compute_dps_and_distances(v_idx)
    print("The chosen index is {idx}".format(idx=v_idx))
    print("With values dp={a}, dp_ch={b}".format(a=dp_old,b=dp_ch_old))
    
    # Get the initial angles for regularization
    bone_angles = get_bone_angles()
    # Run the optimization loop
    for i in range(epochs):
        old_energy = objective_function(v_idx,w1=w1,w2=w2,barrier=barrier)  # NOTE: Change to "energy()" for testing target task.
        for bone in armature.pose.bones:
            if not bone.parent:
                approx_grad_descent(bone,v_idx,step,bone_angles,
                reg_coef,w1=w1,w2=w2,barrier=barrier)
        # Save the current value of the dot products
        dp,dp_ch, _, _ = compute_dps_and_distances(v_idx)
        dp_history.append((dp,dp_ch))
        # Termination criterion without regularization
        new_energy = objective_function(v_idx,w1=w1,w2=w2,barrier=barrier)
        if abs(new_energy - old_energy) < delta:  # NOTE: Change to "energy()" for testing target task.
            break
        
    print('stopped after %i epochs' % i)
        
    # Print final values of dot products
    (dp, dp_ch) = dp_history[-1]
    print("Final values dp={a}, dp_ch={b}".format(a=dp,b=dp_ch))
    print("Variations: dp_new-dp_old={a}, dp_ch_new-dp_ch_old={b}".format(a=dp-dp_old,b=dp_ch-dp_ch_old))
    # Update the scene to reflect the changes
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    return np.array(dp_history)

#################################################
def reorient(v_idx, disp_balls=False,ball_size=0.001):
    # Get the COM and vertex (contains some duplicate code)
    # 1 – COM
#    tmesh = get_trimesh_object()
    COM = get_com()
    # 2 – Vertex
    dmesh = get_deformed_mesh()
    bm = bmesh.new()
    bm.from_mesh(dmesh)
    bm.verts.ensure_lookup_table()
    vertex = mesh.matrix_world @ bm.verts[v_idx].co
    
    # Get and select the armature
    armature = bpy.data.objects.get(armature_name)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    
    # Set the origin of the armature to the COM (to rotate about it)
    old_origin_loc = armature.location.copy()
    bpy.context.scene.cursor.location = COM
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    bpy.context.scene.cursor.location = (0,0,0)
    # Rotate so that COM to vertex vector is pointing up
    v = (vertex - mathutils.Vector(COM))
    v.normalize()
    rotation_axis = v.cross(mathutils.Vector((0, 0, 1)))
    rotation_angle = v.angle(mathutils.Vector((0, 0, 1)), 0)
    rotation_quaternion = mathutils.Quaternion(rotation_axis, rotation_angle)
    armature.rotation_mode = 'QUATERNION'
    armature.rotation_quaternion = rotation_quaternion @ armature.rotation_quaternion
    # Wwhat if current rotation is not null??
    # Set the origin of the armature back where it was (just for cleanliness)
    bpy.context.scene.cursor.location = old_origin_loc
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    bpy.context.scene.cursor.location = (0,0,0)
    
    # Show vertex & COM if needed
    if disp_balls:
        show_com(ball_size=ball_size)
        show_vertex(v_idx, ball_size=ball_size)
        show_comch(ball_size=ball_size)
    
    # Refresh
    bpy.context.view_layer.update()
    
#################################################
# Computes Gaussian curvature of every vertex
def gaussian_curvature():
    # Get the mesh
    mesh = get_deformed_mesh()
    nV = len(mesh.vertices)
    curvature = np.zeros(nV)
    # Iterate over all vertices
    for vertex_index in range(nV):
        vertex = mesh.vertices[vertex_index]
        edges = [edge for edge in mesh.edges if vertex_index in edge.vertices]
        angles_sum = 0

        # Loop through edges connected to the vertex
        for edge in edges:
            other_vertex_index = edge.vertices[0] if edge.vertices[1] == vertex_index else edge.vertices[1]
            other_vertex = mesh.vertices[other_vertex_index]

            # Compute angle between edges connected to the vertex
            v1 = (other_vertex.co - vertex.co).normalized()
            v2 = (mesh.vertices[edge.vertices[0]].co - vertex.co).normalized()
            # Calculate the dot product and clamp it within the valid range
            dot_product = min(1, max(-1, v1.dot(v2)))
            angle = math.acos(dot_product)

            angles_sum += angle

        # Calculate Gaussian curvature
        curvature[vertex_index] = (2 * math.pi - angles_sum) / len(edges)

    return curvature
#################################################
# Selects a specified number of vertices using Gaussian curvature
# and saves the selected indices in a txt file
def select_indices_with_curvature(num_samples,indices_path,alpha=1):
    curvature = gaussian_curvature()
    nV = len(curvature)
    P = np.abs(curvature)**alpha
    P = P/np.sum(P)
    indices = np.random.choice(nV,size=num_samples,p=P,replace=False)
    # Save indices in file
    if os.path.isfile(indices_path):
        old_indices = np.loadtxt(indices_path,dtype=int)
        new_indices = np.append(old_indices,indices)
        np.savetxt(indices_path,new_indices,fmt="%i")
    else:
        np.savetxt(indices_path,indices,fmt="%i")
    return indices
    
#################################################
# Returns a dictionary fo the paths in which we have results saved
# with values being the corresponding vertex indices
def get_result_paths(result_dir,indices_path):
    # Load the indices
    indices =  np.loadtxt(indices_path,dtype=int, ndmin=1)
    # Files containing results
    file_names = os.listdir(result_dir)
    if ".DS_Store" in file_names:
        file_names.remove(".DS_Store")
    # Return a dictionary with index-path correspondence
    paths = {}
    for idx in indices:
        for i, name in enumerate(file_names):
            if str(idx) in name:
                path = os.path.join(result_dir,name)
                paths[idx] = path
                break
    return paths

################################################
### Prepare paths
result_dir = os.path.join(my_dir,"results")
plot_dir = os.path.join(my_dir,"plots")
indices_path = os.path.join(my_dir,"indices.txt")

state = 2  # 0 = choose vertices, 1 = run optimization, 2 = view results

if state == 0:
    # Mesh must be in object mode, after you select the verts!
    obj = bpy.context.object
    me = obj.data
    #bm = bmesh.from_edit_mesh(me)
    bm = bmesh.new()
    bm.from_mesh(me)
    indices = [v.index for v in bm.verts if v.select]
    np.savetxt(indices_path,indices,fmt="%i")
    
elif state == 1:
    ####################################################
    ### RUNNING THE OPTIMIZATION ON SEVERAL VERTICES
    ### Select indices for experiments
    #num_trials = 4
    #indices = select_indices_with_curvature(num_trials,indices_path,alpha=2)
    indices = np.loadtxt(indices_path,dtype=int)
#    indices = [2460]#[279, 491, 506, 696]
    # Set parameters for optimization
    epochs = 10;step = 0.01;reg_coef = 1.;delta=0.0001;w1=100.;w2=30.
    v_idx = indices[0]
    dp,dp_ch, _, _ = compute_dps_and_distances(v_idx)
    barrier=dp*0.97#None
    # Interate optimization
#    paths = get_result_paths(result_dir,indices_path)# REMOVE
    for v_idx in indices:
        # Save dot product history and resulting
        dp_history = run_optimization(epochs,step,
        reg_coef,v_idx=v_idx,delta=delta,w1=w1,w2=w2,barrier=barrier)
        filepath = os.path.join(result_dir,"leaf_guy{i}.pkl".format(i=v_idx))
        # Get values for the evolution of dp, dp_ch
        dps = dp_history[:,0]
        dp_chs = dp_history[:,1]
        epochs_completed = range(len(dps))
        # Make plot
        plt.clf()
        plt.plot(epochs_completed,dps,label="dp")
        plt.plot(epochs_completed,dp_chs,label="dp_ch")
        plt.xlabel("Epochs")
        plt.legend(loc="upper right")
        plt.title("v_idx={i}, w1={a}, w2={b}".format(i=v_idx,a=w1,b=w2))
        # Save plot
        plot_path = os.path.join(plot_dir,"leaf_guy{i}_continuation.pdf".format(i=v_idx))
        plt.savefig(plot_path)
        # Save angle information in path
        armature = bpy.data.objects.get(armature_name)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        save_bone_angles(filepath)
        reset_bone_angles()
        
else:
    ################################################
    ## LOOKING AT THE RESULTS
    v_idx = 613 # 22, 83, 106, 245, 262, 540, 547, 697, 765

    paths = get_result_paths(result_dir,indices_path)
    path = paths[v_idx]
    set_bone_angles(path)
#    reset_bone_angles()
    ball_size = 0.02
    # EITHER uncomment below lines OR reorient
#    show_com(ball_size=ball_size)
#    show_vertex(v_idx, ball_size=ball_size)
#    show_comch(ball_size=ball_size)
    reorient(v_idx,disp_balls=True,ball_size=ball_size)
    dp,dp_ch = compute_dot_products(v_idx)
    text = "dp="+str(dp)+", dp_ch=" + str(dp_ch)
    print(text)
    ################################################
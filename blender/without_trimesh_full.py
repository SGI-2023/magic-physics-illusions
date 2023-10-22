import bpy, bmesh, mathutils
import math
import numpy as np, scipy as sp
import matplotlib.pyplot as plt
import json, os, pickle
import time
import copy

mesh_name = "my_hand"
armature_name = "Armature"
my_dir = "/Users/adityaabhyankar/Desktop/Programming/Magic-Physics-Illusions/magic-physics-illusions/blender"
#target = bpy.data.objects["target"]

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
#def get_trimesh_object():
#    # Convert mesh to bmesh
#    dmesh = get_deformed_mesh()
#    bm = bmesh.new()
#    bm.from_mesh(dmesh)
#    # Get the coordinates of all vertices
#    M = mesh.matrix_world
#    V = np.array([M @ v.co for v in bm.verts])
#    # Get face list
#    F = [
#    [v.index for v in f.verts]
#    for f in bm.faces
#    ]
#    # Get the face normals
#    N = [np.array(f.normal) for f in bm.faces]
#    
#    return trimesh.Trimesh(vertices=V,faces=F,face_normals=N)


#################################################
def get_com():
    # Select the object to duplicate
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    mesh.select_set(True)  # Select the object you want to duplicate
    bpy.context.view_layer.objects.active = mesh  # make it "active"
    bpy.ops.object.duplicate()  # duplicate it
    duplicated_obj = bpy.context.active_object  # get the duplicated object, which is the active one
    # Apply armature modifier to duplicate
    bpy.ops.object.modifier_apply(modifier=armature_name)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    # Get COM of duplicate
    COM = copy.copy(duplicated_obj.location)
    # Delete duplicate
    bpy.ops.object.delete()
    return np.array(COM)

#################################################
def get_comch():
    # Select the object to duplicate
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    mesh.select_set(True)  # Select the object you want to duplicate
    bpy.context.view_layer.objects.active = mesh  # make it "active"
    bpy.ops.object.duplicate()  # duplicate it
    duplicated_obj = bpy.context.active_object
    # Apply armature modifier
    bpy.ops.object.modifier_apply(modifier=armature_name)
    # Get the Convex Hull
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.convex_hull()
    bpy.ops.object.mode_set(mode='OBJECT')
    # Get COM of Convex Hull
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    COM = copy.copy(duplicated_obj.location)
    # Delete duplicate (i.e. the convex hull mesh)
    bpy.ops.object.delete()
    return np.array(COM)
    

#################################################

def compute_dot_products(v_idx):
    #tmesh = get_trimesh_object()
    bm = bmesh.new()
    bm.from_mesh(get_deformed_mesh())
    bm.verts.ensure_lookup_table()
    COM = get_com()
    COM_CH = get_comch()
    # Get the vertex normal
    world_rotation = mesh.matrix_world.to_3x3().normalized()
    vertex_normal = world_rotation @ bm.verts[v_idx].normal
    vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)
    # Vectors going to com and com_ch
    vertex_to_com = COM - (mesh.matrix_world @ bm.verts[v_idx].co)
    vertex_to_com = vertex_to_com/np.linalg.norm(vertex_to_com)
    vertex_to_com_ch = COM_CH - (mesh.matrix_world @ bm.verts[v_idx].co)
    vertex_to_com_ch = vertex_to_com_ch/np.linalg.norm(vertex_to_com_ch)
    # Compute the dot produtcs
    dp = np.dot(vertex_to_com,vertex_normal)
    dp_ch = np.dot(vertex_to_com_ch,vertex_normal)
    return dp, dp_ch
#################################################

def compute_all_dot_products():
#    tmesh = get_trimesh_object()
    bm = bmesh.new()
    bm.from_mesh(get_deformed_mesh())
    bm.verts.ensure_lookup_table()
    COM = get_com()
    COM_CH = get_comch()
    # Get the vertex normal
    world_rotation = mesh.matrix_world.to_3x3().normalized()
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
def objective_function(v_idx,w1 = 2, w2 = 1,barrier=None):
    dp, dp_ch = compute_dot_products(v_idx)
    # Return the weighted difference
    # More dp alignment = less energy. More dp_ch alignment = more energy.
    # Best possible case: dp=1, dp_ch=-1, so value=-(w1 - w2).
    
    # Once we have a nice value of dp, say dp = 0.98 we add a term to
    # the objective function which will force dp >= barrier (for example 0.95)
    if barrier == None:
        return w2*dp_ch - w1*dp
    if dp - barrier <= 0:
        raise Exception("Constraint violated in barrier method")
    return w2*dp_ch - w1*np.log(dp - barrier)

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

#################################################
def show_com(ball_size=0.003):
    """Creates an icosahedron at the location of the COM of the mesh"""
    # Get the mesh
#    tmesh = get_trimesh_object()
    COM = get_com()
    # Draw the icosahedron
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1,
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=COM,
                                        scale=(ball_size,ball_size,ball_size))
                                        
                                        
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
# Method that saves the original angles for the armature in a text file and
# returns the angles as a dictionary
def get_bone_angles():
    # First check if the text file exists
    angle_file_name = os.path.join(my_dir,armature_name+"_angles.pkl")
    if os.path.isfile(angle_file_name):
        # If the file exists, we just read it and return it
        with open(angle_file_name, 'rb') as file:
            bone_angles = pickle.load(file)
            return bone_angles
    # If there is no angle file, we make one
    # Get the armature & target
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
# New optimization algo (gradient descent instead of fixed step-size)
def approx_grad_descent(bone,v_idx,step,bone_angles,reg_coef = 0.01,w1=2,w2=1,barrier=None):
    '''Does one gradient descent step on each angle for the bone'''
    
    # Save the initial angles for regularization purposes
#    try:
#        bone["initial_angles"] = bone["initial_angles"]
#    except:
#        bone["initial_angles"] = [bone.rotation_euler[i] for i in range(3)]
    
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
    
    energy_history = []
    # Get the armature & target
    armature = bpy.data.objects.get(armature_name)
    # Set mode to pose + get bone (assuming armature is selected)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Select vertex for the optimization
    if v_idx == None:
        v_idx,dp_old,dp_ch_old = find_best_index()
    else:
        dp_old,dp_ch_old = compute_dot_products(v_idx)
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
        # Termination criterion without regularization
        new_energy = objective_function(v_idx,w1=w1,w2=w2,barrier=barrier)
        energy_history.append(new_energy)
        if abs(new_energy - old_energy) < delta:  # NOTE: Change to "energy()" for testing target task.
            break
        
    print('stopped after %i epochs' % i)
        
    # Print final values of dot products
    dp, dp_ch = compute_dot_products(v_idx)
    print("Final values dp={a}, dp_ch={b}".format(a=dp,b=dp_ch))
    print("Variations: dp_new-dp_old={a}, dp_ch_new-dp_ch_old={b}".format(a=dp-dp_old,b=dp_ch-dp_ch_old))
    # Update the scene to reflect the changes
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    return energy_history

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
    armature.rotation_quaternion = rotation_quaternion
    # Wwhat if current rotation is not null??
    # Set the origin of the armature back where it was (just for cleanliness)
    bpy.context.scene.cursor.location = old_origin_loc
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    bpy.context.scene.cursor.location = (0,0,0)
    
    # Show vertex & COM if needed
    if disp_balls:
        show_com(ball_size=ball_size)
        show_vertex(v_idx, ball_size=ball_size)
    
    # Refresh
    bpy.context.view_layer.update()
    
    
#################################################
start_time = time.time()
finger_tips = [685,857,1763,1251,1360]
#pics_path = "/Users/unaicaja/Documents/GitHub/magic-physics-illusions/blender/pics"
finger_num = 3
v_idx = finger_tips[finger_num-1]
epochs = 30;step = 0.01;reg_coef = 1;delta=0.001;w1=4;w2=1
barrier=None
energy_history = run_optimization(epochs,step,
    reg_coef,v_idx=v_idx,delta=delta,w1=w1,w2=w2,barrier=barrier)
    
print(time.time() - start_time)

# Making energy plot
xx = range(len(energy_history))
plt.clf()
plt.plot(xx,energy_history)
plt.title("Finger {a}, step {b}, reg_coef {c}".format(a=finger_num,b=step,c=reg_coef))
plt.xlabel("Epochs")
plt.ylabel("Obj. value")
path = my_dir + "/energy.pdf"
plt.savefig(path)
reorient(v_idx=v_idx,disp_balls = True,ball_size=0.01)
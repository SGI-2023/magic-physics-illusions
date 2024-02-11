import bpy, bmesh, mathutils
import math, mathutils
import numpy as np
import matplotlib.pyplot as plt
import os, pickle
import time
import trimesh


#################################################
# PARAMETERS
mesh_name = "squirrel"
our_name = "squirrel"
armature_name = "Armature"
useless_bone = "Waist"
my_dir = "/Users/unaicaja/Documents/GitHub/magic-physics-illusions/blender/current lab"
result_dir = os.path.join(my_dir,"results")
plot_dir = os.path.join(my_dir,"plots")
indices_path = os.path.join(my_dir,"indices.txt")
mesh = bpy.data.objects[mesh_name]  # (basically just needed for local to global coord matrix)

# For ball, comment if we are not using one
ABS_density = 1.05 * 0.15 # g/cm3 #? MULTIPLY BY FACTOR
steel_density = 7.9 # g/cm3
alpha = steel_density/ABS_density
metal_ball = bpy.data.objects["metal_ball"]

#################################################
#################################################
# DISPLAYING RESULTS
# From stackoverflow (for printing into console here)
def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")  
#################################################
def show_vertex(params):
    """Creates an icosahedron at the location of a vertex of the mesh"""

    v_idx = params["v_idx"]; ball_size = params["ball_size_for_display"]
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
def show_com(params):
    """Creates an icosahedron at the location of the COM of the mesh"""

    COM = get_com(params)
    COM = mathutils.Vector(COM)
    # Draw the icosahedron
    ball_size = params["ball_size_for_display"]
    bpy.ops.mesh.primitive_ico_sphere_add(radius=1,
                                        enter_editmode=False, 
                                        align='WORLD', 
                                        location=COM,
                                        scale=(ball_size,ball_size,ball_size))
#################################################
def reorient(params,disp_balls=False):
    # Get parameters
    v_idx = params["v_idx"]
    # Get the COM and vertex (contains some duplicate code)
    # 1 – COM
    COM = get_com(params)
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
        show_com(params)
        show_vertex(params)
    
    # Refresh
    bpy.context.view_layer.update()                             
#################################################
# SAVING/LOADING THE RESULTS.
# Method that reads the original angles for the armature from a text file and
# returns the angles as a dictionary. The ball location and radius is also saved
def get_scene_parameters(params,filepath = None):
    """Gets the scene parameters specified by filepath. If no filepath is given, it returns the initial setting."""

    # Set the armature as the active object
    armature = bpy.data.objects.get(armature_name)
    bpy.context.view_layer.objects.active = armature

    if filepath == None:
        filepath = os.path.join(my_dir,our_name+"_scene_parameters.pkl")
    # Check if the text file exists
    if os.path.isfile(filepath):
        # If the file exists, we just read it and return it
        with open(filepath, 'rb') as file:
            scene_param = pickle.load(file)
            return scene_param
    
    # If there is no parameter file, we make one
    # Get the armature
    armature = bpy.data.objects.get(armature_name)
    # Set mode to pose + get bone (assuming armature is selected)
    bpy.ops.object.mode_set(mode='POSE')
    # Dictionary in which we will sabe the angles
    scene_param = {}
    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'
        scene_param[bone.name] = bone.rotation_euler[:]
    # In case we are using a ball, we do the same for its location and radius
    if params["using_ball"]:
        scene_param["metal_ball_location"] = metal_ball.location[:]
        scene_param["metal_ball_radius"] = metal_ball.scale[0]
    # Save the dictionary
    with open(filepath, 'wb') as file:
        pickle.dump(scene_param, file)
    return scene_param
#################################################
# Saves the current bone angles to a file and restores the 
# angles to the previous position
def save_scene_parameters(filepath,params):
    # Get the armature
    armature = bpy.data.objects.get(armature_name)
    # Dictionary in which we will sabe the angles
    scene_param = {}
    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'
        scene_param[bone.name] = [0]*3
        for i in range(3):
            scene_param[bone.name][i] = bone.rotation_euler[i]
    # In case we are using a ball, we do the same for its location and radius
    if params["using_ball"]:
        scene_param["metal_ball_location"] = metal_ball.location[:]
        scene_param["metal_ball_radius"] = metal_ball.scale[0]
    # Save the dictionary
    with open(filepath, 'wb') as file:
        pickle.dump(scene_param, file)

#################################################
# Sets the angles of the armature to whatever they where at the beginning
def reset_scene_parameters(params):
    """
    Sets the armature angles and ball location, radious to the initial setting.
    """

    # Set the armature as the only active object
    armature = bpy.data.objects.get(armature_name)
    if not bpy.context.object.mode == "POSE":
        # bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        bpy.ops.object.mode_set(mode='POSE')
    # Initial parameters
    initial_params = get_scene_parameters(params)
    # Set mode to pose + get bone (assuming armature is selected)
    # Dictionary in which we will sabe the angles
    for bone in armature.pose.bones:
        bone.rotation_mode = 'XYZ'
        for i in range(3):
            a = bone.rotation_euler[i]
            bone.rotation_euler[i] += (initial_params[bone.name][i] - a)
    # If we are using a ball we do the same
    if params["using_ball"]:
        initial_location = initial_params["metal_ball_location"]
        initial_radius = initial_params["metal_ball_radius"]
        init_scale = mathutils.Vector([initial_radius]*3)
        metal_ball.location = mathutils.Vector(initial_location)
        metal_ball.scale = init_scale


#################################################
# sets the bone angles equal to the ones stored in a file
def set_scene_parameters(filepath,params):
    if not os.path.isfile(filepath):
        raise Exception("No parameters file found")
    
    # Set the armature as the only active object
    armature = bpy.data.objects.get(armature_name)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
        
    with open(filepath, 'rb') as file:
        scene_params = pickle.load(file)
        for bone in armature.pose.bones:
            bone.rotation_mode = 'XYZ'
            for i in range(3):
                a = bone.rotation_euler[i]
                bone.rotation_euler[i] += (scene_params[bone.name][i] - a)

        # If we are using a ball we do the same
        if params["using_ball"]:
            location = scene_params["metal_ball_location"]
            radius = scene_params["metal_ball_radius"]
            init_scale = mathutils.Vector([radius]*3)
            metal_ball.location = mathutils.Vector(location)
            metal_ball.scale = init_scale
#################################################
# Returns a dictionary fo the paths in which we have results saved
# with values being the corresponding vertex indices
def get_result_paths(result_dir,indices_path):
    # Load the indices
    indices =  np.loadtxt(indices_path,dtype=int)
    # Files containing results
    file_names = os.listdir(result_dir)
    if ".DS_Store" in file_names:
        file_names.remove(".DS_Store")
    # Return a dictionary with index-path correspondence
    paths = {}
    for idx in indices:
        for name in file_names:
            if str(idx) in name:
                path = os.path.join(result_dir,name)
                paths[idx] = path
                break
    return paths
#################################################
#################################################
# MISCELLANEOUS FUNCTIONS FOR MESHES.
def get_deformed_mesh():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return mesh.evaluated_get(depsgraph).to_mesh()
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
#################################################
# GETING INFO FROM MESHES.
def get_com(params):
    tmesh = get_trimesh_object()
    if not params["using_ball"]:
        return tmesh.center_mass
    # There is a metal ball in the mesh    
    # Now we use formula described in notes
    comR0 = np.array(metal_ball.location)
    r = metal_ball.scale[0]# Radius of the ball
    volR = 4/3*np.pi*r**3
    comB0 = tmesh.center_mass
    volB = tmesh.volume
    # Sanity check
    if volB < volR:
        text = "volB = {a} < {b} = volR".format(a=volB,b=volR)
        raise Exception(text)
        
    theta = volB/((alpha-1)*volR+volB)
    com = (1-theta)*comR0 + theta*comB0
    return com
#################################################
def get_comch():
    tmesh = get_trimesh_object()
    return tmesh.convex_hull.center_mass
#################################################
def compute_dps_and_distances(params):
    tmesh = get_trimesh_object()
    # Compute center of mass
    if params["using_ball"]:
        # Now we use formula described in notes
        comR0 = np.array(metal_ball.location)
        r = np.abs(metal_ball.scale[0])# Radius of the ball
        volR = 4/3*np.pi*r**3
        comB0 = tmesh.center_mass
        volB = tmesh.volume
        # Sanity check
        if volB < volR:
            text = "volB = {a} < {b} = volR".format(a=volB,b=volR)
            raise Exception(text)
            
        theta = volB/((alpha-1)*volR+volB)
        COM = (1-theta)*comR0 + theta*comB0
    else:
        COM = tmesh.center_mass
    # center of mass of convex hull
    COM_CH = tmesh.convex_hull.center_mass

    # Distance from vertex to centers of mass
    v_idx = params["v_idx"]
    vertex = tmesh.vertices[v_idx]
    dist_com = np.linalg.norm(vertex-COM)
    dist_com_ch = np.linalg.norm(vertex-COM_CH)

    # Get the vertex normal
    vertex_normal = tmesh.vertex_normals[v_idx]
    vertex_normal = vertex_normal / np.linalg.norm(vertex_normal)
    # Vectors going to com and com_ch
    vertex_to_com = (COM - vertex)/dist_com
    vertex_to_com_ch = (COM_CH - vertex)/dist_com_ch
    # Compute the dot produtcs
    dp = np.dot(vertex_to_com,vertex_normal)
    dp_ch = np.dot(vertex_to_com_ch,vertex_normal)

    return dp, dp_ch, dist_com, dist_com_ch
#################################################
#################################################
# OPTIMISATION CODE
def barrier_fun(x,x_min):
    """
    Evaluate barrier function at x.

    It is of class C1, 0 when x >= x_max and goes to +infinity when x approaches 0.
    """
    if x >= x_min:
        return 0
    else:
        return (x-x_min)*np.log(x/x_min)

def dist_ball_to_outside():
    """Computes the signed distance from the ball to the outside of the mesh."""
    # Get vertex list
    tmesh = get_trimesh_object()
    V = tmesh.vertices
    # Compute minimun distance to mesh points
    center = np.array(metal_ball.location)
    i_min = np.argmin(np.linalg.norm(V-center, axis=1))
    # To give the distance a sign, we dot v_min - center times the normal
    v_min = V[i_min,:]; n_min = tmesh.vertex_normals[i_min]
    n_min = n_min/np.linalg.norm(n_min)
    dist = np.dot(v_min - center,n_min)
    # Compute the distance depending on the sign
    radius = metal_ball.scale[0]
    if abs(dist) <= radius:
        return 0
    if dist > 0:
        return dist - radius
    else:
        return dist + radius
#################################################
def objective_function(params):
    """
    Objective function to optimise:

    Parameters:
            w1,w2: Weights for the terms of the objective function
            ball_coef: Weight for the ball penalty term
            mdto: Minimum distance that the ball can be to the outside
    """
    
    # get necessary parameters
    using_ball = params["using_ball"]; barrier = params["barrier"]
    w1 = params["w1"]; w2 = params["w2"]
    # First compute the term for the metal ball
    obj_fun = 0
    if using_ball:
        # We define a barrier term
        dist_to_outside = dist_ball_to_outside()
        mdto = params["mdto"]
        obj_fun = dist_to_outside-mdto
        if obj_fun <= 0:
            raise Exception("Ball too close to the surface")
        obj_fun = barrier_fun(x=dist_to_outside-mdto,x_min=5*mdto)
        
    dp, dp_ch, dist_com, dist_com_ch = compute_dps_and_distances(params)
    
    # Add penalization when com and com_ch are too close to the vertex
    dist_coef = params["dist_coef"]; mdtv = params["mdtv"]
    obj_fun += dist_coef*barrier_fun(x=dist_com,x_min=mdtv)
    obj_fun += dist_coef*barrier_fun(x=dist_com_ch,x_min=mdtv)

    if barrier == None:
        return obj_fun + w1*(dp-1)**2  + w2*(dp_ch + 1)**2
    
    # the objective function which will force dp >= barrier (for example 0.95)
    if dp - barrier <= 0:
        raise Exception("dp decreased too much")
    return obj_fun + w2*(dp_ch + 1)**2 + w1*barrier_fun(x=dp-barrier,x_min=1-barrier)
#################################################
# New optimization algo (gradient descent instead of fixed step-size)
def bone_grad_descent(bone,init_angles,params):
    '''
    Does one gradient descent step on each angle for the bone

    Parameters:
            diff_step: Step size for numerical differentiation
            mdto: Minimum distance that the ball can be to the outside of the mesh
    '''

    # The useless bone is one that rotates the whole mesh
    # For that bone I skip the optimization
    if bone.name == useless_bone:
        # Recurse
        for child in bone.children:
            bone_grad_descent(bone=child,init_angles=init_angles,params=params)
        return

    old_energy = objective_function(params)
    # Read necessary parameters
    reg_coef = params["reg_coef"]; diff_step = params["diff_step_bone"]
    step = params["armature_steps"][bone.name]
    # Read the initial angles
    initial_angles = init_angles[bone.name]
    # Optimization for bone angles
    bone.rotation_mode = 'XYZ'
    # Compute gradient
    grad = np.array([0.0]*3)
    for i in range(3):
        # Compute energies at two points
        bone.rotation_euler[i] += diff_step
        plus_angle = bone.rotation_euler[i]
        plus_reg = (initial_angles[i] - plus_angle)**2*reg_coef
        plus_energy = objective_function(params)
        plus_energy += plus_reg  # NOTE: Change to "energy()" for testing target task!
        # Repeat
        bone.rotation_euler[i] -= 2 * diff_step
        minus_angle = bone.rotation_euler[i]
        minus_reg = (initial_angles[i] - minus_angle)**2*reg_coef
        minus_energy = objective_function(params)
        minus_energy += minus_reg
        # Set the angle back to its original value
        bone.rotation_euler[i] += diff_step
        # Approximate the derivative at that point
        grad[i] = (plus_energy - minus_energy)/(2*diff_step)

    #Normalize gradient and update
    # grad /= np.linalg.norm(grad)
    #Update the angle
    for i in range(3):
        bone.rotation_euler[i] -= step*grad[i]
    # Update bone steps for future iterations
    new_energy = objective_function(params)
    if new_energy < old_energy:
        step *= params["amplification_factor"]
    else:
        step *= params["contraction_factor"]
    params["armature_steps"][bone.name] = step
    
    # Recurse
    for child in bone.children:
        bone_grad_descent(bone=child,init_angles=init_angles,params=params)

#################################################
def ball_grad_descent(params):
    """
    Optimization for ball location and radius
    """

    old_energy = objective_function(params)
    # Get necessary parameters
    mesh_size = max(mesh.dimensions)
    diff_step = params["diff_step_ball"]*mesh_size
    step = params["ball_step"]

    # Ball radius
    r = np.abs(metal_ball.scale[0])
    diff_step = np.min([diff_step,r])
    # Numerical differentiation
    plus_r = r + diff_step
    metal_ball.scale = mathutils.Vector([plus_r]*3)
    plus_energy = objective_function(params)
    minus_r = r - diff_step
    metal_ball.scale = mathutils.Vector([minus_r]*3)
    minus_energy = objective_function(params)
    # Update radius
    sign = np.sign(plus_energy - minus_energy)
    r -= sign*step
    # Make it so that the ball does not get too small
    r = np.max([r,params["min_r"]])
    metal_ball.scale = mathutils.Vector([r]*3)

    # Ball location
    loc_gradient = np.array([0.0]*3)
    for i in range(3):
        metal_ball.location[i] += diff_step
        plus_energy = objective_function(params)
        metal_ball.location[i] -= 2*diff_step
        minus_energy = objective_function(params)
        # Set coordinate back to original and compute derivative
        metal_ball.location[i] += diff_step
        loc_gradient[i] = (plus_energy - minus_energy)/(2*diff_step)
    #Normalize gradient and update location
    loc_gradient /= np.linalg.norm(loc_gradient)
    for i in range(3):
        metal_ball.location[i] -= step*loc_gradient[i]
    
    # Update the steps of the algorithm for later iterations
    new_energy = objective_function(params)
    if new_energy < old_energy:
        params["ball_step"] *= params["amplification_factor"]
    else:
        params["ball_step"] *= params["contraction_factor"]

#################################################
def run_optimization(params):
    """Runs the optimization for the balancing object. If requested, it also creates an animation of the process"""

    # Set the armature as the only active object
    armature = bpy.data.objects.get(armature_name)
    current_mode = bpy.context.object.mode
    if current_mode != "POSE":
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        bpy.ops.object.mode_set(mode='POSE')

    if params["animation"]:
        # Set total number of frames
        frames_per_key = params["frames_per_key"]
        init_frame = params["init_frame"]
        bpy.context.scene.frame_end = init_frame + frames_per_key*params["epochs"]
        # Set initial keyframe
        make_keyframes(armature=armature,frame=init_frame)
        num_keys = 1

    dp_old,dp_ch_old, dist_com_old, dist_com_ch_old= compute_dps_and_distances(params)
    print("The chosen index is {idx}".format(idx=params["v_idx"]))
    text = "With dp={dp}, dp_ch = {dp_ch}".format(dp=dp_old,dp_ch=dp_ch_old)
    print(text)
    
    # Get the initial angles for regularization
    init_angles = get_scene_parameters(params)
    # Run the optimization loop
    dp_history = []
    for i in range(params["epochs"]):
        old_energy = objective_function(params)

        # Gradient descent on bones
        for bone in armature.pose.bones:
            if not bone.parent:
                bone_grad_descent(bone=bone,init_angles=init_angles,params=params)
        
        # Gradient descent on ball
        if params["using_ball"]:
            ball_grad_descent(params=params)
            
        
        # Save the data for plotting
        dp, dp_ch, dist_com, dist_com_ch = compute_dps_and_distances(params)
        dp_history.append((dp,dp_ch,dist_com,dist_com_ch))

        # Animation code
        if params["animation"]: 
            make_keyframes(armature=armature,frame=init_frame+num_keys*frames_per_key)
            num_keys += 1

        # Termination criterion without regularization
        new_energy = objective_function(params)
        if abs(new_energy - old_energy) < params["delta"]:  
            break
        
    print('stopped after %i epochs' % i)
        
    # Print final values of dot products
    print("Final values dp={a}, dp_ch={b}".format(a=dp,b=dp_ch))
    print("Variations: dp_new-dp_old={a}, dp_ch_new-dp_ch_old={b}".format(a=dp-dp_old,b=dp_ch-dp_ch_old))
    # Update the scene to reflect the changes
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    return np.array(dp_history)


def make_keyframes(armature,frame):
    """
    Sets keyframes for every element of the optimization.
    """

    # Armature
    for bone in armature.pose.bones:
            bone.keyframe_insert("rotation_euler",frame=frame)
    # Ball
    if params["using_ball"]:
        metal_ball.keyframe_insert("location",frame=frame)
        metal_ball.keyframe_insert("scale",frame=frame)

    

################################################
################################################
# TODO
# 1. Try blender extension and debugging 
# 2. Debug saving and loading ball info from files
# 3. Try optimizing but only changing the ball
# 4. Test it for real
################################################
################################################
# RUNNING THE OPTIMIZATION ON SEVERAL VERTICES
# Select indices for experiments
indices = [347]

params = {
    "v_idx" : indices[0],
    "using_ball" : True,
    "epochs" : 50,
    "init_armature_step" : 1e-3, # Initial step for gradient descent on the ball
    "ball_step" : 1e-2, # Initial step for gradient descent on the ball
    "w1" : 3, # Coefficient for dp
    "w2" : 0, # Coefficient for dp_ch
    "ball_coef" : 1,
    "dist_coef" : 0.5, # Coefficients for the barrier term involving the distance to the vertex
    "min_r" : 6e-2,# Minimum radius for the ball
    "mdto" : 2e-2, # The barrier term blows up when distance(ball,outside) = mdto
    "mdtv" : 1e-3, # If distance(com,vertex) < mdtv then the barrier term activates
    "reg_coef" : 3,
    "amplification_factor" : 1.05, #Amplification factor to update step of the algorithm
    "contraction_factor" : 0.95, #Contraction factor to update step of the algorithm
    "diff_step_bone": 1e-2,
    "diff_step_ball": 1e-5,
    "delta" : 1e-5,
    "barrier" : None,#0.95, #0.95
    "ball_size_for_display": 0.1,
    "animation": True,
    "frames_per_key": 5,
    "init_frame" : 250
}
# Make a step for each bone
armature = bpy.data.objects.get(armature_name)
armature_steps = {bone.name: params["init_armature_step"] for bone in armature.pose.bones}
params["armature_steps"] = armature_steps

dp, dp_ch, dist_com, dist_com_ch = compute_dps_and_distances(params)
# Parameters for algorithm
#reset_scene_parameters(params)

text = "The initial dot products are dp={dp}, dp_ch={dp_ch}".format(dp=dp,dp_ch=dp_ch)
print(text)
# Interate optimization
#for idx in indices:
#    params["v_idx"] = idx
#    # Save dot product history and resulting
#    dp_history = run_optimization(params)
#    # Get values for the evolution of dp, dp_ch
#    dps = dp_history[:,0]
#    dp_chs = dp_history[:,1]
#    epochs_completed = range(len(dps))
#    # Make plot
#    plt.clf()
#    plt.plot(epochs_completed,dps,label="dp")
#    plt.plot(epochs_completed,dp_chs,label="dp_ch")
#    plt.xlabel("Epochs")
#    plt.legend(loc="lower right")
#    plt.title("v_idx={i}, w1={a}, w2={b}".format(i=idx,a=params["w1"],b=params["w2"]))
#    # Save plot
#    plot_path = os.path.join(plot_dir,"mewto{i}_metal_ball.pdf".format(i=idx))
#    plt.savefig(plot_path)
#    # Save angle information in path
#    armature = bpy.data.objects.get(armature_name)
#    bpy.context.view_layer.objects.active = armature
#    bpy.ops.object.mode_set(mode='POSE')





import trimesh,os, numpy as np

# obj_filepath = "/Users/unaicaja/Documents/GitHub/magic-physics-illusions/rignet_watertight/17271.obj"
# mesh = trimesh.load(obj_filepath, force='mesh')
# v = mesh.vertices[0]
# print(f"The mesh has a vertex {str(v)}")



# objs_path = "/Users/unaicaja/Documents/GitHub/magic-physics-illusions/Rignet/ModelResource_RigNetv1_preproccessed/obj_remesh"
# fbx_path = "/Users/unaicaja/Documents/GitHub/magic-physics-illusions/Rignet/ModelsResource-RigNetv1-fbx/fbx"
# destination_path = "/Users/unaicaja/Documents/GitHub/magic-physics-illusions/rignet_watertight"
# valid_meshes = []
# for file_name in os.listdir(objs_path):
#     if file_name == ".DS_Store":
#         continue
#     obj_filepath = os.path.join(objs_path,file_name)
#     mesh = trimesh.load(obj_filepath, force='mesh')

#     if not mesh.is_watertight:
#         continue
    
#     mesh_number = os.path.splitext(file_name)[0]
#     fbx_filepath = os.path.join(fbx_path,mesh_number+".fbx")
#     new_obj_filepath = os.path.join(destination_path,file_name)
#     new_fbx_filepath = os.path.join(destination_path,mesh_number+".fbx")
#     os.rename(obj_filepath,new_obj_filepath)
#     os.rename(fbx_filepath,new_fbx_filepath)
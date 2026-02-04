import argparse
import numpy as np
import os
import sys
import time
from PIL import Image
sys.path.append(os.path.dirname(__file__))
sys.path.append('/attached/data/remote-home2/zzq/07-splat-image')

import util
import blender_interface
from fep_nbv.viewpoint_utils.generate_viewpoints import generate_HEALPix_viewpoints


model_path = '/attached/data/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133'
mesh_fpath = os.path.join(model_path,'models/model_normalized.obj')
output_path = '/attached/data/remote-home2/zzq/data/test/blender_renderer'

renderer = blender_interface.BlenderInterface(resolution=128)


obj_location = np.zeros((1,3))

# 测试目标旋转
absolute_viewpoint_poses = generate_HEALPix_viewpoints(n_side=2)
# 打印前五个候选视点
print("前五个候选视点:")
for i in range(5):
    print(absolute_viewpoint_poses[i])
obj_location = np.zeros((1,3))
rot_mat = np.eye(3)
hom_coords = np.array([[0., 0., 0., 1.]]).reshape(1, 4)
obj_pose = np.concatenate((rot_mat, obj_location.reshape(3,1)), axis=-1)
obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)
print(obj_pose)
renderer.import_mesh(mesh_fpath, scale=1., object_world_matrix=obj_pose)
time1 = time.time()
gt_images = renderer.render(output_path, absolute_viewpoint_poses, write_cam_params=True)
time2 = time.time()
elapsed_time=time2-time1
print(f"{str(renderer.scene.render.engine) }代码运行时间: {elapsed_time:.6f} 秒")
# for index,gt_image in enumerate(gt_images):
#         image = Image.fromarray((gt_image*255).cpu().detach().numpy().astype(np.uint8))
#         image.save(os.path.join(output_path,f'{index}_gt.png'))



# nvf
# blender_poses = util.xyz2pose(cam_locations[:,0], cam_locations[:,1], cam_locations[:,2])
# # blender_poses = [util.to_transform(m).numpy() for m in cv_poses]

# # cv_poses = util.look_at(cam_locations, obj_location)
# # blender_poses = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]
# # blender_poses = cv_poses

# shapenet_rotation_mat = np.array([[1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#                                   [0.0000000e+00, -1.0000000e+00, -1.2246468e-16],
#                                   [0.0000000e+00,  1.2246468e-16, -1.0000000e+00]])
# rot_mat = np.eye(3)
# hom_coords = np.array([[0., 0., 0., 1.]]).reshape(1, 4)
# obj_pose = np.concatenate((rot_mat, obj_location.reshape(3,1)), axis=-1)
# obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)

# renderer.import_mesh(opt.mesh_fpath, scale=1., object_world_matrix=obj_pose)
# renderer.render(instance_dir, blender_poses, write_cam_params=True)
# 把存下来的camera和util算出来的camera和healpix转换出来的camera对比一下
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))
sys.path.append('/home/zhengquan/06-splatter-image')
import blender_interface
import util
from fep_nbv.viewpoint_utils.generate_viewpoints import generate_HEALPix_viewpoints

# set renderer
model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/03691459/102f9164c5ae3846205b5aa6ba4df9c8'
mesh_fpath = os.path.join(model_path,'models/model_normalized.obj')
output_path = '/home/zhengquan/06-splatter-image/out/test_out/blender_render_2'
obj_location = np.zeros((1,3))
rot_mat = np.eye(3)
hom_coords = np.array([[0., 0., 0., 1.]]).reshape(1, 4)
obj_pose = np.concatenate((rot_mat, obj_location.reshape(3,1)), axis=-1)
obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)
renderer = blender_interface.BlenderInterface(resolution=128)
renderer.import_mesh(mesh_fpath, scale=1., object_world_matrix=obj_pose)

pose_index = 0
poses_saved_by_renderer_path = '/home/zhengquan/06-splatter-image/out/test_out/blender_render/pose'
absolute_viewpoint_poses = generate_HEALPix_viewpoints(n_side=2)
cam_location = absolute_viewpoint_poses[0,4:]

saved_pose_path = os.path.join(poses_saved_by_renderer_path, '%06d.txt'%pose_index)
c2w = np.loadtxt(saved_pose_path, dtype=np.float32).reshape(4, 4)
w2c = np.linalg.inv(c2w)
R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
T = w2c[:3, 3]
w2c[:3,:3] = R


# 
gt_images = renderer.render(output_path, absolute_viewpoint_poses[pose_index].unsqueeze(0), write_cam_params=True)
camera = renderer.camera
cv_pose = util.look_at(cam_location.unsqueeze(0).numpy(), np.zeros((1,3)))
saved_pose = w2c

a = 1
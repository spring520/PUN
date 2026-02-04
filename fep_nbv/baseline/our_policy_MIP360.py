# 根据不确定性的分布结果进行视角选择的策略test,使用相对视角分布
import tyro
import sys
from tqdm import tqdm
from pathlib import Path
import os
import random
import numpy as np
from PIL import Image
import time
import math
from datetime import timedelta, datetime
import healpy as hp
import re
import timm
import argparse
import torch
from mathutils import Vector
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append("/home/zhengquan/03-nips-rebuttal")
sys.path.append("/home/zhengquan/03-nips-rebuttal/08-vit-train")

NMR_dataset_path = '/attached/remote-home2/zzq/data/splat-image/NMR_Dataset'
gt_images_path = '/mnt/hdd/zhengquan/data/splat-image/gt_temp'
Shapenet_path = '/attached/remote-home2/zzq/data/shapenet/ShapeNetCore.v2'
root_path = '/home/zhengquan/03-nips-rebuttal'
SINGLE = True

from config import *
import config
from fep_nbv.utils.utils import *
from fep_nbv.env.utils import * # 什么鬼啊
from fep_nbv.env.shapenet_env import set_env
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints,generate_fibonacci_viewpoints,generate_polar_viewpoints,index2pose_HEALPix
from fep_nbv.visualization.uncertainty_distribution_visualization import * 
from nvf.metric.MetricTracker import RefMetricTracker
from regress_model import ViTRegressor  # 你的 ViT 回归模型定义
from blender_utils import blender_interface
from nvf.active_mapping.agents import *
from nvf.active_mapping.mapping_utils import to_pose
from fep_nbv.env.read_write_model import read_cameras_binary,read_images_binary

import warnings
warnings.filterwarnings("ignore")

# 找到最后一个视角不确定性最小的n个点的index，并从working_index中删除，如果他们还在working_index中
# 找到最后一个视角，归一化不确定性的10%对应的点，从working_index中删除，如果他们还在working_index中
# 每次选取不确定性中不确定性变化最大的那个点

def calculate_raidus_scale(NMR_dataset_path, model_path):
    class_offset = model_path.split('/')[-2]
    model_index = model_path.split('/')[-1]
    if os.path.exists(os.path.join(NMR_dataset_path,class_offset,model_index)):
        _coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        _coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        _pixelnerf_to_colmap = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        print('load radius from npz')
        all_cam = np.load(os.path.join(NMR_dataset_path,class_offset,model_index,'cameras.npz'))
        radius = np.sqrt((all_cam['world_mat_inv_0']**2).sum()).item()
        c2w_cmo = all_cam['world_mat_inv_0']
        c2w_cmo = (
                _coord_trans_world
                @ torch.tensor(c2w_cmo, dtype=torch.float32)
                @ _coord_trans_cam # to pixelnerf coordinate system
                @ _pixelnerf_to_colmap # to colmap coordinate system
            ) 
        radius = np.sqrt((c2w_cmo[:3, 3] ** 2).sum()).item()
        scale = 2.0
        print(f'radius of {class_offset}/{model_index} is {radius} loaded from npz')
    else:
        radius = 2.7319998741149902
        scale= 2.0
        print(f'radius of {class_offset}/{model_index} is {radius}')
    return radius,scale

def load_model(model_path, model_name, device="cuda"):
    """加载训练好的 ViT 回归模型"""
    model = ViTRegressor(model_name=model_name, output_dim=48).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设为推理模式
    return model

def view_select_last(absolute_uncertainty_his=None,candidate_viewpoint_poses=None,mode=None):
    # 根据上一个uncertainty distribution进行选择
    uncertainty_absolute = absolute_uncertainty_his[-1]
    if mode=='uncertainty' or mode=='MSE' or mode=='LPIPS':
        extreme_index = np.argmax(uncertainty_absolute)
    elif mode=='PSNR' or mode=='SSIM':
        extreme_index = np.argmin(uncertainty_absolute)
    else:
        raise NotImplementedError

    result_pose = candidate_viewpoint_poses[extreme_index]
    print(f'select viewpoint of {extreme_index} from all')
    return result_pose

def view_select_all(absolute_uncertainty_his=None,candidate_viewpoint_poses=None,mode=None):
    # print(f'len in absolute_uncertainty_his {absolute_uncertainty_his.shape} len in candidate {candidate_viewpoint_poses.shape[0]}')
    current_uncertainty = np.prod(absolute_uncertainty_his, axis=0)
    # print(f'current uncertainty {current_uncertainty.shape}')
    if mode=='uncertainty' or mode=='MSE' or mode=='LPIPS':
        extreme_index = np.argmax(current_uncertainty)
    elif mode=='PSNR' or mode=='SSIM':
        extreme_index = np.argmin(current_uncertainty)
    else:
        raise NotImplementedError

    result_pose = candidate_viewpoint_poses[extreme_index]
    print(f'select viewpoint of {extreme_index} from all')
    return result_pose

def view_select_diff(absolute_uncertainty_his=None,candidate_viewpoint_poses=None,mode=None,angle_threshold_degrees=5):
    angle_threshold = np.radians(angle_threshold_degrees)
    T, N = absolute_uncertainty_his.shape

    # 计算所有 pose 的方向向量（最后三维是方向单位向量）
    candidate_dirs = candidate_viewpoint_poses[:, 4:]
    candidate_dirs = candidate_dirs / np.linalg.norm(candidate_dirs, axis=1, keepdims=True)

    # 计算每一对 pose 之间的球面夹角（N x N）
    dot_products = np.clip(candidate_dirs @ candidate_dirs.T, -1.0, 1.0)
    angle_matrix = np.arccos(dot_products)  # 每对之间的夹角

    max_diffs = np.zeros_like(absolute_uncertainty_his[-1])

    for i in range(N):
        neighbors = np.where((angle_matrix[i] < angle_threshold) & (angle_matrix[i] > 1e-4))[0]
        if len(neighbors) == 0:
            # 如果没有邻居，选出最近的
            closest = np.argmin(angle_matrix[i] + np.eye(N)[i] * 999)  # 排除自己
            neighbors = [closest]

        diffs = []
        for j in neighbors:
            diff = abs(absolute_uncertainty_his[-1][i] - absolute_uncertainty_his[-1][j])
            angle = angle_matrix[i, j]
            if angle > 1e-5:
                diffs.append(diff / angle)

        if diffs:
            max_diffs[i] = np.max(diffs)

    # 找出 diff 最大的视角
    selected_index = np.argmax(max_diffs)

    print(f'select viewpoint of {selected_index} from all')
    return candidate_viewpoint_poses[selected_index]


def filter_viewpoints_single(absolute_uncertainty_his=None,pose_his=None,candidate_viewpoint_poses=None,mode=None):
    """
    从 candidate_viewpoint_poses 中移除与 pose_his 中任一 pose 球面夹角小于 angle_threshold_deg 的视角。

    参数:
        pose_his: shape [M, >=7] 的 numpy 数组，历史视角，其中 [:,4:] 是三维位置向量
        candidate_viewpoint_poses: shape [N, >=7] 的 numpy 数组，候选视角，其中 [:,4:] 是三维位置向量
        angle_threshold_deg: 阈值角度（单位：度），默认 5°

    返回:
        working_index: 被保留的 candidate_viewpoint_poses 的下标列表
    """
    angle_threshold_rad = np.radians(5.0)

    # 获取所有三维位置（单位向量）
    candidate_dirs = candidate_viewpoint_poses[:, 4:]
    candidate_dirs = candidate_dirs / np.linalg.norm(candidate_dirs, axis=1, keepdims=True)

    pose_his_dirs = pose_his[:, 4:]
    pose_his_dirs = pose_his_dirs / np.linalg.norm(pose_his_dirs, axis=1, keepdims=True)

    # 记录哪些 candidate 保留
    working_index = []

    for i, cand_dir in enumerate(candidate_dirs):
        cos_angles = np.dot(pose_his_dirs, cand_dir)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        if np.all(angles >= angle_threshold_rad):
            working_index.append(i)
    print(f'left {len(working_index)} candidates')
    return candidate_viewpoint_poses[working_index]

def filter_viewpoints_small(absolute_uncertainty_his=None,pose_his=None,candidate_viewpoint_poses=None,mode=None,threshold=0.00):
    normalized_uncertainty = np.zeros_like(absolute_uncertainty_his)
    for i in range(absolute_uncertainty_his.shape[0]):
        row = absolute_uncertainty_his[i]
        min_val = row.min()
        max_val = row.max()
        if max_val > min_val:
            normalized_uncertainty[i] = (row - min_val) / (max_val - min_val)
        else:
            normalized_uncertainty[i] = 0  # or 0.5, if all values are equal

    if mode in ['uncertainty', 'MSE', 'LPIPS']:
        # 标记那些在任意时刻不确定性过小的视角
        bad_mask = np.any(normalized_uncertainty <= threshold, axis=0)
    elif mode in ['PSNR', 'SSIM']:
        # 对于 PSNR/SSIM，数值越高越好；小于 1 - threshold 视为不确定性低
        bad_mask = np.any(normalized_uncertainty > (1 - threshold), axis=0)
    else:
        raise NotImplementedError(f"Mode {mode} not supported.")

    # 保留那些所有时刻都“没有”出现不确定性低的视角
    keep_mask = ~bad_mask

    print(f'left {np.sum(keep_mask)} candidates')
    return candidate_viewpoint_poses[keep_mask]

def filter_viewpoints_no(absolute_uncertainty_his=None,pose_his=None,candidate_viewpoint_poses=None,mode=None,threshold=0.1):

    return candidate_viewpoint_poses

def filter_viewpoints_top_n(absolute_uncertainty_his=None,pose_his=None,candidate_viewpoint_poses=None,mode=None,top_n=5):
    T, N = absolute_uncertainty_his.shape
    bad_mask = np.zeros(N, dtype=bool)
    for t in range(T):
        row = absolute_uncertainty_his[t]
        if mode in ['uncertainty', 'MSE', 'LPIPS']:
            # 找出当前时间步下 top-n 最大的不确定性 index
            top_indices = np.argpartition(row, -top_n)[-top_n:]
        elif mode in ['PSNR', 'SSIM']:
            # 越大越好，排除最小的 top-n（即 uncertainty 大）
            top_indices = np.argpartition(-row, -top_n)[-top_n:]
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        bad_mask[top_indices] = True

    keep_mask = ~bad_mask
    print(f'left {np.sum(keep_mask)} candidates')
    return candidate_viewpoint_poses[keep_mask]

def policy_template(viewpoints_sampler=None,mode='uncertainty',select_method='last',delete_method='single',renderer=None,prediction_model=None,transform=None):
    candidate_viewpoint_poses_index = np.array(env.scene.candidate_poses)
    candidate_viewpoint_poses = env.scene.poses[candidate_viewpoint_poses_index]
    obs = renderer.render([0]).unsqueeze(0).float() # 1 128 128 3
    obs_his = []
    pose_his = []
    relative_uncertainty_his = []
    pose_his.append(env.scene.poses[0])
    obs_his.append(obs)
    relative_uncertainty = model(transform(obs.permute(0,3,1,2)[:,:3]).to(args.device))[0].detach().cpu().numpy() # 1 48
    relative_uncertainty_his.append(relative_uncertainty)
    # absolute_uncertainty_his = interpolate_uncertainty(relative_uncertainty_his, pose_his, candidate_viewpoint_poses) # 每个视角 offset_phi=0的时候的值

    while len(pose_his)<20:
        absolute_uncertainty_his = interpolate_uncertainty(relative_uncertainty_his, torch.stack(pose_his,axis=0), candidate_viewpoint_poses)
        candidate_viewpoint_poses_index = eval('filter_viewpoints_'+delete_method)(absolute_uncertainty_his,torch.stack(pose_his,axis=0),candidate_viewpoint_poses_index,mode)
        candidate_viewpoint_poses = env.scene.poses[candidate_viewpoint_poses_index]
        absolute_uncertainty_his = interpolate_uncertainty(relative_uncertainty_his, torch.stack(pose_his,axis=0), candidate_viewpoint_poses)


        result_pose_index = eval('view_select_'+select_method)(absolute_uncertainty_his,candidate_viewpoint_poses_index,mode)
        candidate_viewpoint_poses_index = candidate_viewpoint_poses_index[candidate_viewpoint_poses_index != result_pose_index] # 删除已经选择的视角
        candidate_viewpoint_poses = env.scene.poses[candidate_viewpoint_poses_index]
        pose_his.append(env.scene.poses[result_pose_index])
        obs = renderer.render([result_pose_index]).unsqueeze(0).float() # 1 128 128 3
        obs_his.append(obs)
        relative_uncertainty = model(transform(obs.permute(0,3,1,2)[:,:3]).to(args.device))[0].detach().cpu().numpy()
        relative_uncertainty_his.append(relative_uncertainty)
    
    return pose_his,np.stack(obs_his,axis=1).squeeze()


def blender_pose_to_pose7d(pose, distance=1.0):
    """
    输入:
        pose: (4, 4) Blender 格式 cam2world pose
        distance: 相机从原点后退的距离（默认 1.0）

    输出:
        pose_7d: [x, y, z, w, tx, ty, tz] 四元数 + 位置（新位置使相机指向原点）
    """
    assert pose.shape == (4, 4)

    # 提取原始朝向
    R = pose[:3, :3]
    forward = -R[:, 2]  # Blender 相机的前向是 -Z
    forward = Vector(forward).normalized()

    # 相机新位置 = 沿 forward 方向反向移动，使其看到原点
    position = -distance * forward  # 相机位置后退 distance 米

    # 方向保持不变，生成 quaternion
    quat = forward.to_track_quat('-Z', 'Y')  # 保持 forward 不变

    # 构造四元数 [x, y, z, w] + 平移
    quat_xyzw = [quat.x, quat.y, quat.z, quat.w]
    pose_7d = np.array(quat_xyzw + list(position), dtype=np.float32)
    # print(f'forward {forward}')
    # print(f'camera position {pose[:3, 3]}')
    # print(forward+Vector(pose[:3, 3]))
    # print(f'position {position}')
    return pose_7d

def interpolate_uncertainty(relative_uncertainty_his, pose_his, candidate_viewpoint_poses):
    absolute_uncertainty_his = []
    for index_time,relative_uncertainty in enumerate(relative_uncertainty_his):
        pose_7d = blender_pose_to_pose7d(pose_his[index_time])
        corresponding_poses = generate_HEALPix_viewpoints(n_side=2,original_viewpoint=pose_7d[4:],radius=2.7319998741149902)
        uncertainty_absolute = []
        for index_pose,pose_ in enumerate(candidate_viewpoint_poses):
            # 从uncertainty_relative中插值出pose位置上的不确定性值
            # cos_angles = np.dot(corresponding_poses[:,4:]/2, pose[4:]/2)
            pose = blender_pose_to_pose7d(pose_)
            cos_angles = np.dot(corresponding_poses[:,4:]/np.linalg.norm(corresponding_poses[:,4:],axis=1,keepdims=True), pose[4:]/np.linalg.norm(pose[4:])) # correct
            angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

            # 筛选距离小于阈值的采样点
            angle_threshold = np.radians(30)  # 10 度阈值
            nearby_indices = np.where(angles < angle_threshold)[0]
            weights = np.exp(-angles[nearby_indices])
            weights = weights/sum(weights)
            temp_uncertainty = np.dot(weights,relative_uncertainty_his[index_time][nearby_indices])
            uncertainty_absolute.append(temp_uncertainty)
        absolute_uncertainty_his.append(np.array(uncertainty_absolute))

    return np.array(absolute_uncertainty_his)


def extract_a_b_from_filename(filename):
    """
    从文件名中提取 a 和 b 的值，假设 a 和 b 都是整数。

    :param filename: 文件名字符串
    :return: (a, b) 元组
    """
    match = re.search(r'viewpoint_(\d+)_offset_phi_(\d+)', filename)
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        return a, b
    else:
        raise ValueError("文件名格式不符合预期")


if __name__=='__main__':
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.mip # eval视角均匀采样
    # cfg.sampler = SamplerType.discrete # 从N个里面随机采样一个

    # load vit model
    parser = argparse.ArgumentParser(description="Evaluate ViT regression model on a single image")
    parser.add_argument('--vit_logs', type=str, default='08-vit-train/logs')
    parser.add_argument('--vit_ckpt_path', type=str, default='vit_small_patch16_224_PSNR_250425172703')
    parser.add_argument('--scene_name', type=str, default='bicycle')
    parser.add_argument('--all_dataset',action='store_true',default=True)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    mode = args.vit_ckpt_path.split('_')[-3]
    mode_name = args.vit_ckpt_path.split('_')[-3]+'_'+args.vit_ckpt_path.split('_')[-2]
    if mode not in ['LPIPS','PSNR','SSIM','MSE']:
        mode_name=mode
        mode = args.vit_ckpt_path.split('_')[-2]
    vit_used = args.vit_ckpt_path.split(f'_{mode}')[0]
    scene_name = args.scene_name
    category = scene_name

    if not args.all_dataset:
        ckpt_path = os.path.join(root_path, args.vit_logs, f'{category}_single_splited',args.vit_ckpt_path, 'best_vit_regressor.pth')
    else:
        ckpt_path = os.path.join(root_path, args.vit_logs, f'all_11class_90modeltrain',args.vit_ckpt_path, 'best_vit_regressor.pth')
    cfg.env.target_path = scene_name
    cfg.env.radius=2.7319998741149902
    camera_path = '/mnt/hdd/zhengquan/mipnerf360/bonsai/sparse/0/cameras.bin'.replace('bonsai',scene_name)
    cameras = read_cameras_binary(camera_path)
    # cfg.env.resolution = (cameras[1].height,cameras[1].width)
    cfg.env.resolution = (410,618)
    args.device = 'cuda'
    print(f'scene name {scene_name}')
    print(f'ckpt_path {ckpt_path}')

    model = load_model(ckpt_path, vit_used, args.device)
    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    gif_current_uncertainty = GIFSaver()
    gif_obs = GIFSaver()
    obs_his = []

    # gt enviroment
    env = set_env(cfg)
    config.env = env
    
    for select_method in ['all']:
        for delete_method in ['small']:
            time1 = time.time()
            t1 = time.time()
            pose_his,obs_his = policy_template(viewpoints_sampler=None,mode=mode,select_method=select_method,delete_method=delete_method,renderer=env.scene,prediction_model=model,transform=transform)
            t2 = time.time()
            print(f'20 viewpoint selection in {timedelta(seconds=t2-t1)}')
            # print(pose_his)
            # sys.exit(1)
            # continue

            cfg.exp_name = f'data/test/policy_MIP/{select_method}_{delete_method}/{mode}'
            print(f'cfg name {cfg.exp_name}')
            save_path = os.path.join(cfg.exp_name,'images')
            if os.path.exists(os.path.join(cfg.exp_name,'metrics.xlsx')):
                del env
                continue

            tracker = RefMetricTracker(cfg, env=env)
            tracker.setup_writer(f'{cfg.exp_name}')
            NeRF_pipeline = NeRF_init(cfg)
            os.makedirs(cfg.exp_name,exist_ok=True)
            os.makedirs(save_path,exist_ok=True)

            empty_cache()
            NeRF_pipeline.reset()
            # obs_his,_,_,_ = env.step(torch.stack(pose_his,dim=0)) 
            # print(index_his)
            # print(pose_his.shape)
            # print(obs_his[0].max())

            pose_his_temp = []
            for pose in pose_his:
                pose_his_temp.append(to_pose(pose))
            pose_his = torch.stack(pose_his_temp,dim=0)

            tracker.init_trajectory(pose_his[0:1])


            NeRF_pipeline.add_image(images=obs_his,poses=pose_his,model_option=None)
            tracker.update(NeRF_pipeline, pose_his, 0)
            for i,obs in enumerate(obs_his):
                save_img(obs,f'{cfg.exp_name}/obs_{i}.png')

            tracker.gif.save(f'{cfg.exp_name}/eval.gif')
            np.savetxt(f'{cfg.exp_name}/pose_his_0.txt', pose_his)
            save_dict_to_excel(f'{cfg.exp_name}/metrics.xlsx', tracker.metric_hist)
            
            time2 = time.time()
            print(f'20 viewpoint selection in {timedelta(seconds=time2-time1)}')
            del env
            del NeRF_pipeline


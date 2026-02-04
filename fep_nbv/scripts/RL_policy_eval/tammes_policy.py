import numpy as np
import os
import tyro
import sys
import json
import mathutils
import torch
root_path = os.getenv('nbv_root_path', '/default/path')
shapenet_path = os.getenv('shapenet_path', '/default/shapenet/path')
distribution_dataset_path = os.getenv('distribution_dataset_path', '/default/distribution/dataset/path')
if not os.path.exists(root_path):
    root_path=root_path.replace('/attached/data','/attached')
    shapenet_path=shapenet_path.replace('/attached/data','/attached')
    distribution_dataset_path=distribution_dataset_path.replace('/attached/data','/attached')
sys.path.append(root_path)

from nvf.metric.MetricTracker import RefMetricTracker
from config import *
from fep_nbv.env.shapenet_RL_env import set_env
from fep_nbv.utils.utils import save_dict_to_excel
from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints

import warnings
warnings.filterwarnings("ignore")

def find_tammes_subset(views, num_points):
    """
    从给定的视角集合 (views) 中选择 num_points 个点，使得最小角距离最大化。
    :param views: (N, 3) 的 numpy 数组，每行是一个单位球面上的 (x, y, z) 视角向量。
    :param num_points: 需要选择的视角数量。
    :return: 选定的视角索引列表。
    """
    views = views[:,-3:]/2
    
    # 计算所有点之间的角距离矩阵
    dot_products = np.clip(np.dot(views, views.T), -1.0, 1.0)
    angles = np.arccos(dot_products)  # 角距离矩阵 (N x N)
    
    # 初始化，从第一个点开始
    selected_indices = [0]  
    for _ in range(1, num_points):
        # 计算每个未选点到已选集合的最小角距离
        min_distances = np.min(angles[selected_indices], axis=0)
        
        # 选择最大化最小距离的点
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)
    
    return selected_indices

if __name__=='__main__':
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133'
    # model_path = cfg.env.target_path
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path

    # env = set_env(cfg)
    # cfg.exp_name = f'data/test/RL_env_test/tammes_policy_discrete'
    # if not os.path.exists(cfg.exp_name):
    #     os.makedirs(cfg.exp_name,exist_ok=True)
    # tracker = RefMetricTracker(cfg, env=env)
    # tracker.setup_writer(f'{cfg.exp_name}')

    # print(env.action_space)

    candidate_viewpoints = generate_HEALPix_viewpoints(n_side=2)
    selected_viewpoint_indexes = find_tammes_subset(candidate_viewpoints,num_points=20)
    print(selected_viewpoint_indexes)

    # for i in range(20):
    #     action = selected_viewpoint_indexes[i]
    #     obs,reward,_,_ = env.step(action)
    #     print(action)
    #     print(reward)
    #     if i==0:
    #         tracker.init_trajectory(env.pose_history[-1].unsqueeze(0))
    #     tracker.update(env.reconstruction_algorithm, env.pose_history[-1].unsqueeze(0), i)
    #     if i>1:
    #         tracker.gif.save(f'{cfg.exp_name}/eval.gif')
    #         save_dict_to_excel(f'{cfg.exp_name}/metrics.xlsx', tracker.metric_hist)
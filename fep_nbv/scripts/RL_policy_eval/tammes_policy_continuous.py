import numpy as np
import os
import tyro
import sys
import json
import mathutils
import torch
from scipy.spatial import distance_matrix
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
from fep_nbv.utils.transform_viewpoints import xyz2polar

import warnings
warnings.filterwarnings("ignore")

def normalize(vectors):
    """ 归一化向量到单位球面 """
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def repulsive_forces(positions, lr=0.01, iterations=500):
    """ 在球面上优化N个点，使其尽量均匀分布（基于电荷排斥力） """
    N = positions.shape[0]
    for _ in range(iterations):
        # 计算所有点之间的欧氏距离
        distances = distance_matrix(positions, positions)
        np.fill_diagonal(distances, np.inf)  # 避免自相互作用
        
        # 计算力的方向（负梯度）
        forces = np.zeros_like(positions)
        for i in range(N):
            diffs = positions[i] - positions
            forces[i] = np.sum(diffs / (distances[i, :, None] ** 3), axis=0)
        
        # 梯度下降更新
        positions += lr * forces
        positions = normalize(positions)  # 归一化到球面
    
    return positions


if __name__=='__main__':
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    cfg.env.action_space_mode = ActionType.sphere
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133'
    # model_path = cfg.env.target_path
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path

    env = set_env(cfg)
    cfg.exp_name = f'data/test/RL_env_test/tammes_policy_continuous'
    if not os.path.exists(cfg.exp_name):
        os.makedirs(cfg.exp_name,exist_ok=True)
    tracker = RefMetricTracker(cfg, env=env)
    tracker.setup_writer(f'{cfg.exp_name}')

    print(env.action_space)

    # 初始化 N 个点，随机分布在单位球面上
    N = 20  # 需要的点数
    np.random.seed(42)
    points = normalize(np.random.randn(N, 3))

    # 运行优化，使点均匀分布
    tammes_points = repulsive_forces(points)*2

    for i in range(20):
        action = tammes_points[i]
        action = xyz2polar(action[0],action[1],action[2])
        obs,reward,_,_ = env.step(action)
        print(action)
        print(reward)
        if i==0:
            tracker.init_trajectory(env.pose_history[-1].unsqueeze(0))
        tracker.update(env.reconstruction_algorithm, env.pose_history[-1].unsqueeze(0), i)
        if i>1:
            tracker.gif.save(f'{cfg.exp_name}/eval.gif')
            save_dict_to_excel(f'{cfg.exp_name}/metrics.xlsx', tracker.metric_hist)
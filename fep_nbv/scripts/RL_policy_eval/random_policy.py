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

import warnings
warnings.filterwarnings("ignore")

if __name__=='__main__':
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet
    model_path = '/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133'
    # model_path = cfg.env.target_path
    obj_file_path = model_path+'/models/model_normalized.obj'
    cfg.env.target_path = obj_file_path

    for index in range(10):
        print(f'epoch {index}')
        env = set_env(cfg)
        cfg.exp_name = f'data/test/RL_env_test/random_policy/{index}'
        if not os.path.exists(cfg.exp_name):
            os.makedirs(cfg.exp_name,exist_ok=True)
        tracker = RefMetricTracker(cfg, env=env)
        tracker.setup_writer(f'{cfg.exp_name}')

        print(env.action_space)

        for i in range(20):
            action = env.action_space.sample()
            obs,reward,_,_ = env.step(action)
            print(action)
            print(reward)
            if i==0:
                tracker.init_trajectory(env.pose_history[-1].unsqueeze(0))
            tracker.update(env.reconstruction_algorithm, env.pose_history[-1].unsqueeze(0), i)
            if i>1:
                tracker.gif.save(f'{cfg.exp_name}/eval.gif')
                save_dict_to_excel(f'{cfg.exp_name}/metrics.xlsx', tracker.metric_hist)
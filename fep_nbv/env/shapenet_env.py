import gym
import numpy as np
import os
import tyro
import sys
import json
import mathutils
sys.path.append('/attached/remote-home2/zzq/04-fep-nbv')

from config import *
from fep_nbv.env.shapenet_scene import ShapeNetScene
from nvf.env.Enviroment import Enviroment
from fep_nbv.env.utils import *
from fep_nbv.utils import *

def set_env(cfg):
    if cfg.env.scene.name == 'hubble':
        # breakpoint()
        #aabb = ([[-0.92220873, -1.00288355, -1.03578806],
    #    [ 0.92220724,  1.05716348,  1.75192416]])
        cfg.object_aabb = torch.tensor([[-1, -1.1, -1.1], [1, 1.1, 1.8]])#*1.1
        factor = 2.5
        cfg.target_aabb = cfg.object_aabb*factor
        cfg.camera_aabb = cfg.object_aabb*factor
        cfg.env.scale = 0.3333 * 0.5
        cfg.density_threshold = 1e-3
    elif cfg.env.scene.name =='lego':
        # array([[-0.6377874 , -1.14001584, -0.34465557],
    #    [ 0.63374418,  1.14873755,  1.00220573]])
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-0.7, -1.2, -0.345], [0.7, 1.2, 1.1]])
        
        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb
    elif cfg.env.scene.name =='drums':
        # array([[-1.12553668, -0.74590737, -0.49164271],
        #[ 1.1216414 ,  0.96219957,  0.93831432]])
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-1.2, -0.8, -0.49164271], [1.2, 1.0, 1.0]])
        
        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        # cfg.camera_aabb = cfg.object_aabb[[0],:] + torch.stack([ torch.zeros(3), (cfg.object_aabb[1,:] - cfg.object_aabb[0,:])*factor])
        cfg.target_aabb = cfg.camera_aabb

        cfg.cycles_samples = 50000
        # cfg.env.n_init_views = 5
    elif cfg.env.scene.name =='hotdog':
        # wrong aabb [[-1.22326267 -1.31131911 -0.19066653]
        # [ 1.22326279  1.13520646  0.32130781]]
        
        # correct aabb [[-1.19797897 -1.28603494 -0.18987501]
        # [ 1.19797897  1.10992301  0.31179601]]
        # factor = 3
        cfg.object_aabb = torch.tensor([[-1.3, -1.4, -0.18987501], [1.3, 1.2, 0.5]])

        diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        # cfg.env.n_init_views = 5
        # cfg.check_density = True
    elif cfg.env.scene.name =='ship':
        # [[-1.27687299 -1.29963005 -0.54935801]
        # [ 1.37087297  1.34811497  0.728508  ]]
        cfg.object_aabb = torch.tensor([[-1.35, -1.35,-0.54935801], [1.45, 1.45, 0.73]])
        
        diff_box = torch.tensor([[-1.7,-1.7,0.43], [1.7,1.7,3.3]])
        
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        # cfg.env.n_init_views = 3

        # if cfg.d0 > 0.: cfg.d0=0.8
    elif cfg.env.scene.name =='chair':
        # [[-0.72080803 -0.69497311 -0.99407679]
        # [ 0.65813684  0.70561057  1.050102  ]]

        cfg.object_aabb = torch.tensor([[-0.8, -0.8,-0.99407679], [0.8, 0.8, 1.1]])
        
        diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

    elif cfg.env.scene.name =='mic':
    #     array([[-1.25128937, -0.90944701, -0.7413525 ],
    #    [ 0.76676297,  1.08231235,  1.15091646]])
        # factor = 2.5
        cfg.object_aabb = torch.tensor([[-1.3, -1.0,-0.7413525], [0.8, 1.2, 1.2]])
        diff_box = torch.tensor([[-1.7,-1.7,0.], [1.7,1.7,4.5]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        
        # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
        

        cfg.target_aabb = cfg.camera_aabb
        # cfg.env.n_init_views = 5
        # breakpoint()
    elif cfg.env.scene.name =='materials':
        # [[-1.12267101 -0.75898403 -0.23194399]
        # [ 1.07156599  0.98509198  0.199104  ]]
        # factor = torch.tensor([2.5, 2.5, 3.5]).reshape(-1,3)
        cfg.object_aabb = torch.tensor([[-1.2, -0.8,-0.23194399], [1.2, 1.0, 0.3]])
        # ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        # cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base

        diff_box = torch.tensor([[-1.5,-1.5,0.], [1.5,1.5,3.]])
        cfg.camera_aabb = cfg.object_aabb+diff_box
        cfg.target_aabb = cfg.camera_aabb

        cfg.target_aabb = cfg.camera_aabb
        # breakpoint()
    elif cfg.env.scene.name =='ficus':
        #[[-0.37773791 -0.85790569 -1.03353798]
        #[ 0.55573422  0.57775307  1.14006007]]
        factor = 2.5
        cfg.object_aabb = torch.tensor([[-0.4, -0.9, -1.03353798], [0.6, 0.6, 1.2]])

        ref_base = torch.tensor([0.,0.,cfg.object_aabb[0,2]]).reshape(-1,3)

        cfg.camera_aabb = (cfg.object_aabb-ref_base)*factor+ref_base
        cfg.target_aabb = cfg.camera_aabb

        # cfg.env.n_init_views = 5
    elif cfg.env.scene.name =='shapenet':
        cfg.object_aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]])#*1.1
        factor = 2
        cfg.target_aabb = cfg.object_aabb
        cfg.camera_aabb = cfg.object_aabb*2.8
        # cfg.env.scale = 0.3333 * 0.5
        cfg.density_threshold = 1e-3
    elif cfg.env.scene.name =='nerf':
        cfg.object_aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]])#*1.1
        factor = 2
        cfg.target_aabb = cfg.object_aabb
        cfg.camera_aabb = cfg.object_aabb*2.8
        # cfg.env.scale = 0.3333 * 0.5
        cfg.density_threshold = 1e-3
    else:
        raise NotImplementedError
    print(cfg.scene.name)
    if cfg.env.scene.name =='shapenet':
        env = ShapeNetEnviroment(cfg.env)
    else:
        env = Enviroment(cfg.env)
    # breakpoint()
    return env


class ShapeNetEnviroment(gym.Env):
    # init_images = 10
    # horizon = 20
    """Custom Environment that follows gym interface"""
    def __init__(self, cfg,):
        super(ShapeNetEnviroment, self).__init__()

        self.horizon = cfg.horizon
        self.max_steps = cfg.horizon
        
        self.pose_history = []
        self.obs_history = []

        # set env
        cfg.object_aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]])#*1.1
        factor = 2
        cfg.target_aabb = cfg.object_aabb
        cfg.camera_aabb = cfg.object_aabb*factor
        cfg.density_threshold = 1e-3

        self.cfg = cfg

        self.scene = eval(self.cfg.scene)(self.cfg)

        # Initialize state
        self.reset()

        # print(f'env info: \n object_aabb {self.cfg.object_aabb} \n target_aabb {self.cfg.target_aabb} \n camera_aabb {self.cfg.camera_aabb} \n density_threshold {self.cfg.density_threshold}')

    def step(self, action):
        if action.ndim==1:
            action = action.unsqueeze(0)
        position = self.state

        # new_poses = []
        new_images = []
        for pose in action:
            img = self.scene.render_pose(pose)

            # self.scene.set_camera_pose(pose)
            # result = self.scene.render()
            # # img = result['mask']*result['image']
            # img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
            # input(img.shape)
            # new_poses.append(pose)
            new_images.append(img)
        

        
        # self.pipeline.add_image(images=new_images, poses=action)

        self.pose_history += action
        self.obs_history += new_images

        done = False
        if self.steps >= self.max_steps:
            done = True

        reward = 0. # TODO

        self.state = action[-1]
        self.steps += 1


        return new_images, reward, done, {}

    def reset(self):
        # transforms = get_transforms(idx_start=0, idx_end=100)

        # np_images = []
        # for pose in transforms:
        #     self.scene.set_camera_pose(pose)
        #     result = self.scene.render()
        #     img = result['mask']*result['image']
        #     img = rgb_to_rgba(img)
        #     # input(img.shape)
        #     # new_poses.append(pose)
        #     np_images.append(img)
        
        np_images, transforms = self.get_images(mode='init')

        self.pose_history = transforms
        self.obs_history = np_images
        self.state = np.array(self.pose_history[-1])  # Reset position to the center

        # self.pipeline.add_image(images=np_images, poses=transforms)
        
        self.steps = 0
        return np_images  # reward, done, info can't be included
    
    def get_images(self, mode, return_quat=True):
        return self.gen_data(mode, return_quat=return_quat)


        # file = f'data/{self.cfg.scene.name}/{mode}/transforms.json'
        # if not os.path.exists(file) or (hasattr(self.cfg, f'gen_{mode}') and getattr(self.cfg, f'gen_{mode}')):
        #     return self.gen_data(mode, return_quat=return_quat)
        # else:
        #     img, transforms = self.scene.load_data(file=file, return_quat=return_quat)
        #     if mode == 'init':
        #         if len(img) != self.cfg.n_init_views:
        #             return self.gen_data(mode, return_quat=return_quat)
        #     return img, transforms
          
    def gen_data(self, mode, return_quat=False):
        file = f'data/{self.cfg.scene.name}/{mode}/transforms.json'
        print(f'Generating data for {mode}')
        poses = self.scene.gen_data_fn[mode]()

        images = []
        for pose in poses:
            image = self.scene.render_pose(pose)
            images.append(image)
        
        if getattr(self.cfg, f'save_data'):
            print(f'saving data to {file}')
            self.scene.save_data(file, poses, images)
        
        if return_quat:
            poses = [pose2tensor(pose) for pose in poses]
        else:
            poses = [torch.FloatTensor(pose) for pose in poses]
        return images, poses
 


if __name__=='__main__':
    # 环境创建初始化
    # 函数：step, reset, render, close
    cfg = tyro.cli(ExpConfig)
    cfg.env.scene = SceneType.shapenet

    target_path = random_shapenet_model_path()
    obj_file_path = target_path+'/models/model_normalized.obj'
    json_path = obj_file_path[:-4]+'.json'
    with open(json_path, 'r') as file:
        data = json.load(file)
        centroid = data.get("centroid", None)
    cfg.env.target_path = obj_file_path # type: ignore

    env = ShapeNetEnviroment(cfg.env)

    # 渲染图像
    camera_position = mathutils.Vector((1, 1, 1))
    target_position = mathutils.Vector((0, 0, 0))
    direction = target_position - camera_position
    rot_quat = torch.tensor(direction.normalized().to_track_quat('-Z', 'Y')) # wxyz
    fixed_pose = torch.concat((rot_quat[[1,2,3,0]], torch.tensor(camera_position))) # xyzwxyz
    obs,_,_,_ = env.step(fixed_pose)

    print(obs[0].shape)
    print(obs[0].max())
    print('env test success')


    pass 
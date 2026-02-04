import os
import shutil

if __name__=='__main__':
    select_method = 'all'
    delete_method = 'small'
    policy_name = select_method+'_'+delete_method
    metric_name = 'PSNR'

    origin_path = ''

    # save_path = os.path.join('/attached/remote-home2/zzq/04-fep-nbv/data/test/GS_poses/',origin_path.split('/')[-2],origin_path.split('/')[-1])
    # os.makedirs(save_path, exist_ok=True) 
    # i=1
    # for model_name in os.listdir(origin_path):
    #     model_path = os.path.join(origin_path, model_name)
    #     os.makedirs(os.path.join(save_path,model_name), exist_ok=True) 
    #     pose_path = os.path.join(model_path, 'run-0','pose_his_0.txt') # not uniform
    #     pose_path = os.path.join(model_path, 'pose_his_0.txt')
    #     if not os.path.exists(pose_path):
    #         print(f'Pose file not found for {model_name}, skipping...')
    #         continue
    #     target_filename = f"{i}.txt"
    #     target_path = os.path.join(save_path, model_name,target_filename)

    #     cfg_path = os.path.join(model_path, 'cfg.yaml')
    #     cfg_target_path = os.path.join(save_path, model_name,'cfg.yaml')

    #     shutil.copy(pose_path, target_path)
    #     shutil.copy(cfg_path, cfg_target_path)
    #     print(f'✅ Copied: {target_path}')
    #     i+=1


    # for ours poses
    # save_path = os.path.join(save_path,origin_path.split('/')[-1])
    # os.makedirs(save_path, exist_ok=True) 
    # for model_name in os.listdir(origin_path):
    #     model_path = os.path.join(origin_path, model_name)
    #     pose_path = os.path.join(model_path, policy_name,metric_name,'pose_his_0.txt')
    #     if not os.path.exists(pose_path):
    #         print(f'Pose file not found for {model_name}, skipping...')
    #         continue
    #     target_filename = f"{model_name}_{policy_name}_{metric_name}.txt"
    #     target_path = os.path.join(save_path, target_filename)

    #     shutil.copy(pose_path, target_path)
    #     print(f'✅ Copied: {target_path}')


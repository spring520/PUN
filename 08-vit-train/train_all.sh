#!/bin/bash

# tiny_vit_11m_224 Train Loss: 3.063997 | Val Loss: 4.191974
# tiny_vit_5m_224 | Train Loss: 1.260856 | Val Loss: 6.200608
# vit_base_r50_s16_224 
# vit_small_patch16_224 Train Loss: 0.086282 | Val Loss: 3.457165
# tiny_vit_21m_224 | Train Loss: 2.582350 | Val Loss: 4.216508
# vit_huge_patch14_224

vit_models=(
    "tiny_vit_11m_224"
    "tiny_vit_5m_224"
    "vit_base_r50_s16_224"
    "vit_small_patch16_224"
    "tiny_vit_21m_224"
    # "vit_huge_patch14_224"
)



# 遍历训练所有模型
# for vit_model in "${vit_models[@]}"; do
#     echo "Training $vit_model..."
#     python test_train.py --vit_used vit_small_patch16_224 --epochs 100 --batch_size 32 --lr 1e-4
# done

uncertainty_modes=("PSNR" "SSIM" "MSE")
dataset_paths=(
    "/mnt/hdd/zhengquan/single_datasets/splat-image-distribution-dataset-chair"
    "/mnt/hdd/zhengquan/single_datasets/splat-image-distribution-dataset-car"
    "/mnt/hdd/zhengquan/single_datasets/splat-image-distribution-dataset-vessel"
)
# 遍历训练所有模型
for dataset_path in "${dataset_paths[@]}"; do
    for uncertainty_mode in "${uncertainty_modes[@]}"; do
        echo "Training $uncertainty_mode $dataset_path..."
        python test_train.py --vit_used vit_small_patch16_224 --epochs 100 --batch_size 32 --lr 1e-4 --uncertainty_mode $uncertainty_mode --dataset_path $dataset_path
    done
done
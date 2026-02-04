#!/bin/bash

# 设置模型路径列表
TARGET_PATHS=(

)

vit_ckpt_paths=(
  "vit_small_patch16_224_PSNR_250425172703" # metric ablation
  # "vit_small_patch16_224_SSIM_250425203806"
  # "vit_small_patch16_224_MSE_250425234848"
  # "vit_small_patch16_224_LPIPS_250427190809"

  # "vit_small_patch16_224_PSNR_1_250510002759" # dataset size ablation
  # "vit_small_patch16_224_PSNR_2_250510013619"
  # "vit_small_patch16_224_PSNR_4_250509232004"

  # "vit_small_patch16_224_PSNR_airplane_250611155057" # 3 run
  # "vit_small_patch16_224_PSNR_cabinet_250611155130"

)

# conda activate splat-image-bpy

# 遍历路径并运行 Python 脚本
count=0
max_retries=100
for vit_ckpt_path in "${vit_ckpt_paths[@]}"
do
  for target_path in "${TARGET_PATHS[@]}"
  do
    echo "🔁 Running with target: $target_path using $vit_ckpt_path"
    until python fep_nbv/baseline/our_policy_single.py \
      "--vit_ckpt_path" "$vit_ckpt_path" \
      "--model_3d_path" "$target_path" \
      "--all_dataset"; do
        ((count++))
        if [ $count -ge $max_retries ]; then
            echo "❌ 已经重试 $max_retries 次，依然失败，跳过。"
            break
        fi
        echo "⚠️ 运行失败，第 $count 次重试..."
    done
  done
done
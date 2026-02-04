This repository provides the **official implementation** of **_Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction_**.

## news
We release our code and model.

## Dataset

The following datasets are used in our experiments:

- **ShapeNet** (https://shapenet.org)
- **NeRF Synthetic Dataset** (data/assets/blend_files)

## Installation

### Dependencies

This repository depends on **NVF** and **Nerfstudio**.  
Please follow the official NVF installation instructions:

1. install nvf (https://github.com/GaTech-RL2/nvf_cvpr24.git)

2. install other dependency
    pip install -r requirements.txt

> Refer to the NVF repository for detailed setup instructions of both **NVF** and **Nerfstudio**.

### Tested Environment

The code has been tested under the following environment:

- **GPU**: NVIDIA GTX 3090
- **PyTorch**: 2.0.1
- **CUDA**: 11.7
- **Operating System**: Ubuntu 20.04
- **python**: 3.10

Other configurations may work but are not guaranteed.

---

## Usage

### Inference

To run NBV inference with a trained model:

```bash
python fep_nbv/baseline/our_policy_single.py --vit_ckpt_path vit_small_patch16_224_PSNR_250425172703 --model_3d_path <PATH_TO_ShapeNet>/<category>/<instance index> # some instance example in data/instance_example
```

To train a UPNet using NUM dataset
```bash
cd 08-vit-train
python test_train.py --vit_used vit_small_patch16_224 --epochs 100 --batch_size 32 --lr 1e-4 --uncertainty_mode PSNR --dataset_path data/NUM_example
```
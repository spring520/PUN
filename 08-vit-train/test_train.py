import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from timm.data import resolve_data_config, create_transform
import torch.optim as optim
import json
from torchvision import transforms
from PIL import Image
import os
from regress_model import ViTRegressor
from regress_dataset import RegressionDataset, RegressionDataset_singleclass
from datetime import datetime
import argparse
from tqdm import tqdm

# tiny_vit_11m_224
# tiny_vit_5m_224
# vit_base_r50_s16_224
# vit_small_patch16_224
# tiny_vit_21m_224
# vit_huge_patch14_224

def get_current_timestamp():
    """返回当前时间，格式为 yymmddhhmmss"""
    return datetime.now().strftime("%y%m%d%H%M%S")

if __name__=='__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train a ViT model for regression")
    parser.add_argument("--vit_used", type=str, default="vit_base_patch16_224.augreg_in21k", help="Choose ViT model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--uncertainty_mode",type=str,default='PSNR')
    parser.add_argument("--dataset_path",type=str,default='/mnt/hdd/zhengquan/single_datasets/splat-image-distribution-dataset-chair')
    args = parser.parse_args()
    
    
    log_path = f"logs/{args.dataset_path.split('-')[-1]}_single_splited/{args.vit_used}_{args.uncertainty_mode}_{get_current_timestamp()}/"
    if os.path.exists(log_path):
        pass
    else:
        log_path = f"logs/{args.dataset_path.split('-')[-1]}_all/{args.vit_used}_{args.uncertainty_mode}_{get_current_timestamp()}/"
    os.makedirs(log_path, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(log_path,"vit_regressor_checkpoint.pth")
    HISTORY_PATH = os.path.join(log_path,"train_history.json")

    # load transform
    model = ViTRegressor(model_name=args.vit_used,output_dim=48).to(args.device)
    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    print(model.backbone.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    
    train_dataset = RegressionDataset_singleclass(
        root_dir=args.dataset_path,  # 替换为你的路径
        transform=transform,
        mode='PSNR',
        split='train'
    )
    test_dataset = RegressionDataset_singleclass(
        root_dir=args.dataset_path,  # 替换为你的路径
        transform=transform,
        mode='PSNR',
        split='test'
    )
    # **拆分训练/验证集**
    train_size = int(8/9 * len(train_dataset))
    val_size = int(1/9 * len(train_dataset))
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f'train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}')

    # === 4. 训练配置 ===
    criterion = nn.MSELoss()  # 均方误差
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # train loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        

        # === 验证 ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # === 测试 ===
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        history["test_loss"].append(avg_test_loss)

       

        # 打印训练状态
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # **保存最优模型**
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(log_path,"best_vit_regressor.pth"))
            print("✅ Model Saved (New Best)")
        # **保存 Checkpoint**
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        # train
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
    # 保存训练历史
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        outputs = model(images)
        print(outputs[0,:5])
        print(labels[0,:5])


# # 2. 加载模型
# model_name = 'vit_base_patch16_224'
# model = timm.create_model(model_name, pretrained=True, num_classes=1)  # 注意：num_classes=1 表示回归任务
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)


# # 3. 数据预处理（自动读取模型推荐的 mean/std）
# config = resolve_data_config({}, model=model)
# transform = create_transform(**config)

# # 4. 加载数据
# train_dataset = RegressionDataset('train_images', 'train_labels.txt', transform=transform)
# train_loader = DataLoader(train_dataset, batchsize=32, shuffle=True)

# # 5. 损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # 6. 训练循环
# model.train()
# for epoch in range(10):
#     epoch_loss = 0
#     for imgs, labels in train_loader:
#         imgs, labels = imgs.to(device), labels.to(device)

#         preds = model(imgs)
#         loss = criterion(preds, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
import torch.nn as nn
import torch
import timm

class ViTRegressor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', output_dim=48, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)  # Remove classification head
        self.regressor = nn.Linear(in_features, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTRegressor(output_dim=48).to(device)

    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    # test input
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model(x)
    print(output.shape)  # torch.Size([2, 48])
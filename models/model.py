import torch
from torch import nn
import torch.nn.functional as F
from models.modules import *

segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}

# 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
# 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
# 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}}

class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = segmentation_head_dict[segmentation_head](backbone_out, **model_config)
        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_head_out = self.segmentation_head(backbone_out)
        y = segmentation_head_out
        return y


if __name__ == '__main__':
    device = torch.device('cuda:0')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model_config = {
        'backbone': 'shufflenetv2',
        'fpem_repeat': 2,  # fpem模块重复的次数
        'pretrained': True,  # backbone 是否使用imagesnet的预训练模型
        'segmentation_head': 'FPN'  # 分割头，FPN or FPEM_FFM
    }
    model = Model(model_config=model_config).to(device)
    y = model(x)

    # print(model)
    # torch.save(model.state_dict(), 'PAN.pth')

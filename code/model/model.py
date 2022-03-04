import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        x = self.model(x)
        return x


class vit_small_r26_s32_224_in21k(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.model=timm.create_model('vit_small_r26_s32_224_in21k', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class efficientnetv2_rw_t(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.model=timm.create_model('efficientnetv2_rw_t', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


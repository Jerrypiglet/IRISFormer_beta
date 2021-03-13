
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision.models import resnet

class netCS(nn.Module):
    # Use PhotoShape material classifier architecture
    def __init__(self, opt, inChannels=5, num_classes=886, base_model=resnet.resnet18, if_est_scale=False, if_est_sup=True):
        super(netCS,  self).__init__()
        self.if_est_sup = if_est_sup

        self.conv1 = nn.Conv2d(
            inChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if hasattr(base_model, 'forward'):
            self.base_model = base_model
        else:
            self.base_model = base_model()

        self.fc_material = nn.Linear(
            512 * resnet.BasicBlock.expansion, num_classes)

        if if_est_sup:
            self.fc_material_sup = nn.Linear( # https://github.com/keunhong/photoshape/blob/6e795512e059bc5a6bdac748fda961f66d51c6f6/src/terial/classifier/network.py   
                512 * resnet.BasicBlock.expansion, opt.cfg.MODEL_MATCLS.num_classes_sup + 1) # + 1 for non-assigned

        self.if_est_scale = if_est_scale
        if self.if_est_scale:
            self.fc_scales = nn.Linear(512 * resnet.BasicBlock.expansion, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)

        output = {}

        output['material'] = self.fc_material(x)

        if self.if_est_scale:
            output['scale'] = self.fc_scales(x)

        if self.if_est_sup:
            output['material_sup'] = self.fc_material_sup(x)
            

        return output

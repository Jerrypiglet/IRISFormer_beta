import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision.models import resnet


class encoder(nn.Module):
    def __init__(self, cascadeLevel=0, isSeg=False):
        super(encoder, self).__init__()

        self.pad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=7, out_channels=64, kernel_size=4, stride=2, bias=True)

        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=64)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True)
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(self, x):
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x))), True)  # 64, h/2, w/2
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1))), True)  # 128, h/4, w/4
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2))), True)  # 256, h/8, w/8
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3))),
                    True)  # 256, h/16, w/16
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4))),
                    True)  # 512, h/32, w/32
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5))),
                    True)  # 1024, h/32, w/32

        return x1, x2, x3, x4, x5, x6


class decoderWPN(nn.Module):
    def __init__(self, imgH=240, imgW=320):
        super(decoderWPN, self).__init__()

        self.fl = nn.Flatten()

        self.dconv1 = nn.Conv2d(
            in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.dconv1_n = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        #self.dbn1_n   = nn.BatchNorm2d(num_features=1)
        self.dfc1_s = nn.Linear(
            in_features=512*(imgH//32)*(imgW//32), out_features=1024, bias=True)

        self.dconv2 = nn.Conv2d(
            in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256)
        self.dconv2_n = nn.Conv2d(
            in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        #self.dbn2_n   = nn.BatchNorm2d(num_features=2)
        self.dfc2_s = torch.nn.Linear(
            in_features=256*(imgH//16)*(imgW//16), out_features=1024, bias=True)

        self.dconv3 = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256)
        self.dconv3_n = nn.Conv2d(
            in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        #self.dbn3_n   = nn.BatchNorm2d(num_features=2)
        self.dfc3_s = torch.nn.Linear(
            in_features=256*(imgH//8)*(imgW//8), out_features=1024, bias=True)

        self.dconv4 = nn.Conv2d(
            in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=128)
        self.dconv4_n = nn.Conv2d(
            in_channels=128, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        #self.dbn4_n   = nn.BatchNorm2d(num_features=2)
        self.dfc4_s = torch.nn.Linear(
            in_features=128*(imgH//4)*(imgW//4), out_features=1024, bias=True)

        self.dconv5 = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64)
        self.dconv5_n = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        #self.dbn5_n   = nn.BatchNorm2d(num_features=2)
        self.dfc5_s = torch.nn.Linear(
            in_features=64*(imgH//2)*(imgW//2), out_features=1024, bias=True)

        self.dconv6 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=4, num_channels=64)
        self.dconv6_n = nn.Conv2d(
            in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        #self.dbn6_n   = nn.BatchNorm2d(num_features=2)
        self.dfc6_s = torch.nn.Linear(
            in_features=64*(imgH)*(imgW), out_features=1024, bias=True)

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, bias=True)
        self.dgn7 = nn.GroupNorm(num_groups=2, num_channels=32)
        self.dconv7_n = nn.Conv2d(
            in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        self.dfc7_s = torch.nn.Linear(
            in_features=32*(imgH)*(imgW), out_features=1024, bias=True)

    def forward(self, im, x1, x2, x3, x4, x5, x6):
        dx1 = F.relu(self.dgn1(self.dconv1(x6)))  # 512, h/32, w/32
        dx1_n = self.dconv1_n(F.interpolate(
            dx1, size=[4, 4], mode='bilinear'))  # 1, 4, 4
        dx1_s = self.dfc1_s(self.fl(dx1))

        xin1 = torch.cat([dx1, x5], dim=1)  # 1024, h/32, w/32
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(
            xin1, scale_factor=2, mode='bilinear'))), True)  # 256, h/16, w/16
        dx2_n = self.dconv2_n(F.interpolate(
            dx2, size=[8, 8], mode='bilinear'))  # 2, 8, 8
        dx2_s = self.dfc2_s(self.fl(dx2))

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        xin2 = torch.cat([dx2, x4], dim=1)  # 512, h/16, w/16
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(
            xin2, scale_factor=2, mode='bilinear'))), True)  # 256, h/8, w/8
        dx3_n = self.dconv3_n(F.interpolate(
            dx3, size=[16, 16], mode='bilinear'))  # 2, 16, 16
        dx3_s = self.dfc3_s(self.fl(dx3))

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        xin3 = torch.cat([dx3, x3], dim=1)  # 512, h/8, w/8
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(
            xin3, scale_factor=2, mode='bilinear'))), True)  # 128, h/4, w/4
        dx4_n = self.dconv4_n(F.interpolate(
            dx4, size=[32, 32], mode='bilinear'))  # 2, 32, 32
        dx4_s = self.dfc4_s(self.fl(dx4))

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        xin4 = torch.cat([dx4, x2], dim=1)  # 256, h/4, w/4
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(
            xin4, scale_factor=2, mode='bilinear'))), True)  # 64, h/2, w/2
        dx5_n = self.dconv5_n(F.interpolate(
            dx5, size=[64, 64], mode='bilinear'))  # 2, 64, 64
        dx5_s = self.dfc5_s(self.fl(dx5))

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        xin5 = torch.cat([dx5, x1], dim=1)  # 128, h/2, w/2
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(
            xin5, scale_factor=2, mode='bilinear'))), True)  # 64, h, w
        dx6_n = self.dconv6_n(F.interpolate(
            dx6, size=[128, 128], mode='bilinear'))  # 2, 128, 128
        dx6_s = self.dfc6_s(self.fl(dx6))

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear')
        x_orig = F.relu(self.dgn7(self.dconvFinal(
            self.dpadFinal(dx6))), True)  # 32, h, w
        dx7_n = self.dconv7_n(F.interpolate(
            dx_orig, size=[256, 256], mode='bilinear'))  # 2, 256, 256
        dx7_s = self.dfc7_s(self.fl(dx7))

        return dx1_n, dx2_n, dx3_n, dx4_n, dx5_n, dx6_n, dx7_n, dx1_s, dx2_s, dx3_s, dx4_s, dx5_s, dx6_s, dx7_s


class netCS(nn.Module):
    # Use PhotoShape material classifier architecture
    def __init__(self, inChannels=5, num_classes=886, base_model=resnet.resnet18):
        super(netCS,  self).__init__()

        self.conv1 = nn.Conv2d(
            inChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if hasattr(base_model, 'forward'):
            self.base_model = base_model
        else:
            self.base_model = base_model()

        self.fc_material = nn.Linear(
            512 * resnet.BasicBlock.expansion, num_classes)
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
        output['scale'] = self.fc_scales(x)

        return output

class netW(nn.Module):a
    # Use PhotoShape material classifier architecture
    def __init__(self, inChannels=5, num_dims=512, base_model=resnet.resnet18):
        super(netW,  self).__init__()

        self.conv1 = nn.Conv2d(
            inChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if hasattr(base_model, 'forward'):
            self.base_model = base_model
        else:
            self.base_model = base_model()

        self.fc_material = nn.Linear(
            512 * resnet.BasicBlock.expansion, num_dims)

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

        return output
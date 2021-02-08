import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class encoderLight(nn.Module ):
    def __init__(self, SGNum, cascadeLevel = 0 ):
        super(encoderLight, self).__init__()

        self.cascadeLevel = cascadeLevel
        self.SGNum = SGNum

        self.preProcess = nn.Sequential(
                nn.ReplicationPad2d(1),
                nn.Conv2d(in_channels=11, out_channels=32, kernel_size=4, stride=2, bias =True),
                nn.GroupNorm(num_groups=2, num_channels=32),
                nn.ReLU(inplace = True ),

                nn.ZeroPad2d(1),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=True),
                nn.GroupNorm(num_groups=4, num_channels=64 ),
                nn.ReLU(inplace = True )
                )

        self.pad1 = nn.ReplicationPad2d(1)
        if self.cascadeLevel == 0:
            self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, bias = True)
        else:
            self.conv1 = nn.Conv2d(in_channels=64 + SGNum * 7, out_channels=128, kernel_size=4, stride=2, bias =True)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True)
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(self, inputBatch, envs = None):

        input1 = self.preProcess(inputBatch )
        input2 = envs

        if self.cascadeLevel == 0:
            x = input1
        else:
            x = torch.cat([input1, input2], dim=1)

        x1 = F.relu(self.gn1(self.conv1(self.pad1(x) ) ), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1) ) ), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2) ) ), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3) ) ), True)
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4) ) ), True)
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5) ) ), True)

        return x1, x2, x3, x4, x5, x6

class decoderLight(nn.Module ):
    def __init__(self, SGNum,  mode = 0):
        super(decoderLight, self).__init__()
        self.SGNum = SGNum

        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn2 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn4 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv5 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn5 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dconv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn6 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dpadFinal = nn.ReplicationPad2d(1)

        if mode == 0 or mode == 2:
            self.dconvFinal = nn.Conv2d(in_channels=128, out_channels = 3*SGNum, kernel_size=3, stride=1, bias=True )
        elif mode == 1:
            self.dconvFinal = nn.Conv2d(in_channels=128, out_channels = SGNum, kernel_size=3, stride=1, bias=True )

        self.mode = mode

    def forward(self, x1, x2, x3, x4, x5, x6, env = None):
        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )

        xin1 = torch.cat([dx1, x5], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear') ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        xin2 = torch.cat([dx2, x4], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear') ) ), True)

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear') ) ), True)

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        xin4 = torch.cat([dx4, x2], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear') ) ), True)

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        xin5 = torch.cat([dx5, x1], dim=1 )
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear') ) ), True)

        if dx6.size(3) != env.size(3) or dx6.size(2) != env.size(2):
            dx6 = F.interpolate(dx6, [env.size(2), env.size(3)], mode='bilinear')
        x_orig = self.dconvFinal(self.dpadFinal(dx6 ) )

        x_out = 1.01 * torch.tanh(self.dconvFinal(self.dpadFinal(dx6) ) )

        if self.mode == 1 or self.mode == 2:
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, 0, 1)
        elif self.mode == 0:
            bn, _, row, col = x_out.size()
            x_out = x_out.view(bn, self.SGNum, 3, row, col)
            x_out = x_out / torch.clamp(torch.sqrt(torch.sum(x_out * x_out,
                dim=2).unsqueeze(2) ), min = 1e-6).expand_as(x_out )
        return x_out


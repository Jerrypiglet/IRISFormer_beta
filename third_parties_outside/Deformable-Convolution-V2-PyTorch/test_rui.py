import imageio, torch
import numpy as np
import torch.nn as nn

im_path = 'test.png'
im_uint8 = imageio.imread(im_path)
im_float = im_uint8.astype(np.float32)
im_tensor = torch.from_numpy(im_float).float().permute(2, 0, 1).unsqueeze(0).cuda()
zeros = im_tensor * 0.
zeros.requires_grad = True
im_tensor = im_tensor + zeros

N, inC, inH, inW = im_tensor.shape 
deformable_groups = 1
inC = 3
outC = 3
kH, kW = 7, 7
kernel_size=(kH, kW)
stride = (1, 1)
padding = 3
dilation=1
groups=1
deformable_groups=1
im2col_step=1
bias=True

# deform-conv itself
weight = nn.Parameter(torch.Tensor(
    outC, inC//groups, *kernel_size)).cuda()
bias = nn.Parameter(torch.Tensor(outC)).cuda()
weight.data.fill_(1.)
bias.data.fill_(1.0)

conv_offset_op = torch.nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                        kernel_size=(kH, kW),
                        stride=(1, 1),
                        padding=padding,
                        bias=True).cuda()
# conv_offset_op.weight.data.fill_(0.0)
# conv_offset_op.bias.data.fill_(0.0)
offsets = conv_offset_op(im_tensor)
print('Offsets:', offsets.shape)

import sys
sys.path.insert(0, '/home/ruizhu/Documents/Projects/semanticInverse/third-parties_outside/Deformable-Convolution-V2-PyTorch/functions')
from deform_conv_func import DeformIm2colFunction, DeformConvFunction
DCN_func_output = DeformConvFunction.apply(im_tensor.contiguous(), offsets, 
                          weight, 
                          bias, 
                          stride, 
                          padding, 
                          dilation, 
                          groups,
                          deformable_groups,
                          im2col_step)
print(DCN_func_output.grad_fn)
loss = torch.sum(DCN_func_output)
loss.backward()
# print(im_tensor.grad)
print('grad on offsets:', offsets.grad, offsets.requires_grad)
print('grad on conv_offset_op.weight:', conv_offset_op.weight.grad, conv_offset_op.weight.requires_grad)
print('grad on zeros:', zeros.grad, zeros.requires_grad)
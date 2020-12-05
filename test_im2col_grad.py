import imageio, torch
import numpy as np
import torch.nn as nn

import sys
sys.path.insert(0, '/home/ruizhu/Documents/Projects/semanticInverse/third-parties_outside/Deformable-Convolution-V2-PyTorch/functions')
from deform_conv_func import DeformIm2colFunction

# im_path = './notebooks/test.png'
# im_uint8 = imageio.imread(im_path)
# im_float = im_uint8.astype(np.float32)
# im_tensor_ori = torch.from_numpy(im_float).float().permute(2, 0, 1).unsqueeze(0).cuda()
im_tensor_ori = torch.ones((1, 3, 10, 8)).cuda().float() / 3.14

N, inC, inH, inW = im_tensor_ori.shape 
deformable_groups = 1
inC = 3
outC = 3
stride = (1, 1)

# kH, kW = 3, 3
# padding = 1
# dilation=1

kH, kW = 3, 3
padding = 2
dilation=2

groups=1
deformable_groups=1
im2col_step=1
bias=True
kernel_size=(kH, kW)

im_tensor_op = nn.Conv2d(3, 3,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1), 
                        bias=True).cuda()

im_tensor_op.weight.data.fill_(0.01)
im_tensor_op.bias.data.fill_(0.0)
im_tensor = im_tensor_op(im_tensor_ori)
# im_tensor = torch.zeros_like(im_tensor, requires_grad=True, device=im_tensor.device).cuda()

# deform-conv itself
weight = nn.Parameter(torch.Tensor(
    outC, inC//groups, *kernel_size)).cuda()
bias = nn.Parameter(torch.Tensor(outC)).cuda()
weight.data.fill_(0.01)
bias.data.fill_(0.0)

conv_offset_op = torch.nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                        kernel_size=(kH, kW),
                        stride=stride,
                        padding=padding,
                        dilation=dilation, 
                        bias=True).cuda()
conv_offset_op.weight.data.fill_(0.001)
conv_offset_op.bias.data.fill_(0.0)
offsets = conv_offset_op(im_tensor)
print('Offsets:', offsets.shape)

# inp_unf = torch.nn.functional.unfold(im_tensor, kernel_size, stride=stride, padding=padding, dilation=dilation) # torch.Size([1, 27, 6])
inp_unf = DeformIm2colFunction.apply(im_tensor, offsets, kernel_size, stride, padding, dilation)
loss = torch.mean(inp_unf)

# out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
# torch_out = torch.nn.functional.fold(out_unf, (240, 320), (1, 1))
# loss = torch.mean(torch_out) - 2

loss.backward()

print(im_tensor_op.weight.grad, loss)
print(conv_offset_op.weight.grad)
# print('grad on im_tensor:', im_tensor_op.weight.grad.shape, im_tensor_op.weight.grad[:, :, :3, 0], im_tensor_op.weight.requires_grad)

# for x in [conv_offset_op.weight.grad, conv_offset_op.bias.grad, im_tensor_op.weight.grad, im_tensor_op.bias.grad]:
#     x.data.zero_()
'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

import os, cv2, torch, numpy as np


def m_makedir(dirpath):
	os.makedirs(dirpath, exist_ok=True)

def CamIntrinsic_to_cuda(CamIntrinsic, device, members = ['intrinsic_M_cuda', 'unit_ray_array_2D']):
    '''put the necessary members in the CamIntrinsic structure to the specified device'''
    for mem in members:
        CamIntrinsic[mem] = CamIntrinsic[mem].to(device)

def get_init_inbound_masks(H, W, spixel_dim, device='cuda'):
    '''
    input:
    H, W - image size
    spixel_dim - (nspixel_horizontal, nspixel_vertical)

    output: 
    9 x H x W

    get the 3x3 spixel in_bound_mask
    
    3x3 index for local neighborhood indexing:
    0 1 2 
    3 4 5
    6 7 8
    '''
    import math
    in_bound_masks = torch.ones( (9, H, W), device=device)
    spixel_width = math.ceil(float(W) / float(spixel_dim[0]))
    spixel_height= math.ceil(float(H) / float(spixel_dim[1]))

    Y, X = torch.meshgrid( torch.arange(0, H).float(), torch.arange(0, W).float())
    X, Y = X.to(in_bound_masks), Y.to(in_bound_masks)

    in_bound_masks[0] = torch.logical_and( X-spixel_width>=0., Y-spixel_height>=0. ).float()
    in_bound_masks[1] = (Y-spixel_height>=0.).float()
    in_bound_masks[2] = torch.logical_and( X+spixel_width <W, Y-spixel_height >=0. ).float()

    in_bound_masks[3] = ( X-spixel_width >=0. ).float()
    in_bound_masks[5] = ( X+spixel_width <W ).float()

    in_bound_masks[6] = torch.logical_and( X-spixel_width >=0., Y+spixel_height <H ).float()
    in_bound_masks[7] = (Y+spixel_height<H ).float()
    in_bound_masks[8] = torch.logical_and( X+spixel_width <W,   Y+spixel_height <H ).float()

    return in_bound_masks

def inpaint_depth(depth):
    '''
    depth- NCHW tensor
    '''
    mask = (depth.squeeze().cpu().numpy() <.001 ).astype(np.uint8)
    dst = cv2.inpaint(depth.squeeze().cpu().numpy(), mask.astype(np.uint8), 3 , cv2.INPAINT_NS)
    return torch.from_numpy(dst).expand_as(depth).to(depth)


def split_frame_list(frame_list, t_win_r):
    r'''
    split the frame_list into two : ref_frame (an array) and src_frames (a list),
    where ref_frame = frame_list[t_win_r]; src_frames = [0:t_win_r, t_win_r :]
    '''
    nframes = len(frame_list)
    ref_frame = frame_list[t_win_r]
    src_frames = [ frame_list[idx] for idx in range( nframes) if idx != t_win_r ]
    return ref_frame, src_frames

def get_entries_list_dict(list_dict, keyname):
    r'''
    Given the list of dicts, and the keyname
    return the list [list_dict[0][keyname] ,... ]
    '''
    return [_dict[keyname] for _dict in list_dict ]

def outer_prod(A, B):
    '''
    get the outerproduct between the vectors stored in A and B
    A - batch x dim
    B - batch x dim

    output: AB_outer: batch x dim x dim 
    AB_outer[ibatch] is the outer product of A[ibatch, :] and B[ibatch, :]
    '''
    assert A.shape[0] == B.shape[0], 'operants should have the same batch size!'
    AB_outer = torch.einsum('bi,bj->bij', (A,B) )
    return AB_outer

def outer_prod_batch(A, B):
    '''
    get the outerproduct between the vectors stored in A and B
    A - b x batch x dim
    B - b x batch x dim

    output: AB_outer: batch x dim x dim 
    AB_outer[ibatch] is the outer product of A[ibatch, :] and B[ibatch, :]
    '''
    assert A.shape[:2] == B.shape[:2], 'operants should have the same batch size!'
    AB_outer = torch.einsum('...bi,...bj->...bij', (A,B) )
    return AB_outer

def msavefig(tensor, fname, vmin=None, vmax=None, scale=False, cmap=None, text=None):
    tensor_ = tensor.detach().squeeze().cpu()
    assert tensor_.shape[0]>2

    import matplotlib.pyplot as plt
    mimshow(tensor, vmin=vmin, vmax=vmax, scale=scale, cmap=cmap) 

    if text is not None: 
        plt.text(10, 10, text, {'color': 'red'})

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(fname, bbox_inches='tight')
    plt.close() 
    return 
def mimshow(tensor, vmin=None, vmax=None, title=None, scale=False, cmap=None):
    if type(tensor) is torch.Tensor:
        tensor_ = tensor.detach().squeeze().cpu()
    else:
        tensor_ = torch.FloatTensor(tensor)

    assert tensor_.shape[0]>2
    if tensor_.shape[0]==3:
        tensor_ = tensor_.permute(1, 2, 0)
    if scale: # scale to [0, 1]
        tensor_ = (tensor_ - tensor_.min())  / (tensor_.max() - tensor_.min())

    import matplotlib.pyplot as plt

    if cmap is not None:
        fig = plt.imshow(tensor_.numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        fig = plt.imshow(tensor_.numpy(), vmin=vmin, vmax=vmax, cmap='gray')

    plt.title(title)
    return fig
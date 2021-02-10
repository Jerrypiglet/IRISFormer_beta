import numpy as np
import torch
from torch.autograd import Variable
# import models
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import time
import torchvision.utils as vutils
from utils.loss import hinge_embedding_loss, surface_normal_loss, parameter_loss, \
    class_balanced_cross_entropy_loss
from utils.match_segmentation import MatchSegmentation
from utils.utils_vis import vis_index_map, reindex_output_map, colorize
from utils.utils_training import reduce_loss_dict, time_meters_to_string
from utils.utils_misc import *

def get_input_dict_semseg(data_batch, opt):
    input_dict = {}

    input_dict['im_paths'] = data_batch['imPath']
    # if opt.if_hdr_input_mat_seg:
    #     im_cpu = data_batch['im']
    # else:
    #     im_cpu = data_batch['image_transformed']
    input_dict['im_batch_semseg_fixed'] = data_batch['image_transformed_fixed'].cuda(non_blocking=True)
    input_dict['im_batch_semseg'] = data_batch['im_semseg_transformed_trainval'].cuda(non_blocking=True)
    input_dict['semseg_label'] = data_batch['semseg_label'].cuda(non_blocking=True)

    input_dict['im_RGB_uint8'] = data_batch['im_RGB_uint8'].cuda(non_blocking=True)

    return input_dict

def postprocess_semseg(input_dict, output_dict, loss_dict, opt, time_meters):
    loss_dict['loss_semseg-main'] = output_dict['PSPNet_main_loss']
    loss_dict['loss_semseg-aux'] = output_dict['PSPNet_aux_loss']
    loss_dict['loss_semseg-ALL'] = loss_dict['loss_semseg-main'] + loss_dict['loss_semseg-aux'] * opt.semseg_configs.aux_weight


    return output_dict, loss_dict
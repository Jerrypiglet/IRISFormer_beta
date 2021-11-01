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
from utils.utils_vis import vis_index_map, reindex_output_map, vis_disp_colormap, colorize
from utils.utils_training import reduce_loss_dict, time_meters_to_string
from utils.utils_misc import *
from utils.utils_semseg import intersectionAndUnionGPU
import torchvision.utils as vutils
import torch.distributed as dist
import cv2
from PIL import Image

from train_funcs_matseg import get_labels_dict_matseg, postprocess_matseg, val_epoch_matseg
from train_funcs_semseg import get_labels_dict_semseg, postprocess_semseg
from train_funcs_brdf import get_labels_dict_brdf, postprocess_brdf
from train_funcs_light import get_labels_dict_light, postprocess_light
from train_funcs_layout_object_emitter import get_labels_dict_layout_emitter, postprocess_layout_object_emitter
from train_funcs_matcls import get_labels_dict_matcls, postprocess_matcls
from train_funcs_detectron import postprocess_detectron, gather_lists
# from utils.comm import synchronize
from train_funcs_iiw import get_labels_dict_iiw, postprocess_iiw

from utils.utils_metrics import compute_errors_depth_nyu
from train_funcs_matcls import getG1IdDict, getRescaledMatFromID
# from pytorch_lightning.metrics import Precision, Recall, F1, Accuracy
from pytorch_lightning.metrics import Accuracy

from icecream import ic
import pickle
import matplotlib.pyplot as plt

from train_funcs_layout_object_emitter import vis_layout_emitter

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader,DatasetCatalog, MetadataCatalog
from utils.utils_dettectron import py_cpu_nms
# from detectron2.utils.visualizer import Visualizer, ColorMode

from contextlib import ExitStack, contextmanager
from skimage.segmentation import mark_boundaries
from skimage.transform import resize as scikit_resize

def get_time_meters_joint_iiw():
    time_meters = {}
    time_meters['ts'] = 0.
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss_iiw'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    
    return time_meters

def get_labels_dict_joint_iiw(data_batch, opt):

    # prepare input_dict from data_batch (from dataloader)
    labels_dict = {'im_fixedscale_SDR': data_batch['im_fixedscale_SDR'].cuda(non_blocking=True), 'batch_idx': data_batch['image_index']}
    if 'im_fixedscale_SDR_next' in data_batch:
        labels_dict['im_fixedscale_SDR_next'] = data_batch['im_fixedscale_SDR_next'].cuda(non_blocking=True)

    input_batch_iiw, labels_dict_iiw = get_labels_dict_iiw(data_batch, opt, return_input_batch_as_list=True)
    # list_from_iiw = [input_batch_iiw, labels_dict_iiw, pre_batch_dict_iiw]
    labels_dict.update({'input_batch_brdf': torch.cat(input_batch_iiw, dim=1)})
    labels_dict.update(labels_dict_iiw)


    return labels_dict

def forward_joint_iiw(is_train, labels_dict, model, opt, time_meters, if_vis=False, if_loss=True, tid=-1, loss_dict=None):
    # forward model + compute losses

    # Forward model
    output_dict = model(labels_dict, if_has_gt_BRDF=False)
    time_meters['forward'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    # Post-processing and computing losses
    if loss_dict is None:
        loss_dict = {}

    output_dict, loss_dict = postprocess_iiw(labels_dict, output_dict, loss_dict, opt, time_meters, tid=tid, if_loss=True)
    time_meters['loss_iiw'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    return output_dict, loss_dict

def val_epoch_joint_iiw(iiw_loader_val, model, params_mis):
    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']

    logger.info(red('=== [IIW] Evaluating for %d batches'%len(iiw_loader_val)))

    model.eval()
    
    loss_keys = [
        'loss_iiw-eq', 
        'loss_iiw-darker', 
        'loss_iiw-ALL', 
    ]
        
    loss_meters = {loss_key: AverageMeter() for loss_key in loss_keys}
    time_meters = get_time_meters_joint_iiw()

    with torch.no_grad():

        iiw_dataset_val = params_mis['iiw_dataset_val']
        count_samples_this_rank = 0

        for batch_id, data_batch in enumerate(iiw_loader_val):

            ts_iter_start = time.time()

            input_dict = get_labels_dict_joint_iiw(data_batch, opt)

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = forward_joint_iiw(False, input_dict, model, opt, time_meters)

            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            
            # ======= update loss
            if len(loss_dict_reduced.keys()) != 0:
                for loss_key in loss_dict_reduced:
                    loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

    if opt.is_master:
        for loss_key in loss_dict_reduced:
            writer.add_scalar('IIW_loss_val/%s'%loss_key, loss_meters[loss_key].avg, tid)
            logger.info('[IIW] Logged val loss for %s'%loss_key)

    logger.info(red('[IIW] Evaluation timings: ' + time_meters_to_string(time_meters)))


def vis_val_epoch_joint_iiw(iiw_loader_val, model, params_mis):

    writer, logger, opt, tid, batch_size = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['batch_size_val_vis']
    logger.info(red('=== [IIW] [vis_val_epoch_joint] Visualizing for %d batches on rank %d'%(len(iiw_loader_val), opt.rank)))

    model.eval()
    opt.if_vis_debug_pac = True

    time_meters = get_time_meters_joint_iiw()

    im_paths_list = []
    albedoBatch_list = []
    normalBatch_list = []
    roughBatch_list = []
    depthBatch_list = []
    imBatch_list = []

    diffusePreBatch_list = []
    specularPreBatch_list = []
    renderedImBatch_list = []
    
    albedoPreds_list = []
    normalPreds_list = []
    roughPreds_list = []
    depthPreds_list = []

    # ===== Gather vis of N batches
    with torch.no_grad():
        im_single_list = []
        for batch_id, data_batch in enumerate(iiw_loader_val):
            if batch_size*batch_id >= opt.cfg.TEST.vis_max_samples:
                break

            input_dict = get_labels_dict_joint_iiw(data_batch, opt)

            # ======= Forward
            output_dict, _ = forward_joint_iiw(False, input_dict, model, opt, time_meters, if_vis=True)
            
            # ======= Vis imagges

            for sample_idx_batch, (im_single, im_path) in enumerate(zip(data_batch['im_fixedscale_SDR'], data_batch['image_path'])):
                sample_idx = sample_idx_batch+batch_size*batch_id
                print('[Image] Visualizing %d sample...'%sample_idx, batch_id, sample_idx_batch)
                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                    break

                im_single = im_single.numpy().squeeze()
                im_single_list.append(im_single)

                if opt.is_master:
                    writer.add_image('IIW_VAL_im/%d'%(sample_idx), im_single, tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((im_single*255.).astype(np.uint8)).save('{0}/IIW_{1}_im_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx) )

                    writer.add_text('IIW_VAL_image_name/%d'%(sample_idx), im_path, tid)
                    assert sample_idx == data_batch['image_index'][sample_idx_batch]

            
            # ===== Vis BRDF 1/2
            if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
                im_paths_list.append(input_dict['im_paths'])

                imBatch_list.append(input_dict['imBatch'])

                if opt.cascadeLevel > 0:
                    diffusePreBatch_list.append(input_dict['pre_batch_dict_brdf']['diffusePreBatch'])
                    specularPreBatch_list.append(input_dict['pre_batch_dict_brdf']['specularPreBatch'])
                    renderedImBatch_list.append(input_dict['pre_batch_dict_brdf']['renderedImBatch'])
                n = 0

                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    albedoPreds_list.append(output_dict['albedoPreds'][n])

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normalPreds_list.append(output_dict['normalPreds'][n])

                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    roughPreds_list.append(output_dict['roughPreds'][n])

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depthPreds_list.append(output_dict['depthPreds'][n])


    # ===== Vis BRDF 2/2
    # ===== logging top N to TB
    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        im_paths_list = flatten_list(im_paths_list)

        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedoPreds_vis = torch.cat(albedoPreds_list)
            
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normalPreds_vis = torch.cat(normalPreds_list)

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            roughPreds_vis = torch.cat(roughPreds_list)

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthPreds_vis = torch.cat(depthPreds_list)

        imBatch_vis = torch.cat(imBatch_list)

        if opt.cascadeLevel > 0:
            diffusePreBatch_vis = torch.cat(diffusePreBatch_list)
            specularPreBatch_vis = torch.cat(specularPreBatch_list)
            renderedImBatch_vis = torch.cat(renderedImBatch_list)

        print('Saving vis to ', '{0}'.format(opt.summary_vis_path_task, tid))
        im_batch_vis_sdr = ( (imBatch_vis)**(1.0/2.2) ).data

        # ==== Preds
        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr = ( (albedoPreds_vis ) ** (1.0/2.2) ).data
            if opt.is_master:
                vutils.save_image(albedo_pred_batch_vis_sdr,
                        '{0}/IIW_{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr = ( 0.5*(normalPreds_vis + 1) ).data
            if opt.is_master:
                vutils.save_image(normal_pred_batch_vis_sdr,
                        '{0}/IIW_{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr = ( 0.5*(roughPreds_vis + 1) ).data
            if opt.is_master:
                vutils.save_image(rough_pred_batch_vis_sdr,
                        '{0}/IIW_{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)
            depth_pred_batch_vis_sdr = depthOut.data
            if opt.is_master:
                vutils.save_image(depth_pred_batch_vis_sdr,
                        '{0}/IIW_{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr_numpy = albedo_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr_numpy = normal_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr_numpy = rough_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depth_pred_batch_vis_sdr_numpy = depth_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

        if opt.is_master:
            for sample_idx in tqdm(range(im_batch_vis_sdr.shape[0])):
                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('IIW_VAL_brdf-albedo_PRED/%d'%sample_idx, albedo_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((albedo_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/IIW_{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('IIW_VAL_brdf-normal_PRED/%d'%sample_idx, normal_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/IIW_{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('IIW_VAL_brdf-rough_PRED/%d'%sample_idx, rough_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((rough_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8).squeeze()).save('{0}/IIW_{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_not_normalized_pred = vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True)[0]
                    writer.add_image('IIW_VAL_brdf-depth_PRED/%d'%sample_idx, depth_not_normalized_pred, tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((depth_not_normalized_pred).astype(np.uint8)).save('{0}/IIW_{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))

    logger.info(red('Evaluation VIS timings: ' + time_meters_to_string(time_meters)))
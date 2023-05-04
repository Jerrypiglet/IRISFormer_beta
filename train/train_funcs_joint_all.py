from math import e
from random import sample
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
from utils.comm import synchronize

from train_funcs_matseg import get_labels_dict_matseg, postprocess_matseg, val_epoch_matseg
from train_funcs_semseg import get_labels_dict_semseg, postprocess_semseg
from train_funcs_brdf import get_labels_dict_brdf, postprocess_brdf
from train_funcs_light import get_labels_dict_light, postprocess_light
from train_funcs_matcls import get_labels_dict_matcls, postprocess_matcls
# from utils.comm import synchronize

from utils.utils_metrics import compute_errors_depth_nyu
from train_funcs_matcls import getG1IdDict, getRescaledMatFromID
# from pytorch_lightning.metrics import Precision, Recall, F1, Accuracy
from torchmetrics import Accuracy

from icecream import ic
import pickle
import matplotlib.pyplot as plt

from contextlib import ExitStack, contextmanager
from skimage.segmentation import mark_boundaries
from skimage.transform import resize as scikit_resize

def get_time_meters_joint():
    time_meters = {}
    time_meters['ts'] = 0.
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss_brdf'] = AverageMeter()
    time_meters['loss_light'] = AverageMeter()
    time_meters['loss_matseg'] = AverageMeter()
    time_meters['loss_semseg'] = AverageMeter()
    time_meters['loss_matcls'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    
    return time_meters

def get_semseg_meters():
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    semseg_meters = {'intersection_meter': intersection_meter, 'union_meter': union_meter, 'target_meter': target_meter,}
    return semseg_meters

def get_brdf_meters(opt):
    brdf_meters = {}
    if 'no' in opt.cfg.MODEL_BRDF.enable_list:
        normal_mean_error_meter = AverageMeter('normal_mean_error_meter')
        normal_median_error_meter = AverageMeter('normal_median_error_meter')
        # inv_depth_mean_error_meter = AverageMeter('inv_depth_mean_error_meter')
        # inv_depth_median_error_meter = AverageMeter('inv_depth_median_error_meter')
        brdf_meters.update({'normal_mean_error_meter': normal_mean_error_meter, 'normal_median_error_meter': normal_median_error_meter})
    if 'de' in opt.cfg.MODEL_BRDF.enable_list:
        brdf_meters.update(get_depth_meters(opt))
    return brdf_meters

def get_depth_meters(opt):
    return {metric: AverageMeter(metric) for metric in opt.depth_metrics}

def get_light_meters(opt):
    # render_error_meter = AverageMeter('render_error_meter')
    # recon_error_meter = AverageMeter('recon_error_meter')
    # light_meters = {'render_error_meter': render_error_meter, 'recon_error_meter': recon_error_meter}
    light_meters = {}
    return light_meters

def get_matcls_meters(opt):
    matcls_meters = {'pred_labels_list': ListMeter(), 'gt_labels_list': ListMeter()}
    if opt.cfg.MODEL_MATCLS.num_classes_sup:
        matcls_meters.update({'pred_labels_sup_list': ListMeter(), 'gt_labels_sup_list': ListMeter()})
    return matcls_meters

def get_labels_dict_joint(data_batch, opt):

    # prepare input_dict from data_batch (from dataloader)
    labels_dict = {'im_trainval_SDR': data_batch['im_trainval_SDR'].cuda(non_blocking=True), 'im_fixedscale_SDR': data_batch['im_fixedscale_SDR'].cuda(non_blocking=True), 'batch_idx': data_batch['image_index']}
    if 'im_fixedscale_SDR_next' in data_batch:
        labels_dict['im_fixedscale_SDR_next'] = data_batch['im_fixedscale_SDR_next'].cuda(non_blocking=True)

    if opt.cfg.DATA.load_matseg_gt:
        labels_dict_matseg = get_labels_dict_matseg(data_batch, opt)
    else:
        labels_dict_matseg = {}
    labels_dict.update(labels_dict_matseg)

    if opt.cfg.DATA.load_semseg_gt:
        labels_dict_semseg = get_labels_dict_semseg(data_batch, opt)
    else:
        labels_dict_semseg = {}
    labels_dict.update(labels_dict_semseg)

    # if opt.cfg.DATA.load_brdf_gt:
    input_batch_brdf, labels_dict_brdf, pre_batch_dict_brdf = get_labels_dict_brdf(data_batch, opt, return_input_batch_as_list=True)
    list_from_brdf = [input_batch_brdf, labels_dict_brdf, pre_batch_dict_brdf]
    labels_dict.update({'input_batch_brdf': torch.cat(input_batch_brdf, dim=1), 'pre_batch_dict_brdf': pre_batch_dict_brdf})
    # else:
    #     labels_dict_brdf = {}    
    #     list_from_brdf = None
    labels_dict.update(labels_dict_brdf)

    if opt.cfg.DATA.load_light_gt:
        input_batch_light, labels_dict_light, pre_batch_dict_light, extra_dict_light = get_labels_dict_light(data_batch, opt, list_from_brdf=list_from_brdf, return_input_batch_as_list=True)
        labels_dict.update({'input_batch_light': torch.cat(input_batch_light, dim=1), 'pre_batch_dict_light': pre_batch_dict_light})
    else:
        labels_dict_light = {}
        extra_dict_light = {}
    labels_dict.update(labels_dict_light)
    labels_dict.update(extra_dict_light)

    if opt.cfg.DATA.load_layout_emitter_gt:
        labels_dict_layout_emitter = get_labels_dict_layout_emitter(labels_dict, data_batch, opt)
    else:
        labels_dict_layout_emitter = {}
    labels_dict.update(labels_dict_layout_emitter)

    if opt.cfg.DATA.load_matcls_gt:
        labels_dict_matcls = get_labels_dict_matcls(data_batch, opt)
    else:
        labels_dict_matcls = {}
    labels_dict.update(labels_dict_matcls)

    return labels_dict

def forward_joint(is_train, labels_dict, model, opt, time_meters, if_vis=False, if_loss=True, tid=-1, loss_dict=None):
    # forward model + compute losses

    # Forward model
    a = time.time()
    output_dict = model(labels_dict)
    # print(time.time()-a)
    time_meters['forward'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    # Post-processing and computing losses
    if loss_dict is None:
        loss_dict = {}

    if opt.cfg.MODEL_SEMSEG.enable:
        output_dict, loss_dict = postprocess_semseg(labels_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_semseg'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
    
    if opt.cfg.MODEL_MATSEG.enable:
        output_dict, loss_dict = postprocess_matseg(labels_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_matseg'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
    
    if opt.cfg.MODEL_BRDF.enable:
        if_loss_brdf = if_loss and opt.cfg.DATA.load_brdf_gt and (not opt.cfg.DATASET.if_no_gt_BRDF)
        output_dict, loss_dict = postprocess_brdf(labels_dict, output_dict, loss_dict, opt, time_meters, tid=tid, if_loss=if_loss_brdf)
        time_meters['loss_brdf'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_LIGHT.enable:
        output_dict, loss_dict = postprocess_light(labels_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_light'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_MATCLS.enable:
        output_dict, loss_dict = postprocess_matcls(labels_dict, output_dict, loss_dict, opt, time_meters, if_vis=if_vis)
        time_meters['loss_matcls'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
        # synchronize()

    return output_dict, loss_dict

def val_epoch_joint(brdf_loader_val, model, params_mis):
    writer, logger, opt, tid, bin_mean_shift = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['bin_mean_shift']
    ENABLE_SEMSEG = opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable
    ENABLE_MATSEG = opt.cfg.MODEL_MATSEG.enable
    ENABLE_BRDF = opt.cfg.MODEL_BRDF.enable and opt.cfg.DATA.load_brdf_gt
    ENABLE_LIGHT = opt.cfg.MODEL_LIGHT.enable
    ENABLE_MATCLS = opt.cfg.MODEL_MATCLS.enable

    logger.info(red('===Evaluating for %d batches'%len(brdf_loader_val)))

    model.eval()
    
    loss_keys = []

    if opt.cfg.MODEL_MATSEG.enable:
        loss_keys += [
            'loss_matseg-ALL', 
            'loss_matseg-pull', 
            'loss_matseg-push', 
            'loss_matseg-binary', 
        ]

    if opt.cfg.MODEL_SEMSEG.enable:
        loss_keys += [
            'loss_semseg-ALL', 
            'loss_semseg-main', 
            'loss_semseg-aux', 
        ]

    if opt.cfg.MODEL_BRDF.enable:
        if opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
            loss_keys += ['loss_brdf-ALL', ]
            if 'al' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-albedo', ]
                if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_albedo:
                    loss_keys += ['loss_brdf-albedo-reg']
            if 'no' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-normal', ]
            if 'ro' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-rough', 'loss_brdf-rough-paper', ]
            if 'de' in opt.cfg.DATA.data_read_list:
                loss_keys += ['loss_brdf-depth', 'loss_brdf-depth-paper']
                if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_depth:
                    loss_keys += ['loss_brdf-depth-reg']

            if opt.cfg.MODEL_BRDF.if_bilateral:
                loss_keys += [
                    'loss_brdf-albedo-bs', ]
                if not opt.cfg.MODEL_BRDF.if_bilateral_albedo_only:
                    loss_keys += [
                        # 'loss_brdf-normal-bs', 
                        'loss_brdf-rough-bs', 
                        'loss_brdf-depth-bs', 
                        'loss_brdf-rough-bs-paper', 
                        'loss_brdf-depth-bs-paper', 
                    ]


        if opt.cfg.MODEL_BRDF.enable_semseg_decoder:
            loss_keys += ['loss_semseg-ALL']

    if opt.cfg.MODEL_LIGHT.enable:
        loss_keys += [
            'loss_light-ALL', 
            'loss_light-reconstErr', 
            'loss_light-renderErr', 
        ]

    if opt.cfg.MODEL_MATCLS.enable:
        loss_keys += [
            'loss_matcls-ALL',
            'loss_matcls-cls', ]
        if opt.cfg.MODEL_MATCLS.if_est_sup:
            loss_keys += [
            'loss_matcls-supcls',]

    loss_meters = {loss_key: AverageMeter() for loss_key in loss_keys}
    time_meters = get_time_meters_joint()
    if ENABLE_SEMSEG:
        semseg_meters = get_semseg_meters()
    if ENABLE_BRDF:
        brdf_meters = get_brdf_meters(opt)
    if ENABLE_LIGHT:
        light_meters = get_light_meters(opt)
    if ENABLE_MATCLS:
        matcls_meters = get_matcls_meters(opt)

    with torch.no_grad():

        brdf_dataset_val = params_mis['brdf_dataset_val']
        count_samples_this_rank = 0

        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):

            if opt.cfg.DATASET.if_binary and opt.distributed:
                count_samples_this_rank += len(data_batch['frame_info'])
                count_samples_gathered = gather_lists([count_samples_this_rank], opt.num_gpus)
                # print('->', i, opt.rank)
                if opt.rank==0:
                    print('-', count_samples_gathered, '-', len(brdf_dataset_val.scene_key_frame_id_list_this_rank))
            
                if max(count_samples_gathered)>=len(brdf_dataset_val.scene_key_frame_id_list_this_rank):
                    break

            ts_iter_start = time.time()

            input_dict = get_labels_dict_joint(data_batch, opt)

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = forward_joint(False, input_dict, model, opt, time_meters)

            # print(loss_dict.keys())
            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            synchronize()
            time_meters['ts'] = time.time()
            # logger.info(green('Training timings: ' + time_meters_to_string(time_meters)))

            # loss = loss_dict['loss_all']
            
            # ======= update loss
            if len(loss_dict_reduced.keys()) != 0:
                for loss_key in loss_dict_reduced:
                    # if loss_dict_reduced[loss_key] != 0:
                    # print(loss_key, loss_meters.keys(), loss_dict_reduced.keys())
                    loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= Metering
            if ENABLE_LIGHT:
                pass

            if ENABLE_BRDF:
                frame_info_list = input_dict['frame_info']
                if opt.cfg.DEBUG.dump_BRDF_offline.enable:
                    if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                        # albedo_output = output_dict['albedoPreds'][0].detach().cpu().numpy()
                        if opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
                            albedo_output = output_dict['albedoPred'].detach().cpu().numpy()
                        else:
                            albedo_output = output_dict['albedoPred_aligned'].detach().cpu().numpy()
                    if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                        # rough_output = output_dict['roughPreds'][0].detach().cpu().numpy()
                        rough_output = output_dict['roughPred'].detach().cpu().numpy()
                    if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                        # normal_output = output_dict['normalPreds'][0].detach().cpu().numpy()
                        normal_output = output_dict['normalPred'].detach().cpu().numpy()
                    if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                        # depth_output = output_dict['depthPreds'][0].detach().cpu().numpy()
                        if opt.cfg.MODEL_BRDF.use_scale_aware_depth:
                            depth_output = output_dict['depthPred'].detach().cpu().numpy()
                        else:
                            depth_output = output_dict['depthPred_aligned'].detach().cpu().numpy()

                    for sample_idx_batch in range(len(frame_info_list)):
                        frame_info = frame_info_list[sample_idx_batch]
                        # print(frame_info)

                        meta_split, scene_name, frame_id = frame_info['meta_split'], frame_info['scene_name'], frame_info['frame_id']
                        scene_path_dump = Path(opt.cfg.DEBUG.dump_BRDF_offline.path_task) / meta_split / scene_name
                        scene_path_dump.mkdir(exist_ok=True)
                        Path(str(scene_path_dump).replace('DiffLight', '')).mkdir(exist_ok=True)
                        Path(str(scene_path_dump).replace('DiffLight', '').replace('DiffMat', '')).mkdir(exist_ok=True)

                        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                            # /data/ruizhu/openrooms_mini/mainDiffMat_xml1/scene0593_01/imbaseColor_12.png/
                            if opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
                                albedo_dump_path = scene_path_dump / ('imbaseColor_%d_dump.png'%frame_id)
                            else:
                                albedo_dump_path = scene_path_dump / ('imbaseColor_scale_invariant_%d_dump.png'%frame_id)
                            albedo_dump_path = Path(str(albedo_dump_path).replace('DiffLight', ''))
                            albedo_sdr = np.clip(albedo_output[sample_idx_batch].transpose(1, 2, 0) ** (1.0/2.2), 0., 1.)
                            Image.fromarray((albedo_sdr*255.).astype(np.uint8)).save(str(albedo_dump_path))
                            print(albedo_dump_path)


                            # print(frame_info['albedo_path'])
                        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                            # /data/ruizhu/openrooms_mini/mainDiffMat_xml1/scene0593_01/imroughness_12.png/
                            rough_dump_path = scene_path_dump / ('imroughness_%d_dump.png'%frame_id)
                            rough_dump_path = Path(str(rough_dump_path).replace('DiffLight', ''))
                            rough_save = np.clip(0.5*(rough_output[sample_idx_batch].squeeze()+1), 0., 1.)
                            Image.fromarray((rough_save*255.).astype(np.uint8)).save(str(rough_dump_path))
                            print(rough_dump_path)
                            print(frame_info['rough_path'])
                        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                            # /data/ruizhu/openrooms_mini/mainDiffMat_xml1/scene0593_01/imnormal_12.png/
                            normal_dump_path = scene_path_dump / ('imnormal_%d_dump.png'%frame_id)
                            normal_dump_path = Path(str(normal_dump_path).replace('DiffLight', '').replace('DiffMat', ''))
                            normal_save = np.clip(0.5*(normal_output[sample_idx_batch].transpose(1, 2, 0)+1), 0., 1.)
                            Image.fromarray((normal_save*255.).astype(np.uint8)).save(str(normal_dump_path))
                            print(normal_dump_path)
                            print(frame_info['normal_path'])
                        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                            # /data/ruizhu/openrooms_mini/mainDiffMat_xml1/scene0593_01/imbaseColor_12.png/
                            if opt.cfg.MODEL_BRDF.use_scale_aware_depth:
                                depth_dump_path = scene_path_dump / ('imdepth_%d_dump.pickle'%frame_id)
                            else:
                                depth_dump_path = scene_path_dump / ('imdepth_scale_invariant_%d_dump.pickle'%frame_id)
                            depth_dump_path = Path(str(depth_dump_path).replace('DiffLight', '').replace('DiffMat', ''))
                            depth_save = depth_output[sample_idx_batch].squeeze().astype(np.float32)
                            with open(str(depth_dump_path),"wb") as f:
                                pickle.dump({'depth_pred': depth_save}, f)
                            print(depth_dump_path)
                            print(frame_info['depth_path'])

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_input = input_dict['depthBatch'].detach().cpu().numpy()
                    depth_output = output_dict['depthPred'].detach().cpu().numpy()
                    seg_obj = data_batch['segObj'].cpu().numpy()
                    min_depth, max_depth = 0.1, 8.
                    # depth_mask = np.logical_and(np.logical_and(seg_obj != 0, depth_input < max_depth), depth_input > min_depth)
                    depth_output = depth_output * seg_obj
                    depth_input = depth_input * seg_obj
                    # np.save('depth_input_%d.npy'%(batch_id), depth_input)
                    # np.save('depth_output_%d.npy'%(batch_id), depth_output)
                    # np.save('seg_obj_%d.npy'%(batch_id), seg_obj)

                    depth_input[depth_input < min_depth] = min_depth
                    depth_output[depth_output < min_depth] = min_depth
                    depth_input[depth_input > max_depth] = max_depth
                    depth_output[depth_output > max_depth] = max_depth

                    for depth_input_single, depth_output_single in zip(depth_input.squeeze(), depth_output.squeeze()):
                        metrics_results = compute_errors_depth_nyu(depth_input_single, depth_output_single)
                        # print(metrics_results)
                        for metric in metrics_results:
                            brdf_meters[metric].update(metrics_results[metric])

                # depth_input = 1. / (depth_input + 1e-6)
                # depth_output = 1. / (depth_output + 1e-6)

                # normal_input = 0.5 * (normal_input + 1)
                # normal_output = 0.5 * (normal_output + 1)

                # inv_depth_error = np.abs(1./(depth_output+1e-6) - 1./(depth_input+1e-6))
                # brdf_meters['inv_depth_mean_error_meter'].update(np.mean(inv_depth_error))
                # brdf_meters['inv_depth_median_error_meter'].update(np.median(inv_depth_error))

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normal_input = input_dict['normalBatch'].detach().cpu().numpy()
                    normal_output = output_dict['normalPred'].detach().cpu().numpy()
                    # np.save('normal_input_%d.npy'%(batch_id), normal_input)
                    # np.save('normal_output_%d.npy'%(batch_id), normal_output)
                    normal_input_Nx3 = np.transpose(normal_input, (0, 2, 3, 1)).reshape(-1, 3)
                    normal_output_Nx3 = np.transpose(normal_output, (0, 2, 3, 1)).reshape(-1, 3)
                    normal_in_n_out_dot = np.sum(np.multiply(normal_input_Nx3, normal_output_Nx3), 1)
                    normal_error = normal_in_n_out_dot / (np.linalg.norm(normal_input_Nx3, axis=1) * np.linalg.norm(normal_output_Nx3, axis=1) + 1e-6)
                    normal_error = np.arccos(normal_error) / np.pi * 180.
                    # print(normal_error.shape, np.mean(normal_error), np.median(normal_error))
                    brdf_meters['normal_mean_error_meter'].update(np.mean(normal_error))
                    brdf_meters['normal_median_error_meter'].update(np.median(normal_error))

            if ENABLE_SEMSEG:
                output = output_dict['semseg_pred'].max(1)[1]
                target = input_dict['semseg_label']
                # print(output_dict['semseg_pred'].shape)
                # print(torch.max(target), torch.min(target))
                intersection, union, target = intersectionAndUnionGPU(output, target, opt.cfg.MODEL_SEMSEG.semseg_classes, opt.cfg.MODEL_SEMSEG.semseg_ignore_label)
                if opt.distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                semseg_meters['intersection_meter'].update(intersection), semseg_meters['union_meter'].update(union), semseg_meters['target_meter'].update(target)
                # accuracy = sum(semseg_meters['intersection_meter'].val) / (sum(semseg_meters['target_meter'].val) + 1e-10)

            if ENABLE_MATCLS:
                output = output_dict['matcls_argmax']
                target = input_dict['mat_label_batch']
                matcls_meters['pred_labels_list'].update(output.cpu().flatten())
                matcls_meters['gt_labels_list'].update(target.cpu().flatten())
                if opt.cfg.MODEL_MATCLS.if_est_sup:
                    output = output_dict['matcls_sup_argmax']
                    target = input_dict['mat_label_sup_batch']
                    matcls_meters['pred_labels_sup_list'].update(output.cpu().flatten())
                    matcls_meters['gt_labels_sup_list'].update(target.cpu().flatten())
                    
    synchronize()

    # ======= Metering
        
    if opt.is_master:
        for loss_key in loss_dict_reduced:
            writer.add_scalar('loss_val/%s'%loss_key, loss_meters[loss_key].avg, tid)
            logger.info('Logged val loss for %s:%.6f'%(loss_key, loss_meters[loss_key].avg))

                
        if ENABLE_SEMSEG:
            iou_class = semseg_meters['intersection_meter'].sum / (semseg_meters['union_meter'].sum + 1e-10)
            accuracy_class = semseg_meters['intersection_meter'].sum / (semseg_meters['target_meter'].sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(semseg_meters['intersection_meter'].sum) / (sum(semseg_meters['target_meter'].sum) + 1e-10)
            logger.info('[SEMSEG] Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(opt.cfg.MODEL_SEMSEG.semseg_classes):
                logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
            writer.add_scalar('VAL/SEMSEG-mIoU_val', mIoU, tid)
            writer.add_scalar('VAL/SEMSEG-mAcc_val', mAcc, tid)
            writer.add_scalar('VAL/SEMSEG-allAcc_val', allAcc, tid)

        if ENABLE_MATCLS:
            # iou_class = matcls_meters['intersection_meter'].sum / (matcls_meters['union_meter'].sum + 1e-10)
            # accuracy_class = matcls_meters['intersection_meter'].sum / (matcls_meters['target_meter'].sum + 1e-10)
            # mIoU = np.mean(iou_class)
            # mAcc = np.mean(accuracy_class)
            # allAcc = sum(matcls_meters['intersection_meter'].sum) / (sum(matcls_meters['target_meter'].sum) + 1e-10)
            # logger.info('[MATCLS] Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            # # for i in range(opt.cfg.MODEL_MATCLS.num_classes):
            # #     logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
            # writer.add_scalar('VAL/MATCLS-mIoU_val', mIoU, tid)
            # writer.add_scalar('VAL/MATCLS-mAcc_val', mAcc, tid)
            # writer.add_scalar('VAL/MATCLS-allAcc_val', allAcc, tid)
            pred_labels = matcls_meters['pred_labels_list'].concat().flatten()
            gt_labels = matcls_meters['gt_labels_list'].concat().flatten()
            # https://pytorch-lightning.readthedocs.io/en/0.8.5/metrics.html
            accuracy = Accuracy()(pred_labels, gt_labels)
            # prec = Precision(num_classes=opt.cfg.MODEL_MATCLS.num_classes)(pred_labels, gt_labels)
            # recall = Recall(num_classes=opt.cfg.MODEL_MATCLS.num_classes)(pred_labels, gt_labels)
            # f1 = F1(num_classes=opt.cfg.MODEL_MATCLS.num_classes)(pred_labels, gt_labels)
            # writer.add_scalar('VAL/MATCLS-precision_val', prec, tid)
            writer.add_scalar('VAL/MATCLS-accuracy_val', accuracy, tid)
            # writer.add_scalar('VAL/MATCLS-recall_val', recall, tid)
            # writer.add_scalar('VAL/MATCLS-F1_val', f1, tid)

            if opt.cfg.MODEL_MATCLS.if_est_sup:
                pred_labels = matcls_meters['pred_labels_sup_list'].concat().flatten()
                gt_labels = matcls_meters['gt_labels_sup_list'].concat().flatten()
                # https://pytorch-lightning.readthedocs.io/en/0.8.5/metrics.html
                accuracy = Accuracy()(pred_labels, gt_labels)
                # prec = Precision(ignore_index=0, num_classes=opt.cfg.MODEL_MATCLS.num_classes_sup+1)(pred_labels, gt_labels)
                # recall = Recall(ignore_index=0, num_classes=opt.cfg.MODEL_MATCLS.num_classes_sup+1)(pred_labels, gt_labels)
                # f1 = F1(num_classes=opt.cfg.MODEL_MATCLS.num_classes_sup+1)(pred_labels, gt_labels)
                # writer.add_scalar('VAL/MATCLS-sup_precision_val', prec, tid)
                writer.add_scalar('VAL/MATCLS-sup_accuracy_val', accuracy, tid)
                # writer.add_scalar('VAL/MATCLS-sup_recall_val', recall, tid)
                # writer.add_scalar('VAL/MATCLS-sup_F1_val', f1, tid)


        if ENABLE_BRDF:
            # writer.add_scalar('VAL/BRDF-inv_depth_mean_val', brdf_meters['inv_depth_mean_error_meter'].avg, tid)
            # writer.add_scalar('VAL/BRDF-inv_depth_median_val', brdf_meters['inv_depth_median_error_meter'].get_median(), tid)
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                for metric in opt.depth_metrics:
                    writer.add_scalar('VAL/BRDF-depth_%s'%metric, brdf_meters[metric].avg, tid)
                logger.info('Val result - depth: ' + ', '.join(['%s: %.4f'%(metric, brdf_meters[metric].avg) for metric in opt.depth_metrics]))
            if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                writer.add_scalar('VAL/BRDF-normal_mean_val', brdf_meters['normal_mean_error_meter'].avg, tid)
                writer.add_scalar('VAL/BRDF-normal_median_val', brdf_meters['normal_median_error_meter'].get_median(), tid)
                logger.info('Val result - normal: mean: %.4f, median: %.4f'%(brdf_meters['normal_mean_error_meter'].avg, brdf_meters['normal_median_error_meter'].get_median()))

    synchronize()
    logger.info(red('Evaluation timings: ' + time_meters_to_string(time_meters)))


def vis_val_epoch_joint(brdf_loader_val, model, params_mis):

    writer, logger, opt, tid, batch_size, bin_mean_shift = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['batch_size_val_vis'], params_mis['bin_mean_shift']
    logger.info(red('=== [vis_val_epoch_joint] Visualizing for %d batches on rank %d'%(len(brdf_loader_val), opt.rank)))

    model.eval()
    opt.if_vis_debug_pac = True

    if opt.cfg.DEBUG.if_test_real:
        if opt.cfg.MODEL_MATCLS.enable:
            matcls_results_path = opt.cfg.MODEL_MATCLS.real_images_list.replace('.txt', '-results.txt')
            f_matcls_results = open(matcls_results_path, 'w')

    time_meters = get_time_meters_joint()

    if opt.cfg.MODEL_MATSEG.enable:
        match_segmentatin = MatchSegmentation()

    im_paths_list = []
    albedoBatch_list = []
    normalBatch_list = []
    roughBatch_list = []
    depthBatch_list = []
    imBatch_list = []
    imBatch_vis_list = []
    segAllBatch_list = []
    segBRDFBatch_list = []
    
    im_w_resized_to_list = []
    im_h_resized_to_list = []

    if opt.cfg.MODEL_BRDF.enable:

        diffusePreBatch_list = []
        specularPreBatch_list = []
        renderedImBatch_list = []
        
        albedoPreds_list = []
        albedoPreds_aligned_list = []
        normalPreds_list = []
        roughPreds_list = []
        depthPreds_list = []
        depthPreds_aligned_list = []

        albedoBsPreds_list = []
        albedoBsPreds_aligned_list = []
        roughBsPreds_list = []
        depthBsPreds_list = []
        depthBsPreds_aligned_list = []


    if opt.cfg.MODEL_MATCLS.enable:
        matG1IdDict = getG1IdDict(opt.cfg.PATH.matcls_matIdG1_path)


    # ===== Gather vis of N batches

    with torch.no_grad():
        im_single_list = []
        real_sample_results_path_list = []
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):
            # for i, x in enumerate(data_batch['image_path']):
            #     ic(i, x)
            if batch_size*batch_id >= opt.cfg.TEST.vis_max_samples:
                break


            # print(batch_id, batch_size)

            # if num_val_brdf_vis >= num_val_vis_MAX or sample_idx_batch >= num_val_vis_MAX:
            #     break

            input_dict = get_labels_dict_joint(data_batch, opt)
            # if batch_id == 0:
            #     print(input_dict['im_paths'])

            # ======= Forward
            output_dict, _ = forward_joint(False, input_dict, model, opt, time_meters, if_vis=True, if_loss=False)

            # synchronize()
            
            # ======= Vis imagges
            colors = np.loadtxt(os.path.join(opt.pwdpath, opt.cfg.PATH.semseg_colors_path)).astype('uint8')
            if opt.cfg.DATA.load_semseg_gt:
                semseg_label = input_dict['semseg_label'].cpu().numpy()
            if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable:
                semseg_pred = output_dict['semseg_pred'].cpu().numpy()


            for sample_idx_batch, (im_single, im_path) in enumerate(zip(data_batch['im_fixedscale_SDR'], data_batch['image_path'])):
                sample_idx = sample_idx_batch+batch_size*batch_id
                print('[Image] Visualizing %d sample...'%sample_idx, batch_id, sample_idx_batch)
                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                    break
                
                im_h_resized_to, im_w_resized_to = data_batch['im_h_resized_to'], data_batch['im_w_resized_to']
                im_h_resized_to_list.append(im_h_resized_to)
                im_w_resized_to_list.append(im_w_resized_to)
                if opt.cfg.DEBUG.if_test_real:
                    real_sample_name = im_path.split('/')[-2]
                    real_sample_results_path = Path(opt.summary_vis_path_task) / real_sample_name
                    real_sample_results_path.mkdir(parents=True, exist_ok=False)
                    real_sample_results_path_list.append([real_sample_results_path, (im_h_resized_to, im_w_resized_to)])

                im_single = im_single.numpy().squeeze()
                im_single_list.append(im_single)
                # im_path = os.path.join('./tmp/', 'im_%d-%d_color.png'%(tid, im_index))
                # color_path = os.path.join('./tmp/', 'im_%d-%d_semseg.png'%(tid, im_index))
                # cv2.imwrite(im_path, im_single * 255.)
                # semseg_color.save(color_path)
                if opt.is_master:
                    writer.add_image('VAL_im/%d'%(sample_idx), im_single, tid, dataformats='HWC')
                    writer.add_image('VAL_pad_mask/%d'%(sample_idx), data_batch['pad_mask'][sample_idx_batch]*255, tid, dataformats='HW')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((im_single*255.).astype(np.uint8)).save('{0}/{1}_im_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx) )

                    writer.add_text('VAL_image_name/%d'%(sample_idx), im_path, tid)
                    assert sample_idx == data_batch['image_index'][sample_idx_batch]
                    # print(sample_idx, im_path)

                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_im_path = real_sample_results_path / 'im_.png'
                        print(im_h_resized_to, im_w_resized_to)
                        assert len(im_h_resized_to) == 1
                        im_h_resized_to, im_w_resized_to = im_h_resized_to[0], im_w_resized_to[0]
                        im_ = Image.fromarray((im_single*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        im_ = im_.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        im_.save(str(real_sample_im_path))
                        if opt.cfg.DEBUG.dump_BRDF_offline.enable:
                            real_sample_name = im_path.split('/')[-2]
                            scene_path_dump = Path(opt.cfg.DEBUG.dump_BRDF_offline.path_task) / real_sample_name
                            # print(opt.cfg.DEBUG.dump_BRDF_offline.path_root_local, Path(opt.cfg.DEBUG.dump_BRDF_offline.path_task), real_sample_name, scene_path_dump)
                            scene_path_dump.mkdir(parents=True, exist_ok=True)
                            # scene_path_dump_list.append(scene_path_dump)
                            # im_h_resized_to, im_w_resized_to = data_batch['im_h_resized_to'], data_batch['im_w_resized_to']
                            # scene_path_dump_list.append([scene_path_dump, (im_h_resized_to, im_w_resized_to)])

            # ======= Vis BRDFsemseg / semseg
            if opt.cfg.DATA.load_semseg_gt:
                for sample_idx_batch in range(semseg_label.shape[0]):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    gray_GT = np.uint8(semseg_label[sample_idx_batch])
                    color_GT = np.array(colorize(gray_GT, colors).convert('RGB'))
                    # print(gray_GT.shape, color_GT.shape) # (241, 321) (241, 321, 3)
                    if opt.is_master:
                        writer.add_image('VAL_semseg_GT/%d'%(sample_idx), color_GT, tid, dataformats='HWC')

            if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable:
                for sample_idx_batch in range(semseg_pred.shape[0]):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    prediction = np.argmax(semseg_pred[sample_idx_batch], 0)
                    gray_pred = np.uint8(prediction)
                    color_pred = np.array(colorize(gray_pred, colors).convert('RGB'))
                    if opt.is_master:
                        writer.add_image('VAL_semseg_PRED/%d'%(sample_idx), color_pred, tid, dataformats='HWC')
                        im_single = data_batch['im_fixedscale_SDR'][sample_idx_batch].detach().cpu().numpy().astype(np.float32)
                        im_single = cv2.resize(im_single, (color_pred.shape[1], color_pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                        color_pred = color_pred.astype(np.float32) / 255.
                        semseg_pred_overlay = im_single * color_pred + im_single * 0.2 * (1. - color_pred)
                        writer.add_image('VAL_semseg_PRED-overlay/%d'%(sample_idx), semseg_pred_overlay, tid, dataformats='HWC')

                        pickle_save_path = Path(opt.summary_vis_path_task) / ('results_semseg_tid%d-%d.pickle'%(tid, sample_idx))
                        save_dict = {'semseg_pred': semseg_pred[sample_idx_batch], }
                        if opt.if_save_pickles:
                            with open(str(pickle_save_path),"wb") as f:
                                pickle.dump(save_dict, f)

            # ======= Vis matcls
            if opt.cfg.MODEL_MATCLS.enable:
                mats_pred_vis_list, prop_list_pred = getRescaledMatFromID(
                    output_dict['matcls_argmax'].cpu().numpy(), np.ones((output_dict['matcls_argmax'].shape[0], 4), dtype=np.float32), opt.cfg.DATASET.matori_path, matG1IdDict, res=256)
                mats_gt_vis_list, prop_list_gt = getRescaledMatFromID(
                    input_dict['mat_label_batch'].cpu().numpy(), np.ones((input_dict['mat_label_batch'].shape[0], 4), dtype=np.float32), opt.cfg.DATASET.matori_path, matG1IdDict, res=256)
                mat_label_batch = input_dict['mat_label_batch'].cpu().numpy()
                mat_pred_batch = output_dict['matcls_argmax'].cpu().numpy()
                mat_sup_pred_batch = output_dict['matcls_sup_argmax'].cpu().numpy()
                for sample_idx_batch, (mats_pred_vis, mats_gt_vis, mat_mask, mat_label, mat_pred, mat_sup_pred) in enumerate(zip(mats_pred_vis_list, mats_gt_vis_list, input_dict['mat_mask_batch'], mat_label_batch, mat_pred_batch, mat_sup_pred_batch)): # torch.Size([3, 768, 256])
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    # print(mats_pred_vis.shape) # torch.Size([3, 256, 768])
                    mat_label = mat_label.item()
                    mat_pred = mat_pred.item()
                    mat_sup_pred = mat_sup_pred.item()
                    im_path = data_batch['image_path'][sample_idx_batch]

                    if not opt.cfg.DATASET.if_no_gt_BRDF:
                        summary_cat = torch.cat([mats_pred_vis, mats_gt_vis], 1).permute(1, 2, 0)
                    else:
                        summary_cat = mats_pred_vis.permute(1, 2, 0) # not showing GT if GT label is 0
                    
                    if opt.is_master:
                        writer.add_image('VAL_matcls_PRED-GT/%d'%(sample_idx), summary_cat, tid, dataformats='HWC')
                        writer.add_image('VAL_matcls_matmask/%d'%(sample_idx), mat_mask.squeeze(), tid, dataformats='HW')
                        im_single = data_batch['im_fixedscale_SDR'][sample_idx_batch].detach().cpu()
                        mat_mask = mat_mask.permute(1, 2, 0).cpu().float()
                        matmask_overlay = im_single * mat_mask + im_single * 0.2 * (1. - mat_mask)
                        writer.add_image('VAL_matcls_matmask-overlay/%d'%(sample_idx), matmask_overlay, tid, dataformats='HWC')
                        if opt.cfg.DEBUG.if_test_real and opt.cfg.MODEL_MATCLS.enable:
                            f_matcls_results.write(' '.join([im_path, opt.matG1Dict[mat_pred+1], opt.valid_sup_classes_dict[mat_sup_pred]]))
                            f_matcls_results.write('\n')
                    

            # ======= Vis clusters for mat-seg
            if opt.cfg.DATA.load_matseg_gt:
                for sample_idx_batch in range(input_dict['mat_aggre_map_cpu'].shape[0]):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    if sample_idx >= opt.cfg.TEST.vis_max_samples:
                        break

                    mat_aggre_map_GT_single = input_dict['mat_aggre_map_cpu'][sample_idx_batch].numpy().squeeze()
                    matAggreMap_GT_single_vis = vis_index_map(mat_aggre_map_GT_single)
                    mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx_batch].numpy().squeeze()
                    if opt.is_master:
                        writer.add_image('VAL_matseg-aggre_map_GT/%d'%(sample_idx), matAggreMap_GT_single_vis, tid, dataformats='HWC')
                        writer.add_image('VAL_matseg-notlight_mask_GT/%d'%(sample_idx), mat_notlight_mask_single, tid, dataformats='HW')

            if opt.cfg.MODEL_MATSEG.enable and opt.cfg.MODEL_MATSEG.embed_dims <= 4:
                b, c, h, w = output_dict['logit'].size()
                for sample_idx_batch, (logit_single, embedding_single) in enumerate(zip(output_dict['logit'].detach(), output_dict['embedding'].detach())):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    if sample_idx >= opt.cfg.TEST.vis_max_samples:
                        break

                    # if sample_idx_batch >= num_val_vis_MAX:
                    #     break
                    ts_start_vis = time.time()

                    
                    # prob_single = torch.sigmoid(logit_single)
                    prob_single = input_dict['mat_notlight_mask_cpu'][sample_idx_batch].to(opt.device).float()
                    # fast mean shift

                    if opt.bin_mean_shift_device == 'cpu':
                        prob_single, logit_single, embedding_single = prob_single.cpu(), logit_single.cpu(), embedding_single.cpu()
                    segmentation, sampled_segmentation = bin_mean_shift.test_forward(
                        prob_single, embedding_single, mask_threshold=0.1)
                    
                    # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
                    # we thus use avg_pool_2d to smooth the segmentation results
                    b = segmentation.t().view(1, -1, h, w)
                    pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
                    b = pooling_b.view(-1, h*w).t()
                    segmentation = b

                    # greedy match of predict segmentation and ground truth segmentation using cross entropy
                    # to better visualization
                    gt_plane_num = input_dict['num_mat_masks_batch'][sample_idx_batch]
                    matching = match_segmentatin(segmentation, prob_single.view(-1, 1), input_dict['instance'][sample_idx_batch], gt_plane_num)

                    # return cluster results
                    predict_segmentation = segmentation.cpu().numpy().argmax(axis=1) # reshape to [h, w]: [0, 1, ..., len(matching)-1]; use this for mask pooling!!!!

                    # reindexing to matching gt segmentation for better visualization
                    matching = matching.cpu().numpy().reshape(-1)
                    used = set([])
                    max_index = max(matching) + 1
                    for i, a in zip(range(len(matching)), matching):
                        if a in used:
                            matching[i] = max_index
                            max_index += 1
                        else:
                            used.add(a)
                    # np.save('tmp/predict_segmentation_ori_%d.npy'%(max_index), predict_segmentation)
                    # np.save('tmp/instance_%d.npy'%(max_index), input_dict['instance'][sample_idx_batch].cpu().numpy().squeeze())
                    # np.save('tmp/gt_%d.npy'%(max_index), input_dict['mat_aggre_map_cpu'][sample_idx_batch].numpy().squeeze())
                    # np.save('tmp/matching_%d.npy'%(max_index), matching)
                    predict_segmentation = matching[predict_segmentation] # matching GT: [0, 1, ... N-1]
                    # np.save('tmp/predict_segmentation_matched%d.npy'%(max_index), predict_segmentation)

                    # mask out non planar region
                    predict_segmentation = predict_segmentation.reshape(h, w) # [0..N-1]
                    predict_segmentation += 1 # 0 for invalid region
                    predict_segmentation[prob_single.cpu().squeeze().numpy() <= 0.1] = opt.invalid_index

                    # ===== vis
                    # im_single = data_batch['im_not_hdr'][sample_idx_batch].numpy().squeeze().transpose(1, 2, 0)

                    # mat_aggre_map_pred_single = reindex_output_map(predict_segmentation.squeeze(), opt.invalid_index)
                    mat_aggre_map_pred_single = predict_segmentation.squeeze()
                    matAggreMap_pred_single_vis = vis_index_map(mat_aggre_map_pred_single)

                    if opt.is_master:
                        writer.add_image('VAL_matseg-aggre_map_PRED/%d'%(sample_idx), matAggreMap_pred_single_vis, tid, dataformats='HWC')
                        if opt.cfg.MODEL_MATSEG.if_save_embedding:
                            npy_path = Path(opt.summary_vis_path_task) / ('matseg_pred_embedding_tid%d_idx%d.npy'%(tid, sample_idx))
                            np.save(str(npy_path), embedding_single.detach().cpu().numpy())

                    logger.info('Vis batch for %.2f seconds.'%(time.time()-ts_start_vis))
            
            # ===== Vis BRDF 1/2
            # print(((input_dict['albedoBatch'] ) ** (1.0/2.2) ).data.shape) # [b, 3, h, w]
            im_paths_list.append(input_dict['im_paths'])
            imBatch_list.append(input_dict['imBatch'])
            imBatch_vis_list.append(data_batch['im_fixedscale_SDR'])
            segAllBatch_list.append(input_dict['segAllBatch'])
            segBRDFBatch_list.append(input_dict['segBRDFBatch'])
            if 'al' in opt.cfg.DATA.data_read_list:
                albedoBatch_list.append(input_dict['albedoBatch'])
            if 'no' in opt.cfg.DATA.data_read_list:
                normalBatch_list.append(input_dict['normalBatch'])
            if 'ro' in opt.cfg.DATA.data_read_list:
                roughBatch_list.append(input_dict['roughBatch'])
            if 'de' in opt.cfg.DATA.data_read_list:
                depthBatch_list.append(input_dict['depthBatch'])

            if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
                if opt.cascadeLevel > 0:
                    diffusePreBatch_list.append(input_dict['pre_batch_dict_brdf']['diffusePreBatch'])
                    specularPreBatch_list.append(input_dict['pre_batch_dict_brdf']['specularPreBatch'])
                    renderedImBatch_list.append(input_dict['pre_batch_dict_brdf']['renderedImBatch'])
                n = 0

                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    # if (not opt.cfg.DATASET.if_no_gt_semantics):
                    # albedoPreds_list.append(output_dict['albedoPreds'][n])
                    albedoPreds_list.append(output_dict['albedoPred'])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        albedoBsPreds_list.append(output_dict['albedoBsPred'])
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                        albedoPreds_aligned_list.append(output_dict['albedoPred_aligned'])
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        # assert output_dict['albedoPreds'][n].shape[0] == 1
                        assert output_dict['albedoPred'].shape[0] == 1
                        # albedoPred_np = output_dict['albedoPreds'][n][0].cpu().numpy()
                        albedoPred_np = output_dict['albedoPred'][0].cpu().numpy()
                        if opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
                            albedo_dump_path = scene_path_dump / ('imbaseColor.png')
                        else:
                            albedo_dump_path = scene_path_dump / ('imbaseColor_scaleInv.png')
                        albedo_sdr = np.clip(albedoPred_np.transpose(1, 2, 0) ** (1.0/2.2), 0., 1.)
                        Image.fromarray((albedo_sdr*255.).astype(np.uint8)).save(str(albedo_dump_path))
                        Image.fromarray((albedo_sdr*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to, :]).save(str(albedo_dump_path).replace('.png','_cropped.png'))
                        print(albedo_dump_path)
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            albedoPredBs_np = output_dict['albedoBsPred'][0].cpu().numpy()
                            albedoBs_dump_path = scene_path_dump / ('imbaseColorBS.png')
                            albedoBs_sdr = np.clip(albedoPredBs_np.transpose(1, 2, 0) ** (1.0/2.2), 0., 1.)
                            Image.fromarray((albedoBs_sdr*255.).astype(np.uint8)).save(str(albedoBs_dump_path))
                            Image.fromarray((albedoBs_sdr*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to, :]).save(str(albedoBs_dump_path).replace('.png','_cropped.png'))

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normalPreds_list.append(output_dict['normalPred'])
                    # print(input_dict['imBatch'][0, 1, :2, :3])
                    # print(output_dict['normalPred'][0, 1, :2, :3])
                    # print(model.MODEL_ALL._.no.scratch.refinenet4.resConfUnit2.conv2.bias)
                    
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        assert output_dict['normalPred'].shape[0] == 1
                        normal_dump_path = scene_path_dump / ('imnormal.png')
                        normalPred_np = output_dict['normalPred'][0].cpu().numpy()
                        normal_save = np.clip(0.5*(normalPred_np.transpose(1, 2, 0)+1), 0., 1.)
                        Image.fromarray((normal_save*255.).astype(np.uint8)).save(str(normal_dump_path))
                        Image.fromarray((normal_save*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to]).save(str(normal_dump_path).replace('.png','_cropped.png'))
                        print(normal_dump_path)

                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    roughPreds_list.append(output_dict['roughPred'])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        roughBsPreds_list.append(output_dict['roughBsPred'])
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        assert output_dict['roughPred'].shape[0] == 1
                        rough_dump_path = scene_path_dump / ('imroughness.png')
                        roughPred_np = output_dict['roughPred'][0].cpu().numpy()
                        rough_save = np.clip(0.5*(roughPred_np.squeeze()+1), 0., 1.)
                        Image.fromarray((rough_save*255.).astype(np.uint8)).save(str(rough_dump_path))
                        Image.fromarray((rough_save*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to]).save(str(rough_dump_path).replace('.png','_cropped.png'))
                        print(rough_dump_path)
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            roughBsPred_np = output_dict['roughBsPred'][0].cpu().numpy()
                            roughBs_dump_path = scene_path_dump / ('imbaseColorBS.png')
                            roughBs_save = np.clip(0.5*(roughBsPred_np.squeeze()+1), 0., 1.)
                            Image.fromarray((roughBs_save*255.).astype(np.uint8)).save(str(roughBs_dump_path))
                            Image.fromarray((roughBs_save*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to, :]).save(str(roughBs_dump_path).replace('.png','_cropped.png'))

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depthPreds_list.append(output_dict['depthPred'])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        depthBsPreds_list.append(output_dict['depthBsPred'])
                    if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                        depthPreds_aligned_list.append(output_dict['depthPred_aligned'])
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            depthBsPreds_aligned_list.append(output_dict['depthBsPred_aligned'])
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        assert output_dict['depthPred'].shape[0] == 1
                        if opt.cfg.MODEL_BRDF.use_scale_aware_depth:
                            depth_dump_path = scene_path_dump / ('imdepth.pickle')
                        else:
                            depth_dump_path = scene_path_dump / ('imdepth_scale_invariant.pickle')
                        depthPred_np = output_dict['depthPred'][0].cpu().numpy()
                        depth_save = depthPred_np.squeeze().astype(np.float32)
                        with open(str(depth_dump_path),"wb") as f:
                            pickle.dump({'depth_pred': depth_save}, f)
                        with open(str(depth_dump_path).replace('.pickle','_cropped.pickle'),"wb") as f:
                            pickle.dump({'depth_pred': depth_save[:im_h_resized_to, :im_w_resized_to]}, f)
                        print(depth_dump_path)
                        _ = depth_save[:im_h_resized_to, :im_w_resized_to]
                        if opt.cfg.MODEL_BRDF.use_scale_aware_depth:
                            depth_dump_vis_path = scene_path_dump / ('imdepth_cropped.png')
                        else:
                            depth_dump_vis_path = scene_path_dump / ('imdepth_scale_invariant_cropped.png')
                        _ = vis_disp_colormap(_, normalize=True)[0]
                        Image.fromarray(_).save(str(depth_dump_vis_path))
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            if opt.cfg.MODEL_BRDF.use_scale_aware_depth:
                                depthBs_dump_path = scene_path_dump / ('imdepthBs.pickle')
                            else:
                                depthBs_dump_path = scene_path_dump / ('imdepthBs_scale_invariant.pickle')
                            depthBsPred_np = output_dict['depthBsPred'][0].cpu().numpy()
                            depthBs_save = depthBsPred_np.squeeze().astype(np.float32)
                            with open(str(depthBs_dump_path),"wb") as f:
                                pickle.dump({'depth_pred': depthBs_save}, f)
                            with open(str(depthBs_dump_path).replace('.pickle','_cropped.pickle'),"wb") as f:
                                pickle.dump({'depth_pred': depthBs_save[:im_h_resized_to, :im_w_resized_to]}, f)
                            _ = depthBs_save[:im_h_resized_to, :im_w_resized_to]
                            if opt.cfg.MODEL_BRDF.use_scale_aware_depth:
                                depthBs_dump_vis_path = scene_path_dump / ('imdepthBs_cropped.png')
                            else:
                                depthBs_dump_vis_path = scene_path_dump / ('imdepthBs_scale_invariant_cropped.png')
                            _ = vis_disp_colormap(_, normalize=True)[0]
                            Image.fromarray(_).save(str(depthBs_dump_vis_path))


            # ===== LIGHT
            if opt.cfg.MODEL_LIGHT.enable or (opt.cfg.MODEL_ALL.enable and 'li' in opt.cfg.MODEL_ALL.enable_list):
                envmapsPredImage = output_dict['envmapsPredImage'].detach().cpu().numpy()
                if not opt.cfg.DATASET.if_no_gt_light:
                    envmapsPredScaledImage = output_dict['envmapsPredScaledImage'].detach().cpu().numpy()
                    envmapsBatch = input_dict['envmapsBatch'].detach().cpu().numpy()
                else:
                    envmapsPredScaledImage = [None] * envmapsPredImage.shape[0]
                    envmapsBatch = [None] * envmapsPredImage.shape[0]
                renderedImPred = output_dict['renderedImPred'].detach().cpu().numpy()
                renderedImPred_sdr = output_dict['renderedImPred_sdr'].detach().cpu().numpy()
                imBatchSmall = output_dict['imBatchSmall'].detach().cpu().numpy()
                if not opt.cfg.DATASET.if_no_gt_light:
                    segEnvBatch = output_dict['segEnvBatch'].detach().cpu().numpy() # (4, 1, 120, 160, 1, 1)
                    reconstErr_loss_map_batch = output_dict['reconstErr_loss_map'].detach().cpu().numpy() # [4, 3, 120, 160, 8, 16]
                    reconstErr_loss_map_2D_batch = reconstErr_loss_map_batch.mean(-1).mean(-1).mean(1)

                # print(reconstErr_loss_map_2D_batch.shape, np.amax(reconstErr_loss_map_2D_batch), np.amin(reconstErr_loss_map_2D_batch),np.median(reconstErr_loss_map_2D_batch)) # (4, 120, 160) 3.9108467 0.0 0.22725311

                for sample_idx_batch in range(envmapsPredImage.shape[0]):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    # assert envmapsPredScaledImage.shape[0] == batch_size
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_env_path = real_sample_results_path / 'env.npz'
                        real_sample_env_path_hdr = real_sample_results_path / 'env.hdr'
                        env_save_half = envmapsPredImage[sample_idx_batch].transpose(1, 2, 3, 4, 0) # -> (120, 160, 8, 16, 3); >>> s = 'third_parties_outside/VirtualObjectInsertion/data_/Example1/env.npz' ['env']: (106, 160, 8, 16, 3)
                        env_save_full = np.zeros((env_save_half.shape[0]*2, env_save_half.shape[1]*2, env_save_half.shape[2], env_save_half.shape[3], env_save_half.shape[4]), dtype=env_save_half.dtype) # (106x2, 160x2, 8, 16, 3) 
                        # env_save_full[::2, ::2] = env_save_half
                        # env_save_full[::2, 1::2] = env_save_half
                        # env_save_full[1::2, 1::2] = env_save_half
                        # env_save_full[1::2, ::2] = env_save_half
                        # # if opt.is_master:
                        # #     np.savez(real_sample_env_path, env=env_save)
                        # print(env_save_half.shape, env_save_full.shape)

                        # Flip to be conincide with our dataset [ Rui: important...... to fix the blue-ish hue of inserted objects]
                        np.savez_compressed(real_sample_env_path,
                            env = np.ascontiguousarray(env_save_half[:, :, :, :, ::-1] ) )
                        # writeEnvToFile(output_dict['envmapsPredImage'][sample_idx_batch], 0, real_sample_env_path_hdr, nrows=24, ncols=16 )

                        I_hdr =envmapsPredImage[sample_idx_batch] * 1000.
                        H_grid, W_grid, h, w = I_hdr.shape[1:]
                        downsize_ratio = 4
                        H_grid_after = H_grid // 4 * 4
                        W_grid_after = W_grid // 4 * 4
                        I_hdr_after = I_hdr[:, :H_grid_after, :W_grid_after, :, :]
                        xx, yy = np.meshgrid(np.arange(0, H_grid_after, downsize_ratio), np.arange(0, W_grid_after, downsize_ratio))
                        I_hdr_downsampled = I_hdr_after[:, xx.T, yy.T, :, :]
                        I_hdr_downsampled = I_hdr_downsampled.transpose(1, 3, 2, 4, 0).reshape(H_grid_after*h//downsize_ratio, W_grid_after*w//downsize_ratio, 3)
                        if opt.is_master:
                            cv2.imwrite('{0}/{1}-{2}_{3}.hdr'.format(opt.summary_vis_path_task, tid, sample_idx, 'light_Pred') , I_hdr_downsampled[:, :, [2, 1, 0]])
                            if opt.cfg.DEBUG.if_dump_full_envmap:
                                # cv2.imwrite('{0}/{1}-{2}_{3}_ori.hdr'.format(opt.summary_vis_path_task, tid, sample_idx, 'light_Pred') , )
                                with open('{0}/{1}-{2}_{3}_ori.pickle'.format(opt.summary_vis_path_task, tid, sample_idx, 'light_Pred'),"wb") as f:
                                    pickle.dump({'env': I_hdr[[2, 1, 0], :, :, :, :]}, f)


                    else:
                        for I_hdr, name_tag in zip([envmapsPredImage[sample_idx_batch], envmapsPredScaledImage[sample_idx_batch], envmapsBatch[sample_idx_batch]], ['light_Pred', 'light_Pred_Scaled', 'light_GT']):
                            if I_hdr is None:
                                continue
                            H_grid, W_grid, h, w = I_hdr.shape[1:]
                            downsize_ratio = 4
                            assert H_grid % downsize_ratio == 0
                            assert W_grid % downsize_ratio == 0
                            xx, yy = np.meshgrid(np.arange(0, H_grid, downsize_ratio), np.arange(0, W_grid, downsize_ratio))
                            I_hdr_downsampled = I_hdr[:, xx.T, yy.T, :, :]
                            I_hdr_downsampled = I_hdr_downsampled.transpose(1, 3, 2, 4, 0).reshape(H_grid*h//downsize_ratio, W_grid*w//downsize_ratio, 3)
                            if opt.is_master:
                                cv2.imwrite('{0}/{1}-{2}_{3}.hdr'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag) , I_hdr_downsampled[:, :, [2, 1, 0]])
                                with open('{0}/{1}-{2}_{3}_ori.pickle'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag),"wb") as f:
                                    pickle.dump({'env': I_hdr[[2, 1, 0], :, :, :, :]}, f)

                    for I_png, name_tag in zip([renderedImPred[sample_idx_batch], renderedImPred_sdr[sample_idx_batch], imBatchSmall[sample_idx_batch], imBatchSmall[sample_idx_batch]**(1./2.2)], ['renderedImPred', 'renderedImPred_sdr', 'imBatchSmall_GT', 'imBatchSmall_GT_sdr']):
                        I_png = np.clip(I_png, 0., 1.)
                        I_png = (I_png.transpose(1, 2, 0) * 255.).astype(np.uint8)
                        if opt.is_master:
                            writer.add_image('VAL_light-%s/%d'%(name_tag, sample_idx), I_png, tid, dataformats='HWC')
                        Image.fromarray(I_png).save('{0}/{1}-{2}_light-{3}.png'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag))

                    if not opt.cfg.DATASET.if_no_gt_light:
                        segEnv = segEnvBatch[sample_idx_batch].squeeze()
                        if opt.is_master:
                            writer.add_image('VAL_light-%s/%d'%('segEnv_mask', sample_idx), segEnv, tid, dataformats='HW')

                        reconstErr_loss_map_2D = reconstErr_loss_map_2D_batch[sample_idx_batch].squeeze()
                        reconstErr_loss_map_2D = reconstErr_loss_map_2D / np.amax(reconstErr_loss_map_2D)
                        if opt.is_master:
                            writer.add_image('VAL_light-%s/%d'%('reconstErr_loss_map_2D', sample_idx), reconstErr_loss_map_2D, tid, dataformats='HW')
        

    # ===== Vis BRDF 2/2
    # ===== logging top N to TB
    im_paths_list = flatten_list(im_paths_list)
    im_h_resized_to_list = flatten_list(im_h_resized_to_list)
    im_w_resized_to_list = flatten_list(im_w_resized_to_list)
    

    if 'al' in opt.cfg.DATA.data_read_list:
        # if (not opt.cfg.DATASET.if_no_gt_semantics):
        albedoBatch_vis = torch.cat(albedoBatch_list)
    if 'no' in opt.cfg.DATA.data_read_list:
        # if (not opt.cfg.DATASET.if_no_gt_semantics):
        normalBatch_vis = torch.cat(normalBatch_list)
    if 'ro' in opt.cfg.DATA.data_read_list:
        # if (not opt.cfg.DATASET.if_no_gt_semantics):
        roughBatch_vis = torch.cat(roughBatch_list)
    if 'de' in opt.cfg.DATA.data_read_list:
        # if (not opt.cfg.DATASET.if_no_gt_semantics):
        depthBatch_vis = torch.cat(depthBatch_list)

    imBatch_vis = torch.cat(imBatch_vis_list)
    segAllBatch_vis = torch.cat(segAllBatch_list)
    segBRDFBatch_vis = torch.cat(segBRDFBatch_list)

    print('Saving vis to ', '{0}'.format(opt.summary_vis_path_task, tid))
    # Save the ground truth and the input
    # albedoBatch_vis_sdr = ( (albedoBatch_vis ) ** (1.0/2.2) ).data # torch.Size([16, 3, 192, 256]) 
    # im_batch_vis_sdr = ( (imBatch_vis)**(1.0/2.2) ).data
    im_batch_vis_sdr = (imBatch_vis ).data.permute(0, 3, 1, 2)

    ## ---- GTs
    # if (not opt.cfg.DATASET.if_no_gt_semantics):
    if 'al' in opt.cfg.DATA.data_read_list:
        albedo_gt_batch_vis_sdr = ( (albedoBatch_vis ) ** (1.0/2.2) ).data
    if 'no' in opt.cfg.DATA.data_read_list:
        normal_gt_batch_vis_sdr = (0.5*(normalBatch_vis + 1) ).data
    if 'ro' in opt.cfg.DATA.data_read_list:
        rough_gt_batch_vis_sdr = (0.5*(roughBatch_vis + 1) ).data
    if 'de' in opt.cfg.DATA.data_read_list:
        depthOut = 1 / torch.clamp(depthBatch_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthBatch_vis) # invert the gt depth just for visualization purposes!
        depth_gt_batch_vis_sdr = ( depthOut*segAllBatch_vis.expand_as(depthBatch_vis) ).data
    
    if opt.is_master:
        vutils.save_image(im_batch_vis_sdr,
                '{0}/{1}_im.png'.format(opt.summary_vis_path_task, tid) )
        if not opt.cfg.DATASET.if_no_gt_BRDF:
            if 'al' in opt.cfg.DATA.data_read_list:
                vutils.save_image(albedo_gt_batch_vis_sdr,
                    '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'no' in opt.cfg.DATA.data_read_list:
                vutils.save_image(normal_gt_batch_vis_sdr,
                    '{0}/{1}_normalGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'ro' in opt.cfg.DATA.data_read_list:
                vutils.save_image(rough_gt_batch_vis_sdr,
                    '{0}/{1}_roughGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'de' in opt.cfg.DATA.data_read_list:
                vutils.save_image(depth_gt_batch_vis_sdr,
                    '{0}/{1}_depthGt.png'.format(opt.summary_vis_path_task, tid) )

    if 'al' in opt.cfg.DATA.data_read_list:
        albedo_gt_batch_vis_sdr_numpy = albedo_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    if 'no' in opt.cfg.DATA.data_read_list:
        normal_gt_batch_vis_sdr_numpy = normal_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    if 'ro' in opt.cfg.DATA.data_read_list:
        rough_gt_batch_vis_sdr_numpy = rough_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    if 'de' in opt.cfg.DATA.data_read_list:
        depth_gt_batch_vis_sdr_numpy = depth_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
    # print('++++', rough_gt_batch_vis_sdr_numpy.shape, depth_gt_batch_vis_sdr_numpy.shape, albedo_gt_batch_vis_sdr_numpy.shape, albedo_gt_batch_vis_sdr_numpy.dtype)
    # print(np.amax(albedo_gt_batch_vis_sdr_numpy), np.amin(albedo_gt_batch_vis_sdr_numpy), np.mean(albedo_gt_batch_vis_sdr_numpy))
    depth_min_and_scale_list = []
    if not opt.cfg.DATASET.if_no_gt_BRDF and opt.is_master:
        for sample_idx in range(im_batch_vis_sdr.shape[0]):
            writer.add_image('VAL_brdf-segBRDF_GT/%d'%sample_idx, segBRDFBatch_vis[sample_idx].cpu().detach().numpy().squeeze(), tid, dataformats='HW')
            segAll = segAllBatch_vis[sample_idx].cpu().detach().numpy().squeeze()
            writer.add_image('VAL_brdf-segAll_GT/%d'%sample_idx, segAll, tid, dataformats='HW')
            if 'al' in opt.cfg.DATA.data_read_list:
                writer.add_image('VAL_brdf-albedo_GT/%d'%sample_idx, albedo_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
            if 'no' in opt.cfg.DATA.data_read_list:
                writer.add_image('VAL_brdf-normal_GT/%d'%sample_idx, normal_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
            if 'ro' in opt.cfg.DATA.data_read_list:
                writer.add_image('VAL_brdf-rough_GT/%d'%sample_idx, rough_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
            if 'de' in opt.cfg.DATA.data_read_list:
                depth_normalized, depth_min_and_scale = vis_disp_colormap(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, valid_mask=segAll==1)
                # print(np.amax(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze()), np.amin(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze()), depth_min_and_scale, '=====')
                depth_min_and_scale_list.append(depth_min_and_scale)
                writer.add_image('VAL_brdf-depth_GT/%d'%sample_idx, depth_normalized, tid, dataformats='HWC')


    ## ---- ESTs
    # if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
    if opt.cfg.MODEL_BRDF.enable:
        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedoPreds_vis = torch.cat(albedoPreds_list)
            if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                albedoPreds_aligned_vis = torch.cat(albedoPreds_aligned_list)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                albedoBsPreds_vis = torch.cat(albedoBsPreds_list)

        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normalPreds_vis = torch.cat(normalPreds_list)

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            roughPreds_vis = torch.cat(roughPreds_list)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                roughBsPreds_vis = torch.cat(roughBsPreds_list)

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthPreds_vis = torch.cat(depthPreds_list)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                depthBsPreds_vis = torch.cat(depthBsPreds_list)
            if not opt.cfg.MODEL_BRDF.use_scale_aware_depth and 'de' in opt.cfg.DATA.data_read_list:
                depthPreds_aligned_vis = torch.cat(depthPreds_aligned_list)
                if opt.cfg.MODEL_BRDF.if_bilateral:
                    depthBsPreds_aligned_vis = torch.cat(depthBsPreds_aligned_list)

        if opt.cascadeLevel > 0:
            diffusePreBatch_vis = torch.cat(diffusePreBatch_list)
            specularPreBatch_vis = torch.cat(specularPreBatch_list)
            renderedImBatch_vis = torch.cat(renderedImBatch_list)

        if opt.cascadeLevel > 0 and opt.is_master:
            vutils.save_image( ( (diffusePreBatch_vis)**(1.0/2.2) ).data,
                    '{0}/{1}_diffusePre.png'.format(opt.summary_vis_path_task, tid) )
            vutils.save_image( ( (specularPreBatch_vis)**(1.0/2.2) ).data,
                    '{0}/{1}_specularPre.png'.format(opt.summary_vis_path_task, tid) )
            vutils.save_image( ( (renderedImBatch_vis)**(1.0/2.2) ).data,
                    '{0}/{1}_renderedImage.png'.format(opt.summary_vis_path_task, tid) )

        # Save the predicted results
        # for n in range(0, len(output_dict['albedoPreds']) ):
        #     vutils.save_image( ( (output_dict['albedoPreds'][n] ) ** (1.0/2.2) ).data,
        #             '{0}/{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
        # for n in range(0, len(output_dict['normalPreds']) ):
        #     vutils.save_image( ( 0.5*(output_dict['normalPred'] + 1) ).data,
        #             '{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
        # for n in range(0, len(output_dict['roughPreds']) ):
        #     vutils.save_image( ( 0.5*(output_dict['roughPred'] + 1) ).data,
        #             '{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
        # for n in range(0, len(output_dict['depthPreds']) ):
        #     depthOut = 1 / torch.clamp(output_dict['depthPred'] + 1, 1e-6, 10) * segAllBatch_vis.expand_as(output_dict['depthPred'])
        #     vutils.save_image( ( depthOut * segAllBatch_vis.expand_as(output_dict['depthPred']) ).data,
        #             '{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        # ==== Preds
        # print(opt.cfg.MODEL_BRDF.enable_list, '>>>>>>>>', n, albedoPreds_vis.shape)
        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr = ( (albedoPreds_vis ) ** (1.0/2.2) ).data
            if opt.cfg.MODEL_BRDF.if_bilateral:
                albedo_bs_pred_batch_vis_sdr = ( (albedoBsPreds_vis ) ** (1.0/2.2) ).data

            if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                albedo_pred_aligned_batch_vis_sdr = ( (albedoPreds_aligned_vis ) ** (1.0/2.2) ).data
            if opt.is_master:
                vutils.save_image(albedo_pred_batch_vis_sdr,
                        '{0}/{1}_albedoPred.png'.format(opt.summary_vis_path_task, tid) )

                if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                    vutils.save_image(albedo_pred_aligned_batch_vis_sdr,
                            '{0}/{1}_albedoPred_aligned.png'.format(opt.summary_vis_path_task, tid) )

        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr = ( 0.5*(normalPreds_vis + 1) ).data
            if opt.is_master:
                vutils.save_image(normal_pred_batch_vis_sdr,
                        '{0}/{1}_normalPred.png'.format(opt.summary_vis_path_task, tid) )

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr = ( 0.5*(roughPreds_vis + 1) ).data
            if opt.cfg.MODEL_BRDF.if_bilateral:
                rough_bs_pred_batch_vis_sdr = ( 0.5*(roughBsPreds_vis + 1)).data
            if opt.is_master:
                vutils.save_image(rough_pred_batch_vis_sdr,
                        '{0}/{1}_roughPred.png'.format(opt.summary_vis_path_task, tid) )

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthPreds_vis)
            depth_pred_batch_vis_sdr = ( depthOut * segAllBatch_vis.expand_as(depthPreds_vis) ).data
            if opt.cfg.MODEL_BRDF.if_bilateral:
                depthBsOut = 1 / torch.clamp(depthBsPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthBsPreds_vis)
                depth_bs_pred_batch_vis_sdr = ( depthBsOut * segAllBatch_vis.expand_as(depthBsPreds_vis) ).data

            depthOut_colored_single_numpy_list = []
            for depthPreds_vis_single_numpy in depthOut.cpu().detach().numpy():
                depthOut_colored_single_numpy_list.append(vis_disp_colormap(depthPreds_vis_single_numpy.squeeze(), normalize=True)[0])
            depthOut_colored_batch = np.stack(depthOut_colored_single_numpy_list).transpose(0, 3, 1, 2).astype(np.float32) / 255.
            depth_pred_batch_vis_sdr_colored = ( torch.from_numpy(depthOut_colored_batch).cuda() * segAllBatch_vis.expand_as(depthPreds_vis) ).data

            if opt.is_master:
                vutils.save_image(depth_pred_batch_vis_sdr,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                vutils.save_image(depth_pred_batch_vis_sdr_colored,
                    '{0}/{1}_depthPred_colored_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr_numpy = albedo_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                albedo_pred_aligned_batch_vis_sdr_numpy = albedo_pred_aligned_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                albedo_bs_pred_batch_vis_sdr_numpy = albedo_bs_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr_numpy = normal_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr_numpy = rough_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                rough_bs_pred_batch_vis_sdr_numpy = rough_bs_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depth_pred_batch_vis_sdr_numpy = depth_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            if opt.cfg.MODEL_BRDF.if_bilateral:
                depth_bs_pred_batch_vis_sdr_numpy = depth_bs_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

        if opt.is_master:
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                writer.add_histogram('VAL_brdf-depth_PRED', np.clip(depthPreds_vis.cpu().numpy().flatten(), 0., 200.), tid)

            for sample_idx in tqdm(range(im_batch_vis_sdr.shape[0])):

                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-albedo_PRED/%d'%sample_idx, albedo_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        writer.add_image('VAL_brdf-albedo_PRED-BS/%d'%sample_idx, albedo_bs_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                        writer.add_image('VAL_brdf-albedo_scaleAligned_PRED/%d'%sample_idx, albedo_pred_aligned_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((albedo_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((albedo_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_albedoGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_albedo_path = real_sample_results_path / 'albedo.png'
                        albedo = Image.fromarray((albedo_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        albedo = albedo.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        albedo.save(str(real_sample_albedo_path))
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            real_sample_albedo_bs_path = real_sample_results_path / 'albedo_bs.png'
                            albedo_bs = Image.fromarray((albedo_bs_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                            albedo_bs = albedo_bs.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                            albedo_bs.save(str(real_sample_albedo_bs_path))


                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-normal_PRED/%d'%sample_idx, normal_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'no' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((normal_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_normalGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_normal_path = real_sample_results_path / 'normal.png'
                        normal = Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        normal = normal.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        normal.save(str(real_sample_normal_path))

                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-rough_PRED/%d'%sample_idx, rough_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        writer.add_image('VAL_brdf-rough_PRED-BS/%d'%sample_idx, rough_bs_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')

                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((rough_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8).squeeze()).save('{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'ro' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((rough_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8).squeeze()).save('{0}/{1}_roughGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_rough_path = real_sample_results_path / 'rough.png'
                        rough = Image.fromarray((rough_pred_batch_vis_sdr_numpy[sample_idx].squeeze()*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        rough = rough.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        rough.save(str(real_sample_rough_path))
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            real_sample_rough_bs_path = real_sample_results_path / 'rough_bs.png'
                            rough_bs = Image.fromarray((rough_bs_pred_batch_vis_sdr_numpy[sample_idx].squeeze()*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                            rough_bs = rough_bs.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                            rough_bs.save(str(real_sample_rough_bs_path))

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'de' in opt.cfg.DATA.data_read_list:
                        depth_normalized_pred, _ = vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, min_and_scale=depth_min_and_scale_list[sample_idx])
                        writer.add_image('VAL_brdf-depth_syncScale_PRED/%d'%sample_idx, depth_normalized_pred, tid, dataformats='HWC')
                    _ = depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze()
                    im_h_resized_to, im_w_resized_to = im_h_resized_to_list[sample_idx], im_w_resized_to_list[sample_idx]
                    # print(im_h_resized_to, im_w_resized_to)
                    _ = _[:im_h_resized_to, :im_w_resized_to]
                    depth_not_normalized_pred = vis_disp_colormap(_, normalize=True)[0]
                    writer.add_image('VAL_brdf-depth_PRED/%d'%sample_idx, depth_not_normalized_pred, tid, dataformats='HWC')
                    writer.add_image('VAL_brdf-depth_PRED_thres%.2f/%d'%(opt.cfg.MODEL_LIGHT.depth_thres, sample_idx), depthPreds_vis[sample_idx].cpu().numpy().squeeze()>opt.cfg.MODEL_LIGHT.depth_thres, tid, dataformats='HW')
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        _ = depth_bs_pred_batch_vis_sdr_numpy[sample_idx].squeeze()
                        im_h_resized_to, im_w_resized_to = im_h_resized_to_list[sample_idx], im_w_resized_to_list[sample_idx]
                        _ = _[:im_h_resized_to, :im_w_resized_to]
                        depth_bs_not_normalized_pred = vis_disp_colormap(_, normalize=True)[0]
                        writer.add_image('VAL_brdf-depth_PRED-BS/%d'%sample_idx, depth_bs_not_normalized_pred, tid, dataformats='HWC')
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'de' in opt.cfg.DATA.data_read_list:
                            depth_bs_normalized_pred = vis_disp_colormap(_, normalize=True, min_and_scale=depth_min_and_scale_list[sample_idx])[0]
                            writer.add_image('VAL_brdf-depth_syncScale_PRED-BS/%d'%sample_idx, depth_bs_normalized_pred, tid, dataformats='HWC')

                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((depth_not_normalized_pred).astype(np.uint8)).save('{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'de' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((depth_normalized_pred).astype(np.uint8)).save('{0}/{1}_depthPred_syncScale_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                            depth_normalized_gt, _ = vis_disp_colormap(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True)
                            Image.fromarray((depth_normalized_gt).astype(np.uint8)).save('{0}/{1}_depthGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))

                    pickle_save_path = Path(opt.summary_vis_path_task) / ('results_depth_%d.pickle'%sample_idx)
                    save_dict = {'depthPreds_vis': depthPreds_vis[sample_idx].detach().cpu().squeeze().numpy()}
                    if opt.if_save_pickles:
                        with open(str(pickle_save_path),"wb") as f:
                            pickle.dump(save_dict, f)

                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_depth_path = real_sample_results_path / 'depth.png'
                        depth = Image.fromarray(depth_not_normalized_pred)
                        depth = depth.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        depth.save(str(real_sample_depth_path))
                        if opt.cfg.MODEL_BRDF.if_bilateral:
                            real_sample_depth_bs_path = real_sample_results_path / 'depth_bs.png'
                            depth_bs = Image.fromarray(depth_bs_not_normalized_pred)
                            depth_bs = depth_bs.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                            depth_bs.save(str(real_sample_depth_bs_path))

    if opt.cfg.DEBUG.if_load_dump_BRDF_offline and opt.cfg.DEBUG.if_test_real:
        if opt.is_master:
            for sample_idx in tqdm(range(im_batch_vis_sdr.shape[0])):
                real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                real_sample_albedo_path = real_sample_results_path / 'albedo.png'
                albedo = Image.fromarray((albedo_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                albedo = albedo.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                albedo.save(str(real_sample_albedo_path))
                print(real_sample_albedo_path)

                real_sample_rough_path = real_sample_results_path / 'rough.png'
                rough = Image.fromarray((rough_gt_batch_vis_sdr_numpy[sample_idx].squeeze()*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                rough = rough.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                rough.save(str(real_sample_rough_path))

                real_sample_normal_path = real_sample_results_path / 'normal.png'
                normal = Image.fromarray((normal_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                normal = normal.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                normal.save(str(real_sample_normal_path))


    logger.info(red('Evaluation VIS timings: ' + time_meters_to_string(time_meters)))

    # synchronize()
    opt.if_vis_debug_pac = False


def writeEnvToFile(envmaps, envId, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envmap = envmaps[envId, :, :, :, :, :].data.cpu().numpy()
    envmap = np.transpose(envmap, [1, 2, 3, 4, 0] )
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows )
    interX = int(envCol / ncols )

    lnrows = len(np.arange(0, envRow, interY) )
    lncols = len(np.arange(0, envCol, interX) )

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY ):
        for c in range(0, envCol, interX ):
            rId = int(r / interY )
            cId = int(c / interX )

            rs = rId * (envHeight + gap )
            cs = cId * (envWidth + gap )
            envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * (envmapLarge ** (1.0/2.2) ) ).astype(np.uint8 )
    cv2.imwrite(envName, envmapLarge[:, :, ::-1] )
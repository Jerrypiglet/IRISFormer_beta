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
from utils.comm import synchronize

from utils.utils_metrics import compute_errors_depth_nyu
from train_funcs_matcls import getG1IdDict, getRescaledMatFromID
# from pytorch_lightning.metrics import Precision, Recall, F1, Accuracy
from pytorch_lightning.metrics import Accuracy

from icecream import ic
import pickle
import matplotlib.pyplot as plt

from train_funcs_layout_object_emitter import vis_layout_emitter

def get_time_meters_joint():
    time_meters = {}
    time_meters['ts'] = 0.
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss_brdf'] = AverageMeter()
    time_meters['loss_light'] = AverageMeter()
    time_meters['loss_layout_emitter'] = AverageMeter()
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
    labels_dict = {'im_trainval_RGB': data_batch['im_trainval_RGB'].cuda(non_blocking=True), 'im_SDR_RGB': data_batch['im_SDR_RGB'].cuda(non_blocking=True)}
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

    if opt.cfg.DATA.load_brdf_gt:
        input_batch_brdf, labels_dict_brdf, pre_batch_dict_brdf = get_labels_dict_brdf(data_batch, opt, return_input_batch_as_list=True)
        list_from_brdf = [input_batch_brdf, labels_dict_brdf, pre_batch_dict_brdf]
        labels_dict.update({'input_batch_brdf': torch.cat(input_batch_brdf, dim=1), 'pre_batch_dict_brdf': pre_batch_dict_brdf})
    else:
        labels_dict_brdf = {}    
        list_from_brdf = None
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

    # labels_dict = {**labels_dict_matseg, **labels_dict_brdf}
    return labels_dict

def forward_joint(is_train, labels_dict, model, opt, time_meters, if_vis=False, ):
    # forward model + compute losses

    # Forward model
    output_dict = model(labels_dict)
    time_meters['forward'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    # Post-processing and computing losses
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
        output_dict, loss_dict = postprocess_brdf(labels_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_brdf'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_LIGHT.enable:
        output_dict, loss_dict = postprocess_light(labels_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_light'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_LAYOUT_EMITTER.enable:
        output_dict, loss_dict = postprocess_layout_object_emitter(labels_dict, output_dict, loss_dict, opt, time_meters, is_train=is_train, if_vis=if_vis)
        time_meters['loss_layout_emitter'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_MATCLS.enable:
        output_dict, loss_dict = postprocess_matcls(labels_dict, output_dict, loss_dict, opt, time_meters, if_vis=if_vis)
        time_meters['loss_matcls'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    # ic(output_dict.keys(), output_dict['results_emitter'].keys())

    return output_dict, loss_dict

def val_epoch_joint(brdf_loader_val, model, bin_mean_shift, params_mis):
    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']
    ENABLE_SEMSEG = opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable
    ENABLE_MATSEG = opt.cfg.MODEL_MATSEG.enable
    ENABLE_BRDF = opt.cfg.MODEL_BRDF.enable
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
            loss_keys += [
                'loss_brdf-albedo', 
                'loss_brdf-normal', 
                'loss_brdf-rough', 
                'loss_brdf-depth', 
                'loss_brdf-ALL', 
                'loss_brdf-rough-paper', 
                'loss_brdf-depth-paper'
            ]
        if opt.cfg.MODEL_BRDF.enable_semseg_decoder:
            loss_keys += ['loss_semseg-ALL']

    if opt.cfg.MODEL_LIGHT.enable:
        loss_keys += [
            'loss_light-ALL', 
            'loss_light-reconstErr', 
            'loss_light-renderErr', 
        ]

    if opt.cfg.MODEL_LAYOUT_EMITTER.enable:
        if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
            loss_keys += [
                'loss_layout-pitch_cls', 
                'loss_layout-pitch_reg', 
                'loss_layout-roll_cls', 
                'loss_layout-roll_reg', 
                'loss_layout-lo_ori_cls', 
                'loss_layout-lo_ori_reg', 
                'loss_layout-lo_centroid', 
                'loss_layout-lo_coeffs', 
                'loss_layout-lo_corner', 
                'loss_layout-ALL'
            ]
        if 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
            loss_keys += [
                'loss_object-size_reg', 
                'loss_object-ori_cls', 
                'loss_object-ori_reg', 
                'loss_object-centroid_cls', 
                'loss_object-centroid_reg', 
                'loss_object-offset_2D', 
                'loss_object-ALL', 
            ]
        if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list and 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
            loss_keys += [
                'loss_joint-phy', 
                'loss_joint-bdb2D', 
                'loss_joint-corner', 
                'loss_joint-ALL', 
            ]

        if 'mesh' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
            if opt.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'SVRLoss':
                loss_keys += [
                    'loss_mesh-chamfer', 
                    'loss_mesh-face', 
                    'loss_mesh-edge', 
                    'loss_mesh-boundary', 
                ]
            elif opt.cfg.MODEL_LAYOUT_EMITTER.mesh.loss == 'ReconLoss':
                loss_keys += [
                    'loss_mesh-point', 
                ]
            loss_keys += [
                    'loss_mesh-ALL', 
                ]

        if 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
            loss_keys += [
                'loss_emitter-light_ratio', 
                'loss_emitter-cell_cls', 
                'loss_emitter-cell_axis', 
                'loss_emitter-cell_intensity', 
                'loss_emitter-cell_lamb', 
                'loss_emitter-ALL', 
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
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):
            ts_iter_start = time.time()

            input_dict = get_labels_dict_joint(data_batch, opt)

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = forward_joint(False, input_dict, model, opt, time_meters)

            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            # loss = loss_dict['loss_all']
            
            # ======= update loss
            for loss_key in loss_dict_reduced:
                loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= Metering
            if ENABLE_LIGHT:
                pass

            if ENABLE_BRDF:
                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_input = input_dict['depthBatch'].detach().cpu().numpy()
                    depth_output = output_dict['depthPreds'][0].detach().cpu().numpy()
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
                    normal_output = output_dict['normalPreds'][0].detach().cpu().numpy()
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
            logger.info('Logged val loss for %s'%loss_key)

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

        logger.info(red('Evaluation timings: ' + time_meters_to_string(time_meters)))


def vis_val_epoch_joint(brdf_loader_val, model, bin_mean_shift, params_mis):

    writer, logger, opt, tid, batch_size = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['batch_size_val_vis']
    logger.info(red('=== [vis_val_epoch_joint] Visualizing for %d batches on rank %d'%(len(brdf_loader_val), opt.rank)))

    model.eval()
    opt.if_vis_debug_pac = True

    if opt.test_real:
        if opt.cfg.MODEL_MATCLS.enable:
            matcls_results_path = opt.cfg.MODEL_MATCLS.real_images_list.replace('.txt', '-results.txt')
            f_matcls_results = open(matcls_results_path, 'w')

    time_meters = get_time_meters_joint()

    if opt.cfg.MODEL_MATSEG.enable:
        match_segmentatin = MatchSegmentation()

    if opt.cfg.MODEL_BRDF.enable:
        im_paths_list = []
        albedoBatch_list = []
        normalBatch_list = []
        roughBatch_list = []
        depthBatch_list = []
        imBatch_list = []
        segAllBatch_list = []

        diffusePreBatch_list = []
        specularPreBatch_list = []
        renderedImBatch_list = []
        
        albedoPreds_list = []
        normalPreds_list = []
        roughPreds_list = []
        depthPreds_list = []
    
    if opt.cfg.MODEL_MATCLS.enable:
        matG1IdDict = getG1IdDict(opt.cfg.PATH.matcls_matIdG1_path)


    # opt.albedo_pooling_debug = True

    # ===== Gather vis of N batches
    with torch.no_grad():
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):
            # ic(batch_id)
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
            output_dict, loss_dict = forward_joint(False, input_dict, model, opt, time_meters, if_vis=True)

            synchronize()
            
            # ======= Vis imagges
            colors = np.loadtxt(os.path.join(opt.pwdpath, opt.cfg.PATH.semseg_colors_path)).astype('uint8')
            if opt.cfg.DATA.load_semseg_gt:
                semseg_label = input_dict['semseg_label'].cpu().numpy()
            if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable:
                semseg_pred = output_dict['semseg_pred'].cpu().numpy()

            for sample_idx_batch, (im_single, im_path) in enumerate(zip(data_batch['im_SDR_RGB'], data_batch['image_path'])):
                sample_idx = sample_idx_batch+batch_size*batch_id
                print('Visualizing %d image...'%sample_idx, batch_id, sample_idx_batch)
                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                    break

                im_single = im_single.numpy().squeeze()
                # im_path = os.path.join('./tmp/', 'im_%d-%d_color.png'%(tid, im_index))
                # color_path = os.path.join('./tmp/', 'im_%d-%d_semseg.png'%(tid, im_index))
                # cv2.imwrite(im_path, im_single * 255.)
                # semseg_color.save(color_path)
                if opt.is_master:
                    writer.add_image('VAL_im/%d'%(sample_idx), im_single, tid, dataformats='HWC')
                    if opt.cfg.MODEL_MATSEG.albedo_pooling_debug and not opt.if_cluster:
                        os.makedirs('tmp/demo_%s'%(opt.task_name), exist_ok=True)
                        np.save('tmp/demo_%s/im_trainval_RGB_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), im_single)
                        print('Saved to' + 'tmp/demo_%s/im_trainval_RGB_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx))

                    writer.add_text('VAL_image_name/%d'%(sample_idx), im_path, tid)
                    assert sample_idx == data_batch['image_index'][sample_idx_batch]
                    # ic(sample_idx, data_batch['image_index'][sample_idx_batch], batch_id, batch_size, im_path)
                    # print(sample_idx, im_path)

                if (opt.cfg.MODEL_MATSEG.if_albedo_pooling or opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or opt.cfg.MODEL_MATSEG.if_albedo_pac_pool or opt.cfg.MODEL_MATSEG.if_albedo_safenet) and opt.cfg.MODEL_MATSEG.albedo_pooling_debug:
                    if opt.is_master:
                        if output_dict['im_trainval_RGB_mask_pooled_mean'] is not None:
                            im_trainval_RGB_mask_pooled_mean = output_dict['im_trainval_RGB_mask_pooled_mean'][sample_idx_batch]
                            im_trainval_RGB_mask_pooled_mean = im_trainval_RGB_mask_pooled_mean.cpu().numpy().squeeze().transpose(1, 2, 0)
                            writer.add_image('VAL_im_trainval_RGB_mask_pooled_mean/%d'%(sample_idx), im_trainval_RGB_mask_pooled_mean, tid, dataformats='HWC')
                        if not opt.if_cluster:
                            if 'kernel_list' in output_dict and not output_dict['kernel_list'] is None:
                                kernel_list = output_dict['kernel_list']
                                # print(len(kernel_list), kernel_list[0].shape)
                                np.save('tmp/demo_%s/kernel_list_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), kernel_list[0].detach().cpu().numpy())
                            if output_dict['im_trainval_RGB_mask_pooled_mean'] is not None:
                                np.save('tmp/demo_%s/im_trainval_RGB_mask_pooled_mean_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), im_trainval_RGB_mask_pooled_mean)
                            if 'embeddings' in output_dict and output_dict['embeddings'] is not None:
                                np.save('tmp/demo_%s/embeddings_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), output_dict['embeddings'].detach().cpu().numpy())
                            if 'affinity' in output_dict:
                                affinity = output_dict['affinity']
                                sample_ij = output_dict['sample_ij']
                                if affinity is not None:
                                    np.save('tmp/demo_%s/affinity_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), affinity[0].detach().cpu().numpy())
                                    np.save('tmp/demo_%s/sample_ij_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), sample_ij)

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
                        im_single = data_batch['im_SDR_RGB'][sample_idx_batch].detach().cpu().numpy().astype(np.float32)
                        im_single = cv2.resize(im_single, (color_pred.shape[1], color_pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                        color_pred = color_pred.astype(np.float32) / 255.
                        semseg_pred_overlay = im_single * color_pred + im_single * 0.2 * (1. - color_pred)
                        writer.add_image('VAL_semseg_PRED-overlay/%d'%(sample_idx), semseg_pred_overlay, tid, dataformats='HWC')

                        pickle_save_path = Path(opt.summary_vis_path_task) / ('results_semseg_tid%d-%d.pickle'%(tid, sample_idx))
                        save_dict = {'semseg_pred': semseg_pred[sample_idx_batch], }
                        if opt.if_save_pickles:
                            with open(str(pickle_save_path),"wb") as f:
                                pickle.dump(save_dict, f)

            # ======= Vis layout-emitter
            if opt.cfg.MODEL_LAYOUT_EMITTER.enable:
                output_vis_dict = vis_layout_emitter(input_dict, output_dict, opt, time_meters)
                # output_dict['output_layout_emitter_vis_dict'] = output_vis_dict
                if_real_image = False
                draw_mode = 'both' if not if_real_image else 'prediction'

                scene_box_list, layout_info_dict_list, emitter_info_dict_list = output_vis_dict['scene_box_list'], output_vis_dict['layout_info_dict_list'], output_vis_dict['emitter_info_dict_list']
                if opt.is_master:
                    logger.info('emitter_layout -------> ' + str(Path(opt.summary_vis_path_task)))

                    if 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                        patch_flattened = data_batch['boxes_batch']['patch'] # [B, 3, ?, ?]
                        patch_batch = [patch_flattened[x[0]:x[1]].numpy() for x in data_batch['obj_split'].cpu().numpy()] # [[?, 3, D, D], [?, 3, D, D], ...]
                        # print([x.shape[0] for x in patch_batch], data_batch['obj_split'].cpu().numpy(), patch_flattened.shape)
                        # print([[x[0], x[1]] for x in data_batch['obj_split'].cpu().numpy()])
                        assert sum([x.shape[0] for x in patch_batch])==patch_flattened.shape[0]


                    for sample_idx_batch, (scene_box, layout_info_dict, emitter_info_dict) in enumerate(zip(scene_box_list, layout_info_dict_list, emitter_info_dict_list)):
                        sample_idx = sample_idx_batch+batch_size*batch_id
                        save_prefix = ('results_LABEL_tid%d-%d'%(tid, sample_idx))

                        if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                            output_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'layout') + '.png')
                            fig_2d, _ = scene_box.draw_projected_layout(draw_mode, return_plt=True, if_use_plt=True) # with plt plotting
                            fig_2d.savefig(str(output_path))
                            plt.close(fig_2d)
                            
                        pickle_save_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'layout_info') + '.pickle')
                        save_dict = {'rgb_img_path': data_batch['image_path'][sample_idx_batch],  'bins_tensor': opt.bins_tensor}
                        save_dict.update(layout_info_dict)
                        if opt.if_save_pickles:
                            with open(str(pickle_save_path),"wb") as f:
                                pickle.dump(save_dict, f)

                        if 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                            output_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'obj') + '.png')
                            fig_2d = scene_box.draw_projected_bdb3d(draw_mode, if_vis_2dbbox=True, return_plt=True, if_use_plt=True)
                            fig_2d.savefig(str(output_path))
                            plt.close(fig_2d)

                            patch_batch_sample = patch_batch[sample_idx_batch]
                            for patch_idx, patch_single in enumerate(patch_batch_sample):
                                writer.add_image('VAL_bdb2d_patch/%d-%d'%(sample_idx, patch_idx), patch_single.transpose(1, 2, 0), tid, dataformats='HWC')

                        if 'mesh' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                            mask_all = np.zeros((240, 320))
                            obj_idx = 1
                            for x in data_batch['boxes_batch']['mask'][sample_idx_batch]:
                                if x is None:
                                    continue
                                x1, y1, x2, y2 = x['msk_bdb']
                                dest_mask = mask_all[y1:y2+1, x1:x2+1]==0
                                ori_mask = x['msk'].astype(np.uint8)
                                ori_mask[ori_mask==1] += obj_idx
                                mask_all[y1:y2+1, x1:x2+1][dest_mask] = ori_mask[dest_mask]
                                obj_idx += 1

                            mask_all = mask_all.astype(np.float32)
                            mask_all = mask_all / np.amax(mask_all)
                            writer.add_image('VAL_bdb2d_masks/%d'%(sample_idx), mask_all, tid, dataformats='HW')

                            fig_3d, ax_3ds = scene_box.draw_3D_scene_plt(draw_mode, if_show_objs=True, hide_random_id=False, if_debug=False, hide_cells=True, if_dump_to_mesh=True, if_show_emitter=False, pickle_id=sample_idx)

                            if opt.cfg.MODEL_LAYOUT_EMITTER.mesh.if_use_vtk:
                                im_meshes_GT = scene_box.draw3D('GT', if_return_img=True, if_save_img=False, if_save_obj=False, save_path_without_suffix = 'recon')['im']
                                im_meshes_pred = scene_box.draw3D('prediction', if_return_img=True, if_save_img=False, if_save_obj=False, save_path_without_suffix = 'recon')['im']
                                writer.add_image('VAL_mesh_GT/%d'%(sample_idx), im_meshes_GT, tid, dataformats='HWC')
                                writer.add_image('VAL_mesh_pred/%d'%(sample_idx), im_meshes_pred, tid, dataformats='HWC')



                        if 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                            output_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'emitter') + '.png')
                            fig_3d, ax_3ds = scene_box.draw_3D_scene_plt(draw_mode, if_return_cells_vis_info=True, if_show_emitter=not(if_real_image))
                            fig_3d.savefig(str(output_path))
                            plt.close(fig_3d)
                            cells_vis_info_list = ax_3ds[-1]
                            save_dict = {'cells_vis_info_list': cells_vis_info_list}
                            save_dict.update(emitter_info_dict)
                            pickle_save_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'cells_vis_info_list') + '.pickle')
                            if opt.if_save_pickles:
                                with open(str(pickle_save_path),"wb") as f:
                                    pickle.dump(save_dict, f)

                            emitter_input_dict = {'hdr_scale': data_batch['hdr_scale'].cpu().numpy()[sample_idx_batch]}

                            if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.enable:
                                fig_3d, ax_3d = scene_box.draw_3D_scene_plt('GT', )
                                ax_3d[1] = fig_3d.add_subplot(122, projection='3d')
                                scene_box.draw_3D_scene_plt('GT', fig_or_ax=[ax_3d[1], ax_3d[0]], hide_cells=True)
                                
                                lightAccu_color_array_GT = emitter_info_dict['envmap_lightAccu_mean_vis_GT'].transpose(0, 2, 3, 1) # -> [6, 8, 8, 3]
                                hdr_scale = data_batch['hdr_scale'][sample_idx_batch].item()
                                # ic(sample_idx, hdr_scale)
                                # lightAccu_color_array_GT = np.clip(lightAccu_color_array_GT * hdr_scale, 0., 1.)
                                lightAccu_color_array_GT = np.clip(lightAccu_color_array_GT, 0., 1.)
                                scene_box.draw_all_cells(ax_3d[1], scene_box.gt_layout, lightnet_array_GT=lightAccu_color_array_GT, alpha=1.)

                                output_path = Path(opt.summary_vis_path_task) / (('results_LABEL-%d'%(sample_idx)).replace('LABEL', 'lightAccu_view1') + '.png')
                                fig_3d.savefig(str(output_path))

                                az = 92
                                elev = 113
                                ax_3d[1].view_init(elev=elev, azim=az)
                                ax_3d[0].view_init(elev=elev, azim=az)
                                output_path = str(output_path).replace('lightAccu_view1', 'lightAccu_view2')
                                fig_3d.savefig(str(output_path))

                                plt.close(fig_3d)
        
                                if opt.if_save_pickles:
                                    results_emitter_pickle_save_path = Path(opt.summary_vis_path_task) / ('results_emitter_%d.pickle'%sample_idx)
                                    results_emitter_save_dict = {}
                                    if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version in ['V2', 'V3']:
                                        results_emitter_save_dict.update({
                                            'envmap_lightAccu_mean': output_dict['emitter_est_result']['envmap_lightAccu_mean'].detach().cpu().numpy()[sample_idx_batch], \
                                            'points_sampled_mask_expanded': output_dict['emitter_est_result']['points_sampled_mask_expanded'].detach().cpu().numpy()[sample_idx_batch], \
                                            'cell_normal_outside_label': input_dict['emitter_labels']['cell_normal_outside'].detach().cpu().numpy()[sample_idx_batch], 
                                            'emitter_cell_axis_abs_est': output_dict['results_emitter']['emitter_cell_axis_abs_est'].detach().cpu().numpy()[sample_idx_batch], \
                                            'emitter_cell_axis_abs_gt': output_dict['results_emitter']['emitter_cell_axis_abs_gt'].detach().cpu().numpy()[sample_idx_batch], \
                                            'window_mask': output_dict['results_emitter']['window_mask'].detach().cpu().numpy()[sample_idx_batch], 
                                        })
                                        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_sampled_img_feats_as_input:
                                            emitter_input_dict.update({'img_feat_map_sampled': output_dict['emitter_input']['img_feat_map_sampled'].detach().cpu().numpy()[sample_idx_batch], })

                                    if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.version in ['V3']:
                                        results_emitter_save_dict.update({
                                            'envmap_lightAccu': output_dict['emitter_est_result']['envmap_lightAccu'].detach().cpu().numpy()[sample_idx_batch], \
                                            'scattered_light': output_dict['emitter_est_result']['scattered_light'].detach().cpu().numpy()[sample_idx_batch], \
                                            'emitter_outdirs_meshgrid_Total3D_outside': output_dict['emitter_est_result']['emitter_outdirs_meshgrid_Total3D_outside'].detach().cpu().numpy()[sample_idx_batch], \
                                            'normal_outside_Total3D': output_dict['emitter_est_result']['normal_outside_Total3D'].detach().cpu().numpy()[sample_idx_batch], \
                                        })
                                        if opt.cfg.MODEL_LAYOUT_EMITTER.emitter.light_accu_net.use_weighted_axis:
                                            results_emitter_save_dict.update({'cell_axis_weights': output_dict['emitter_est_result']['cell_axis_weights'].detach().cpu().numpy()[sample_idx_batch]})

                                    # envmap_lightAccu (3, 384, 120, 160)
                                    # envmap_lightAccu_mean (6, 3, 8, 8)
                                    # points_sampled_mask_expanded (1, 1, 120, 160)
                                    # scattered_light (384, 8, 16, 3)
                                    # cell_normal_outside_label (6, 8, 8, 3)
                                    # emitter_outdirs_meshgrid_Total3D_outside (384, 8, 16, 3)
                                    # normal_outside_Total3D (384, 1, 1, 3)
                                    # emitter_cell_axis_abs_est (6, 64, 3)
                                    # emitter_cell_axis_abs_gt (6, 64, 3)
                                    # window_mask (6, 64)
                                    # cell_axis_weights (384, 8, 16, 1)
                                    if opt.if_save_pickles:
                                        with open(str(results_emitter_pickle_save_path),"wb") as f:
                                            pickle.dump(results_emitter_save_dict, f)

                                emitter_input_dict.update({x: output_dict['emitter_input'][x].detach().cpu().numpy()[sample_idx_batch] for x in output_dict['emitter_input']})
                                emitter_input_dict.update({'env_scale': data_batch['env_scale'].cpu().numpy()[sample_idx_batch], 'envmap_path': data_batch['envmap_path'][sample_idx_batch]})

                            pickle_save_path = Path(opt.summary_vis_path_task) / ('results_emitter_input_%d.pickle'%sample_idx)
                            # normalPred_lightAccu (3, 240, 320)
                            # depthPred_lightAccu (240, 320)
                            # envmapsPredImage_lightAccu (3, 120, 160, 8, 16)
                            if opt.if_save_pickles:
                                with open(str(pickle_save_path),"wb") as f:
                                    pickle.dump(emitter_input_dict, f)


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

                    if not opt.test_real:
                        summary_cat = torch.cat([mats_pred_vis, mats_gt_vis], 1).permute(1, 2, 0)
                    else:
                        summary_cat = mats_pred_vis.permute(1, 2, 0) # not showing GT if GT label is 0
                    
                    if opt.is_master:
                        writer.add_image('VAL_matcls_PRED-GT/%d'%(sample_idx), summary_cat, tid, dataformats='HWC')
                        writer.add_image('VAL_matcls_matmask/%d'%(sample_idx), mat_mask.squeeze(), tid, dataformats='HW')
                        im_single = data_batch['im_SDR_RGB'][sample_idx_batch].detach().cpu()
                        mat_mask = mat_mask.permute(1, 2, 0).cpu().float()
                        matmask_overlay = im_single * mat_mask + im_single * 0.2 * (1. - mat_mask)
                        writer.add_image('VAL_matcls_matmask-overlay/%d'%(sample_idx), matmask_overlay, tid, dataformats='HWC')
                        if opt.test_real and opt.cfg.MODEL_MATCLS.enable:
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
                        if opt.cfg.MODEL_MATSEG.albedo_pooling_debug and not opt.if_cluster:
                            np.save('tmp/demo_%s/matseg_pred_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), matAggreMap_pred_single_vis)

                    logger.info('Vis batch for %.2f seconds.'%(time.time()-ts_start_vis))

            
            # ===== Vis BRDF 1/2
            # print(((input_dict['albedoBatch'] ) ** (1.0/2.2) ).data.shape) # [b, 3, h, w]
            if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
                # if opt.is_master:
                    # print(input_dict.keys())
                im_paths_list.append(input_dict['im_paths'])
                # print(input_dict['im_paths'])

                imBatch_list.append(input_dict['imBatch'])
                segAllBatch_list.append(input_dict['segAllBatch'])

                if opt.cascadeLevel > 0:
                    diffusePreBatch_list.append(input_dict['pre_batch_dict_brdf']['diffusePreBatch'])
                    specularPreBatch_list.append(input_dict['pre_batch_dict_brdf']['specularPreBatch'])
                    renderedImBatch_list.append(input_dict['pre_batch_dict_brdf']['renderedImBatch'])
                n = 0
                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    albedoBatch_list.append(input_dict['albedoBatch'])
                    albedoPreds_list.append(output_dict['albedoPreds'][n])
                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normalBatch_list.append(input_dict['normalBatch'])
                    normalPreds_list.append(output_dict['normalPreds'][n])
                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    roughBatch_list.append(input_dict['roughBatch'])
                    roughPreds_list.append(output_dict['roughPreds'][n])
                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depthBatch_list.append(input_dict['depthBatch'])
                    depthPreds_list.append(output_dict['depthPreds'][n])

            # ===== LIGHT
            if opt.cfg.MODEL_LIGHT.enable:
                envmapsPredScaledImage = output_dict['envmapsPredScaledImage'].detach().cpu().numpy()
                envmapsBatch = input_dict['envmapsBatch'].detach().cpu().numpy()
                renderedImPred = output_dict['renderedImPred'].detach().cpu().numpy()
                renderedImPred_sdr = output_dict['renderedImPred_sdr'].detach().cpu().numpy()
                imBatchSmall = output_dict['imBatchSmall'].detach().cpu().numpy()
                for sample_idx_batch in range(envmapsPredScaledImage.shape[0]):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    # assert envmapsPredScaledImage.shape[0] == batch_size
                    for I_hdr, name_tag in zip([envmapsPredScaledImage[sample_idx_batch], envmapsBatch[sample_idx_batch]], ['light_Pred', 'light_GT']):
                        H_grid, W_grid, h, w = I_hdr.shape[1:]
                        downsize_ratio = 4
                        assert H_grid % downsize_ratio == 0
                        assert W_grid % downsize_ratio == 0
                        xx, yy = np.meshgrid(np.arange(0, H_grid, downsize_ratio), np.arange(0, W_grid, downsize_ratio))
                        I_hdr_downsampled = I_hdr[:, xx.T, yy.T, :, :]
                        I_hdr_downsampled = I_hdr_downsampled.transpose(1, 3, 2, 4, 0).reshape(H_grid*h//downsize_ratio, W_grid*w//downsize_ratio, 3)
                        cv2.imwrite('{0}/{1}-{2}_{3}.hdr'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag) , I_hdr_downsampled[:, :, [2, 1, 0]])

                    for I_png, name_tag in zip([renderedImPred[sample_idx_batch], renderedImPred_sdr[sample_idx_batch], imBatchSmall[sample_idx_batch]], ['renderedIm_Pred', 'renderedImPred_sdr', 'imBatchSmall_GT']):
                        I_png = (I_png.transpose(1, 2, 0) * 255.).astype(np.uint8)
                        if opt.is_master:
                            writer.add_image('VAL_light-%s/%d'%(name_tag, sample_idx), I_png, tid, dataformats='HWC')
                        Image.fromarray(I_png).save('{0}/{1}-{2}_light-{3}.png'.format(opt.summary_vis_path_task, tid, sample_idx, name_tag))
        
            synchronize()

    synchronize()

    # ===== Vis BRDF 2/2
    # ===== logging top N to TB
    if opt.cfg.MODEL_BRDF.enable and opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        im_paths_list = flatten_list(im_paths_list)
        # print(im_paths_list)

        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedoBatch_vis = torch.cat(albedoBatch_list)
            albedoPreds_vis = torch.cat(albedoPreds_list)
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normalBatch_vis = torch.cat(normalBatch_list)
            normalPreds_vis = torch.cat(normalPreds_list)
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            roughBatch_vis = torch.cat(roughBatch_list)
            roughPreds_vis = torch.cat(roughPreds_list)
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthBatch_vis = torch.cat(depthBatch_list)
            depthPreds_vis = torch.cat(depthPreds_list)

        imBatch_vis = torch.cat(imBatch_list)
        segAllBatch_vis = torch.cat(segAllBatch_list)

        if opt.cascadeLevel > 0:
            diffusePreBatch_vis = torch.cat(diffusePreBatch_list)
            specularPreBatch_vis = torch.cat(specularPreBatch_list)
            renderedImBatch_vis = torch.cat(renderedImBatch_list)


        print('Saving vis to ', '{0}'.format(opt.summary_vis_path_task, tid))
        # Save the ground truth and the input
        # albedoBatch_vis_sdr = ( (albedoBatch_vis ) ** (1.0/2.2) ).data # torch.Size([16, 3, 192, 256]) 
        im_batch_vis_sdr = ( (imBatch_vis)**(1.0/2.2) ).data

        # ==== GTs
        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_gt_batch_vis_sdr = ( (albedoBatch_vis ) ** (1.0/2.2) ).data
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_gt_batch_vis_sdr = (0.5*(normalBatch_vis + 1) ).data
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_gt_batch_vis_sdr = (0.5*(roughBatch_vis + 1) ).data
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthOut = 1 / torch.clamp(depthBatch_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthBatch_vis)
            depth_gt_batch_vis_sdr = ( depthOut*segAllBatch_vis.expand_as(depthBatch_vis) ).data

        if not opt.test_real and opt.is_master:
            vutils.save_image(im_batch_vis_sdr,
                    '{0}/{1}_im.png'.format(opt.summary_vis_path_task, tid) )
            if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                vutils.save_image(albedo_gt_batch_vis_sdr,
                    '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                vutils.save_image(normal_gt_batch_vis_sdr,
                    '{0}/{1}_normalGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                vutils.save_image(rough_gt_batch_vis_sdr,
                    '{0}/{1}_roughGt.png'.format(opt.summary_vis_path_task, tid) )
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                vutils.save_image(depth_gt_batch_vis_sdr,
                    '{0}/{1}_depthGt.png'.format(opt.summary_vis_path_task, tid) )

        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_gt_batch_vis_sdr_numpy = albedo_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_gt_batch_vis_sdr_numpy = normal_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_gt_batch_vis_sdr_numpy = rough_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depth_gt_batch_vis_sdr_numpy = depth_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        # print('++++', rough_gt_batch_vis_sdr_numpy.shape, depth_gt_batch_vis_sdr_numpy.shape, albedo_gt_batch_vis_sdr_numpy.shape, albedo_gt_batch_vis_sdr_numpy.dtype)
        # print(np.amax(albedo_gt_batch_vis_sdr_numpy), np.amin(albedo_gt_batch_vis_sdr_numpy), np.mean(albedo_gt_batch_vis_sdr_numpy))
        depth_min_and_scale_list = []
        if not opt.test_real and opt.is_master:
            for sample_idx in range(im_batch_vis_sdr.shape[0]):
                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-albedo_GT/%d'%sample_idx, albedo_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-normal_GT/%d'%sample_idx, normal_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-rough_GT/%d'%sample_idx, rough_gt_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_normalized, depth_min_and_scale = vis_disp_colormap(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True)
                    # ic('--> GT', np.amax(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze()), np.amin(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze()), np.median(depth_gt_batch_vis_sdr_numpy[sample_idx].squeeze()))
                    depth_min_and_scale_list.append(depth_min_and_scale)
                    writer.add_image('VAL_brdf-depth_GT/%d'%sample_idx, depth_normalized, tid, dataformats='HWC')
                    # ic('----gt', depth_min_and_scale, np.amax(depth_normalized), np.amin(depth_normalized), np.median(depth_normalized))



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
        #     vutils.save_image( ( 0.5*(output_dict['normalPreds'][n] + 1) ).data,
        #             '{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
        # for n in range(0, len(output_dict['roughPreds']) ):
        #     vutils.save_image( ( 0.5*(output_dict['roughPreds'][n] + 1) ).data,
        #             '{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
        # for n in range(0, len(output_dict['depthPreds']) ):
        #     depthOut = 1 / torch.clamp(output_dict['depthPreds'][n] + 1, 1e-6, 10) * segAllBatch_vis.expand_as(output_dict['depthPreds'][n])
        #     vutils.save_image( ( depthOut * segAllBatch_vis.expand_as(output_dict['depthPreds'][n]) ).data,
        #             '{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        # ==== Preds
        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr = ( (albedoPreds_vis ) ** (1.0/2.2) ).data
            if opt.is_master:
                vutils.save_image(albedo_pred_batch_vis_sdr,
                        '{0}/{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr = ( 0.5*(normalPreds_vis + 1) ).data
            if opt.is_master:
                vutils.save_image(normal_pred_batch_vis_sdr,
                        '{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr = ( 0.5*(roughPreds_vis + 1) ).data
            if opt.is_master:
                vutils.save_image(rough_pred_batch_vis_sdr,
                        '{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            # print(torch.min(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.max(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.mean(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.median(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)))
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthPreds_vis)
            depth_pred_batch_vis_sdr = ( depthOut * segAllBatch_vis.expand_as(depthPreds_vis) ).data
            if opt.is_master:
                vutils.save_image(depth_pred_batch_vis_sdr,
                        '{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

        if 'al' in opt.cfg.MODEL_BRDF.enable_list:
            albedo_pred_batch_vis_sdr_numpy = albedo_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'no' in opt.cfg.MODEL_BRDF.enable_list:
            normal_pred_batch_vis_sdr_numpy = normal_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
            rough_pred_batch_vis_sdr_numpy = rough_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depth_pred_batch_vis_sdr_numpy = depth_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
        if opt.is_master:
            for sample_idx in range(im_batch_vis_sdr.shape[0]):
                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-albedo_PRED/%d'%sample_idx, albedo_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-normal_PRED/%d'%sample_idx, normal_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-rough_PRED/%d'%sample_idx, rough_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_normalized, _ = vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, min_scale=depth_min_and_scale_list[sample_idx])
                    # ic('--> EST', np.amax(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze()), np.amin(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze()), np.median(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze()))
                    # ic(np.amax(depth_normalized), np.amin(depth_normalized), np.median(depth_normalized))
                    writer.add_image('VAL_brdf-depth_syncScale_PRED/%d'%sample_idx, depth_normalized, tid, dataformats='HWC')
                    writer.add_image('VAL_brdf-depth_PRED/%d'%sample_idx, vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True)[0], tid, dataformats='HWC')
                    pickle_save_path = Path(opt.summary_vis_path_task) / ('results_depth_%d.pickle'%sample_idx)
                    save_dict = {'depthPreds_vis': depthPreds_vis[sample_idx].detach().cpu().squeeze().numpy()}
                    if opt.if_save_pickles:
                        with open(str(pickle_save_path),"wb") as f:
                            pickle.dump(save_dict, f)
                            # ic(str(pickle_save_path))


    logger.info(red('Evaluation VIS timings: ' + time_meters_to_string(time_meters)))

    # opt.albedo_pooling_debug = False

    synchronize()
    opt.if_vis_debug_pac = False
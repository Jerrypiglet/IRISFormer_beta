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

from train_funcs_matseg import get_labels_dict_matseg, postprocess_matseg, val_epoch_matseg
from train_funcs_semseg import get_labels_dict_semseg, postprocess_semseg
from train_funcs_brdf import get_labels_dict_brdf, postprocess_brdf
from train_funcs_light import get_labels_dict_light, postprocess_light
from train_funcs_layout_object_emitter import get_labels_dict_layout_emitter, postprocess_layout_object_emitter
from train_funcs_matcls import get_labels_dict_matcls, postprocess_matcls
from train_funcs_detectron import postprocess_detectron, gather_lists
# from utils.comm import synchronize

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
    time_meters['loss_detectron'] = AverageMeter()
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

    if opt.cfg.DATA.load_detectron_gt:
        labels_dict_detectron = {'detectron_dict_list': data_batch['detectron_sample_dict']}
    else:
        labels_dict_detectron = {}
    labels_dict.update(labels_dict_detectron)

    # labels_dict = {**labels_dict_matseg, **labels_dict_brdf}
    return labels_dict

def forward_joint(is_train, labels_dict, model, opt, time_meters, if_vis=False, if_loss=True, tid=-1, loss_dict=None):
    # forward model + compute losses

    # Forward model
    output_dict = model(labels_dict)
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

    if opt.cfg.MODEL_LAYOUT_EMITTER.enable:
        output_dict, loss_dict = postprocess_layout_object_emitter(labels_dict, output_dict, loss_dict, opt, time_meters, is_train=is_train, if_vis=if_vis)
        time_meters['loss_layout_emitter'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_MATCLS.enable:
        output_dict, loss_dict = postprocess_matcls(labels_dict, output_dict, loss_dict, opt, time_meters, if_vis=if_vis)
        time_meters['loss_matcls'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
        # synchronize()

    if opt.cfg.MODEL_DETECTRON.enable:
        output_dict, loss_dict = postprocess_detectron(labels_dict, output_dict, loss_dict, opt, time_meters, if_vis=if_vis, is_train=is_train)
        time_meters['loss_detectron'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
        # synchronize()


    return output_dict, loss_dict

def val_epoch_joint(brdf_loader_val, model, params_mis):
    writer, logger, opt, tid, bin_mean_shift = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['bin_mean_shift']
    if_register_detectron_only = params_mis['if_register_detectron_only']
    ENABLE_SEMSEG = opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable
    ENABLE_MATSEG = opt.cfg.MODEL_MATSEG.enable
    ENABLE_BRDF = opt.cfg.MODEL_BRDF.enable and opt.cfg.DATA.load_brdf_gt
    ENABLE_LIGHT = opt.cfg.MODEL_LIGHT.enable
    ENABLE_MATCLS = opt.cfg.MODEL_MATCLS.enable
    if if_register_detectron_only:
        ENABLE_SEMSEG = False
        ENABLE_MATSEG = False
        ENABLE_BRDF = False
        ENABLE_LIGHT = False
        ENABLE_MATCLS = False

    ENABLE_DETECTRON = opt.cfg.MODEL_DETECTRON.enable


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
                        'loss_brdf-normal-bs', 
                        'loss_brdf-rough-bs', 
                        'loss_brdf-depth-bs', 
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

    if opt.cfg.MODEL_DETECTRON.enable:
        loss_keys += [
            'loss_detectron-ALL',
            'loss_detectron-cls', 
            'loss_detectron-box_reg', 
            'loss_detectron-mask', 
            'loss_detectron-rpn_cls', 
            'loss_detectron-rpn_loc', ]
        
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
        if ENABLE_DETECTRON:
            detectron_evaluator = COCOEvaluator(None, ("bbox", "segm"), opt.distributed, output_dir=str(opt.summary_vis_path_task), logger=logger)
            detectron_evaluator.reset()
            coco_dictt_labels = []

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

            print(loss_dict.keys())
            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            # loss = loss_dict['loss_all']
            
            # ======= update loss
            if len(loss_dict_reduced.keys()) != 0:
                for loss_key in loss_dict_reduced:
                    # if loss_dict_reduced[loss_key] != 0:
                    print(loss_key, loss_meters.keys(), loss_dict_reduced.keys())
                    loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= Metering
            if ENABLE_LIGHT:
                pass

            if ENABLE_BRDF:
                frame_info_list = input_dict['frame_info']
                if opt.cfg.DEBUG.dump_BRDF_offline.enable:
                    if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                        albedo_output = output_dict['albedoPreds'][0].detach().cpu().numpy()
                    if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                        rough_output = output_dict['roughPreds'][0].detach().cpu().numpy()
                    if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                        normal_output = output_dict['normalPreds'][0].detach().cpu().numpy()
                    if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                        depth_output = output_dict['depthPreds'][0].detach().cpu().numpy()

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
                            albedo_dump_path = scene_path_dump / ('imbaseColor_%d_dump.png'%frame_id)
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
                            depth_dump_path = scene_path_dump / ('imdepth_%d_dump.pickle'%frame_id)
                            depth_dump_path = Path(str(depth_dump_path).replace('DiffLight', '').replace('DiffMat', ''))
                            depth_save = depth_output[sample_idx_batch].squeeze().astype(np.float32)
                            with open(str(depth_dump_path),"wb") as f:
                                pickle.dump({'depth_pred': depth_save}, f)
                            print(depth_dump_path)
                            print(frame_info['depth_path'])

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

            if ENABLE_DETECTRON:
                detectron_dict_list, output_detectron = input_dict['detectron_dict_list'], output_dict['output_detectron']
                if opt.distributed:
                    detectron_dict_list_gathered = gather_lists(detectron_dict_list, opt.num_gpus)
                #     output_detectron = gather_lists(output_detectron, opt.num_gpus)
                detectron_evaluator.process(detectron_dict_list, output_detectron)
                coco_dictt_labels += detectron_dict_list_gathered
                # print('----', opt.rank, [x['image_id'] for x in input_dict['detectron_dict_list']])


            # synchronize()

    # ======= Metering

    if ENABLE_DETECTRON:
        # if opt.distributed:
        #     coco_dictt_labels = gather_lists(coco_dictt_labels, opt.num_gpus)
        #     # print(len(coco_dictt_labels), '<<<<<<<<<<', opt.rank)
        #     coco_dictt_labels_allgather = [None for _ in range(opt.num_gpus)]
        #     dist.all_gather_object(coco_dictt_labels_allgather, coco_dictt_labels)
        #     # print(len(coco_dictt_labels_allgather), len(coco_dictt_labels_allgather[0]), '<<<<<<<<<<-------', opt.rank)
        #     coco_dictt_labels = [item for sublist in coco_dictt_labels_allgather for item in sublist]
        pickle_path = Path(opt.summary_vis_path_task) / ('OR_detectron_%s.pth'%params_mis['detectron_dataset_name'])
        if opt.is_master:
            # --- dump processed labels for the entire val set for use in detectron_evaluator.evaluate()
            if not pickle_path.exists():
                torch.save(coco_dictt_labels, str(pickle_path))
                logger.info(green('[Detectron] Dumped detectron pickle: %s'%str(pickle_path)))
        # synchronize()

        coco_dictt_labels = torch.load(str(pickle_path))
        # brdf_dataset_val.run(OR_detectron_path)
        # val_dict = brdf_dataset_val.dict
        detectron_dataset_name = "OR_detectron_" + params_mis['detectron_dataset_name']
        if detectron_dataset_name not in DatasetCatalog.list():
            DatasetCatalog.register(detectron_dataset_name, lambda d=params_mis['detectron_dataset_name']:coco_dictt_labels)
            assert len(opt.OR_classes)==opt.cfg_detectron.MODEL.ROI_HEADS.NUM_CLASSES
            MetadataCatalog.get(detectron_dataset_name).set(thing_classes=opt.OR_classes)
            logger.info(green('[Detectron] Registered dataCatalog: %s'%detectron_dataset_name))

        # --- eval with the dumped labels
        # opt.OR_detectron_metadata_val = MetadataCatalog.get("OR_detectron_val")
        if if_register_detectron_only:
            return
        detectron_evaluator.init_dataset(detectron_dataset_name)
        detectron_results = detectron_evaluator.evaluate()
        if detectron_results != {}:
            for task_idx, task_name in enumerate(['bbox', 'segm']):
                dict_items = list(detectron_results.items())[task_idx]
                assert dict_items[0]==task_name
                for metric in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']:
                    if opt.is_master:
                        writer.add_scalar('VAL/DETECTRON-%s-%s_val'%(task_name, metric), dict_items[1][metric], tid)
        # writer.add_scalar('VAL/DETECTRON-mAcc_val', mAcc, tid)
        # writer.add_scalar('VAL/DETECTRON-allAcc_val', allAcc, tid)
        # synchronize()

        
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

    # synchronize()
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

        albedoBsPreds_list = []


    if opt.cfg.MODEL_GMM.enable:
        output_GMM_Q_list = []
    
    if opt.cfg.MODEL_MATCLS.enable:
        matG1IdDict = getG1IdDict(opt.cfg.PATH.matcls_matIdG1_path)


    # opt.albedo_pooling_debug = True

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

                    if opt.cfg.MODEL_MATSEG.albedo_pooling_debug and not opt.if_cluster:
                        os.makedirs('tmp/demo_%s'%(opt.task_name), exist_ok=True)
                        np.save('tmp/demo_%s/im_trainval_SDR_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), im_single)
                        print('Saved to' + 'tmp/demo_%s/im_trainval_SDR_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx))

                    writer.add_text('VAL_image_name/%d'%(sample_idx), im_path, tid)
                    assert sample_idx == data_batch['image_index'][sample_idx_batch]
                    # print(sample_idx, im_path)

                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_im_path = real_sample_results_path / 'im_.png'
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


                            
                if (opt.cfg.MODEL_MATSEG.if_albedo_pooling or opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or opt.cfg.MODEL_MATSEG.if_albedo_pac_pool or opt.cfg.MODEL_MATSEG.if_albedo_safenet) and opt.cfg.MODEL_MATSEG.albedo_pooling_debug:
                    if opt.is_master:
                        if output_dict['im_trainval_SDR_mask_pooled_mean'] is not None:
                            im_trainval_SDR_mask_pooled_mean = output_dict['im_trainval_SDR_mask_pooled_mean'][sample_idx_batch]
                            im_trainval_SDR_mask_pooled_mean = im_trainval_SDR_mask_pooled_mean.cpu().numpy().squeeze().transpose(1, 2, 0)
                            writer.add_image('VAL_im_trainval_SDR_mask_pooled_mean/%d'%(sample_idx), im_trainval_SDR_mask_pooled_mean, tid, dataformats='HWC')
                        if not opt.if_cluster:
                            if 'kernel_list' in output_dict and not output_dict['kernel_list'] is None:
                                kernel_list = output_dict['kernel_list']
                                # print(len(kernel_list), kernel_list[0].shape)
                                np.save('tmp/demo_%s/kernel_list_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), kernel_list[0].detach().cpu().numpy())
                            if output_dict['im_trainval_SDR_mask_pooled_mean'] is not None:
                                np.save('tmp/demo_%s/im_trainval_SDR_mask_pooled_mean_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), im_trainval_SDR_mask_pooled_mean)
                            if 'embeddings' in output_dict and output_dict['embeddings'] is not None:
                                np.save('tmp/demo_%s/embeddings_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), output_dict['embeddings'].detach().cpu().numpy())
                            if 'affinity' in output_dict:
                                affinity = output_dict['affinity']
                                sample_ij = output_dict['sample_ij']
                                if affinity is not None:
                                    np.save('tmp/demo_%s/affinity_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), affinity[0].detach().cpu().numpy())
                                    np.save('tmp/demo_%s/sample_ij_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), sample_ij)

            # === BRDF feat_ssn superpixel segmentation
            # print(output_dict['encoder_outputs']['brdf_extra_output_dict'].keys(), output_dict['albedo_extra_output_dict'].keys()) # dict_keys(['x1_affinity', 'x2_affinity', 'x3_affinity']) dict_keys(['dx3_affinity', 'dx4_affinity', 'dx5_affinity'])
            if opt.cfg.MODEL_GMM.enable and opt.cfg.MODEL_GMM.feat_recon.enable:
                for encoder_key in ['x1', 'x2', 'x3']:
                    if encoder_key in opt.cfg.MODEL_GMM.feat_recon.layers_list:
                        affinity_matrix = output_dict['encoder_outputs']['brdf_extra_output_dict']['%s_affinity'%encoder_key]
                        affinity_matrix_label = torch.argmax(affinity_matrix, 1).detach().cpu().numpy()

                        for sample_idx_batch, affinity_matrix_label_single in enumerate(affinity_matrix_label):
                            sample_idx = sample_idx_batch+batch_size*batch_id
                            if sample_idx >= opt.cfg.TEST.vis_max_samples:
                                break
                            
                            im_single_resized = scikit_resize(im_single_list[sample_idx], (affinity_matrix_label_single.shape[0], affinity_matrix_label_single.shape[1]))

                            if opt.is_master:
                                im_single_ssn_result = mark_boundaries(im_single_resized, affinity_matrix_label_single)
                                writer.add_image('VAL_GMM_encoder_SSN_%s/%d'%(encoder_key, sample_idx), im_single_ssn_result, tid, dataformats='HWC')

                for mode in opt.cfg.MODEL_GMM.appearance_recon.modalities:
                    modality = {'al': 'albedo', 'ro': 'rough'}[mode]
                    if not '%s_extra_output_dict'%modality in output_dict:
                        continue

                    for decoder_key in ['dx3', 'dx4', 'dx5']:
                        if decoder_key in opt.cfg.MODEL_GMM.feat_recon.layers_list:
                            affinity_matrix = output_dict['%s_extra_output_dict'%modality]['%s_affinity'%decoder_key]
                            affinity_matrix_label = torch.argmax(affinity_matrix, 1).detach().cpu().numpy()

                            for sample_idx_batch, affinity_matrix_label_single in enumerate(affinity_matrix_label):
                                sample_idx = sample_idx_batch+batch_size*batch_id
                                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                                    break

                                im_single_resized = scikit_resize(im_single_list[sample_idx], (affinity_matrix_label_single.shape[0], affinity_matrix_label_single.shape[1]))

                                if opt.is_master:
                                    im_single_ssn_result = mark_boundaries(im_single_resized, affinity_matrix_label_single)
                                    writer.add_image('VAL_GMM_decoder-%s_SSN_%s/%d'%(modality, decoder_key, sample_idx), im_single_ssn_result, tid, dataformats='HWC')

            # === BRDF-DPT superpixel segmentation
            if opt.cfg.MODEL_BRDF.DPT_baseline.enable:
                mode = opt.cfg.MODEL_BRDF.DPT_baseline.modality
                assert mode=='enabled'
                # modality = {'al': 'albedo', 'de': 'depth'}[mode]
                for modality in opt.cfg.MODEL_BRDF.enable_list:
                    if not '%s_extra_output_dict'%modality in output_dict:
                        pass
                    else:
                        if opt.cfg.MODEL_BRDF.DPT_baseline.model == 'dpt_hybrid_SSN' or opt.cfg.MODEL_BRDF.DPT_baseline.if_vis_CA_SSN_affinity:
                            decoder_key = 'matseg'
                            affinity_matrix = output_dict['%s_extra_output_dict'%modality]['%s_affinity'%decoder_key]
                            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_ssn_matseg_on_lower_res:
                                affinity_matrix = F.interpolate(affinity_matrix, scale_factor=4., mode='bilinear')
                            # print(affinity_matrix.shape, affinity_matrix[0].sum(-1).sum(-1)) # should be either normalized by J, or RAW dist matrix
                            affinity_matrix_label = torch.argmax(affinity_matrix, 1).detach().cpu().numpy()

                            for sample_idx_batch, affinity_matrix_label_single in enumerate(affinity_matrix_label):
                                sample_idx = sample_idx_batch+batch_size*batch_id
                                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                                    break

                                im_single_resized = scikit_resize(im_single_list[sample_idx], (affinity_matrix_label_single.shape[0], affinity_matrix_label_single.shape[1]))

                                if opt.is_master:
                                    im_single_ssn_result = mark_boundaries(im_single_resized, affinity_matrix_label_single)
                                    writer.add_image('VAL_DPT-SSN_%s_SSN_%s/%d'%(modality, decoder_key, sample_idx), im_single_ssn_result, tid, dataformats='HWC')


                if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_unet_backbone and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.if_debug_unet:
                    assert 'albedo_pred_unet' in output_dict['albedo_extra_output_dict']
                    albedo_pred_unet = output_dict['albedo_extra_output_dict']['albedo_pred_unet'].cpu().numpy().transpose(0, 2, 3, 1)
                    for sample_idx_batch, albedo_pred_unet_single in enumerate(albedo_pred_unet):
                        sample_idx = sample_idx_batch+batch_size*batch_id
                        if opt.is_master:
                            writer.add_image('VAL_DPT-SSN_albedo_PRED/%d'%sample_idx, albedo_pred_unet_single, tid, dataformats='HWC')

                if opt.cfg.MODEL_BRDF.DPT_baseline.if_vis_CA_proj_coef:
                    assert 'proj_coef_dict' in output_dict['albedo_extra_output_dict']
                    assert 'hooks' in output_dict['albedo_extra_output_dict']
                    proj_coef_dict = output_dict['albedo_extra_output_dict']['proj_coef_dict']
                    hooks = output_dict['albedo_extra_output_dict']['hooks'] if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers == -1 else range(opt.cfg.MODEL_BRDF.DPT_baseline.dpt_SSN.keep_N_layers)
                    start_im_hw = [256//4, 320//4]
                    patch_size = opt.cfg.MODEL_BRDF.DPT_baseline.patch_size
                    spixel_hw = [256//patch_size, 320//patch_size]
                    # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_ssn_matseg_on_lower_res:
                    #     spixel_hw = [spixel_hw[0]//4, spixel_hw[1]//4]
                    head = 0
                    for idx, hook in enumerate(hooks):
                        proj_coef_matrix = proj_coef_dict['proj_coef_%d'%hook].detach() # torch.Size([1, 2, 5120, 320])
                        # print(idx, proj_coef_matrix[0, 0, :5, 0])
                        # print(idx, proj_coef_matrix[0, 0, :5, -1])
                        # print(proj_coef_matrix.shape, np.min(proj_coef_matrix.cpu().numpy()), np.max(proj_coef_matrix.cpu().numpy()), np.median(proj_coef_matrix.cpu().numpy()))
                        if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg_if_inject_proj_coef:
                            proj_coef_matrix = (proj_coef_matrix / (proj_coef_matrix.sum(2, keepdims=True)+1e-8)).cpu().numpy() # softmax in pixels dim
                        else:
                            proj_coef_matrix = F.softmax(proj_coef_matrix, dim=2).cpu().numpy() # softmax in pixels dim

                        # print(proj_coef_matrix.shape, np.min(proj_coef_matrix), np.max(proj_coef_matrix), np.median(proj_coef_matrix))
                        for sample_idx_batch, proj_coef_matrix_single in enumerate(proj_coef_matrix):
                            sample_idx = sample_idx_batch+batch_size*batch_id
                            if opt.is_master:
                                proj_coef_matrix_single_vis = proj_coef_matrix_single[head].reshape(start_im_hw[0], start_im_hw[1], spixel_hw[0], spixel_hw[1])
                                # print('>>>>', idx, sample_idx_batch, proj_coef_matrix_single_vis[0, :5, 0, 0])
                                # print('>>>>', idx, sample_idx_batch, proj_coef_matrix_single_vis[0, :5, -1, -1])
                                a_list = []
                                if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_SSN and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg:
                                    for spixel_h in [0, 1]:
                                        for spixel_w in [0, 1]:
                                            a_list.append([spixel_h, spixel_w])
                                else:
                                    for spixel_h in [spixel_hw[0]//3, spixel_hw[0]//3*2, spixel_hw[0]-1]:
                                        for spixel_w in [spixel_hw[1]//3, spixel_hw[1]//3*2, spixel_hw[1]-1]:
                                            a_list.append([spixel_h, spixel_w])

                                for spixel_h, spixel_w in a_list:
                                    # print(spixel_h, spixel_w, proj_coef_matrix_single_vis.shape)
                                    proj_coef_matrix_single_token_vis = proj_coef_matrix_single_vis[:, :, spixel_h, spixel_w]
                                    # print(proj_coef_matrix_single_token_vis.shape, np.sum(proj_coef_matrix_single_token_vis))
                                    print(spixel_h, spixel_w, hook, np.min(proj_coef_matrix_single_token_vis), np.max(proj_coef_matrix_single_token_vis), np.median(proj_coef_matrix_single_token_vis), np.amax(proj_coef_matrix_single_vis))
                                    # proj_coef_matrix_single_token_vis = proj_coef_matrix_single_token_vis - np.amin(proj_coef_matrix_single_token_vis)
                                    # proj_coef_matrix_single_token_vis = proj_coef_matrix_single_token_vis / (np.amax(proj_coef_matrix_single_vis)+1e-6)
                                    # proj_coef_matrix_single_token_vis = proj_coef_matrix_single_token_vis / (np.sum(proj_coef_matrix_single_token_vis) + 1e-6)
                                    proj_coef_matrix_single_token_vis = np.clip(proj_coef_matrix_single_token_vis * start_im_hw[0] * start_im_hw[1] / 10., 0., 1.)
                                    proj_coef_matrix_single_token_vis = cv2.resize(proj_coef_matrix_single_token_vis, dsize=(320, 256), interpolation=cv2.INTER_NEAREST)
                                    print('->', np.min(proj_coef_matrix_single_token_vis), np.max(proj_coef_matrix_single_token_vis), np.median(proj_coef_matrix_single_token_vis))

                                    writer.add_image('VAL_DPT-CA_proj_coef_sample%d/head%d_spixel(%d)%d-%d_PRED/%d'%(sample_idx, head, spixel_h*spixel_hw[1]+spixel_w, spixel_h*patch_size, spixel_w*patch_size, hook), \
                                        vis_disp_colormap(proj_coef_matrix_single_token_vis, normalize=False, cmap_name='viridis')[0], tid, dataformats='HWC')

                        if not opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_not_reduce_res:
                            start_im_hw = start_im_hw[0]//2, start_im_hw[1]//2


                if opt.cfg.MODEL_BRDF.DPT_baseline.if_vis_CA_SSN_affinity:
                    assert 'abs_affinity_normalized_by_pixels' in output_dict['albedo_extra_output_dict']
                    abs_affinity_normalized_by_pixels_input = output_dict['albedo_extra_output_dict']['abs_affinity_normalized_by_pixels'] # torch.Size([1, 320, 64, 80])
                    patch_size = opt.cfg.MODEL_BRDF.DPT_baseline.patch_size
                    spixel_hw = [256//patch_size, 320//patch_size]
                    # if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_ssn_matseg_on_lower_res:
                    #     spixel_hw = [spixel_hw[0]//4, spixel_hw[1]//4]

                    abs_affinity_normalized_by_pixels_input = abs_affinity_normalized_by_pixels_input.view(-1, spixel_hw[0], spixel_hw[1], abs_affinity_normalized_by_pixels_input.shape[-2], abs_affinity_normalized_by_pixels_input.shape[-1]).detach().cpu().numpy()
                    for sample_idx_batch, abs_affinity_normalized_by_pixels_input_single in enumerate(abs_affinity_normalized_by_pixels_input):
                        sample_idx = sample_idx_batch+batch_size*batch_id
                        if opt.is_master:
                            a_list = []
                            if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_SSN and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg:
                                for spixel_h in [0, 1]:
                                    for spixel_w in [0, 1]:
                                        a_list.append([spixel_h, spixel_w])
                            else:
                                for spixel_h in [spixel_hw[0]//3, spixel_hw[0]//3*2, spixel_hw[0]-1]:
                                    for spixel_w in [spixel_hw[1]//3, spixel_hw[1]//3*2, spixel_hw[1]-1]:
                                        a_list.append([spixel_h, spixel_w])

                            for spixel_h, spixel_w in a_list:
                                abs_affinity_normalized_by_pixels_input_vis = abs_affinity_normalized_by_pixels_input_single[spixel_h, spixel_w, :, :]
                                # print(abs_affinity_normalized_by_pixels_input.shape, abs_affinity_normalized_by_pixels_input_vis.shape)
                                writer.add_histogram('VAL_hist_DPT-SSN_abs_affinity_normalized_by_pixels_sample%d/head%d_spixel(%d)%d-%d_PRED'%(sample_idx, head, spixel_h*spixel_hw[1]+spixel_w, spixel_h*patch_size, spixel_w*patch_size), \
                                    abs_affinity_normalized_by_pixels_input_vis, tid)

                                # print(np.sum(abs_affinity_normalized_by_pixels_input_vis), '----------')
                                # abs_affinity_normalized_by_pixels_input_vis = abs_affinity_normalized_by_pixels_input_vis / (np.amax(abs_affinity_normalized_by_pixels_input_vis)+1e-6)
                                abs_affinity_normalized_by_pixels_input_vis = np.clip(abs_affinity_normalized_by_pixels_input_vis * 256 * 320 / 10., 0., 1.)
                                writer.add_image('VAL_DPT-SSN_abs_affinity_normalized_by_pixels_sample%d/head%d_spixel(%d)%d-%d_PRED'%(sample_idx, head, spixel_h*spixel_hw[1]+spixel_w, spixel_h*patch_size, spixel_w*patch_size), \
                                    vis_disp_colormap(abs_affinity_normalized_by_pixels_input_vis, normalize=False, cmap_name='viridis')[0], tid, dataformats='HWC')

                if opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.if_use_SSN and opt.cfg.MODEL_BRDF.DPT_baseline.dpt_hybrid.CA.SSN.if_gt_matseg:
                    assert 'num_mat_masks' in output_dict['albedo_extra_output_dict'] and 'instance' in output_dict['albedo_extra_output_dict']
                    for sample_idx_batch, abs_affinity_normalized_by_pixels_input_single in enumerate(abs_affinity_normalized_by_pixels_input):
                        sample_idx = sample_idx_batch+batch_size*batch_id
                        if opt.is_master:
                            num_mat_masks = output_dict['albedo_extra_output_dict']['num_mat_masks'][sample_idx_batch].item()
                            instance = output_dict['albedo_extra_output_dict']['instance'][sample_idx_batch].detach().cpu().numpy()
                            # print(instance.shape, instance.sum(-1).sum(-1), instance.dtype)
                            for mask_idx in range(num_mat_masks+1):
                                writer.add_image('VAL_DPT-SSN_gt_matseg_instance_sample%d/%d'%(sample_idx, mask_idx), \
                                    instance[mask_idx].astype(np.float32), tid, dataformats='HW')

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

            # ======= Vis Detectron2
            if opt.cfg.MODEL_DETECTRON.enable:
                detectron_dataset_name = "OR_detectron_" + params_mis['detectron_dataset_name']
                if detectron_dataset_name not in DatasetCatalog.list():
                    with torch.no_grad():
                        params_mis['if_register_detectron_only'] = True
                        val_epoch_joint(brdf_loader_val, model, params_mis)
                        params_mis['if_register_detectron_only'] = False
                for sample_idx_batch, detectron_output_dict in enumerate(output_dict['output_detectron']):
                    sample_idx = sample_idx_batch+batch_size*batch_id
                    if opt.is_master:
                        print('[detectron] Visualizing %d sample ...'%sample_idx, batch_id, sample_idx_batch)
                    output_dict = detectron_output_dict['instances']._fields
                    # put all the predction of the current image into a list, them apply NMS
                    scores=output_dict['scores'].cpu().numpy()
                    mask_pre_list=(output_dict['pred_masks'].cpu().numpy()*255).astype('uint8')[scores > opt.cfg.MODEL_DETECTRON.thresh]
                    scores=scores[scores > opt.cfg.MODEL_DETECTRON.thresh]
                    # os.makedirs(osp.join(pred,filesubpath), exist_ok=True)
                    # os.makedirs(osp.join(viz,filesubpath), exist_ok=True)

                    ## visualize the prediction mask
                    if(mask_pre_list.shape[0]==0):
                        mask=np.zeros((opt.cfg.DATA.im_height, opt.cfg.DATA.im_width))
                    else:
                        mask_pre_list=mask_pre_list[py_cpu_nms(mask_pre_list, scores, opt.cfg.MODEL_DETECTRON.nms_thresh)]
                        for mask_idx, mask in enumerate(mask_pre_list):
                            # cv2.imwrite(osp.join(pred, filesubpath,'mask{}.png'.format(index)), mask)
                            if opt.is_master:
                                writer.add_image('VAL_DETECTRON_mask_PRED_%d/%d'%(sample_idx, mask_idx), mask, tid, dataformats='HW')
                        # mask_list=map1[d["file_name"]]

                    im_single = (data_batch['im_fixedscale_SDR'][sample_idx_batch].detach().cpu().numpy().astype(np.float32) * 255.).astype(np.uint8)
                    v = Visualizer(im_single, metadata=MetadataCatalog.get("OR_detectron_val"), scale=1)
                    v_pred = v.draw_instance_predictions(detectron_output_dict["instances"].to("cpu"))
                    # cv2.imwrite(osp.join(viz, filesubpath,'viz.png'),v.get_image()[:, :, ::-1])
                    if opt.is_master:
                        writer.add_image('VAL_DETECTRON_vis_PRED/%d'%(sample_idx), v_pred.get_image(), tid, dataformats='HWC')
                    
                    v = Visualizer(im_single, metadata=MetadataCatalog.get("OR_detectron_val"), scale=1)
                    v_gt = v.draw_dataset_dict(input_dict['detectron_dict_list'][sample_idx_batch])
                    if opt.is_master:
                        writer.add_image('VAL_DETECTRON_vis_GT/%d'%(sample_idx), v_gt.get_image(), tid, dataformats='HWC')


            # ======= Vis layout-emitter
            if opt.cfg.MODEL_LAYOUT_EMITTER.enable:
                output_vis_dict = vis_layout_emitter(input_dict, output_dict, data_batch, opt, time_meters=time_meters, batch_size_id=[batch_size, batch_id])
                # output_dict['output_layout_emitter_vis_dict'] = output_vis_dict
                if_real_image = False
                draw_mode = 'both' if not if_real_image else 'prediction'

                scene_box_list, layout_info_dict_list, emitter_info_dict_list = output_vis_dict['scene_box_list'], output_vis_dict['layout_info_dict_list'], output_vis_dict['emitter_info_dict_list']
                if opt.is_master:
                    logger.info('emitter_layout -------> ' + str(Path(opt.summary_vis_path_task)))

                    if 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                        patch_flattened = data_batch['boxes_batch']['patch'] # [B, 3, ?, ?]
                        patch_batch = [patch_flattened[x[0]:x[1]].numpy() for x in data_batch['obj_split'].cpu().numpy()] # [[?, 3, D, D], [?, 3, D, D], ...]
                        # [x.shape[0] for x in patch_batch], data_batch['obj_split'].cpu().numpy(), patch_flattened.shape)
                        # print([[x[0], x[1]] for x in data_batch['obj_split'].cpu().numpy()])
                        assert sum([x.shape[0] for x in patch_batch])==patch_flattened.shape[0]


                    for sample_idx_batch, (scene_box, layout_info_dict, emitter_info_dict) in enumerate(zip(scene_box_list, layout_info_dict_list, emitter_info_dict_list)):
                        sample_idx = sample_idx_batch+batch_size*batch_id
                        save_prefix = ('results_LABEL_tid%d-%d'%(tid, sample_idx))

                        if 'lo' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                            output_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'layout') + '.png')
                            fig_2d, ax_2d, _ = scene_box.draw_projected_layout(draw_mode, return_plt=True, if_use_plt=True) # with plt plotting
                            # fig_2d.savefig(str(output_path))
                            # plt.close(fig_2d)
                            fig_2d.tight_layout(pad=0)
                            ax_2d.margins(0) # To remove the huge white borders
                            fig_2d.canvas.draw()
                            image_from_plot = np.frombuffer(fig_2d.canvas.tostring_rgb(), dtype=np.uint8)
                            image_from_plot = image_from_plot.reshape(fig_2d.canvas.get_width_height()[::-1] + (3,))
                            writer.add_image('VAL_layout_PRED/%d'%(sample_idx), image_from_plot, tid, dataformats='HWC')

                            
                        pickle_save_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'layout_info') + '.pickle')
                        save_dict = {'rgb_img_path': data_batch['image_path'][sample_idx_batch],  'bins_tensor': opt.bins_tensor}
                        save_dict.update(layout_info_dict)
                        if opt.if_save_pickles:
                            with open(str(pickle_save_path),"wb") as f:
                                pickle.dump(save_dict, f)

                        if 'ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                            output_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'obj') + '.png')
                            fig_2d = scene_box.draw_projected_bdb3d(draw_mode, if_vis_2dbbox=False, return_plt=True, if_use_plt=True)
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

                            fig_3d, _, ax_3ds = scene_box.draw_3D_scene_plt(draw_mode, if_show_objs=True, hide_random_id=False, if_debug=False, hide_cells=True, if_dump_to_mesh=True, if_show_emitter=False, pickle_id=sample_idx)

                            if opt.cfg.MODEL_LAYOUT_EMITTER.mesh.if_use_vtk:
                                im_meshes_GT = scene_box.draw3D('GT', if_return_img=True, if_save_img=False, if_save_obj=False, save_path_without_suffix = 'recon')['im']
                                im_meshes_pred = scene_box.draw3D('prediction', if_return_img=True, if_save_img=False, if_save_obj=False, save_path_without_suffix = 'recon')['im']
                                writer.add_image('VAL_mesh_GT/%d'%(sample_idx), im_meshes_GT, tid, dataformats='HWC')
                                writer.add_image('VAL_mesh_pred/%d'%(sample_idx), im_meshes_pred, tid, dataformats='HWC')



                        if 'em' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list:
                            output_path = Path(opt.summary_vis_path_task) / (save_prefix.replace('LABEL', 'emitter') + '.png')
                            fig_3d, _, ax_3ds = scene_box.draw_3D_scene_plt(draw_mode, if_return_cells_vis_info=True, if_show_emitter=not(if_real_image), if_show_objs='ob' in opt.cfg.MODEL_LAYOUT_EMITTER.enable_list, \
                                if_show_cell_normals=False, if_show_cell_meshgrid=False)
                            # az = 90
                            # elev = 0
                            # ax_3ds[1].view_init(elev=elev, azim=az)
                            # ax_3ds[0].view_init(elev=elev, azim=az)

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
                                fig_3d, _, ax_3d = scene_box.draw_3D_scene_plt('GT', )
                                ax_3d[1] = fig_3d.add_subplot(122, projection='3d')
                                scene_box.draw_3D_scene_plt('GT', fig_or_ax=[ax_3d[1], ax_3d[0]], hide_cells=True)
                                
                                lightAccu_color_array_GT = emitter_info_dict['envmap_lightAccu_mean_vis_GT'].transpose(0, 2, 3, 1) # -> [6, 8, 8, 3]
                                hdr_scale = data_batch['hdr_scale'][sample_idx_batch].item()
                                # lightAccu_color_array_GT = np.clip(lightAccu_color_array_GT * hdr_scale, 0., 1.)
                                lightAccu_color_array_GT = np.clip(lightAccu_color_array_GT, 0., 1.)
                                scene_box.draw_all_cells(ax_3d[1], scene_box.gt_layout, current_type='GT', lightnet_array_GT=lightAccu_color_array_GT, alpha=1.)

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
                                            'depthPred': output_dict['emitter_est_result']['depthPred'].detach().cpu().numpy()[sample_idx_batch], \
                                            'points_backproj': output_dict['emitter_est_result']['points_backproj'].detach().cpu().numpy()[sample_idx_batch]}) \
                                            # 'points': output_dict['emitter_est_result']['points'].detach().cpu().numpy()[sample_idx_batch], \
                                        if opt.cfg.DEBUG.if_dump_anything:
                                            results_emitter_save_dict.update({
                                                'depthGT': input_dict['depthBatch'].detach().cpu().numpy()[sample_idx_batch], \
                                                'normalGT': input_dict['normalBatch'].detach().cpu().numpy()[sample_idx_batch], \
                                                'normalPred': output_dict['normalPred'].detach().cpu().numpy()[sample_idx_batch], \
                                                'albedoGT': input_dict['albedoBatch'].detach().cpu().numpy()[sample_idx_batch], \
                                                'roughGT': input_dict['roughBatch'].detach().cpu().numpy()[sample_idx_batch], \
                                                'albedoPred': output_dict['albedoPred'].detach().cpu().numpy()[sample_idx_batch], \
                                                'roughPred': output_dict['roughPred'].detach().cpu().numpy()[sample_idx_batch], \
                                                'T_LightNet2Total3D_rightmult': output_dict['emitter_est_result']['T_LightNet2Total3D_rightmult'].detach().cpu().numpy()[sample_idx_batch], \
                                                'semseg_label': input_dict['semseg_label_ori'].detach().cpu().numpy()[sample_idx_batch], \
                                                'envmapsBatch': input_dict['envmapsBatch'].detach().cpu().numpy()[sample_idx_batch], \
                                                'envmapsPredScaledImage': output_dict['envmapsPredScaledImage'].detach().cpu().numpy()[sample_idx_batch], \
                                                'envmapsPredImage': output_dict['envmapsPredImage'].detach().cpu().numpy()[sample_idx_batch], \
                                                # 'envmapsPredScaledImage_LightNetCoords': output_dict['emitter_input']['envmapsPredScaledImage_LightNetCoords'].detach().cpu().numpy()[sample_idx_batch], \
                                                # 'envmapsPred_LightNetCoords': output_dict['emitter_input']['envmapsPred_LightNetCoords'].detach().cpu().numpy()[sample_idx_batch], \
                                            })
                                            if opt.cfg.MODEL_LIGHT.if_transform_to_LightNet_coords:
                                                LightNet_misc = {}
                                                for key in output_dict['emitter_misc']['LightNet_misc']:
                                                    LightNet_misc[key] = output_dict['emitter_misc']['LightNet_misc'][key].detach().cpu().numpy()[sample_idx_batch]
                                                results_emitter_save_dict['LightNet_misc'] = LightNet_misc
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
                        if opt.cfg.MODEL_MATSEG.albedo_pooling_debug and not opt.if_cluster:
                            np.save('tmp/demo_%s/matseg_pred_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx), matAggreMap_pred_single_vis)
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
                    albedoPreds_list.append(output_dict['albedoPreds'][n])
                    if opt.cfg.MODEL_BRDF.if_bilateral:
                        albedoBsPreds_list.append(output_dict['albedoBsPred'])
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'al' in opt.cfg.DATA.data_read_list:
                        albedoPreds_aligned_list.append(output_dict['albedoPreds_aligned'][n])
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        assert output_dict['albedoPreds'][n].shape[0] == 1
                        albedoPred_np = output_dict['albedoPreds'][n][0].cpu().numpy()
                        albedo_dump_path = scene_path_dump / ('imbaseColor.png')
                        albedo_sdr = np.clip(albedoPred_np.transpose(1, 2, 0) ** (1.0/2.2), 0., 1.)
                        Image.fromarray((albedo_sdr*255.).astype(np.uint8)).save(str(albedo_dump_path))
                        Image.fromarray((albedo_sdr*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to, :]).save(str(albedo_dump_path).replace('.png','_cropped.png'))
                        print(albedo_dump_path)

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    normalPreds_list.append(output_dict['normalPreds'][n])
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        assert output_dict['normalPreds'][n].shape[0] == 1
                        normal_dump_path = scene_path_dump / ('imnormal.png')
                        normalPred_np = output_dict['normalPreds'][n][0].cpu().numpy()
                        normal_save = np.clip(0.5*(normalPred_np.transpose(1, 2, 0)+1), 0., 1.)
                        Image.fromarray((normal_save*255.).astype(np.uint8)).save(str(normal_dump_path))
                        Image.fromarray((normal_save*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to]).save(str(normal_dump_path).replace('.png','_cropped.png'))
                        print(normal_dump_path)

                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    roughPreds_list.append(output_dict['roughPreds'][n])
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        assert output_dict['roughPreds'][n].shape[0] == 1
                        rough_dump_path = scene_path_dump / ('imroughness.png')
                        roughPred_np = output_dict['roughPreds'][n][0].cpu().numpy()
                        rough_save = np.clip(0.5*(roughPred_np.squeeze()+1), 0., 1.)
                        Image.fromarray((rough_save*255.).astype(np.uint8)).save(str(rough_dump_path))
                        Image.fromarray((rough_save*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to]).save(str(rough_dump_path).replace('.png','_cropped.png'))
                        print(rough_dump_path)

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depthPreds_list.append(output_dict['depthPreds'][n])
                    if opt.cfg.DEBUG.if_test_real and opt.cfg.DEBUG.dump_BRDF_offline.enable:
                        assert output_dict['depthPreds'][n].shape[0] == 1
                        depth_dump_path = scene_path_dump / ('imdepth.pickle')
                        depthPred_np = output_dict['depthPreds'][n][0].cpu().numpy()
                        depth_save = depthPred_np.squeeze().astype(np.float32)
                        with open(str(depth_dump_path),"wb") as f:
                            pickle.dump({'depth_pred': depth_save}, f)
                        with open(str(depth_dump_path).replace('.pickle','_cropped.pickle'),"wb") as f:
                            pickle.dump({'depth_pred': depth_save[:im_h_resized_to, :im_w_resized_to]}, f)
                        print(depth_dump_path)
                        _ = depth_save[:im_h_resized_to, :im_w_resized_to]
                        depth_dump_vis_path = scene_path_dump / ('imdepth_cropped.png')
                        _ = vis_disp_colormap(_, normalize=True)[0]
                        Image.fromarray(_).save(str(depth_dump_vis_path))


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
        
            # ===== Vis GMM
            if opt.cfg.MODEL_GMM.enable:
                if opt.cfg.MODEL_GMM.appearance_recon.enable and opt.cfg.MODEL_GMM.appearance_recon.sanity_check:
                    Q_M_batch = output_dict['output_GMM']['gamma_update'].detach().cpu().numpy()
                    im_resampled_GMM_batch = output_dict['output_GMM']['im_resampled_GMM'].detach().cpu().numpy()
                    # print(Q_M_batch.shape) # [b, 315, 240, 320]
                    for sample_idx_batch, (im_single, im_path) in enumerate(zip(data_batch['im_fixedscale_SDR'], data_batch['image_path'])):
                        sample_idx = sample_idx_batch+batch_size*batch_id
                        # Q_M = Q_M_batch[sample_idx_batch]
                        # output_GMM_Q_list.append(Q_M)

                        im_resampled_GMM = im_resampled_GMM_batch[sample_idx_batch].transpose(1, 2, 0)
                        im_resampled_GMM = np.clip(im_resampled_GMM*255., 0., 255.).astype(np.uint8)
                        writer.add_image('VAL_GMM_im_hat/%d'%(sample_idx), im_resampled_GMM, tid, dataformats='HWC')
                        # writer.add_image('VAL_GMM_im_gt/%d'%(sample_idx), im_single, tid, dataformats='HWC')

    # ===== Vis BRDF 2/2
    # ===== logging top N to TB
    im_paths_list = flatten_list(im_paths_list)

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

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthPreds_vis = torch.cat(depthPreds_list)

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
            if opt.is_master:
                vutils.save_image(rough_pred_batch_vis_sdr,
                        '{0}/{1}_roughPred.png'.format(opt.summary_vis_path_task, tid) )

        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthPreds_vis)
            depth_pred_batch_vis_sdr = ( depthOut * segAllBatch_vis.expand_as(depthPreds_vis) ).data

            depthOut_colored_single_numpy_list = []
            for depthPreds_vis_single_numpy in depthOut.cpu().detach().numpy():
                depthOut_colored_single_numpy_list.append(vis_disp_colormap(depthPreds_vis_single_numpy.squeeze(), normalize=True)[0])
            depthOut_colored_batch = np.stack(depthOut_colored_single_numpy_list).transpose(0, 3, 1, 2).astype(np.float32) / 255.
            # print(depthOut_colored_batch.shape, segAllBatch_vis.cpu().detach().numpy().shape)
            depth_pred_batch_vis_sdr_colored = ( torch.from_numpy(depthOut_colored_batch).cuda() * segAllBatch_vis.expand_as(depthPreds_vis) ).data
            # depth_pred_batch_vis_sdr_colored = depthOut_colored_batch * segAllBatch_vis.cpu().detach().numpy()
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
        if 'de' in opt.cfg.MODEL_BRDF.enable_list:
            depth_pred_batch_vis_sdr_numpy = depth_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)

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
                        al = Image.fromarray((albedo_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        al = al.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        al.save(str(real_sample_albedo_path))

                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-normal_PRED/%d'%sample_idx, normal_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'no' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((normal_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)).save('{0}/{1}_normalGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_normal_path = real_sample_results_path / 'normal.png'
                        no = Image.fromarray((normal_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        no = no.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        no.save(str(real_sample_normal_path))

                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-rough_PRED/%d'%sample_idx, rough_pred_batch_vis_sdr_numpy[sample_idx], tid, dataformats='HWC')
                    if opt.cfg.DEBUG.if_dump_perframe_BRDF:
                        Image.fromarray((rough_pred_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8).squeeze()).save('{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                        if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'ro' in opt.cfg.DATA.data_read_list:
                            Image.fromarray((rough_gt_batch_vis_sdr_numpy[sample_idx]*255.).astype(np.uint8).squeeze()).save('{0}/{1}_roughGt_{2}.png'.format(opt.summary_vis_path_task, tid, sample_idx ))
                    if opt.cfg.DEBUG.if_test_real:
                        real_sample_results_path, (im_h_resized_to, im_w_resized_to) = real_sample_results_path_list[sample_idx]
                        real_sample_rough_path = real_sample_results_path / 'rough.png'
                        ro = Image.fromarray((rough_pred_batch_vis_sdr_numpy[sample_idx].squeeze()*255.).astype(np.uint8)[:im_h_resized_to, :im_w_resized_to])
                        ro = ro.resize((im_w_resized_to*2, im_h_resized_to*2), Image.ANTIALIAS)
                        ro.save(str(real_sample_rough_path))

                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    if (not opt.cfg.DATASET.if_no_gt_BRDF) and opt.cfg.DATA.load_brdf_gt and 'de' in opt.cfg.DATA.data_read_list:
                        depth_normalized_pred, _ = vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze(), normalize=True, min_and_scale=depth_min_and_scale_list[sample_idx])
                        writer.add_image('VAL_brdf-depth_syncScale_PRED/%d'%sample_idx, depth_normalized_pred, tid, dataformats='HWC')
                    _ = depth_pred_batch_vis_sdr_numpy[sample_idx].squeeze()
                    im_h_resized_to, im_w_resized_to = im_h_resized_to_list[sample_idx], im_w_resized_to_list[sample_idx]
                    _ = _[:im_h_resized_to, :im_w_resized_to]
                    depth_not_normalized_pred = vis_disp_colormap(_, normalize=True)[0]
                    writer.add_image('VAL_brdf-depth_PRED/%d'%sample_idx, depth_not_normalized_pred, tid, dataformats='HWC')
                    writer.add_image('VAL_brdf-depth_PRED_thres%.2f/%d'%(opt.cfg.MODEL_LIGHT.depth_thres, sample_idx), depthPreds_vis[sample_idx].cpu().numpy().squeeze()>opt.cfg.MODEL_LIGHT.depth_thres, tid, dataformats='HW')
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

                # if opt.cfg.MODEL_GMM.enable:
                #     Q_M = output_GMM_Q_list[sample_idx]
                #     im_single = im_batch_vis_sdr[sample_idx].cpu().detach().numpy().transpose(1, 2, 0) # [H, W, 3]
                #     J = Q_M.shape[0]

                #     # V1.0: appearance recon via SSD
                #     # Q_M_Jnormalized = Q_M / (Q_M.sum(-1, keepdims=True).sum(-2, keepdims=True)+1e-6) # [J, 240, 320]
                #     # im_single_J = Q_M_Jnormalized.reshape(J, -1) @ im_single.reshape(-1, 3) # [J, 3]
                #     # # print(Q_M_Jnormalized.shape, Q_M_Jnormalized.sum(-1).sum(-1)) # should be all 1s
                #     # # print(np.sum(Q_M, 0))
                #     # # print(im_single.shape)
                #     # # print(im_single_J.shape, im_single_J) # (315, 3)
                #     # # print(Q_M.shape, Q_M)
                #     # im_single_hat = Q_M.transpose(1, 2, 0) @ im_single_J # (240, 320, 3)

                #     # im_single_hat = np.clip(im_single_hat*255., 0., 255.).astype(np.uint8)

                #     writer.add_image('VAL_GMM_im_hat/%d'%(sample_idx), im_single_hat, tid, dataformats='HWC')
                #     writer.add_image('VAL_GMM_im_gt/%d'%(sample_idx), im_single, tid, dataformats='HWC')

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

    # opt.albedo_pooling_debug = False

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
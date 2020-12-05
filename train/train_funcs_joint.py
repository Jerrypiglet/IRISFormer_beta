import numpy as np
import torch
from torch.autograd import Variable
import models
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

from train_funcs_matseg import get_input_dict_matseg, process_matseg, val_epoch_matseg
from train_funcs_semseg import get_input_dict_semseg, process_semseg
from train_funcs_brdf import get_input_dict_brdf, process_brdf
from utils.comm import synchronize

from utils.utils_metrics import compute_errors_depth_nyu

def get_time_meters_joint():
    time_meters = {}
    time_meters['ts'] = 0.
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss_brdf'] = AverageMeter()
    time_meters['loss_matseg'] = AverageMeter()
    time_meters['loss_semseg'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    
    return time_meters

def get_semseg_meters():
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    semseg_meters = {'intersection_meter': intersection_meter, 'union_meter': union_meter, 'target_meter': target_meter}
    return semseg_meters

def get_brdf_meters(opt):
    normal_mean_error_meter = AverageMeter('normal_mean_error_meter')
    normal_median_error_meter = AverageMeter('normal_median_error_meter')
    # inv_depth_mean_error_meter = AverageMeter('inv_depth_mean_error_meter')
    # inv_depth_median_error_meter = AverageMeter('inv_depth_median_error_meter')
    brdf_meters = {'normal_mean_error_meter': normal_mean_error_meter, 'normal_median_error_meter': normal_median_error_meter}
    brdf_meters.update(get_depth_meters(opt))
    return brdf_meters

def get_depth_meters(opt):
    return {metric: AverageMeter(metric) for metric in opt.depth_metrics}

def get_input_dict_joint(data_batch, opt):
    input_dict = {'im_trainval_RGB': data_batch['im_trainval_RGB'].cuda(non_blocking=True)}
    if opt.cfg.DATA.load_matseg_gt:
        input_dict_matseg = get_input_dict_matseg(data_batch, opt)
    else:
        input_dict_matseg = {}
    input_dict.update(input_dict_matseg)

    if opt.cfg.DATA.load_semseg_gt:
        input_dict_semseg = get_input_dict_semseg(data_batch, opt)
    else:
        input_dict_semseg = {}
    input_dict.update(input_dict_semseg)

    if opt.cfg.DATA.load_brdf_gt:
        input_batch_brdf, input_dict_brdf, pre_batch_dict_brdf = get_input_dict_brdf(data_batch, opt)
        input_dict.update({'input_batch_brdf': input_batch_brdf, 'pre_batch_dict_brdf': pre_batch_dict_brdf})
    else:
        input_dict_brdf = {}    

    input_dict.update(input_dict_brdf)

    # input_dict = {**input_dict_matseg, **input_dict_brdf}
    return input_dict

def forward_joint(input_dict, model, opt, time_meters):
    output_dict = model(input_dict)
    time_meters['forward'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    loss_dict = {}

    if opt.cfg.MODEL_MATSEG.enable:
        output_dict, loss_dict = process_matseg(input_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_matseg'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
    
    if opt.cfg.MODEL_BRDF.enable:
        output_dict, loss_dict = process_brdf(input_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_brdf'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()

    if opt.cfg.MODEL_SEMSEG.enable:
        output_dict, loss_dict = process_semseg(input_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_semseg'].update(time.time() - time_meters['ts'])
        time_meters['ts'] = time.time()
    
    return output_dict, loss_dict

def val_epoch_joint(brdf_loader_val, model, bin_mean_shift, params_mis):
    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']
    ENABLE_SEMSEG = opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable
    ENABLE_MATSEG = opt.cfg.MODEL_MATSEG.enable
    ENABLE_BRDF = opt.cfg.MODEL_BRDF.enable

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

    if opt.cfg.MODEL_BRDF.enable:
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

    if opt.cfg.MODEL_SEMSEG.enable:
        loss_keys += [
            'loss_semseg-ALL', 
            'loss_semseg-main', 
            'loss_semseg-aux', 
        ]
        
    loss_meters = {loss_key: AverageMeter() for loss_key in loss_keys}
    time_meters = get_time_meters_joint()
    if ENABLE_SEMSEG:
        semseg_meters = get_semseg_meters()
    if ENABLE_BRDF:
        brdf_meters = get_brdf_meters(opt)

    with torch.no_grad():
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):
            ts_iter_start = time.time()

            input_dict = get_input_dict_joint(data_batch, opt)

            time_meters['data_to_gpu'].update(time.time() - ts_iter_start)
            time_meters['ts'] = time.time()

            # ======= Forward
            time_meters['ts'] = time.time()
            output_dict, loss_dict = forward_joint(input_dict, model, opt, time_meters)

            loss_dict_reduced = reduce_loss_dict(loss_dict, mark=tid, logger=logger) # **average** over multi GPUs
            time_meters['ts'] = time.time()
            # loss = loss_dict['loss_all']
            
            # ======= update loss
            for loss_key in loss_dict_reduced:
                # print(loss_key)
                loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= Metering
            if ENABLE_BRDF:
                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    depth_input = input_dict['depthBatch'].detach().cpu().numpy()
                    depth_output = output_dict['depthPreds'][0].detach().cpu().numpy()
                    seg_obj = data_batch['segObj'].cpu().numpy()
                    min_depth, max_depth = 0.1, 8.
                    # depth_mask = np.logical_and(np.logical_and(seg_obj != 0, depth_input < max_depth), depth_input > min_depth)
                    depth_output = depth_output * seg_obj
                    depth_input = depth_input * seg_obj
                    np.save('depth_input_%d.npy'%(batch_id), depth_input)
                    np.save('depth_output_%d.npy'%(batch_id), depth_output)
                    np.save('seg_obj_%d.npy'%(batch_id), seg_obj)

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
                intersection, union, target = intersectionAndUnionGPU(output, target, opt.cfg.DATA.semseg_classes, opt.cfg.DATA.semseg_ignore_label)
                if opt.distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                semseg_meters['intersection_meter'].update(intersection), semseg_meters['union_meter'].update(union), semseg_meters['target_meter'].update(target)
                # accuracy = sum(semseg_meters['intersection_meter'].val) / (sum(semseg_meters['target_meter'].val) + 1e-10)

            print(batch_id)

            # synchronize()

    # ======= Metering
    if ENABLE_SEMSEG:
        iou_class = semseg_meters['intersection_meter'].sum / (semseg_meters['union_meter'].sum + 1e-10)
        accuracy_class = semseg_meters['intersection_meter'].sum / (semseg_meters['target_meter'].sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(semseg_meters['intersection_meter'].sum) / (sum(semseg_meters['target_meter'].sum) + 1e-10)

    if opt.is_master:
        for loss_key in loss_dict_reduced:
            writer.add_scalar('loss_val/%s'%loss_key, loss_meters[loss_key].avg, tid)
            logger.info('Logged val loss for %s'%loss_key)
        if ENABLE_SEMSEG:
            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(opt.cfg.DATA.semseg_classes):
                logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
            writer.add_scalar('VAL/mIoU_val', mIoU, tid)
            writer.add_scalar('VAL/mAcc_val', mAcc, tid)
            writer.add_scalar('VAL/allAcc_val', allAcc, tid)
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

    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']
    logger.info(red('===Visualizing for %d batches on rank %d'%(len(brdf_loader_val), opt.rank)))

    model.eval()
    opt.if_vis_debug_pac = True

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
    
    # opt.albedo_pooling_debug = True

    with torch.no_grad():
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):
            batch_size = len(data_batch['imPath'])

            # print(batch_id, batch_size)

            # if num_val_brdf_vis >= num_val_vis_MAX or sample_idx >= num_val_vis_MAX:
            #     break

            input_dict = get_input_dict_joint(data_batch, opt)
            # if batch_id == 0:
            #     print(input_dict['im_paths'])

            # ======= Forward
            output_dict, loss_dict = forward_joint(input_dict, model, opt, time_meters)

            synchronize()
            
            # ======= Vis imagges
            colors = np.loadtxt(os.path.join(opt.pwdpath, opt.cfg.PATH.semseg_colors_path)).astype('uint8')
            if opt.cfg.DATA.load_semseg_gt:
                semseg_label = input_dict['semseg_label'].cpu().numpy()
            if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable:
                semseg_pred = output_dict['semseg_pred'].cpu().numpy()

            for sample_idx,(im_single, im_path) in enumerate(zip(data_batch['im_SDR_RGB'], data_batch['imPath'])):
                # if sample_idx >= num_val_vis_MAX:
                #     break
                im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
                # im_path = os.path.join('./tmp/', 'im_%d-%d_color.png'%(tid, im_index))
                # color_path = os.path.join('./tmp/', 'im_%d-%d_semseg.png'%(tid, im_index))
                # cv2.imwrite(im_path, im_single * 255.)
                # semseg_color.save(color_path)
                if opt.is_master:
                    writer.add_image('VAL_im/%d'%(sample_idx+batch_size*batch_id), im_single, tid, dataformats='HWC')
                    if opt.cfg.MODEL_MATSEG.albedo_pooling_debug and not opt.if_cluster:
                        os.makedirs('tmp/demo_%s'%(opt.task_name), exist_ok=True)
                        np.save('tmp/demo_%s/im_trainval_RGB_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id), im_single)
                        print('Saved to' + 'tmp/demo_%s/im_trainval_RGB_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id))

                    writer.add_text('VAL_image_name/%d'%(sample_idx+batch_size*batch_id), im_path, tid)
                    print(sample_idx+batch_size*batch_id, im_path)

                if (opt.cfg.MODEL_MATSEG.if_albedo_pooling or opt.cfg.MODEL_MATSEG.if_albedo_asso_pool_conv or opt.cfg.MODEL_MATSEG.if_albedo_pac_pool or opt.cfg.MODEL_MATSEG.if_albedo_safenet) and opt.cfg.MODEL_MATSEG.albedo_pooling_debug:
                    if opt.is_master:
                        if output_dict['im_trainval_RGB_mask_pooled_mean'] is not None:
                            im_trainval_RGB_mask_pooled_mean = output_dict['im_trainval_RGB_mask_pooled_mean'][sample_idx]
                            im_trainval_RGB_mask_pooled_mean = im_trainval_RGB_mask_pooled_mean.cpu().numpy().squeeze().transpose(1, 2, 0)
                            writer.add_image('VAL_im_trainval_RGB_mask_pooled_mean/%d'%(sample_idx+batch_size*batch_id), im_trainval_RGB_mask_pooled_mean, tid, dataformats='HWC')
                        if not opt.if_cluster:
                            kernel_list = output_dict['kernel_list']
                            if not kernel_list is None:
                                print(len(kernel_list), kernel_list[0].shape)
                                np.save('tmp/demo_%s/kernel_list_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id), kernel_list[0].detach().cpu().numpy())
                            if output_dict['im_trainval_RGB_mask_pooled_mean'] is not None:
                                np.save('tmp/demo_%s/im_trainval_RGB_mask_pooled_mean_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id), im_trainval_RGB_mask_pooled_mean)
                            if 'embeddings' in output_dict and output_dict['embeddings'] is not None:
                                np.save('tmp/demo_%s/embeddings_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id), output_dict['embeddings'].detach().cpu().numpy())
                            if 'affinity' in output_dict:
                                affinity = output_dict['affinity']
                                sample_ij = output_dict['sample_ij']
                                if affinity is not None:
                                    np.save('tmp/demo_%s/affinity_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id), affinity[0].detach().cpu().numpy())
                                    np.save('tmp/demo_%s/sample_ij_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id), sample_ij)




                # ======= Vis BRDFsemseg / semseg
                if opt.cfg.DATA.load_semseg_gt:
                    gray_GT = np.uint8(semseg_label[sample_idx])
                    color_GT = np.array(colorize(gray_GT, colors).convert('RGB'))
                    if opt.is_master:
                        writer.add_image('VAL_semseg_GT/%d'%(sample_idx+batch_size*batch_id), color_GT, tid, dataformats='HWC')
                if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable:
                    prediction = np.argmax(semseg_pred[sample_idx], 0)
                    gray_pred = np.uint8(prediction)
                    color_pred = np.array(colorize(gray_pred, colors).convert('RGB'))
                    if opt.is_master:
                        writer.add_image('VAL_semseg_PRED/%d'%(sample_idx+batch_size*batch_id), color_pred, tid, dataformats='HWC')

            
                
            # ======= visualize clusters for mat-seg
            if opt.cfg.DATA.load_matseg_gt:
                for sample_idx in range(input_dict['mat_aggre_map_cpu'].shape[0]):
                    mat_aggre_map_GT_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
                    matAggreMap_GT_single_vis = vis_index_map(mat_aggre_map_GT_single)
                    mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()
                    if opt.is_master:
                        writer.add_image('VAL_matseg-aggre_map_GT/%d'%(sample_idx+batch_size*batch_id), matAggreMap_GT_single_vis, tid, dataformats='HWC')
                        writer.add_image('VAL_matseg-notlight_mask_GT/%d'%(sample_idx+batch_size*batch_id), mat_notlight_mask_single, tid, dataformats='HW')

            if opt.cfg.MODEL_MATSEG.enable and opt.cfg.MODEL_MATSEG.embed_dims <= 4:
                b, c, h, w = output_dict['logit'].size()
                for sample_idx, (logit_single, embedding_single) in enumerate(zip(output_dict['logit'].detach(), output_dict['embedding'].detach())):

                    # if sample_idx >= num_val_vis_MAX:
                    #     break
                    ts_start_vis = time.time()

                    
                    # prob_single = torch.sigmoid(logit_single)
                    prob_single = input_dict['mat_notlight_mask_cpu'][sample_idx].to(opt.device).float()
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
                    gt_plane_num = input_dict['num_mat_masks_batch'][sample_idx]
                    matching = match_segmentatin(segmentation, prob_single.view(-1, 1), input_dict['instance'][sample_idx], gt_plane_num)

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
                    # np.save('tmp/instance_%d.npy'%(max_index), input_dict['instance'][sample_idx].cpu().numpy().squeeze())
                    # np.save('tmp/gt_%d.npy'%(max_index), input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze())
                    # np.save('tmp/matching_%d.npy'%(max_index), matching)
                    predict_segmentation = matching[predict_segmentation] # matching GT: [0, 1, ... N-1]
                    # np.save('tmp/predict_segmentation_matched%d.npy'%(max_index), predict_segmentation)

                    # mask out non planar region
                    predict_segmentation = predict_segmentation.reshape(h, w) # [0..N-1]
                    predict_segmentation += 1 # 0 for invalid region
                    predict_segmentation[prob_single.cpu().squeeze().numpy() <= 0.1] = opt.invalid_index

                    # ===== vis
                    # im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)

                    # mat_aggre_map_pred_single = reindex_output_map(predict_segmentation.squeeze(), opt.invalid_index)
                    mat_aggre_map_pred_single = predict_segmentation.squeeze()
                    matAggreMap_pred_single_vis = vis_index_map(mat_aggre_map_pred_single)

                    if opt.is_master:
                        writer.add_image('VAL_matseg-aggre_map_PRED/%d'%(sample_idx+batch_size*batch_id), matAggreMap_pred_single_vis, tid, dataformats='HWC')
                        if opt.cfg.MODEL_MATSEG.albedo_pooling_debug and not opt.if_cluster:
                            np.save('tmp/demo_%s/matseg_pred_tid%d_idx%d.npy'%(opt.task_name, tid, sample_idx+batch_size*batch_id), matAggreMap_pred_single_vis)

                    logger.info('Vis batch for %.2f seconds.'%(time.time()-ts_start_vis))


            # ======= visualize clusters for mat-seg
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

    synchronize()

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


        if opt.is_master:
            print('Saving to ', '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid))
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

            if not opt.test_real:
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
            if not opt.test_real:
                for image_idx in range(albedo_gt_batch_vis_sdr.shape[0]):
                    if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                        writer.add_image('VAL_brdf-albedo_GT/%d'%image_idx, albedo_gt_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                    if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                        writer.add_image('VAL_brdf-normal_GT/%d'%image_idx, normal_gt_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                    if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                        writer.add_image('VAL_brdf-rough_GT/%d'%image_idx, rough_gt_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                    if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                        writer.add_image('VAL_brdf-depth_GT/%d'%image_idx, vis_disp_colormap(depth_gt_batch_vis_sdr_numpy[image_idx].squeeze()), tid, dataformats='HWC')

            if opt.cascadeLevel > 0:
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
                vutils.save_image(albedo_pred_batch_vis_sdr,
                        '{0}/{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                normal_pred_batch_vis_sdr = ( 0.5*(normalPreds_vis + 1) ).data
                vutils.save_image(normal_pred_batch_vis_sdr,
                        '{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                rough_pred_batch_vis_sdr = ( 0.5*(roughPreds_vis + 1) ).data
                vutils.save_image(rough_pred_batch_vis_sdr,
                        '{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            # print(torch.min(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.max(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.mean(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.median(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)))
            if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthPreds_vis)
                depth_pred_batch_vis_sdr = ( depthOut * segAllBatch_vis.expand_as(depthPreds_vis) ).data
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
            for image_idx in range(albedo_pred_batch_vis_sdr.shape[0]):
                if 'al' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-albedo_PRED/%d'%image_idx, albedo_pred_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                if 'no' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-normal_PRED/%d'%image_idx, normal_pred_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                if 'ro' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-rough_PRED/%d'%image_idx, rough_pred_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                if 'de' in opt.cfg.MODEL_BRDF.enable_list:
                    writer.add_image('VAL_brdf-depth_PRED/%d'%image_idx, vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[image_idx].squeeze()), tid, dataformats='HWC')

    logger.info(red('Evaluation VIS timings: ' + time_meters_to_string(time_meters)))

    # opt.albedo_pooling_debug = False

    synchronize()
    opt.if_vis_debug_pac = False




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

from train_funcs_mat_seg import get_input_dict_mat_seg, process_mat_seg
from train_funcs_semseg import get_input_dict_semseg, process_semseg
from train_funcs_brdf import get_input_dict_brdf, process_brdf
from utils.comm import synchronize

def get_time_meters_joint():
    time_meters = {}
    time_meters['ts'] = 0.
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss_brdf'] = AverageMeter()
    time_meters['loss_mat_seg'] = AverageMeter()
    time_meters['loss_semseg'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    
    return time_meters

def get_semseg_meters():
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    semseg_meters = {'intersection_meter': intersection_meter, 'union_meter': union_meter, 'target_meter': target_meter}
    return semseg_meters


def get_input_dict_joint(data_batch, opt):
    input_dict = {}
    if opt.cfg.MODEL_SEG.enable:
        input_dict_mat_seg = get_input_dict_mat_seg(data_batch, opt)
    else:
        input_dict_mat_seg = {}
    input_dict.update(input_dict_mat_seg)

    if opt.cfg.MODEL_SEMSEG.enable:
        input_dict_semseg = get_input_dict_semseg(data_batch, opt)
    else:
        input_dict_semseg = {}
    input_dict.update(input_dict_semseg)

    if opt.cfg.MODEL_BRDF.enable:
        input_batch_brdf, input_dict_brdf, pre_batch_dict_brdf = get_input_dict_brdf(data_batch, opt)
    else:
        input_dict_brdf = {}    
    input_dict.update(input_dict_brdf)

    if opt.cfg.MODEL_BRDF.enable:
        input_dict.update({'input_batch_brdf': input_batch_brdf, 'pre_batch_dict_brdf': pre_batch_dict_brdf})

    # input_dict = {**input_dict_mat_seg, **input_dict_brdf}
    return input_dict

def forward_joint(input_dict, model, opt, time_meters):
    output_dict = model(input_dict)
    time_meters['forward'].update(time.time() - time_meters['ts'])
    time_meters['ts'] = time.time()

    loss_dict = {}

    if opt.cfg.MODEL_SEG.enable:
        output_dict, loss_dict = process_mat_seg(input_dict, output_dict, loss_dict, opt, time_meters)
        time_meters['loss_mat_seg'].update(time.time() - time_meters['ts'])
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

    logger.info(red('===Evaluating for %d batches'%len(brdf_loader_val)))

    model.eval()
    
    loss_keys = []

    if opt.cfg.MODEL_SEG.enable:
        loss_keys += [
            'loss_mat_seg-ALL', 
            'loss_mat_seg-pull', 
            'loss_mat_seg-push', 
            'loss_mat_seg-binary', 
        ]

    if opt.cfg.MODEL_BRDF.enable:
        loss_keys += [
            'loss_brdf-albedo', 
            'loss_brdf-normal', 
            'loss_brdf-rough', 
            'loss_brdf-depth', 
            'loss_brdf-ALL', 
        ]
        if opt.cfg.MODEL_BRDF.enable_semseg_decoder:
            loss_keys += ['loss_brdf-semseg']

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
                loss_meters[loss_key].update(loss_dict_reduced[loss_key].item())

            # ======= Metering
            if ENABLE_SEMSEG:
                output = output_dict['semseg_pred'].max(1)[1]
                target = input_dict['semseg_label']
                intersection, union, target = intersectionAndUnionGPU(output, target, opt.cfg.DATA.semseg_classes, opt.cfg.DATA.semseg_ignore_label)
                if opt.distributed:
                    dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                semseg_meters['intersection_meter'].update(intersection), semseg_meters['union_meter'].update(union), semseg_meters['target_meter'].update(target)
                # accuracy = sum(semseg_meters['intersection_meter'].val) / (sum(semseg_meters['target_meter'].val) + 1e-10)

            # print(batch_id)

            # synchronize()

    # ======= Metering
    if ENABLE_SEMSEG:
        iou_class = semseg_meters['intersection_meter'].sum / (semseg_meters['union_meter'].sum + 1e-10)
        accuracy_class = semseg_meters['intersection_meter'].sum / (semseg_meters['target_meter'].sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(semseg_meters['intersection_meter'].sum) / (sum(semseg_meters['target_meter'].sum) + 1e-10)
        if opt.is_master:
            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            for i in range(opt.cfg.DATA.semseg_classes):
                logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))


    if opt.is_master:
        for loss_key in loss_dict_reduced:
            writer.add_scalar('loss_val/%s'%loss_key, loss_meters[loss_key].avg, tid)
        if ENABLE_SEMSEG:
            writer.add_scalar('VAL/mIoU_val', mIoU, tid)
            writer.add_scalar('VAL/mAcc_val', mAcc, tid)
            writer.add_scalar('VAL/allAcc_val', allAcc, tid)


        logger.info(red('Evaluation timings: ' + time_meters_to_string(time_meters)))


def vis_val_epoch_joint(brdf_loader_val, model, bin_mean_shift, params_mis):

    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']
    logger.info(red('===Visualizing for %d batches on rank %d'%(len(brdf_loader_val), opt.rank)))

    model.eval()

    num_val_mat_seg_vis = 0
    num_val_im_vis = 0
    num_val_semseg_vis = 0
    num_val_brdf_vis = 0
    # num_val_vis_MAX = 100

    time_meters = get_time_meters_joint()

    if opt.cfg.MODEL_SEG.enable:
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


    with torch.no_grad():
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):

            # print(batch_id)

            # if num_val_brdf_vis >= num_val_vis_MAX or num_val_mat_seg_vis >= num_val_vis_MAX:
            #     break

            input_dict = get_input_dict_joint(data_batch, opt)
            # if batch_id == 0:
            #     print(input_dict['im_paths'])

            # ======= Forward
            output_dict, loss_dict = forward_joint(input_dict, model, opt, time_meters)

            synchronize()
            
            # ======= Vis imagges
            if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable:
                colors = np.loadtxt(os.path.join(opt.pwdpath, opt.cfg.PATH.semseg_colors_path)).astype('uint8')
                semseg_pred = output_dict['semseg_pred'].cpu().numpy()
                semseg_label = input_dict['semseg_label'].cpu().numpy()

            for sample_idx,(im_single, im_path) in enumerate(zip(data_batch['im_SDR_RGB'], data_batch['imPath'])):
                # if num_val_im_vis >= num_val_vis_MAX:
                #     break
                im_single = im_single.numpy().squeeze().transpose(1, 2, 0)
                # im_path = os.path.join('./tmp/', 'im_%d-%d_color.png'%(tid, im_index))
                # color_path = os.path.join('./tmp/', 'im_%d-%d_semseg.png'%(tid, im_index))
                # cv2.imwrite(im_path, im_single * 255.)
                # semseg_color.save(color_path)
                if opt.is_master:
                    writer.add_image('VAL_im/%d'%num_val_im_vis, im_single, tid, dataformats='HWC')
                    writer.add_text('VAL_image_name/%d'%num_val_im_vis, im_path, tid)


                # ======= Vis BRDFsemseg / semseg
                if opt.cfg.MODEL_BRDF.enable_semseg_decoder or opt.cfg.MODEL_SEMSEG.enable:
                    gray_GT = np.uint8(semseg_label[sample_idx])
                    color_GT = np.array(colorize(gray_GT, colors).convert('RGB'))
                    prediction = np.argmax(semseg_pred[sample_idx], 0)
                    gray_pred = np.uint8(prediction)
                    color_pred = np.array(colorize(gray_pred, colors).convert('RGB'))
                    if opt.is_master:
                        writer.add_image('VAL_semseg_GT/%d'%num_val_im_vis, color_GT, tid, dataformats='HWC')
                        writer.add_image('VAL_semseg_PRED/%d'%num_val_im_vis, color_pred, tid, dataformats='HWC')

                num_val_im_vis += 1
            
                
            # ======= visualize clusters for mat-seg
            if opt.cfg.MODEL_SEG.enable:
                b, c, h, w = output_dict['logit'].size()
                for sample_idx, (logit_single, embedding_single) in enumerate(zip(output_dict['logit'].detach(), output_dict['embedding'].detach())):

                    # if num_val_mat_seg_vis >= num_val_vis_MAX:
                    #     break
                    
                    # prob_single = torch.sigmoid(logit_single)
                    prob_single = input_dict['mat_notlight_mask_cpu'][sample_idx].to(opt.device).float()
                    # fast mean shift
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
                    predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

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
                    predict_segmentation = matching[predict_segmentation]

                    # mask out non planar region
                    predict_segmentation[prob_single.cpu().numpy().reshape(-1) <= 0.1] = opt.invalid_index
                    predict_segmentation = predict_segmentation.reshape(h, w)

                    # ===== vis
                    # im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)


                    mat_aggre_map_GT_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
                    matAggreMap_GT_single_vis = vis_index_map(mat_aggre_map_GT_single)

                    mat_aggre_map_pred_single = reindex_output_map(predict_segmentation.squeeze(), opt.invalid_index)
                    matAggreMap_pred_single_vis = vis_index_map(mat_aggre_map_pred_single)


                    mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()


                    if opt.is_master:
                        # writer.add_image('VAL_im/%d'%num_val_mat_seg_vis, im_single, tid, dataformats='HWC')

                        writer.add_image('VAL_mat_seg-aggre_map_GT/%d'%num_val_mat_seg_vis, matAggreMap_GT_single_vis, tid, dataformats='HWC')

                        writer.add_image('VAL_mat_seg-aggre_map_PRED/%d'%num_val_mat_seg_vis, matAggreMap_pred_single_vis, tid, dataformats='HWC')

                        writer.add_image('VAL_mat_seg-notlight_mask_GT/%d'%num_val_mat_seg_vis, mat_notlight_mask_single, tid, dataformats='HW')

                        writer.add_text('VAL_im_path/%d'%num_val_mat_seg_vis, input_dict['im_paths'][sample_idx], tid)


                    num_val_mat_seg_vis += 1

            # ======= visualize clusters for mat-seg
            # print(((input_dict['albedoBatch'] ) ** (1.0/2.2) ).data.shape) # [b, 3, h, w]
            if opt.cfg.MODEL_BRDF.enable:
                # if opt.is_master:
                    # print(input_dict.keys())
                im_paths_list.append(input_dict['im_paths'])
                # print(input_dict['im_paths'])

                albedoBatch_list.append(input_dict['albedoBatch'])
                normalBatch_list.append(input_dict['normalBatch'])
                roughBatch_list.append(input_dict['roughBatch'])
                depthBatch_list.append(input_dict['depthBatch'])
                imBatch_list.append(input_dict['imBatch'])
                segAllBatch_list.append(input_dict['segAllBatch'])

                if opt.cascadeLevel > 0:
                    diffusePreBatch_list.append(input_dict['pre_batch_dict_brdf']['diffusePreBatch'])
                    specularPreBatch_list.append(input_dict['pre_batch_dict_brdf']['specularPreBatch'])
                    renderedImBatch_list.append(input_dict['pre_batch_dict_brdf']['renderedImBatch'])
                n = 0
                albedoPreds_list.append(output_dict['albedoPreds'][n])
                normalPreds_list.append(output_dict['normalPreds'][n])
                roughPreds_list.append(output_dict['roughPreds'][n])
                depthPreds_list.append(output_dict['depthPreds'][n])

                num_val_brdf_vis += len(input_dict['im_paths'])


    synchronize()

    if opt.cfg.MODEL_BRDF.enable:
        im_paths_list = flatten_list(im_paths_list)
        # print(im_paths_list)

        albedoBatch_vis = torch.cat(albedoBatch_list)
        normalBatch_vis = torch.cat(normalBatch_list)
        roughBatch_vis = torch.cat(roughBatch_list)
        depthBatch_vis = torch.cat(depthBatch_list)
        imBatch_vis = torch.cat(imBatch_list)
        segAllBatch_vis = torch.cat(segAllBatch_list)

        if opt.cascadeLevel > 0:
            diffusePreBatch_vis = torch.cat(diffusePreBatch_list)
            specularPreBatch_vis = torch.cat(specularPreBatch_list)
            renderedImBatch_vis = torch.cat(renderedImBatch_list)

        albedoPreds_vis = torch.cat(albedoPreds_list)
        normalPreds_vis = torch.cat(normalPreds_list)
        roughPreds_vis = torch.cat(roughPreds_list)
        depthPreds_vis = torch.cat(depthPreds_list)

        if opt.is_master:
            print('Saving to ', '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid))
            # Save the ground truth and the input
            # albedoBatch_vis_sdr = ( (albedoBatch_vis ) ** (1.0/2.2) ).data # torch.Size([16, 3, 192, 256]) 
            im_batch_vis_sdr = ( (imBatch_vis)**(1.0/2.2) ).data

            # ==== GTs
            albedo_gt_batch_vis_sdr = ( (albedoBatch_vis ) ** (1.0/2.2) ).data
            normal_gt_batch_vis_sdr = (0.5*(normalBatch_vis + 1) ).data
            rough_gt_batch_vis_sdr = (0.5*(roughBatch_vis + 1) ).data
            depthOut = 1 / torch.clamp(depthBatch_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthBatch_vis)
            depth_gt_batch_vis_sdr = ( depthOut*segAllBatch_vis.expand_as(depthBatch_vis) ).data

            if not opt.test_real:
                vutils.save_image(im_batch_vis_sdr,
                        '{0}/{1}_im.png'.format(opt.summary_vis_path_task, tid) )
                vutils.save_image(albedo_gt_batch_vis_sdr,
                        '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid) )
                vutils.save_image(normal_gt_batch_vis_sdr,
                        '{0}/{1}_normalGt.png'.format(opt.summary_vis_path_task, tid) )
                vutils.save_image(rough_gt_batch_vis_sdr,
                        '{0}/{1}_roughGt.png'.format(opt.summary_vis_path_task, tid) )
                vutils.save_image(depth_gt_batch_vis_sdr,
                        '{0}/{1}_depthGt.png'.format(opt.summary_vis_path_task, tid) )

            albedo_gt_batch_vis_sdr_numpy = albedo_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            normal_gt_batch_vis_sdr_numpy = normal_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            rough_gt_batch_vis_sdr_numpy = rough_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            depth_gt_batch_vis_sdr_numpy = depth_gt_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            # print('++++', rough_gt_batch_vis_sdr_numpy.shape, depth_gt_batch_vis_sdr_numpy.shape, albedo_gt_batch_vis_sdr_numpy.shape, albedo_gt_batch_vis_sdr_numpy.dtype)
            # print(np.amax(albedo_gt_batch_vis_sdr_numpy), np.amin(albedo_gt_batch_vis_sdr_numpy), np.mean(albedo_gt_batch_vis_sdr_numpy))
            if not opt.test_real:
                for image_idx in range(albedo_gt_batch_vis_sdr.shape[0]):
                    writer.add_image('VAL_brdf-albedo_GT/%d'%image_idx, albedo_gt_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                    writer.add_image('VAL_brdf-normal_GT/%d'%image_idx, normal_gt_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                    writer.add_image('VAL_brdf-rough_GT/%d'%image_idx, rough_gt_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
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
            albedo_pred_batch_vis_sdr = ( (albedoPreds_vis ) ** (1.0/2.2) ).data
            normal_pred_batch_vis_sdr = ( 0.5*(normalPreds_vis + 1) ).data
            rough_pred_batch_vis_sdr = ( 0.5*(roughPreds_vis + 1) ).data
            # print(torch.min(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.max(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.mean(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)), torch.median(1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10)))

            depthOut = 1 / torch.clamp(depthPreds_vis + 1, 1e-6, 10) * segAllBatch_vis.expand_as(depthPreds_vis)
            depth_pred_batch_vis_sdr = ( depthOut * segAllBatch_vis.expand_as(depthPreds_vis) ).data

            vutils.save_image(albedo_pred_batch_vis_sdr,
                    '{0}/{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            vutils.save_image(normal_pred_batch_vis_sdr,
                    '{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            vutils.save_image(rough_pred_batch_vis_sdr,
                    '{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
            vutils.save_image(depth_pred_batch_vis_sdr,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )

            albedo_pred_batch_vis_sdr_numpy = albedo_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            normal_pred_batch_vis_sdr_numpy = normal_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            rough_pred_batch_vis_sdr_numpy = rough_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            depth_pred_batch_vis_sdr_numpy = depth_pred_batch_vis_sdr.cpu().numpy().transpose(0, 2, 3, 1)
            for image_idx in range(albedo_pred_batch_vis_sdr.shape[0]):
                writer.add_image('VAL_brdf-albedo_PRED/%d'%image_idx, albedo_pred_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                writer.add_image('VAL_brdf-normal_PRED/%d'%image_idx, normal_pred_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                writer.add_image('VAL_brdf-rough_PRED/%d'%image_idx, rough_pred_batch_vis_sdr_numpy[image_idx], tid, dataformats='HWC')
                writer.add_image('VAL_brdf-depth_PRED/%d'%image_idx, vis_disp_colormap(depth_pred_batch_vis_sdr_numpy[image_idx].squeeze()), tid, dataformats='HWC')
            



    logger.info(red('Evaluation VIS timings: ' + time_meters_to_string(time_meters)))

    synchronize()



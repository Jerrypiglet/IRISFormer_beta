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
from utils.misc import AverageMeter
from utils.utils_vis import vis_index_map, reindex_output_map
from utils.utils_training import reduce_loss_dict, time_meters_to_string
from utils.utils_misc import *
import torchvision.utils as vutils

from train_funcs_mat_seg import get_input_dict_mat_seg, process_mat_seg
from train_funcs_brdf import get_input_dict_brdf, process_brdf



def get_input_dict_mat_seg_joint(data_batch, opt):
    input_dict_mat_seg = get_input_dict_mat_seg(data_batch, opt)
    input_batch_brdf, input_dict_brdf, pre_batch_dict_brdf = get_input_dict_brdf(data_batch, opt)
    input_dict = {**input_dict_mat_seg, **input_dict_brdf}
    input_dict.update({'input_batch_brdf': input_batch_brdf, 'pre_batch_dict_brdf': pre_batch_dict_brdf})
    return input_dict

def forward_joint(input_dict, model, opt, time_meters):
    output_dict = model(input_dict)
    loss_dict = {}
    output_dict, loss_dict = process_mat_seg(input_dict, output_dict, loss_dict, opt, time_meters)
    output_dict, loss_dict = process_brdf(input_dict, output_dict, loss_dict, opt, time_meters)
    return output_dict, loss_dict

def val_epoch_joint(brdfLoaderVal, model, bin_mean_shift, params_mis):

    writer, logger, opt, tid = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid']
    logger.info(red('===Evaluating for %d batches'%len(brdfLoaderVal)))

    model.eval()

    loss_keys = [
        'loss_mat_seg-ALL', 
        'loss_mat_seg-pull', 
        'loss_mat_seg-push', 
        'loss_mat_seg-binary', 
        'loss_brdf-albedo', 
        'loss_brdf-normal', 
        'loss_brdf-rough', 
        'loss_brdf-depth', 
        'loss_brdf-ALL', 
    ]
    loss_meters = {loss_key: AverageMeter() for loss_key in loss_keys}

    time_meters = {}
    time_meters['data_to_gpu'] = AverageMeter()
    time_meters['forward'] = AverageMeter()
    time_meters['loss'] = AverageMeter()
    time_meters['backward'] = AverageMeter()    


    match_segmentatin = MatchSegmentation()

    with torch.no_grad():
        for batch_id, data_batch in tqdm(enumerate(brdfLoaderVal)):
            ts_iter_start = time.time()

            input_dict = get_input_dict_mat_seg_joint(data_batch, opt)

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

            # ======= visualize clusters for mat-seg
            if batch_id == 0 and opt.is_master:

                b, c, h, w = output_dict['logit'].size()

                for j, (logit_single, embedding_single) in enumerate(zip(output_dict['logit'].detach(), output_dict['embedding'].detach())):
                    sample_idx = j

                    # prob_single = torch.sigmoid(logit_single)
                    prob_single = input_dict['mat_notlight_mask_cpu'][j].to(opt.device).float()
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
                    gt_plane_num = input_dict['num_mat_masks_batch'][j]
                    matching = match_segmentatin(segmentation, prob_single.view(-1, 1), input_dict['instance'][j], gt_plane_num)

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
                    im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)

                    writer.add_image('VAL_im/%d'%sample_idx, im_single, tid, dataformats='HWC')

                    mat_aggre_map_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
                    matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
                    writer.add_image('VAL_mat_aggre_map_GT/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')

                    mat_aggre_map_single = reindex_output_map(predict_segmentation.squeeze(), opt.invalid_index)
                    matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
                    writer.add_image('VAL_mat_aggre_map_PRED/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')


                    mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()
                    writer.add_image('VAL_mat_notlight_mask_GT/%d'%sample_idx, mat_notlight_mask_single, tid, dataformats='HW')

                    writer.add_text('VAL_im_path/%d'%sample_idx, input_dict['im_paths'][sample_idx], tid)

                    if j > 12:
                        break

            # ======= visualize clusters for mat-seg
            # print(((input_dict['albedoBatch'] ) ** (1.0/2.2) ).data.shape) # [b, 3, h, w]
            if batch_id == 0 and opt.is_master:
                print('Saving to ', '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid))
                # Save the ground truth and the input
                vutils.save_image(( (input_dict['albedoBatch'] ) ** (1.0/2.2) ).data,
                        '{0}/{1}_albedoGt.png'.format(opt.summary_vis_path_task, tid) )
                vutils.save_image( (0.5*(input_dict['normalBatch'] + 1) ).data,
                        '{0}/{1}_normalGt.png'.format(opt.summary_vis_path_task, tid) )
                vutils.save_image( (0.5*(input_dict['roughBatch'] + 1) ).data,
                        '{0}/{1}_roughGt.png'.format(opt.summary_vis_path_task, tid) )
                vutils.save_image( ( (input_dict['imBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_im.png'.format(opt.summary_vis_path_task, tid) )
                depthOut = 1 / torch.clamp(input_dict['depthBatch'] + 1, 1e-6, 10) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'])
                vutils.save_image( ( depthOut*input_dict['segAllBatch'].expand_as(input_dict['depthBatch']) ).data,
                        '{0}/{1}_depthGt.png'.format(opt.summary_vis_path_task, tid) )

                if opt.cascadeLevel > 0:
                    vutils.save_image( ( (input_dict['pre_batch_dict_brdf']['diffusePreBatch'])**(1.0/2.2) ).data,
                            '{0}/{1}_diffusePre.png'.format(opt.summary_vis_path_task, tid) )
                    vutils.save_image( ( (input_dict['pre_batch_dict_brdf']['specularPreBatch'])**(1.0/2.2) ).data,
                            '{0}/{1}_specularPre.png'.format(opt.summary_vis_path_task, tid) )
                    vutils.save_image( ( (input_dict['pre_batch_dict_brdf']['renderedImBatch'])**(1.0/2.2) ).data,
                            '{0}/{1}_renderedImage.png'.format(opt.summary_vis_path_task, tid) )

                # Save the predicted results
                for n in range(0, len(output_dict['albedoPreds']) ):
                    vutils.save_image( ( (output_dict['albedoPreds'][n] ) ** (1.0/2.2) ).data,
                            '{0}/{1}_albedoPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                for n in range(0, len(output_dict['normalPreds']) ):
                    vutils.save_image( ( 0.5*(output_dict['normalPreds'][n] + 1) ).data,
                            '{0}/{1}_normalPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                for n in range(0, len(output_dict['roughPreds']) ):
                    vutils.save_image( ( 0.5*(output_dict['roughPreds'][n] + 1) ).data,
                            '{0}/{1}_roughPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )
                for n in range(0, len(output_dict['depthPreds']) ):
                    depthOut = 1 / torch.clamp(output_dict['depthPreds'][n] + 1, 1e-6, 10) * input_dict['segAllBatch'].expand_as(output_dict['depthPreds'][n])
                    vutils.save_image( ( depthOut * input_dict['segAllBatch'].expand_as(output_dict['depthPreds'][n]) ).data,
                            '{0}/{1}_depthPred_{2}.png'.format(opt.summary_vis_path_task, tid, n) )


    if opt.is_master:
        for loss_key in loss_dict_reduced:
            writer.add_scalar('loss_val/%s'%loss_key, loss_meters[loss_key].avg, tid)

        logger.info('Evaluation timings: ' + time_meters_to_string(time_meters))



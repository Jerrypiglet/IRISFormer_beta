import torch
from torch.autograd import Variable
# import models
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import torchvision.utils as vutils
from icecream import ic
from models_def.loss_midas import ScaleAndShiftInvariantLoss, GradientLoss

regularization_loss = GradientLoss(scales=4, reduction='batch-based')

def get_labels_dict_brdf(data_batch, opt, return_input_batch_as_list=False):
    input_dict = {}
    
    input_dict['im_paths'] = data_batch['image_path']
    # Load the image from cpu to gpu
    # im_cpu = (data_batch['im_trainval'].permute(0, 3, 1, 2) )
    im_cpu = data_batch['im_trainval']
    input_dict['imBatch'] = im_cpu.cuda(non_blocking=True).contiguous()

    input_dict['brdf_loss_mask'] = data_batch['brdf_loss_mask'].cuda(non_blocking=True).contiguous()

    if_load_mask = opt.cfg.DATA.load_brdf_gt
    
    if opt.cfg.DATA.load_brdf_gt:
        # Load data from cpu to gpu
        if 'al' in opt.cfg.DATA.data_read_list:
            albedo_cpu = data_batch['albedo']
            input_dict['albedoBatch'] = albedo_cpu.cuda(non_blocking=True)

        if 'no' in opt.cfg.DATA.data_read_list:
            normal_cpu = data_batch['normal']
            input_dict['normalBatch'] = normal_cpu.cuda(non_blocking=True)

        if 'ro' in opt.cfg.DATA.data_read_list:
            rough_cpu = data_batch['rough']
            input_dict['roughBatch'] = rough_cpu.cuda(non_blocking=True)

        if 'de' in opt.cfg.DATA.data_read_list:
            depth_cpu = data_batch['depth']
            input_dict['depthBatch'] = depth_cpu.cuda(non_blocking=True)
            if 'depth_next' in data_batch:
                depth_cpu_next = data_batch['depth_next']
                input_dict['depthBatch_next'] = depth_cpu_next.cuda(non_blocking=True)

        # if (not opt.cfg.DATASET.if_no_gt_semantics):
        if if_load_mask:
            if 'mask' in data_batch:
                mask_cpu = data_batch['mask'].permute(0, 3, 1, 2) # [b, 3, h, w]
                input_dict['maskBatch'] = mask_cpu.cuda(non_blocking=True)

            segArea_cpu = data_batch['segArea']
            segEnv_cpu = data_batch['segEnv']
            segObj_cpu = data_batch['segObj']

            seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
            segBatch = seg_cpu.cuda(non_blocking=True)

            input_dict['segBRDFBatch'] = segBatch[:, 2:3, :, :]
            input_dict['segAllBatch'] = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]
        else:
            input_dict['segBRDFBatch'] = torch.ones((im_cpu.shape[0], 1, im_cpu.shape[2], im_cpu.shape[3]), dtype=torch.float32).cuda(non_blocking=True)
            input_dict['segAllBatch'] = input_dict['segBRDFBatch']

    if opt.cfg.DATA.load_semseg_gt:
        input_dict['semseg_label'] = data_batch['semseg_label'].cuda(non_blocking=True)
        input_dict['semseg_label_ori'] = data_batch['semseg_label_ori'].cuda(non_blocking=True)

    if opt.cfg.DATA.load_matseg_gt:
        matAggreMap_cpu = data_batch['mat_aggre_map'].permute(0, 3, 1, 2) # [b, 1, h, w]
        input_dict['matAggreMapBatch'] = matAggreMap_cpu.cuda(non_blocking=True)

    preBatchDict = {}
    if opt.cfg.DATA.load_brdf_gt:
        if opt.cascadeLevel > 0:
            albedoPre_cpu = data_batch['albedoPre']
            albedoPreBatch = albedoPre_cpu.cuda(non_blocking=True)

            normalPre_cpu = data_batch['normalPre']
            normalPreBatch = normalPre_cpu.cuda(non_blocking=True)

            roughPre_cpu = data_batch['roughPre']
            roughPreBatch = roughPre_cpu.cuda(non_blocking=True)

            depthPre_cpu = data_batch['depthPre']
            depthPreBatch = depthPre_cpu.cuda(non_blocking=True)

            diffusePre_cpu = data_batch['diffusePre']
            diffusePreBatch = diffusePre_cpu.cuda(non_blocking=True)

            specularPre_cpu = data_batch['specularPre']
            specularPreBatch = specularPre_cpu.cuda(non_blocking=True)

            if albedoPreBatch.size(2) < opt.imHeight or albedoPreBatch.size(3) < opt.imWidth:
                albedoPreBatch = F.interpolate(albedoPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if normalPreBatch.size(2) < opt.imHeight or normalPreBatch.size(3) < opt.imWidth:
                normalPreBatch = F.interpolate(normalPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if roughPreBatch.size(2) < opt.imHeight or roughPreBatch.size(3) < opt.imWidth:
                roughPreBatch = F.interpolate(roughPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if depthPreBatch.size(2) < opt.imHeight or depthPreBatch.size(3) < opt.imWidth:
                depthPreBatch = F.interpolate(depthPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

            # Regress the diffusePred and specular Pred
            envRow, envCol = diffusePreBatch.size(2), diffusePreBatch.size(3)
            imBatchSmall = F.adaptive_avg_pool2d(input_dict['imBatch'], (envRow, envCol) )
            diffusePreBatch, specularPreBatch = models.LSregressDiffSpec(
                    diffusePreBatch, specularPreBatch, imBatchSmall,
                    diffusePreBatch, specularPreBatch )

            if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
                diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
                specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

            renderedImBatch = diffusePreBatch + specularPreBatch

        # if opt.cascadeLevel == 0:
        assert opt.cascadeLevel == 0
        input_batch = [input_dict['imBatch']]
        if opt.ifMatMapInput:
            input_batch.append(input_dict['matAggreMapBatch'])
        if not return_input_batch_as_list:
            input_batch = torch.cat(input_batch, 1)

        # elif opt.cascadeLevel > 0:
        #     input_batch = torch.cat([input_dict['imBatch'], albedoPreBatch,
        #         normalPreBatch, roughPreBatch, depthPreBatch,
        #         diffusePreBatch, specularPreBatch], dim=1)

        #     preBatchDict.update({'albedoPreBatch': albedoPreBatch, 'normalPreBatch': normalPreBatch, 'roughPreBatch': roughPreBatch, 'depthPreBatch': depthPreBatch, 'diffusePreBatch': diffusePreBatch, 'specularPreBatch': specularPreBatch})
        #     preBatchDict['renderedImBatch'] = renderedImBatch

    return input_batch, input_dict, preBatchDict

def postprocess_brdf(input_dict, output_dict, loss_dict, opt, time_meters, eval_module_list=[], tid=-1):
    if opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
        opt.albeW, opt.normW, opt.rougW, opt.deptW = opt.cfg.MODEL_BRDF.albedoWeight, opt.cfg.MODEL_BRDF.normalWeight, opt.cfg.MODEL_BRDF.roughWeight, opt.cfg.MODEL_BRDF.depthWeight

        pixelObjNum = (torch.sum(input_dict['segBRDFBatch'] ).cpu().data).item()
        pixelAllNum = (torch.sum(input_dict['segAllBatch'] ).cpu().data).item()

        if opt.cfg.MODEL_LIGHT.enable:
            extra_dict = []

        if opt.cfg.MODEL_BRDF.enable_BRDF_decoders:
            loss_dict['loss_brdf-ALL'] = 0.

        if 'al' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            albedoPreds = []
            if opt.cfg.MODEL_BRDF.use_scale_aware_albedo:
                albedoPred = output_dict['albedoPred']
            else:
                albedoPred = output_dict['albedoPred_aligned']
            albedoPreds.append(albedoPred ) 
            # if (not opt.cfg.DATASET.if_no_gt_semantics):
            loss_dict['loss_brdf-albedo'] = []
            assert len(albedoPreds) == 1
            for n in range(0, len(albedoPreds) ):
                loss = torch.sum( (albedoPreds[n] - input_dict['albedoBatch'])
                    * (albedoPreds[n] - input_dict['albedoBatch']) * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch'] ) ) / pixelObjNum / 3.0
                if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_albedo:
                    reg_loss = regularization_loss(albedoPreds[n], input_dict['albedoBatch'], input_dict['segBRDFBatch'].squeeze())
                    loss += opt.cfg.MODEL_BRDF.loss.reg_loss_albedo_weight * reg_loss
                loss_dict['loss_brdf-albedo'].append(loss)

            loss_dict['loss_brdf-ALL'] += 4 * opt.albeW * loss_dict['loss_brdf-albedo'][-1]
            # output_dict.update({'mat_seg-albedoPreds': albedoPreds})
            loss_dict['loss_brdf-albedo'] = loss_dict['loss_brdf-albedo'][-1]
            output_dict['albedoPreds'] = [output_dict['albedoPred']]
            output_dict['albedoPreds_aligned'] = [output_dict['albedoPred_aligned']]

        if 'no' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            normalPreds = []
            normalPred = output_dict['normalPred']
            normalPreds.append(normalPred )
            # if (not opt.cfg.DATASET.if_no_gt_semantics):
            loss_dict['loss_brdf-normal'] = []
            for n in range(0, len(normalPreds) ):
                loss_dict['loss_brdf-normal'].append( torch.sum( (normalPreds[n] - input_dict['normalBatch'])
                    * (normalPreds[n] - input_dict['normalBatch']) * input_dict['segAllBatch'].expand_as(input_dict['normalBatch']) ) / pixelAllNum / 3.0)
            loss_dict['loss_brdf-ALL'] += opt.normW * loss_dict['loss_brdf-normal'][-1]
            # output_dict.update({'mat_seg-normalPreds': normalPreds})
            loss_dict['loss_brdf-normal'] = loss_dict['loss_brdf-normal'][-1]
            output_dict['normalPreds'] = normalPreds

        if 'ro' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            roughPreds = []
            roughPred = output_dict['roughPred']
            roughPreds.append(roughPred )
            # if (not opt.cfg.DATASET.if_no_gt_semantics):
            loss_dict['loss_brdf-rough'] = []
            for n in range(0, len(roughPreds) ):
                loss_dict['loss_brdf-rough'].append( torch.sum( (roughPreds[n] - input_dict['roughBatch'])
                    * (roughPreds[n] - input_dict['roughBatch']) * input_dict['segBRDFBatch'] ) / pixelObjNum )
            loss_dict['loss_brdf-ALL'] += opt.rougW * loss_dict['loss_brdf-rough'][-1]
            # output_dict.update({'mat_seg-roughPreds': roughPreds}) 
            loss_dict['loss_brdf-rough'] = loss_dict['loss_brdf-rough'][-1]
            loss_dict['loss_brdf-rough-paper'] = loss_dict['loss_brdf-rough'] / 4.
            output_dict['roughPreds'] = roughPreds

        if 'de' in opt.cfg.MODEL_BRDF.enable_list + eval_module_list:
            depthPreds = []
            depthPred = output_dict['depthPred']
            depthPreds.append(depthPred )
            # if (not opt.cfg.DATASET.if_no_gt_semantics):
            loss_dict['loss_brdf-depth'] = []
            for n in range(0, len(depthPreds ) ):
                if opt.cfg.MODEL_BRDF.loss.if_use_midas_loss_depth:
                    # alpha = 0.5 if (tid!=-1 and tid>100) else 0.
                    # alpha = 0.
                    alpha = 0.5
                    midas_loss_func = ScaleAndShiftInvariantLoss(alpha=alpha)
                    invd_pred = 1./(depthPreds[n].squeeze(1)+1.)
                    invd_gt = 1./(input_dict['depthBatch'].squeeze(1)+1.)
                    loss = midas_loss_func(invd_pred, invd_gt, mask=input_dict['segAllBatch'].squeeze())
                else:
                    loss =  torch.sum( (torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) )
                        * ( torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) ) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) ) / pixelAllNum 
                    if opt.cfg.MODEL_BRDF.loss.if_use_reg_loss_depth:
                        reg_loss = regularization_loss(depthPreds[n], input_dict['depthBatch'], input_dict['segAllBatch'].squeeze())
                        # print(reg_loss.item(), loss.item())
                        loss += opt.cfg.MODEL_BRDF.reg_loss_depth_weight * reg_loss

                loss_dict['loss_brdf-depth'].append(loss)
            loss_dict['loss_brdf-ALL'] += opt.deptW * loss_dict['loss_brdf-depth'][-1]
            # output_dict.update({'mat_seg-depthPreds': depthPreds})
            loss_dict['loss_brdf-depth'] = loss_dict['loss_brdf-depth'][-1]
            loss_dict['loss_brdf-depth-paper'] = torch.sum( (torch.log(depthPreds[-1]+0.001) - torch.log(input_dict['depthBatch']+0.001) )
                * ( torch.log(depthPreds[-1]+0.001) - torch.log(input_dict['depthBatch']+0.001) ) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) ) / pixelAllNum
            output_dict['depthPreds'] = depthPreds

    if opt.cfg.MODEL_BRDF.enable_semseg_decoder:
        semsegPred = output_dict['semseg_pred']
        semsegLabel = input_dict['semseg_label']
        # ic(semsegPred.shape, semsegLabel.shape)
        loss_dict['loss_semseg-ALL'] = opt.semseg_criterion(semsegPred, semsegLabel)

    return output_dict, loss_dict

def train_step(input_dict, output_dict, preBatchDict, optimizer, opt, if_train=True):
    if if_train:
        # Clear the gradient in optimizer
        optimizer['opEncoder'].zero_grad()
        optimizer['opAlbedo'].zero_grad()
        optimizer['opNormal'].zero_grad() 
        optimizer['opRough'].zero_grad()
        optimizer['opDepth'].zero_grad()

    ########################################################
    # Build the cascade network architecture #
    albedoPreds = []
    normalPreds = []
    roughPreds = []
    depthPreds = []

    # print(output_dict.keys())

    albedoPred, normalPred, roughPred, depthPred = output_dict['albedoPred'], output_dict['normalPred'], output_dict['roughPred'], output_dict['depthPred']
    # # Initial Prediction
    # x1, x2, x3, x4, x5, x6 = model['encoder'](input_batch)
    # albedoPred = 0.5 * (model['albedoDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6) + 1)
    # normalPred = model['normalDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6)
    # roughPred = model['roughDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6)
    # depthPred = 0.5 * (model['depthDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6 ) + 1)

    # input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
    # albedoPred = models.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred ),
    #         input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred )
    # albedoPred = torch.clamp(albedoPred, 0, 1)

    # depthPred = models.LSregress(depthPred *  input_dict['segAllBatch'].expand_as(depthPred),
    #         input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred )

    albedoPreds.append(albedoPred )
    normalPreds.append(normalPred )
    roughPreds.append(roughPred )
    depthPreds.append(depthPred )

    ########################################################

    # Compute the error
    errors = {}
    errors['albedoErrs'] = []
    errors['normalErrs'] = []
    errors['roughErrs'] = []
    errors['depthErrs'] = []
    opt.albeW, opt.normW, opt.rougW, opt.deptW = opt.cfg.MODEL_BRDF.albedoWeight, opt.cfg.MODEL_BRDF.normalWeight, opt.cfg.MODEL_BRDF.roughWeight, opt.cfg.MODEL_BRDF.depthWeight

    pixelObjNum = (torch.sum(input_dict['segBRDFBatch'] ).cpu().data).item()
    pixelAllNum = (torch.sum(input_dict['segAllBatch'] ).cpu().data).item()
    for n in range(0, len(albedoPreds) ):
       errors['albedoErrs'].append( torch.sum( (albedoPreds[n] - input_dict['albedoBatch'])
            * (albedoPreds[n] - input_dict['albedoBatch']) * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch'] ) ) / pixelObjNum / 3.0 )
    for n in range(0, len(normalPreds) ):
        errors['normalErrs'].append( torch.sum( (normalPreds[n] - input_dict['normalBatch'])
            * (normalPreds[n] - input_dict['normalBatch']) * input_dict['segAllBatch'].expand_as(input_dict['normalBatch']) ) / pixelAllNum / 3.0)
    for n in range(0, len(roughPreds) ):
        errors['roughErrs'].append( torch.sum( (roughPreds[n] - input_dict['roughBatch'])
            * (roughPreds[n] - input_dict['roughBatch']) * input_dict['segBRDFBatch'] ) / pixelObjNum )
    for n in range(0, len(depthPreds ) ):
        errors['depthErrs'].append( torch.sum( (torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) )
            * ( torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) ) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) ) / pixelAllNum )

    # Back propagate the gradients
    totalErr = 4 * opt.albeW * errors['albedoErrs'][-1] + opt.normW * errors['normalErrs'][-1] \
            + opt.rougW * errors['roughErrs'][-1] + opt.deptW * errors['depthErrs'][-1]

    if if_train:
        totalErr.backward()

        # Update the network parameter
        optimizer['opEncoder'].step()
        optimizer['opAlbedo'].step()
        optimizer['opNormal'].step()
        optimizer['opRough'].step()
        optimizer['opDepth'].step()

    preBatchDict.update({'albedoPreds': albedoPreds, 'normalPreds': normalPreds, 'roughPreds': roughPreds, 'depthPreds': depthPreds})

    return errors

def val_epoch_brdf(brdfLoaderVal, model, optimizer, writer, opt, tid):
    print('===Evaluating for %d batches'%len(brdfLoaderVal))

    for key in model:
        model[key].eval()

    loss_dict = {'loss_albedo': [], 'loss_normal': [], 'loss_rough': [], 'loss_depth': []}
    for i, data_batch in tqdm(enumerate(brdfLoaderVal)):
        
        input_batch, input_dict, preBatchDict = get_labels_dict_brdf(data_batch, opt)

        errors = train_step(input_batch, input_dict, preBatchDict, optimizer, model, opt)
        loss_dict['loss_albedo'].append(errors['albedoErrs'][0].item())
        loss_dict['loss_normal'].append(errors['normalErrs'][0].item())
        loss_dict['loss_rough'].append(errors['roughErrs'][0].item())
        loss_dict['loss_depth'].append(errors['depthErrs'][0].item())

        if i == 0:
            # if j == 1 or j% 2000 == 0:
            # Save the ground truth and the input
            vutils.save_image(( (input_dict['albedoBatch'] ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoGt.png'.format(opt.experiment, tid) )
            vutils.save_image( (0.5*(input_dict['normalBatch'] + 1) ).data,
                    '{0}/{1}_normalGt.png'.format(opt.experiment, tid) )
            vutils.save_image( (0.5*(input_dict['roughBatch'] + 1) ).data,
                    '{0}/{1}_roughGt.png'.format(opt.experiment, tid) )
            vutils.save_image( ( (input_dict['imBatch'])**(1.0/2.2) ).data,
                    '{0}/{1}_im.png'.format(opt.experiment, tid) )
            depthOut = 1 / torch.clamp(input_dict['depthBatch'] + 1, 1e-6, 10) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'])
            vutils.save_image( ( depthOut*input_dict['segAllBatch'].expand_as(input_dict['depthBatch']) ).data,
                    '{0}/{1}_depthGt.png'.format(opt.experiment, tid) )

            if opt.cascadeLevel > 0:
                vutils.save_image( ( (preBatchDict['diffusePreBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_diffusePre.png'.format(opt.experiment, tid) )
                vutils.save_image( ( (preBatchDict['specularPreBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_specularPre.png'.format(opt.experiment, tid) )
                vutils.save_image( ( (preBatchDict['renderedImBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_renderedImage.png'.format(opt.experiment, tid) )

            # Save the predicted results
            for n in range(0, len(preBatchDict['albedoPreds']) ):
                vutils.save_image( ( (preBatchDict['albedoPreds'][n] ) ** (1.0/2.2) ).data,
                        '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, tid, n) )
            for n in range(0, len(preBatchDict['normalPreds']) ):
                vutils.save_image( ( 0.5*(preBatchDict['normalPreds'][n] + 1) ).data,
                        '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, tid, n) )
            for n in range(0, len(preBatchDict['roughPreds']) ):
                vutils.save_image( ( 0.5*(preBatchDict['roughPreds'][n] + 1) ).data,
                        '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, tid, n) )
            for n in range(0, len(preBatchDict['depthPreds']) ):
                depthOut = 1 / torch.clamp(preBatchDict['depthPreds'][n] + 1, 1e-6, 10) * input_dict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n])
                vutils.save_image( ( depthOut * input_dict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n]) ).data,
                        '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, tid, n) )

    writer.add_scalar('loss_eval/loss_albedo', statistics.mean(loss_dict['loss_albedo']), tid)
    writer.add_scalar('loss_eval/loss_normal', statistics.mean(loss_dict['loss_normal']), tid)
    writer.add_scalar('loss_eval/loss_rough', statistics.mean(loss_dict['loss_rough']), tid)
    writer.add_scalar('loss_eval/loss_depth', statistics.mean(loss_dict['loss_depth']), tid)

    
    for key in model:
        model[key].train()

    print('===Evaluating finished.')



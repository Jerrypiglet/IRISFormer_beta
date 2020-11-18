import torch
from torch.autograd import Variable
import models
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import torchvision.utils as vutils

def get_input_dict_brdf(data_batch, opt):
    input_dict = {}
    
    input_dict['im_paths'] = data_batch['imPath']

    # Load data from cpu to gpu
    albedo_cpu = data_batch['albedo']
    input_dict['albedoBatch'] = Variable(albedo_cpu ).cuda(non_blocking=True)

    normal_cpu = data_batch['normal']
    input_dict['normalBatch'] = Variable(normal_cpu ).cuda(non_blocking=True)

    rough_cpu = data_batch['rough']
    input_dict['roughBatch'] = Variable(rough_cpu ).cuda(non_blocking=True)

    depth_cpu = data_batch['depth']
    input_dict['depthBatch'] = Variable(depth_cpu ).cuda(non_blocking=True)

    mask_cpu = data_batch['mask'].permute(0, 3, 1, 2) # [b, 3, h, w]
    input_dict['maskBatch'] = Variable(mask_cpu ).cuda(non_blocking=True)

    matAggreMap_cpu = data_batch['mat_aggre_map'].permute(0, 3, 1, 2) # [b, 1, h, w]
    input_dict['matAggreMapBatch'] = Variable(matAggreMap_cpu ).cuda(non_blocking=True)

    segArea_cpu = data_batch['segArea']
    segEnv_cpu = data_batch['segEnv']
    segObj_cpu = data_batch['segObj']

    seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
    segBatch = Variable(seg_cpu ).cuda(non_blocking=True)

    input_dict['segBRDFBatch'] = segBatch[:, 2:3, :, :]
    input_dict['segAllBatch'] = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]

    # Load the image from cpu to gpu
    im_cpu = (data_batch['im'] )
    input_dict['imBatch'] = Variable(im_cpu ).cuda(non_blocking=True)

    if opt.cfg.DATA.load_semseg_gt:
        input_dict['semseg_label'] = data_batch['semseg_label'].cuda(non_blocking=True)

    if opt.cascadeLevel > 0:
        albedoPre_cpu = data_batch['albedoPre']
        albedoPreBatch = Variable(albedoPre_cpu ).cuda(non_blocking=True)

        normalPre_cpu = data_batch['normalPre']
        normalPreBatch = Variable(normalPre_cpu ).cuda(non_blocking=True)

        roughPre_cpu = data_batch['roughPre']
        roughPreBatch = Variable(roughPre_cpu ).cuda(non_blocking=True)

        depthPre_cpu = data_batch['depthPre']
        depthPreBatch = Variable(depthPre_cpu ).cuda(non_blocking=True)

        diffusePre_cpu = data_batch['diffusePre']
        diffusePreBatch = Variable(diffusePre_cpu ).cuda(non_blocking=True)

        specularPre_cpu = data_batch['specularPre']
        specularPreBatch = Variable(specularPre_cpu ).cuda(non_blocking=True)

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

    preBatchDict = {}

    if opt.cascadeLevel == 0:
        if opt.ifMatMapInput:
            # matinput_dict['maskBatch'] = input_dict['maskBatch'][:, 0:1, :, :]
            inputBatch = torch.cat([input_dict['imBatch'], input_dict['matAggreMapBatch']], dim=1)
        else:
            inputBatch = input_dict['imBatch']
    elif opt.cascadeLevel > 0:
        inputBatch = torch.cat([input_dict['imBatch'], albedoPreBatch,
            normalPreBatch, roughPreBatch, depthPreBatch,
            diffusePreBatch, specularPreBatch], dim=1)

        preBatchDict.update({'albedoPreBatch': albedoPreBatch, 'normalPreBatch': normalPreBatch, 'roughPreBatch': roughPreBatch, 'depthPreBatch': depthPreBatch, 'diffusePreBatch': diffusePreBatch, 'specularPreBatch': specularPreBatch})
        preBatchDict['renderedImBatch'] = renderedImBatch

    return inputBatch, input_dict, preBatchDict

# def forward_brdf(input_dict, model):
#     output_dict = model(input_dict)
    # return output_dict

def process_brdf(input_dict, output_dict, loss_dict, opt, time_meters):
    albedoPreds = []
    normalPreds = []
    roughPreds = []
    depthPreds = []

    albedoPred, normalPred, roughPred, depthPred = output_dict['albedoPred'], output_dict['normalPred'], output_dict['roughPred'], output_dict['depthPred']

    albedoPreds.append(albedoPred )
    normalPreds.append(normalPred )
    roughPreds.append(roughPred )
    depthPreds.append(depthPred )
    

    ########################################################
    opt.albeW, opt.normW, opt.rougW, opt.deptW = opt.cfg.MODEL_BRDF.albedoWeight, opt.cfg.MODEL_BRDF.normalWeight, opt.cfg.MODEL_BRDF.roughWeight, opt.cfg.MODEL_BRDF.depthWeight

    # Compute the error
    loss_dict['loss_brdf-albedo'] = []
    loss_dict['loss_brdf-normal'] = []
    loss_dict['loss_brdf-rough'] = []
    loss_dict['loss_brdf-depth'] = []

    pixelObjNum = (torch.sum(input_dict['segBRDFBatch'] ).cpu().data).item()
    pixelAllNum = (torch.sum(input_dict['segAllBatch'] ).cpu().data).item()
    for n in range(0, len(albedoPreds) ):
       loss_dict['loss_brdf-albedo'].append( torch.sum( (albedoPreds[n] - input_dict['albedoBatch'])
            * (albedoPreds[n] - input_dict['albedoBatch']) * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch'] ) ) / pixelObjNum / 3.0 )
    for n in range(0, len(normalPreds) ):
        loss_dict['loss_brdf-normal'].append( torch.sum( (normalPreds[n] - input_dict['normalBatch'])
            * (normalPreds[n] - input_dict['normalBatch']) * input_dict['segAllBatch'].expand_as(input_dict['normalBatch']) ) / pixelAllNum / 3.0)
    for n in range(0, len(roughPreds) ):
        loss_dict['loss_brdf-rough'].append( torch.sum( (roughPreds[n] - input_dict['roughBatch'])
            * (roughPreds[n] - input_dict['roughBatch']) * input_dict['segBRDFBatch'] ) / pixelObjNum )
    for n in range(0, len(depthPreds ) ):
        loss_dict['loss_brdf-depth'].append( torch.sum( (torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) )
            * ( torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) ) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) ) / pixelAllNum )

    # Back propagate the gradients
    loss_dict['loss_brdf-ALL'] = 4 * opt.albeW * loss_dict['loss_brdf-albedo'][-1] + opt.normW * loss_dict['loss_brdf-normal'][-1] \
            + opt.rougW * loss_dict['loss_brdf-rough'][-1] + opt.deptW * loss_dict['loss_brdf-depth'][-1]
        
    output_dict.update({'mat_seg-albedoPreds': albedoPreds, 'mat_seg-normalPreds': normalPreds, 'mat_seg-roughPreds': roughPreds, 'mat_seg-depthPreds': depthPreds})

    loss_dict['loss_brdf-albedo'] = loss_dict['loss_brdf-albedo'][-1]
    loss_dict['loss_brdf-normal'] = loss_dict['loss_brdf-normal'][-1]
    loss_dict['loss_brdf-rough'] = loss_dict['loss_brdf-rough'][-1]
    loss_dict['loss_brdf-depth'] = loss_dict['loss_brdf-depth'][-1]

    output_dict['albedoPreds'] = albedoPreds
    output_dict['normalPreds'] = normalPreds
    output_dict['roughPreds'] = roughPreds
    output_dict['depthPreds'] = depthPreds

    if opt.cfg.MODEL_BRDF.enable_semseg_decoder:
        semsegPred = output_dict['semsegPred']
        semsegLabel = input_dict['semseg_label']
        loss_dict['loss_brdf-semseg'] = opt.semseg_criterion(semsegPred, semsegLabel)
        output_dict['semsegPred'] = semsegPred

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
    # x1, x2, x3, x4, x5, x6 = model['encoder'](inputBatch)
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
        
        inputBatch, input_dict, preBatchDict = get_input_dict_brdf(data_batch, opt)

        errors = train_step(inputBatch, input_dict, preBatchDict, optimizer, model, opt)
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



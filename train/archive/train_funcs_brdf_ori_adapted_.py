import torch

import torch
from torch.autograd import Variable
import models_def.models as models
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import torchvision.utils as vutils

def get_input_batch(dataBatch, opt):
    inputDict = {}
    inputDict['image_path'] = dataBatch['image_path']
    # Load data from cpu to gpu
    albedo_cpu = dataBatch['albedo']
    inputDict['albedoBatch'] = Variable(albedo_cpu ).cuda()

    normal_cpu = dataBatch['normal']
    inputDict['normalBatch'] = Variable(normal_cpu ).cuda()

    rough_cpu = dataBatch['rough']
    inputDict['roughBatch'] = Variable(rough_cpu ).cuda()

    depth_cpu = dataBatch['depth']
    inputDict['depthBatch'] = Variable(depth_cpu ).cuda()

    mask_cpu = dataBatch['mask'].permute(0, 3, 1, 2) # [b, 3, h, w]
    inputDict['maskBatch'] = Variable(mask_cpu ).cuda()

    matAggreMap_cpu = dataBatch['mat_aggre_map'].permute(0, 3, 1, 2) # [b, 1, h, w]
    inputDict['matAggreMapBatch'] = Variable(matAggreMap_cpu ).cuda()

    segArea_cpu = dataBatch['segArea']
    segEnv_cpu = dataBatch['segEnv']
    segObj_cpu = dataBatch['segObj']

    seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
    segBatch = Variable(seg_cpu ).cuda()

    inputDict['segBRDFBatch'] = segBatch[:, 2:3, :, :]
    inputDict['segAllBatch'] = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]

    # Load the image from cpu to gpu
    im_cpu = (dataBatch['im'] )
    inputDict['imBatch'] = Variable(im_cpu ).cuda()

    if opt.cascadeLevel > 0:
        albedoPre_cpu = dataBatch['albedoPre']
        albedoPreBatch = Variable(albedoPre_cpu ).cuda()

        normalPre_cpu = dataBatch['normalPre']
        normalPreBatch = Variable(normalPre_cpu ).cuda()

        roughPre_cpu = dataBatch['roughPre']
        roughPreBatch = Variable(roughPre_cpu ).cuda()

        depthPre_cpu = dataBatch['depthPre']
        depthPreBatch = Variable(depthPre_cpu ).cuda()

        diffusePre_cpu = dataBatch['diffusePre']
        diffusePreBatch = Variable(diffusePre_cpu ).cuda()

        specularPre_cpu = dataBatch['specularPre']
        specularPreBatch = Variable(specularPre_cpu ).cuda()

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
        imBatchSmall = F.adaptive_avg_pool2d(inputDict['imBatch'], (envRow, envCol) )
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
            # matinputDict['maskBatch'] = inputDict['maskBatch'][:, 0:1, :, :]
            input_batch = torch.cat([inputDict['imBatch'], inputDict['matAggreMapBatch']], dim=1)
        else:
            input_batch = inputDict['imBatch']
    elif opt.cascadeLevel > 0:
        input_batch = torch.cat([inputDict['imBatch'], albedoPreBatch,
            normalPreBatch, roughPreBatch, depthPreBatch,
            diffusePreBatch, specularPreBatch], dim=1)

        preBatchDict.update({'albedoPreBatch': albedoPreBatch, 'normalPreBatch': normalPreBatch, 'roughPreBatch': roughPreBatch, 'depthPreBatch': depthPreBatch, 'diffusePreBatch': diffusePreBatch, 'specularPreBatch': specularPreBatch})
        preBatchDict['renderedImBatch'] = renderedImBatch

    return input_batch, inputDict, preBatchDict

    
def train_step(input_batch, inputDict, preBatchDict, optimizer, model, opt, if_train=True):
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

    # Initial Prediction
    x1, x2, x3, x4, x5, x6 = model['encoder'](input_batch)
    albedoPred = 0.5 * (model['albedoDecoder'](inputDict['imBatch'], x1, x2, x3, x4, x5, x6) + 1)
    normalPred = model['normalDecoder'](inputDict['imBatch'], x1, x2, x3, x4, x5, x6)
    roughPred = model['roughDecoder'](inputDict['imBatch'], x1, x2, x3, x4, x5, x6)
    depthPred = 0.5 * (model['depthDecoder'](inputDict['imBatch'], x1, x2, x3, x4, x5, x6 ) + 1)

    inputDict['albedoBatch'] = inputDict['segBRDFBatch'] * inputDict['albedoBatch']
    albedoPred = models.LSregress(albedoPred * inputDict['segBRDFBatch'].expand_as(albedoPred ),
            inputDict['albedoBatch'] * inputDict['segBRDFBatch'].expand_as(inputDict['albedoBatch']), albedoPred )
    albedoPred = torch.clamp(albedoPred, 0, 1)

    depthPred = models.LSregress(depthPred *  inputDict['segAllBatch'].expand_as(depthPred),
            inputDict['depthBatch'] * inputDict['segAllBatch'].expand_as(inputDict['depthBatch']), depthPred )

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

    pixelObjNum = (torch.sum(inputDict['segBRDFBatch'] ).cpu().data).item()
    pixelAllNum = (torch.sum(inputDict['segAllBatch'] ).cpu().data).item()
    for n in range(0, len(albedoPreds) ):
       errors['albedoErrs'].append( torch.sum( (albedoPreds[n] - inputDict['albedoBatch'])
            * (albedoPreds[n] - inputDict['albedoBatch']) * inputDict['segBRDFBatch'].expand_as(inputDict['albedoBatch'] ) ) / pixelObjNum / 3.0 )
    for n in range(0, len(normalPreds) ):
        errors['normalErrs'].append( torch.sum( (normalPreds[n] - inputDict['normalBatch'])
            * (normalPreds[n] - inputDict['normalBatch']) * inputDict['segAllBatch'].expand_as(inputDict['normalBatch']) ) / pixelAllNum / 3.0)
    for n in range(0, len(roughPreds) ):
        errors['roughErrs'].append( torch.sum( (roughPreds[n] - inputDict['roughBatch'])
            * (roughPreds[n] - inputDict['roughBatch']) * inputDict['segBRDFBatch'] ) / pixelObjNum )
    for n in range(0, len(depthPreds ) ):
        errors['depthErrs'].append( torch.sum( (torch.log(depthPreds[n]+1) - torch.log(inputDict['depthBatch']+1) )
            * ( torch.log(depthPreds[n]+1) - torch.log(inputDict['depthBatch']+1) ) * inputDict['segAllBatch'].expand_as(inputDict['depthBatch'] ) ) / pixelAllNum )

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

def val_epoch(brdfLoaderVal, model, optimizer, writer, opt, tid):
    print('===Evaluating for %d batches'%len(brdfLoaderVal))

    for key in model:
        model[key].eval()

    loss_dict = {'loss_albedo': [], 'loss_normal': [], 'loss_rough': [], 'loss_depth': []}
    loss_dict.update({'loss_albedo-reg': [], 'loss_depth-reg': []})

    with torch.no_grad():
        for i, dataBatch in tqdm(enumerate(brdfLoaderVal)):
            
            input_batch, inputDict, preBatchDict = get_input_batch(dataBatch, opt)

            errors = train_step(input_batch, inputDict, preBatchDict, optimizer, model, opt, if_train=False)
            loss_dict['loss_albedo'].append(errors['albedoErrs'][0].item())
            loss_dict['loss_normal'].append(errors['normalErrs'][0].item())
            loss_dict['loss_rough'].append(errors['roughErrs'][0].item())
            loss_dict['loss_depth'].append(errors['depthErrs'][0].item())

            if i == 0:
                print(inputDict['image_path'])
                # if j == 1 or j% 2000 == 0:
                # Save the ground truth and the input
                vutils.save_image(( (inputDict['albedoBatch'] ) ** (1.0/2.2) ).data,
                        '{0}/{1}_albedoGt.png'.format(opt.experiment, tid) )
                vutils.save_image( (0.5*(inputDict['normalBatch'] + 1) ).data,
                        '{0}/{1}_normalGt.png'.format(opt.experiment, tid) )
                vutils.save_image( (0.5*(inputDict['roughBatch'] + 1) ).data,
                        '{0}/{1}_roughGt.png'.format(opt.experiment, tid) )
                vutils.save_image( ( (inputDict['imBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_im.png'.format(opt.experiment, tid) )
                depthOut = 1 / torch.clamp(inputDict['depthBatch'] + 1, 1e-6, 10) * inputDict['segAllBatch'].expand_as(inputDict['depthBatch'])
                vutils.save_image( ( depthOut*inputDict['segAllBatch'].expand_as(inputDict['depthBatch']) ).data,
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
                    depthOut = 1 / torch.clamp(preBatchDict['depthPreds'][n] + 1, 1e-6, 10) * inputDict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n])
                    vutils.save_image( ( depthOut * inputDict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n]) ).data,
                            '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, tid, n) )

    writer.add_scalar('loss_val/loss_brdf-albedo', statistics.mean(loss_dict['loss_albedo']), tid)
    writer.add_scalar('loss_val/loss_brdf-normal', statistics.mean(loss_dict['loss_normal']), tid)
    writer.add_scalar('loss_val/loss_brdf-rough', statistics.mean(loss_dict['loss_rough']), tid)
    writer.add_scalar('loss_val/loss_brdf-depth', statistics.mean(loss_dict['loss_depth']), tid)

    
    for key in model:
        model[key].train()

    print('===Evaluating finished.')

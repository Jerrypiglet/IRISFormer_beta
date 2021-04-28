import torch
from torch.autograd import Variable
# import models
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import torchvision.utils as vutils

from train_funcs_brdf import get_labels_dict_brdf

def get_labels_dict_light(data_batch, opt, list_from_brdf=None, return_input_batch_as_list=True):

    if list_from_brdf is None:
        input_batch, input_dict, preBatchDict = get_labels_dict_brdf(data_batch, opt, return_input_batch_as_list=True)
    else:
        input_batch, input_dict, preBatchDict = list_from_brdf
        
    extra_dict = {}

    if opt.cfg.DATA.load_light_gt:
        envmaps_cpu = data_batch['envmaps']
        envmapsBatch = Variable(envmaps_cpu ).cuda(non_blocking=True)

        envmapsInd_cpu = data_batch['envmapsInd']
        envmapsIndBatch = Variable(envmapsInd_cpu ).cuda(non_blocking=True)

        extra_dict.update({'envmapsBatch': envmapsBatch, 'envmapsIndBatch': envmapsIndBatch})

        if opt.cascadeLevel > 0:

            diffusePre_cpu = data_batch['diffusePre']
            diffusePreBatch = Variable(diffusePre_cpu ).cuda(non_blocking=True)

            specularPre_cpu = data_batch['specularPre']
            specularPreBatch = Variable(specularPre_cpu ).cuda(non_blocking=True)

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

            input_batch += [diffusePreBatch, specularPreBatch]

            preBatchDict.update({'diffusePreBatch': diffusePreBatch, 'specularPreBatch': specularPreBatch})
            preBatchDict['renderedImBatch'] = renderedImBatch

            envmapsPre_cpu = data_batch['envmapsPre']
            envmapsPreBatch = Variable(envmapsPre_cpu ).cuda(non_blocking=True)
            input_dict['envmapsPreBatch'] = envmapsPreBatch


    if not return_input_batch_as_list:
        input_batch = torch.cat(input_batch, dim=1)

    return input_batch, input_dict, preBatchDict, extra_dict

def postprocess_light(input_dict, output_dict, loss_dict, opt, time_meters):
    # Compute the recontructed error
    if opt.cfg.MODEL_LIGHT.use_scale_aware_loss:
        reconstErr = torch.sum( 
                    ( torch.log(output_dict['envmapsPredImage'] + opt.cfg.MODEL_LIGHT.offset) -
                    torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) )
                * 
                    ( torch.log(output_dict['envmapsPredImage'] + opt.cfg.MODEL_LIGHT.offset ) -
                        torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) ) 
                *
                    output_dict['segEnvBatch'].expand_as(output_dict['envmapsPredImage'] ) 
                ) \
            / output_dict['pixelNum_recon'] / 3.0 / opt.cfg.MODEL_LIGHT.envWidth / opt.cfg.MODEL_LIGHT.envHeight
    else:
        reconstErr = torch.sum( 
                    ( torch.log(output_dict['envmapsPredScaledImage'] + opt.cfg.MODEL_LIGHT.offset) -
                    torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) )
                * 
                    ( torch.log(output_dict['envmapsPredScaledImage'] + opt.cfg.MODEL_LIGHT.offset ) -
                        torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) ) 
                *
                    output_dict['segEnvBatch'].expand_as(output_dict['envmapsPredImage'] ) 
                ) \
            / output_dict['pixelNum_recon'] / 3.0 / opt.cfg.MODEL_LIGHT.envWidth / opt.cfg.MODEL_LIGHT.envHeight

    loss_dict['loss_light-reconstErr'] = reconstErr

    # Compute the rendered error
    renderErr = torch.sum( (output_dict['renderedImPred'] - output_dict['imBatchSmall'])
        * (output_dict['renderedImPred'] - output_dict['imBatchSmall']) * output_dict['segBatchSmall'].expand_as(output_dict['imBatchSmall'] )  ) \
        / output_dict['pixelNum_render'] / 3.0
    loss_dict['loss_light-renderErr'] = renderErr

    loss_dict['loss_light-ALL'] = opt.renderWeight * renderErr + opt.reconstWeight * reconstErr

    # torch.Size([4, 3, 120, 160, 8, 16]) torch.Size([4, 3, 120, 160, 8, 16]) torch.Size([4, 3, 120, 160]) torch.Size([4, 3, 120, 160])
    # print(output_dict['envmapsPredScaledImage'].shape, input_dict['envmapsBatch'].shape, output_dict['renderedImPred'].shape, output_dict['imBatchSmall'].shape)
    # import pickle
    # reindexed_pickle_path = 'a.pkl'
    # sequence = {'envmapsPredScaledImage': output_dict['envmapsPredScaledImage'].detach().cpu().numpy(), \
    #     'envmapsBatch': input_dict['envmapsBatch'].detach().cpu().numpy(), \
    #     'renderedImPred': output_dict['renderedImPred'].detach().cpu().numpy(), 
    #     'imBatchSmall': output_dict['imBatchSmall'].detach().cpu().numpy()}
    # with open(reindexed_pickle_path, 'wb') as f:
    #     pickle.dump(sequence, f, protocol=pickle.HIGHEST_PROTOCOL)



    return output_dict, loss_dict

    

# def train_step(input_dict, output_dict, preBatchDict, optimizer, opt, if_train=True):
#     if if_train:
#         # Clear the gradient in optimizer
#         optimizer['opEncoder'].zero_grad()
#         optimizer['opAlbedo'].zero_grad()
#         optimizer['opNormal'].zero_grad() 
#         optimizer['opRough'].zero_grad()
#         optimizer['opDepth'].zero_grad()

#     ########################################################
#     # Build the cascade network architecture #
#     albedoPreds = []
#     normalPreds = []
#     roughPreds = []
#     depthPreds = []

#     # print(output_dict.keys())

#     albedoPred, normalPred, roughPred, depthPred = output_dict['albedoPred'], output_dict['normalPred'], output_dict['roughPred'], output_dict['depthPred']
#     # # Initial Prediction
#     # x1, x2, x3, x4, x5, x6 = model['encoder'](input_batch)
#     # albedoPred = 0.5 * (model['albedoDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6) + 1)
#     # normalPred = model['normalDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6)
#     # roughPred = model['roughDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6)
#     # depthPred = 0.5 * (model['depthDecoder'](input_dict['imBatch'], x1, x2, x3, x4, x5, x6 ) + 1)

#     # input_dict['albedoBatch'] = input_dict['segBRDFBatch'] * input_dict['albedoBatch']
#     # albedoPred = models.LSregress(albedoPred * input_dict['segBRDFBatch'].expand_as(albedoPred ),
#     #         input_dict['albedoBatch'] * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch']), albedoPred )
#     # albedoPred = torch.clamp(albedoPred, 0, 1)

#     # depthPred = models.LSregress(depthPred *  input_dict['segAllBatch'].expand_as(depthPred),
#     #         input_dict['depthBatch'] * input_dict['segAllBatch'].expand_as(input_dict['depthBatch']), depthPred )

#     albedoPreds.append(albedoPred )
#     normalPreds.append(normalPred )
#     roughPreds.append(roughPred )
#     depthPreds.append(depthPred )

#     ########################################################

#     # Compute the error
#     errors = {}
#     errors['albedoErrs'] = []
#     errors['normalErrs'] = []
#     errors['roughErrs'] = []
#     errors['depthErrs'] = []
#     opt.albeW, opt.normW, opt.rougW, opt.deptW = opt.cfg.MODEL_BRDF.albedoWeight, opt.cfg.MODEL_BRDF.normalWeight, opt.cfg.MODEL_BRDF.roughWeight, opt.cfg.MODEL_BRDF.depthWeight

#     pixelObjNum = (torch.sum(input_dict['segBRDFBatch'] ).cpu().data).item()
#     pixelAllNum = (torch.sum(input_dict['segAllBatch'] ).cpu().data).item()
#     for n in range(0, len(albedoPreds) ):
#        errors['albedoErrs'].append( torch.sum( (albedoPreds[n] - input_dict['albedoBatch'])
#             * (albedoPreds[n] - input_dict['albedoBatch']) * input_dict['segBRDFBatch'].expand_as(input_dict['albedoBatch'] ) ) / pixelObjNum / 3.0 )
#     for n in range(0, len(normalPreds) ):
#         errors['normalErrs'].append( torch.sum( (normalPreds[n] - input_dict['normalBatch'])
#             * (normalPreds[n] - input_dict['normalBatch']) * input_dict['segAllBatch'].expand_as(input_dict['normalBatch']) ) / pixelAllNum / 3.0)
#     for n in range(0, len(roughPreds) ):
#         errors['roughErrs'].append( torch.sum( (roughPreds[n] - input_dict['roughBatch'])
#             * (roughPreds[n] - input_dict['roughBatch']) * input_dict['segBRDFBatch'] ) / pixelObjNum )
#     for n in range(0, len(depthPreds ) ):
#         errors['depthErrs'].append( torch.sum( (torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) )
#             * ( torch.log(depthPreds[n]+1) - torch.log(input_dict['depthBatch']+1) ) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'] ) ) / pixelAllNum )

#     # Back propagate the gradients
#     totalErr = 4 * opt.albeW * errors['albedoErrs'][-1] + opt.normW * errors['normalErrs'][-1] \
#             + opt.rougW * errors['roughErrs'][-1] + opt.deptW * errors['depthErrs'][-1]

#     if if_train:
#         totalErr.backward()

#         # Update the network parameter
#         optimizer['opEncoder'].step()
#         optimizer['opAlbedo'].step()
#         optimizer['opNormal'].step()
#         optimizer['opRough'].step()
#         optimizer['opDepth'].step()

#     preBatchDict.update({'albedoPreds': albedoPreds, 'normalPreds': normalPreds, 'roughPreds': roughPreds, 'depthPreds': depthPreds})

#     return errors

# def val_epoch_brdf(brdfLoaderVal, model, optimizer, writer, opt, tid):
#     print('===Evaluating for %d batches'%len(brdfLoaderVal))

#     for key in model:
#         model[key].eval()

#     loss_dict = {'loss_albedo': [], 'loss_normal': [], 'loss_rough': [], 'loss_depth': []}
#     for i, data_batch in tqdm(enumerate(brdfLoaderVal)):
        
#         input_batch, input_dict, preBatchDict = get_labels_dict_brdf(data_batch, opt)

#         errors = train_step(input_batch, input_dict, preBatchDict, optimizer, model, opt)
#         loss_dict['loss_albedo'].append(errors['albedoErrs'][0].item())
#         loss_dict['loss_normal'].append(errors['normalErrs'][0].item())
#         loss_dict['loss_rough'].append(errors['roughErrs'][0].item())
#         loss_dict['loss_depth'].append(errors['depthErrs'][0].item())

#         if i == 0:
#             # if j == 1 or j% 2000 == 0:
#             # Save the ground truth and the input
#             vutils.save_image(( (input_dict['albedoBatch'] ) ** (1.0/2.2) ).data,
#                     '{0}/{1}_albedoGt.png'.format(opt.experiment, tid) )
#             vutils.save_image( (0.5*(input_dict['normalBatch'] + 1) ).data,
#                     '{0}/{1}_normalGt.png'.format(opt.experiment, tid) )
#             vutils.save_image( (0.5*(input_dict['roughBatch'] + 1) ).data,
#                     '{0}/{1}_roughGt.png'.format(opt.experiment, tid) )
#             vutils.save_image( ( (input_dict['imBatch'])**(1.0/2.2) ).data,
#                     '{0}/{1}_im.png'.format(opt.experiment, tid) )
#             depthOut = 1 / torch.clamp(input_dict['depthBatch'] + 1, 1e-6, 10) * input_dict['segAllBatch'].expand_as(input_dict['depthBatch'])
#             vutils.save_image( ( depthOut*input_dict['segAllBatch'].expand_as(input_dict['depthBatch']) ).data,
#                     '{0}/{1}_depthGt.png'.format(opt.experiment, tid) )

#             if opt.cascadeLevel > 0:
#                 vutils.save_image( ( (preBatchDict['diffusePreBatch'])**(1.0/2.2) ).data,
#                         '{0}/{1}_diffusePre.png'.format(opt.experiment, tid) )
#                 vutils.save_image( ( (preBatchDict['specularPreBatch'])**(1.0/2.2) ).data,
#                         '{0}/{1}_specularPre.png'.format(opt.experiment, tid) )
#                 vutils.save_image( ( (preBatchDict['renderedImBatch'])**(1.0/2.2) ).data,
#                         '{0}/{1}_renderedImage.png'.format(opt.experiment, tid) )

#             # Save the predicted results
#             for n in range(0, len(preBatchDict['albedoPreds']) ):
#                 vutils.save_image( ( (preBatchDict['albedoPreds'][n] ) ** (1.0/2.2) ).data,
#                         '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, tid, n) )
#             for n in range(0, len(preBatchDict['normalPreds']) ):
#                 vutils.save_image( ( 0.5*(preBatchDict['normalPreds'][n] + 1) ).data,
#                         '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, tid, n) )
#             for n in range(0, len(preBatchDict['roughPreds']) ):
#                 vutils.save_image( ( 0.5*(preBatchDict['roughPreds'][n] + 1) ).data,
#                         '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, tid, n) )
#             for n in range(0, len(preBatchDict['depthPreds']) ):
#                 depthOut = 1 / torch.clamp(preBatchDict['depthPreds'][n] + 1, 1e-6, 10) * input_dict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n])
#                 vutils.save_image( ( depthOut * input_dict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n]) ).data,
#                         '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, tid, n) )

#     writer.add_scalar('loss_eval/loss_albedo', statistics.mean(loss_dict['loss_albedo']), tid)
#     writer.add_scalar('loss_eval/loss_normal', statistics.mean(loss_dict['loss_normal']), tid)
#     writer.add_scalar('loss_eval/loss_rough', statistics.mean(loss_dict['loss_rough']), tid)
#     writer.add_scalar('loss_eval/loss_depth', statistics.mean(loss_dict['loss_depth']), tid)

    
#     for key in model:
#         model[key].train()

#     print('===Evaluating finished.')



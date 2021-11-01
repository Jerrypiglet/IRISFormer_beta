import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def get_labels_dict_iiw(data_batch, opt, return_input_batch_as_list=False):
    input_dict = {}
    
    input_dict['im_paths'] = data_batch['image_path']
    im_cpu = data_batch['im_trainval']
    input_dict['imBatch'] = im_cpu.cuda(non_blocking=True).contiguous()

    if opt.cfg.DEBUG.if_test_real:
        input_dict['pad_mask'] = data_batch['pad_mask'].cuda(non_blocking=True).contiguous()
        input_dict['im_h_resized_to'] = data_batch['im_h_resized_to']
        input_dict['im_w_resized_to'] = data_batch['im_w_resized_to']


    assert opt.cascadeLevel == 0
    input_batch = [input_dict['imBatch']]
    if not return_input_batch_as_list:
        input_batch = torch.cat(input_batch, 1)

    eq = data_batch['eq']
    darker = data_batch['darker']
    input_dict.update({'eq': eq, 'darker': darker})

    # elif opt.cascadeLevel > 0:
    #     input_batch = torch.cat([input_dict['imBatch'], albedoPreBatch,
    #         normalPreBatch, roughPreBatch, depthPreBatch,
    #         diffusePreBatch, specularPreBatch], dim=1)

    #     preBatchDict.update({'albedoPreBatch': albedoPreBatch, 'normalPreBatch': normalPreBatch, 'roughPreBatch': roughPreBatch, 'depthPreBatch': depthPreBatch, 'diffusePreBatch': diffusePreBatch, 'specularPreBatch': specularPreBatch})
    #     preBatchDict['renderedImBatch'] = renderedImBatch

    return input_batch, input_dict

def postprocess_iiw(input_dict, output_dict, loss_dict, opt, time_meters, eval_module_list=[], tid=-1, if_loss=True):

    if if_loss:
        albedoPred = output_dict['albedoPred']
        output_dict['albedoPreds'] = [output_dict['albedoPred']]
        output_dict['normalPreds'] = [output_dict['normalPred']]
        output_dict['depthPreds'] = [output_dict['depthPred']]
        output_dict['roughPreds'] = [output_dict['roughPred']]

        eq, darker = input_dict['eq'], input_dict['darker']        

        eqLoss, darkerLoss = 0, 0
        for m in range(0, albedoPred.size(0) ):
            eqPoint = eq[m]['point'].astype(np.long )
            eqWeight = eq[m]['weight'].astype(np.float32 )
            eqNum = eq[m]['num']
            eqPoint = eqPoint[0:eqNum, :]
            eqWeight = eqWeight[0:eqNum ]

            darkerPoint = darker[m]['point'].astype(np.long )
            darkerWeight = darker[m]['weight'].astype(np.float32 )
            darkerNum = darker[m]['num']
            darkerPoint = darkerPoint[0:darkerNum, :]
            darkerWeight = darkerWeight[0:darkerNum ]
            eqL, darkerL = \
                BatchRankingLoss(albedoPred[m],
                    eqPoint, eqWeight,
                    darkerPoint, darkerWeight )
            eqLoss += eqL
            darkerLoss += darkerL

        eqLoss = eqLoss / max(albedoPred.size(0 ), 1e-5)
        darkerLoss = darkerLoss / max(albedoPred.size(0), 1e-5)

        loss_dict['loss_iiw-eq'] = eqLoss
        loss_dict['loss_iiw-darker'] = darkerLoss

        rankW = opt.rankWeight
        loss_dict['loss_iiw-ALL'] = rankW * loss_dict['loss_iiw-eq'] + rankW * loss_dict['loss_iiw-darker']

    return output_dict, loss_dict


def BatchRankingLoss(albedoPred, eqPoint, eqWeight, darkerPoint, darkerWeight):
    tau = 0.5
    height, width = albedoPred.size(1), albedoPred.size(2)

    reflectance = torch.mean(albedoPred, dim=0)
    reflectLog = torch.log(reflectance + 0.001)
    reflectLog = reflectLog.view(-1)

    eqPoint = Variable(torch.from_numpy(eqPoint ).long() ).cuda( )
    eqWeight = Variable(torch.from_numpy(eqWeight ).float()  ).cuda( )
    darkerPoint = Variable(torch.from_numpy(darkerPoint ).long() ).cuda( )
    darkerWeight = Variable(torch.from_numpy(darkerWeight ).float() ).cuda( )

    # compute the eq loss
    r1, c1, r2, c2 = torch.split(eqPoint, 1, dim=1)
    p1 = (r1 * width + c1).view(-1)
    p1.requires_grad = False
    p2 = (r2 * width + c2).view(-1)
    p2.requires_grad = False
    rf1 = torch.index_select(reflectLog, 0, p1)
    rf2 = torch.index_select(reflectLog, 0, p2)
    eqWeight = eqWeight.view(-1)

    eqLoss = torch.mean(eqWeight * torch.pow(rf1 - rf2, 2) )

    # compute the darker loss
    r1, c1, r2, c2 = torch.split(darkerPoint, 1, dim=1)
    p1 = (r1 * width + c1).view(-1)
    p1.requires_grad = False
    p2 = (r2 * width + c2).view(-1)
    p2.requires_grad = False
    rf1 = torch.index_select(reflectLog, 0, p1)
    rf2 = torch.index_select(reflectLog, 0, p2)
    darkerWeight = darkerWeight.view(-1)

    darkerLoss = torch.mean(darkerWeight * torch.pow(F.relu(rf2 - rf1 + tau), 2) )

    return eqLoss, darkerLoss

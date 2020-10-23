import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
os.system('touch models/__init__.py')
os.system('touch utils/__init__.py')
print('started.')
import models
import torchvision.utils as vutils
import utils
from dataset_openrooms_noDDP import openrooms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from tqdm import tqdm
import time
from train_funcs import *
from utils.utils_vis import vis_index_map
import torchvision.transforms as T

from models.baseline_same import Baseline as UNet
# import yaml
import os, inspect
pwdpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from utils.config import cfg
from utils.misc import AverageMeter, get_optimizer, get_datetime
from train_funcs_mat_seg_noDDP import get_input_dict_mat_seg, forward_mat_seg, val_epoch_mat_seg
from utils.bin_mean_shift import Bin_Mean_Shift


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to input images')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epochs for training')
parser.add_argument('--nepoch1', type=int, default=10, help='the number of epochs for training')

parser.add_argument('--batchSize0', type=int, default=16, help='input batch size; ALL GPUs')
parser.add_argument('--batchSize1', type=int, default=16, help='input batch size; ALL GPUs')

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to model')
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to model')
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to model')
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to model')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for training model')
# Fine tune the model
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the model')
parser.add_argument('--epochIdFineTune', type=int, default = 0, help='the training of epoch of the loaded model')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for depth component')   

# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# Rui
parser.add_argument('--ifMatMapInput', action='store_true', help='using mask as additional input')
parser.add_argument('--ifDataloaderOnly', action='store_true', help='benchmark dataloading overhead')
parser.add_argument('--ifCluster', action='store_true', help='if using cluster')
parser.add_argument('--if_hdr', action='store_true', help='if using hdr images')
parser.add_argument('--eval_every_iter', type=int, default=2000, help='the casacade level')
parser.add_argument('--invalid_index', type=int, default = 255, help='index for invalid aread (e.g. windows, lights)')
parser.add_argument('--resume', type=str, help='resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp); [to continue the current task, use: resume]', default='NoCkpt')
parser.add_argument(
    "--config-file",
    default=os.path.join(pwdpath, "configs/config.yaml"),
    metavar="FILE",
    help="path to config file",
    type=str,
)

# The detail model setting
opt = parser.parse_args()
print(opt)

cfg.merge_from_file(opt.config_file)
print(cfg)

opt.gpuId = opt.deviceIds[0]

opt.albeW, opt.normW = opt.albedoWeight, opt.normalWeight
opt.rougW = opt.roughWeight
opt.deptW = opt.depthWeight

if opt.cascadeLevel == 0:
    opt.nepoch = opt.nepoch0
    opt.batchSize = opt.batchSize0
    opt.imHeight, opt.imWidth = opt.imHeight0, opt.imWidth0
elif opt.cascadeLevel == 1:
    opt.nepoch = opt.nepoch1
    opt.batchSize = opt.batchSize1
    opt.imHeight, opt.imWidth = opt.imHeight1, opt.imWidth1

if opt.experiment is None:
    opt.experiment = 'check_cascade%d_w%d_h%d' % (opt.cascadeLevel,
            opt.imWidth, opt.imHeight )
if opt.ifCluster:
    opt.experiment = 'logs/' + opt.experiment
else:
    opt.experiment = 'logs/' + get_datetime() + opt.experiment
if opt.ifCluster:
    opt.experiment = '/viscompfs/users/ruizhu/' + opt.experiment
os.system('rm -rf {0}'.format(opt.experiment) )
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp -r train %s' % opt.experiment )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# # Initial model
# encoder = models.encoder0(cascadeLevel = opt.cascadeLevel, in_channels = 3 if not opt.ifMatMapInput else 4)
# albedoDecoder = models.decoder0(mode=0 )
# normalDecoder = models.decoder0(mode=1 )
# roughDecoder = models.decoder0(mode=2 )
# depthDecoder = models.decoder0(mode=4 )
# ####################################################################


# #########################################
# lr_scale = 1
# if opt.isFineTune:
#     print('--- isFineTune=True')
#     encoder.load_state_dict(
#             torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     albedoDecoder.load_state_dict(
#             torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     normalDecoder.load_state_dict(
#             torch.load('{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     roughDecoder.load_state_dict(
#             torch.load('{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     depthDecoder.load_state_dict(
#             torch.load('{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
#     lr_scale = 1.0 / (2.0 ** (np.floor( ( (opt.epochIdFineTune+1) / 10)  ) ) )
# else:
#     opt.epochIdFineTune = -1
# #########################################
# model = {}
# model['encoder'] = nn.DataParallel(encoder, device_ids = opt.deviceIds )
# model['albedoDecoder'] = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
# model['normalDecoder'] = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
# model['roughDecoder'] = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
# model['depthDecoder'] = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

# ##############  ######################
# # Send things into GPU
# if opt.cuda:
#     model['encoder'] = model['encoder'].cuda(opt.gpuId )
#     model['albedoDecoder'] = model['albedoDecoder'].to(opt.device)
#     model['normalDecoder'] = model['normalDecoder'].to(opt.device)
#     model['roughDecoder'] = model['roughDecoder'].to(opt.device)
#     model['depthDecoder'] = model['depthDecoder'].to(opt.device)
# ####################################


# ####################################
# # Optimizer
# optimizer = {}
# optimizer['opEncoder'] = optim.Adam(model['encoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opAlbedo'] = optim.Adam(model['albedoDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opNormal'] = optim.Adam(model['normalDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opRough'] = optim.Adam(model['roughDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# optimizer['opDepth'] = optim.Adam(model['depthDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
# #####################################

# ----------------- Rui from Plane paper 
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build model
model = UNet(cfg.model)
if not (opt.resume == 'NoCkpt'):
    model_dict = torch.load(opt.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)


# load nets into gpu
opt.num_gpus = len(opt.deviceIds)
if opt.num_gpus > 1 and torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
model.to(opt.device)

# set up optimizers
optimizer = get_optimizer(model.parameters(), cfg.solver)

model.train(not cfg.model.fix_bn)

bin_mean_shift = Bin_Mean_Shift(device=opt.device, invalid_index=opt.invalid_index)


####################################
transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
brdfDatasetTrain = openrooms( opt.dataRoot, transforms, opt, 
        imWidth = opt.imWidth, imHeight = opt.imHeight,
        cascadeLevel = opt.cascadeLevel, split = 'train')
brdfLoaderTrain = DataLoader(brdfDatasetTrain, batch_size = opt.batchSize,
        num_workers = 16, shuffle = True, pin_memory=True)

if 'mini' in opt.dataRoot:
    print('=====!!!!===== mini: brdfDatasetVal = brdfDatasetTrain')
    brdfDatasetVal = brdfDatasetTrain
else:
    brdfDatasetVal = openrooms( opt.dataRoot, transforms, opt, 
            imWidth = opt.imWidth, imHeight = opt.imHeight,
            cascadeLevel = opt.cascadeLevel, split = 'val')
# brdfLoaderVal = DataLoader(brdfDatasetVal, batch_size = opt.batchSize,
#         num_workers = 16, shuffle = False, pin_memory=True)

writer = SummaryWriter(log_dir=opt.experiment, flush_secs=10) # relative path


tid = 0
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

ts_iter_end_start_list = []
ts_iter_start_end_list = []
num_mat_masks_MAX = 0

print('=======1')

for epoch in list(range(opt.epochIdFineTune+1, opt.nepoch) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')

    losses = AverageMeter()
    losses_pull = AverageMeter()
    losses_push = AverageMeter()
    losses_binary = AverageMeter()

    ts_epoch_start = time.time()
    # ts = ts_epoch_start
    # ts_iter_start = ts
    ts_iter_end = ts_epoch_start

    print('=======3')
    for i, data_batch in tqdm(enumerate(brdfLoaderTrain)):
        # if opt.eval_every_iter != -1 and tid % opt.eval_every_iter == 0:
        #     val_epoch_mat_seg(brdfLoaderVal, model, bin_mean_shift, writer, opt, tid)
        #     model.train(not cfg.model.fix_bn)
        #     ts_iter_end = time.time()
            
            # break

        # num_mat_masks_MAX = max(np.max(input_dict['num_mat_masks_batch'].numpy()), num_mat_masks_MAX)

        # continue

        tid += 1
        ts_iter_start = time.time()
        if tid > 5:
            ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

        if opt.ifDataloaderOnly:
            continue

        # ======= Load data from cpu to gpu
        input_dict = get_input_dict_mat_seg(data_batch, opt)

        if tid % 1000 == 0:
            for sample_idx in tqdm(range(opt.batchSize0)):
                # im_single = im_cpu[sample_idx].numpy().squeeze().transpose(1, 2, 0)
                # im_single = im_single**(1.0/2.2)
                im_single = data_batch['im_not_hdr'][sample_idx].numpy().squeeze().transpose(1, 2, 0)

                writer.add_image('TRAIN_im/%d'%sample_idx, im_single, tid, dataformats='HWC')

                mat_aggre_map_single = input_dict['mat_aggre_map_cpu'][sample_idx].numpy().squeeze()
                matAggreMap_single_vis = vis_index_map(mat_aggre_map_single)
                writer.add_image('TRAIN_mat_aggre_map/%d'%sample_idx, matAggreMap_single_vis, tid, dataformats='HWC')

                mat_notlight_mask_single = input_dict['mat_notlight_mask_cpu'][sample_idx].numpy().squeeze()
                writer.add_image('TRAIN_mat_notlight_mask/%d'%sample_idx, mat_notlight_mask_single, tid, dataformats='HW')

                writer.add_text('TRAIN_im_path/%d'%sample_idx, input_dict['im_paths'][sample_idx], tid)
            
        # ======= Forward
        output_dict, loss_dict = forward_mat_seg(input_dict, model, opt)
        loss = loss_dict['loss']

        # ======= Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ======= update loss
        losses.update(loss_dict['loss'].item())
        losses_pull.update(loss_dict['loss_pull'].item())
        losses_push.update(loss_dict['loss_push'].item())
        losses_binary.update(loss_dict['loss_binary'].item())

        writer.add_scalar('loss_train/loss_all', loss_dict['loss'].item(), tid)
        writer.add_scalar('loss_train/loss_pull', loss_dict['loss_pull'].item(), tid)
        writer.add_scalar('loss_train/loss_push', loss_dict['loss_push'].item(), tid)
        writer.add_scalar('loss_train/loss_binary', loss_dict['loss_binary'].item(), tid)
        writer.add_scalar('training/epoch', epoch, tid)

        print('Epoch %d - Tid %d - loss_all %.3f = loss_pull %.3f + loss_push %.3f + loss_binary %.3f' % (epoch, tid, loss_dict['loss'].item(), loss_dict['loss_pull'].item(), loss_dict['loss_push'].item(), loss_dict['loss_binary'].item()))

        # End of iteration
        ts_iter_end = time.time()
        if tid > 5:
            ts_iter_start_end_list.append(ts_iter_end - ts_iter_start)
            if tid % 20 == 0:
                print('Rolling end-to-start %.2f, Rolling start-to-end %.2f'%(sum(ts_iter_end_start_list)/len(ts_iter_end_start_list), sum(ts_iter_start_end_list)/len(ts_iter_start_end_list)))
            # print(ts_iter_end_start_list, ts_iter_start_end_list)



        # break


    #     albedo_cpu = data_batch['albedo']
    #     albedoBatch = Variable(albedo_cpu ).to(opt.device)

    #     normal_cpu = data_batch['normal']
    #     normalBatch = Variable(normal_cpu ).to(opt.device)

    #     rough_cpu = data_batch['rough']
    #     roughBatch = Variable(rough_cpu ).to(opt.device)

    #     depth_cpu = data_batch['depth']
    #     depthBatch = Variable(depth_cpu ).to(opt.device)


    #     segArea_cpu = data_batch['segArea']
    #     segEnv_cpu = data_batch['segEnv']
    #     segObj_cpu = data_batch['segObj']

    #     seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
    #     segBatch = Variable(seg_cpu ).to(opt.device)

    #     segBRDFBatch = segBatch[:, 2:3, :, :]
    #     segAllBatch = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]

    #     # Load the image from cpu to gpu
    #     im_cpu = (data_batch['im'] )
    #     im_batch = Variable(im_cpu ).to(opt.device)


    #     if opt.cascadeLevel > 0:
    #         albedoPre_cpu = data_batch['albedoPre']
    #         albedoPreBatch = Variable(albedoPre_cpu ).to(opt.device)

    #         normalPre_cpu = data_batch['normalPre']
    #         normalPreBatch = Variable(normalPre_cpu ).to(opt.device)

    #         roughPre_cpu = data_batch['roughPre']
    #         roughPreBatch = Variable(roughPre_cpu ).to(opt.device)

    #         depthPre_cpu = data_batch['depthPre']
    #         depthPreBatch = Variable(depthPre_cpu ).to(opt.device)

    #         diffusePre_cpu = data_batch['diffusePre']
    #         diffusePreBatch = Variable(diffusePre_cpu ).to(opt.device)

    #         specularPre_cpu = data_batch['specularPre']
    #         specularPreBatch = Variable(specularPre_cpu ).to(opt.device)

    #         if albedoPreBatch.size(2) < opt.imHeight or albedoPreBatch.size(3) < opt.imWidth:
    #             albedoPreBatch = F.interpolate(albedoPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if normalPreBatch.size(2) < opt.imHeight or normalPreBatch.size(3) < opt.imWidth:
    #             normalPreBatch = F.interpolate(normalPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if roughPreBatch.size(2) < opt.imHeight or roughPreBatch.size(3) < opt.imWidth:
    #             roughPreBatch = F.interpolate(roughPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if depthPreBatch.size(2) < opt.imHeight or depthPreBatch.size(3) < opt.imWidth:
    #             depthPreBatch = F.interpolate(depthPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

    #         # Regress the diffusePred and specular Pred
    #         envRow, envCol = diffusePreBatch.size(2), diffusePreBatch.size(3)
    #         im_batchSmall = F.adaptive_avg_pool2d(im_batch, (envRow, envCol) )
    #         diffusePreBatch, specularPreBatch = models.LSregressDiffSpec(
    #                 diffusePreBatch, specularPreBatch, im_batchSmall,
    #                 diffusePreBatch, specularPreBatch )

    #         if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
    #             diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
    #         if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
    #             specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

    #         renderedim_batch = diffusePreBatch + specularPreBatch


    #     # Clear the gradient in optimizer
    #     opEncoder.zero_grad()
    #     opAlbedo.zero_grad()
    #     opNormal.zero_grad()
    #     opRough.zero_grad()
    #     opDepth.zero_grad()

    #     ########################################################
    #     # Build the cascade model architecture #
    #     albedoPreds = []
    #     normalPreds = []
    #     roughPreds = []
    #     depthPreds = []

    #     if opt.cascadeLevel == 0:
    #         if opt.isMatMaskInput:
    #             inputBatch = torch.cat([im_batch, matMaskBatch], dim=1)
    #         else:
    #             inputBatch = im_batch
    #     elif opt.cascadeLevel > 0:
    #         inputBatch = torch.cat([im_batch, albedoPreBatch,
    #             normalPreBatch, roughPreBatch, depthPreBatch,
    #             diffusePreBatch, specularPreBatch], dim=1)

    #     # Initial Prediction
    #     x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    #     albedoPred = 0.5 * (albedoDecoder(im_batch, x1, x2, x3, x4, x5, x6) + 1)
    #     normalPred = normalDecoder(im_batch, x1, x2, x3, x4, x5, x6)
    #     roughPred = roughDecoder(im_batch, x1, x2, x3, x4, x5, x6)
    #     depthPred = 0.5 * (depthDecoder(im_batch, x1, x2, x3, x4, x5, x6 ) + 1)

    #     albedoBatch = segBRDFBatch * albedoBatch
    #     albedoPred = models.LSregress(albedoPred * segBRDFBatch.expand_as(albedoPred ),
    #             albedoBatch * segBRDFBatch.expand_as(albedoBatch), albedoPred )
    #     albedoPred = torch.clamp(albedoPred, 0, 1)

    #     depthPred = models.LSregress(depthPred *  segAllBatch.expand_as(depthPred),
    #             depthBatch * segAllBatch.expand_as(depthBatch), depthPred )

    #     albedoPreds.append(albedoPred )
    #     normalPreds.append(normalPred )
    #     roughPreds.append(roughPred )
    #     depthPreds.append(depthPred )

    #     ########################################################

    #     # Compute the error
    #     albedoErrs = []
    #     normalErrs = []
    #     roughErrs = []
    #     depthErrs = []

    #     pixelObjNum = (torch.sum(segBRDFBatch ).cpu().data).item()
    #     pixelAllNum = (torch.sum(segAllBatch ).cpu().data).item()
    #     for n in range(0, len(albedoPreds) ):
    #         albedoErrs.append( torch.sum( (albedoPreds[n] - albedoBatch)
    #             * (albedoPreds[n] - albedoBatch) * segBRDFBatch.expand_as(albedoBatch ) ) / pixelObjNum / 3.0 )
    #     for n in range(0, len(normalPreds) ):
    #         normalErrs.append( torch.sum( (normalPreds[n] - normalBatch)
    #             * (normalPreds[n] - normalBatch) * segAllBatch.expand_as(normalBatch) ) / pixelAllNum / 3.0)
    #     for n in range(0, len(roughPreds) ):
    #         roughErrs.append( torch.sum( (roughPreds[n] - roughBatch)
    #             * (roughPreds[n] - roughBatch) * segBRDFBatch ) / pixelObjNum )
    #     for n in range(0, len(depthPreds ) ):
    #         depthErrs.append( torch.sum( (torch.log(depthPreds[n]+1) - torch.log(depthBatch+1) )
    #             * ( torch.log(depthPreds[n]+1) - torch.log(depthBatch+1) ) * segAllBatch.expand_as(depthBatch ) ) / pixelAllNum )

    #     # Back propagate the gradients
    #     totalErr = 4 * albeW * albedoErrs[-1] + normW * normalErrs[-1] \
    #             + rougW *roughErrs[-1] + deptW * depthErrs[-1]
    #     totalErr.backward()

    #     # Update the model parameter
    #     opEncoder.step()
    #     opAlbedo.step()
    #     opNormal.step()
    #     opRough.step()
    #     opDepth.step()

    #     # Output training error
    #     utils.writeErrToScreen('albedo', albedoErrs, epoch, j )
    #     utils.writeErrToScreen('normal', normalErrs, epoch, j )
    #     utils.writeErrToScreen('rough', roughErrs, epoch, j )
    #     utils.writeErrToScreen('depth', depthErrs, epoch, j )

    #     utils.writeErrToFile('albedo', albedoErrs, trainingLog, epoch, j )
    #     utils.writeErrToFile('normal', normalErrs, trainingLog, epoch, j )
    #     utils.writeErrToFile('rough', roughErrs, trainingLog, epoch, j )
    #     utils.writeErrToFile('depth', depthErrs, trainingLog, epoch, j )

    #     albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
    #     normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
    #     roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
    #     depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)

    #     if j < 1000:
    #         utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)

    #         utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
    #     else:
    #         utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
    #         utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)

    #         utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
    #         utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)


    #     if j == 1 or j% 2000 == 0:
    #         # Save the ground truth and the input
    #         vutils.save_image(( (albedoBatch ) ** (1.0/2.2) ).data,
    #                 '{0}/{1}_albedoGt.png'.format(opt.experiment, j) )
    #         vutils.save_image( (0.5*(normalBatch + 1) ).data,
    #                 '{0}/{1}_normalGt.png'.format(opt.experiment, j) )
    #         vutils.save_image( (0.5*(roughBatch + 1) ).data,
    #                 '{0}/{1}_roughGt.png'.format(opt.experiment, j) )
    #         vutils.save_image( ( (im_batch)**(1.0/2.2) ).data,
    #                 '{0}/{1}_im.png'.format(opt.experiment, j) )
    #         depthOut = 1 / torch.clamp(depthBatch + 1, 1e-6, 10) * segAllBatch.expand_as(depthBatch)
    #         vutils.save_image( ( depthOut*segAllBatch.expand_as(depthBatch) ).data,
    #                 '{0}/{1}_depthGt.png'.format(opt.experiment, j) )

    #         if opt.cascadeLevel > 0:
    #             vutils.save_image( ( (diffusePreBatch)**(1.0/2.2) ).data,
    #                     '{0}/{1}_diffusePre.png'.format(opt.experiment, j) )
    #             vutils.save_image( ( (specularPreBatch)**(1.0/2.2) ).data,
    #                     '{0}/{1}_specularPre.png'.format(opt.experiment, j) )
    #             vutils.save_image( ( (renderedim_batch)**(1.0/2.2) ).data,
    #                     '{0}/{1}_renderedImage.png'.format(opt.experiment, j) )

    #         # Save the predicted results
    #         for n in range(0, len(albedoPreds) ):
    #             vutils.save_image( ( (albedoPreds[n] ) ** (1.0/2.2) ).data,
    #                     '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
    #         for n in range(0, len(normalPreds) ):
    #             vutils.save_image( ( 0.5*(normalPreds[n] + 1) ).data,
    #                     '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
    #         for n in range(0, len(roughPreds) ):
    #             vutils.save_image( ( 0.5*(roughPreds[n] + 1) ).data,
    #                     '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )
    #         for n in range(0, len(depthPreds) ):
    #             depthOut = 1 / torch.clamp(depthPreds[n] + 1, 1e-6, 10) * segAllBatch.expand_as(depthPreds[n])
    #             vutils.save_image( ( depthOut * segAllBatch.expand_as(depthPreds[n]) ).data,
    #                     '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )

    #     writer.

    # trainingLog.close()

    # # Update the training rate
    # if (epoch + 1) % 10 == 0:
    #     for param_group in opEncoder.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opAlbedo.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opNormal.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opRough.param_groups:
    #         param_group['lr'] /= 2
    #     for param_group in opDepth.param_groups:
    #         param_group['lr'] /= 2
    # # Save the error record
    # np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    # np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    # np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    # np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )

    # # save the models
    # torch.save(encoder.module, '{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(albedoDecoder.module, '{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(normalDecoder.module, '{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(roughDecoder.module, '{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    # torch.save(depthDecoder.module, '{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )

print('num_mat_masks_MAX', num_mat_masks_MAX) 

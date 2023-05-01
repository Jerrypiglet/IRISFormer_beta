import torch
import numpy as np
import cv2
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def initializeEnv(imOrig ):
    im = np.sum(imOrig, axis=2 )
    height, width = im.shape
    im = im.flatten()
    imId = np.argmax(im )
    rowId = int(imId / width )
    colId = imId - rowId * width


    # The weight
    weight = imOrig[rowId, colId, :]
    # The theta
    theta = (rowId + 0.5) / height * np.pi
    phi = ((colId + 0.5 ) / width - 0.5) * np.pi * 2

    ambient = np.ones(3, np.float32 )

    return weight, theta, phi, ambient


class SGEnvOptimSky():
    def __init__(self,
            ambientValue, weightValue, thetaValue, phiValue,
            gpuId = 0, niter = 40, envNum = 1, isCuda = True,
            envWidth = 512, envHeight = 256, SGRow = 1, SGCol = 1, ch = 3,
            isFixLightPos = True ):

        self.SGNum = int(SGRow*SGCol )

        self.envNum = envNum
        self.niter = niter
        self.ch = ch
        self.isFixLightPos = isFixLightPos

        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 ) * 2 * np.pi
        El = ( (np.arange(envHeight) + 0.5) / envHeight ) * np.pi # [0, pi]
        Az, El = np.meshgrid(Az, El )
        Az = Az[:, :, np.newaxis ]
        El = El[:, :, np.newaxis ]
        lx = np.sin(El ) * np.cos(Az )
        ly = np.sin(El ) * np.sin(Az )
        lz = np.cos(El )
        self.ls = np.concatenate( (lx, ly, lz), axis = 2)[np.newaxis, np.newaxis, np.newaxis, :]
        self.ls = Variable(torch.from_numpy(self.ls.astype(np.float32) ) )
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.iterCount = 0

        self.W = Variable(torch.from_numpy(np.sin(El.astype(np.float32) ).reshape( (1, 1, envHeight, envWidth) ) ) )
        self.W[:, :, 0:int(envHeight/2), :] = 0 # set the upper half of the envmap to 0
        self.envmap = Variable(torch.zeros( (self.envNum, self.ch, self.envHeight, self.envWidth), dtype=torch.float32 ) )

        self.isCuda = isCuda
        self.gpuId = gpuId

        thetaValue = max(thetaValue, np.pi/2.0 + 0.01 )
        thetaValue = (thetaValue - np.pi/2.0)/ np.pi*4 - 1
        thetaValue = 0.5 * np.log((1 + thetaValue) / (1 - thetaValue) )
        phiValue = phiValue / np.pi
        phiValue = 0.5 * np.log((1+phiValue) / (1-phiValue) )

        weightValue = np.log(weightValue + 1e-5).squeeze()
        weightValue = weightValue.tolist()
        weight = Variable(torch.zeros( (self.envNum, self.SGNum, self.ch), dtype = torch.float32) )
        weight[:, :, 0] += weightValue[0]
        weight[:, :, 1] += weightValue[1]
        weight[:, :, 2] += weightValue[2]

        ambientValue = np.log(ambientValue + 1e-5).squeeze()
        ambientValue = ambientValue.tolist()
        ambient = Variable(torch.zeros( (self.envNum, self.SGNum, self.ch), dtype = torch.float32 ) )
        ambient[:, :, 0] += ambientValue[0]
        ambient[:, :, 1] += ambientValue[1]
        ambient[:, :, 2] += ambientValue[2]

        theta = Variable(torch.zeros( (self.envNum, self.SGNum, 1), dtype = torch.float32 ) ) + thetaValue
        phi = Variable(torch.zeros( (self.envNum, self.SGNum, 1 ), dtype = torch.float32 ) ) + phiValue
        lamb = torch.log(Variable(torch.ones(self.envNum, self.SGNum, 1) * np.pi * 2.0 ) )

        self.weight = weight.unsqueeze(-1).unsqueeze(-1)
        self.ambient = ambient.unsqueeze(-1).unsqueeze(-1)
        self.theta = theta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.phi = phi.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.lamb = lamb.unsqueeze(-1).unsqueeze(-1)

        if isCuda:
            self.ls = self.ls.cuda(self.gpuId )

            self.ambient = self.ambient.cuda()
            self.weight = self.weight.cuda()
            self.theta = self.theta.cuda()
            self.phi = self.phi.cuda()
            self.lamb = self.lamb.cuda()

            self.envmap = self.envmap.cuda(self.gpuId )
            self.W = self.W.cuda(self.gpuId )

        self.ambient.requires_grad = True
        self.weight.requires_grad = True
        self.theta.requires_grad = True
        self.phi.requires_grad = True
        self.lamb.requires_grad = True

        self.mseLoss = nn.MSELoss(size_average = False )
        if self.isFixLightPos:
            self.optEnvAdam = optim.Adam([self.ambient, self.weight, self.lamb], lr=1e-2 )
        else:
            self.optEnvAdam = optim.Adam([self.weight,
                self.theta, self.phi, self.lamb], lr=1e-2 )

    def renderSG(self, theta, phi, lamb, weight, ambient ):
        axisX = torch.sin(theta ) * torch.cos(phi )
        axisY = torch.sin(theta ) * torch.sin(phi )
        axisZ = torch.cos(theta )

        axis = torch.cat([axisX, axisY, axisZ], dim=5)

        mi = lamb.expand([self.envNum, self.SGNum, 1, self.envHeight, self.envWidth] ) * \
                (torch.sum(
                    axis.expand([self.envNum, self.SGNum, 1, self.envHeight, self.envWidth, 3] ) * \
                            self.ls.expand([self.envNum, self.SGNum, 1, self.envHeight, self.envWidth, 3] ),
                            dim = 5) -1 )
        envmaps = weight.expand([self.envNum, self.SGNum, self.ch, self.envHeight, self.envWidth] ) * \
                torch.exp(mi ).expand([self.envNum, self.SGNum, self.ch,
                    self.envHeight, self.envWidth] )
        # print(torch.max(envmaps), torch.min(envmaps), ambient.squeeze())
        # envmaps = envmaps + ambient
        envmaps[:, :, :, self.envHeight//2:, :] = envmaps[:, :, :, self.envHeight//2:, :] + ambient

        envmap = torch.sum(envmaps, dim=1 )

        return envmap

    def optimizeAdam(self, envmap, if_print=False):
        assert(envmap.shape[0] == self.envNum
                and envmap.shape[1] == self.ch
                and envmap.shape[2] == self.envHeight
                and envmap.shape[3] == self.envWidth )
        self.envmap.data.copy_(torch.from_numpy(envmap.copy()))

        minLoss = 2e20
        recImageBest = None
        thetaBest = None
        phiBest = None
        weightBest = None
        ambientBest = None
        lambBest = None
        self.loss = None
        self.loss_list = []

        for i in range(0, self.niter ):

            if if_print:
                print('Iteration %d' % i )

            for j in range(0, 100):
                theta = 0.25*np.pi * (torch.tanh(self.theta )+1) + np.pi/2.0 + 0.01
                phi = np.pi * torch.tanh(self.phi )
                weight = torch.exp(self.weight )
                ambient = torch.exp(self.ambient )
                lamb = torch.clamp(torch.exp(self.lamb ), max=70 )

                recImage = self.renderSG(theta, phi, lamb, weight, ambient )
                loss = self.mseLoss(
                        recImage * self.W.expand_as(recImage),
                        self.envmap * self.W.expand_as(recImage) )

                self.loss = loss

                self.optEnvAdam.zero_grad()
                loss.backward()
                self.iterCount += 1

                self.optEnvAdam.step()

            step_loss = loss.item() / self.envNum / self.envWidth / self.envHeight / self.ch 
            self.loss_list.append(step_loss)
            if if_print:
                print ('[Iteration %d] Step %d Loss: %f' % (i, self.iterCount, step_loss ) )

            if self.loss.cpu().data.item() < minLoss:
                if torch.isnan(torch.sum(self.theta ) ) or \
                        torch.isnan(torch.sum(self.phi) ) or \
                        torch.isnan(torch.sum(self.weight ) ) or \
                        torch.isnan(torch.sum(self.lamb ) ) or \
                        torch.isnan(torch.sum(self.ambient ) ) or \
                        torch.isinf(torch.sum(self.theta ) ) or \
                        torch.isinf(torch.sum(self.phi ) ) or \
                        torch.isinf(torch.sum(self.weight ) ) or \
                        torch.isinf(torch.sum(self.lamb ) ) or \
                        torch.isinf(torch.sum(self.ambient ) ):
                    break
                else:
                    theta = 0.25*np.pi * (torch.tanh(self.theta )+1) + np.pi/2.0 + 0.01
                    phi = np.pi * torch.tanh(self.phi )
                    weight = torch.exp(self.weight )
                    ambient = torch.exp(self.ambient )
                    lamb = torch.clamp(torch.exp(self.lamb ), max=70 )

                    recImage = self.renderSG(theta, phi, lamb, weight, ambient )

                    recImageBest = recImage.cpu().data.numpy()


                    thetaBest = theta.data.cpu().numpy().reshape( (self.envNum, self.SGNum, 1) )
                    phiBest = phi.data.cpu().numpy().squeeze().reshape( (self.envNum, self.SGNum, 1) )
                    lambBest = lamb.data.cpu().numpy().squeeze().reshape( (self.envNum, self.SGNum, 1) )
                    weightBest = weight.data.cpu().numpy().squeeze().reshape( (self.envNum, self.SGNum, 3) )
                    ambientBest = ambient.data.cpu().numpy().squeeze().reshape( (self.envNum, self.SGNum, 3) )

                    minLoss = self.loss.cpu()

                    del theta, phi, weight, ambient, lamb, recImage
            else:
                break


        return thetaBest, phiBest, lambBest, weightBest, ambientBest, recImageBest

from util import *
from render import *
from descriptor import FeatureLoss, StyleLoss, FeatureMaskLoss
from torchvision.transforms import Normalize
sys.path.insert(1, 'PerceptualSimilarity/')
import models
#import kornia
import torch

np.set_printoptions(precision=4, suppress=True)

def plotAndSave(loss, save_dir):
    plt.figure(figsize=(8,4))
    for i in range(loss.shape[1]):
        plt.plot(loss[:,i], label='%.4f' % loss[-1,i])
    plt.legend()
    plt.savefig(save_dir)
    plt.close()

    plt.figure(figsize=(8,4))
    for i in range(loss.shape[1]):
        plt.plot(np.log1p(loss[:,i]), label='%.4f' % np.log1p(loss[-1,i]))
    plt.legend()
    plt.savefig(save_dir[:-4]+'_log.png')
    plt.close()

def normalize_vgg19(input, isGram):
    if isGram:
        transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
    else:
        transform = Normalize(
            mean=[0.48501961, 0.45795686, 0.40760392],
            std=[1./255, 1./255, 1./255]
        )
    #return transform(input)
    return transform(input.cpu()).cuda()

class Losses:
    def __init__(self, args, textures_init, textures_ref, rendered_ref, res, size, lp, cp):
        self.args = args
        self.textures_init = textures_init
        self.textures_ref = textures_ref
        self.rendered_ref = rendered_ref
        self.res = res
        self.size = size
        self.lp = lp
        self.cp = cp


        if args.applyMask or args.applyMask2:
            self.criterion = th.nn.MSELoss(reduction='none').cuda()
            png = Image.open(args.mask_path)
            if png.height != 256:
                png = png.resize((png.width // png.height * 256, 256))
            png = gyPIL2Array(png)
            self.mask = torch.from_numpy(png).unsqueeze(0).unsqueeze(1).cuda()

            # # Fill in Ref unseen pixels with average color of seen pixels
            # avgColor = torch.sum(self.textures_ref * self.mask, dim=(3, 2), keepdim=True) / torch.sum(self.mask)
            # self.textures_ref = torch.ones_like(self.textures_ref) * avgColor * (1-self.mask) + self.textures_ref * self.mask

            if args.findInit:
                initMap = torch.sum(self.textures_ref * self.mask, dim=(3, 2), keepdim=True) / torch.sum(self.mask)
                self.textures_ref = torch.ones_like(self.textures_ref) * initMap
                self.mask = torch.ones_like(self.textures_ref[:, :3, :, :])
        else:
            self.criterion = th.nn.MSELoss().cuda()
            self.mask = None

        self.precompute()

    def precompute(self):

        self.FL_w = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_w)
        for p in self.FL_w.parameters():
            p.requires_grad = False

        self.FL_n = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_n)
        for p in self.FL_n.parameters():
            p.requires_grad = False

        self.FL_wn = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
        for p in self.FL_wn.parameters():
            p.requires_grad = False

        self.SL = StyleLoss()
        for p in self.SL.parameters():
            p.requires_grad = False

        #self.LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        #self.Laplacian = kornia.filters.Laplacian(3, normalized=False)

        if self.args.embed_tex:
            self.albedo_ref_pix, self.normal_ref_pix, self.rough_ref_pix, self.specular_ref_pix, \
            self.albedo_ref_vgg, self.normal_ref_vgg, self.rough_ref_vgg, self.specular_ref_vgg = \
            self.eval_texture_vgg(self.textures_ref)

            # if self.mask is not None:
            #     print(self.albedo_ref_pix.shape, self.mask.shape)
            #     self.albedo_ref_pix = self.albedo_ref_pix * self.mask
            #     self.normal_ref_pix = self.normal_ref_pix * self.mask
            #     self.rough_ref_pix = self.rough_ref_pix * self.mask
            #     self.specular_ref_pix = self.specular_ref_pix * self.mask

            self.fl_albedo_ref_w   = self.FL_w(self.albedo_ref_vgg)
            self.fl_normal_ref_w   = self.FL_w(self.normal_ref_vgg)
            self.fl_rough_ref_w    = self.FL_w(self.rough_ref_vgg)
            self.fl_specular_ref_w = self.FL_w(self.specular_ref_vgg)

            self.fl_albedo_ref_n   = self.FL_n(self.albedo_ref_vgg)
            self.fl_normal_ref_n   = self.FL_n(self.normal_ref_vgg)
            self.fl_rough_ref_n    = self.FL_n(self.rough_ref_vgg)
            self.fl_specular_ref_n = self.FL_n(self.specular_ref_vgg)

            self.fl_albedo_ref_wn   = self.FL_wn(self.albedo_ref_vgg)
            self.fl_normal_ref_wn   = self.FL_wn(self.normal_ref_vgg)
            self.fl_rough_ref_wn    = self.FL_wn(self.rough_ref_vgg)
            self.fl_specular_ref_wn = self.FL_wn(self.specular_ref_vgg)

            if (self.args.applyMask or self.args.applyMask2) and self.args.alignPixStd:
                if self.args.alignVGG:
                    self.FML_wn = FeatureMaskLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
                    for p in self.FML_wn.parameters():
                        p.requires_grad = False
                    self.fml_albedo_ref_wn   = self.FML_wn(self.albedo_ref_vgg, self.mask)
                    self.fml_normal_ref_wn   = self.FML_wn(self.normal_ref_vgg, self.mask)
                    self.fml_rough_ref_wn    = self.FML_wn(self.rough_ref_vgg, self.mask)
                    self.fml_specular_ref_wn = self.FML_wn(self.specular_ref_vgg, self.mask)

                self.std_albedo_list = []
                if not self.args.diffuseOnly or self.args.applyMask2:
                    self.std_normal_list = []
                    self.std_rough_list = []
                    self.std_specular_list = []
                for c in range(3):
                    self.std_albedo_list.append(torch.std(torch.masked_select(self.albedo_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                    if not self.args.diffuseOnly:
                        self.std_normal_list.append(torch.std(torch.masked_select(self.normal_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                        self.std_specular_list.append(torch.std(torch.masked_select(self.specular_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                        if c == 0:
                            self.std_rough_list.append(torch.std(torch.masked_select(self.rough_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )

        else:
            self.rendered_ref_pix = self.rendered_ref
            self.rendered_ref_vgg = self.eval_render_vgg(self.rendered_ref, self.args.jittering)

            if self.args.jittering:
                self.sl_rendered_ref = self.SL(self.rendered_ref_vgg)
            else:
                self.fl_rendered_ref_w = self.FL_w(self.rendered_ref_vgg)
                self.fl_rendered_ref_n = self.FL_n(self.rendered_ref_vgg)
                self.fl_rendered_ref_wn = self.FL_wn(self.rendered_ref_vgg)

    def eval_texture_vgg(self, textures):
        albedo, normal, rough, specular = tex2map(textures)
        albedo    = albedo.clamp(eps,1) ** (1/2.2)
        normal    = (normal+1)/2
        rough     = rough.clamp(eps,1) ** (1/2.2)
        specular  = specular.clamp(eps,1) ** (1/2.2)

        albedo_vgg = normalize_vgg19(albedo[0,:].cpu(), False).unsqueeze(0).cuda()
        normal_vgg = normalize_vgg19(normal[0,:].cpu(), False).unsqueeze(0).cuda()
        rough_vgg  = normalize_vgg19(rough[0,:].cpu(), False).unsqueeze(0).cuda()
        specular_vgg  = normalize_vgg19(specular[0,:].cpu(), False).unsqueeze(0).cuda()

        return albedo, normal, rough, specular, albedo_vgg, normal_vgg, rough_vgg, specular_vgg

    def eval_render_vgg(self, rendered, isGram):
        rendered_tmp = rendered.clone()
        for i in range(self.args.num_render_used):
            rendered_tmp[i,:] = normalize_vgg19(rendered[i,:], isGram)
        return rendered_tmp

    def eval_render_jitter(self, textures, li):

        renderOBJ = Microfacet(res=self.res, size=self.size)
        rendered = th.zeros(self.args.num_render_used, 3, self.res, self.res).cuda()
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:]
            rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())

        renderOBJ_jitter = Microfacet(res=self.res, size=self.size)
        rendered_jitter = th.zeros(self.args.num_render_used, 3, self.res, self.res).cuda()
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:] + np.random.randn(*self.lp[i,:].shape) * 0.1
            rendered_jitter[i,:] = renderOBJ_jitter.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())

        return rendered.clamp(eps,1)**(1/2.2), rendered_jitter.clamp(eps,1)**(1/2.2)

    def eval_render(self, textures, li):

        renderOBJ = Microfacet(res=self.res, size=self.size)
        rendered = th.zeros(self.args.num_render_used, 3, self.res, self.res).cuda()
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:]
            rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())
        return rendered.clamp(eps,1)**(1/2.2)


    def eval(self, textures_pre, light, type, epoch):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        if self.args.embed_tex:
            albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
            albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
            self.eval_texture_vgg(textures_pre)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl  = self.criterion(albedo_pre_pix, self.albedo_ref_pix)
                if not self.args.diffuseOnly:
                    loss_pl += self.criterion(normal_pre_pix, self.normal_ref_pix)
                    loss_pl += self.criterion(rough_pre_pix,  self.rough_ref_pix)
                    loss_pl += self.criterion(specular_pre_pix,  self.specular_ref_pix)

                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()


            if self.args.loss_weight[1] > -eps:
                # feature loss
                if type == 'L':
                    loss_fl  = self.criterion(self.FL_w(albedo_pre_vgg), self.fl_albedo_ref_w) * 0.1
                    if not self.args.diffuseOnly:
                        loss_fl += self.criterion(self.FL_w(normal_pre_vgg), self.fl_normal_ref_w) * 0.7
                        loss_fl += self.criterion(self.FL_w(rough_pre_vgg),  self.fl_rough_ref_w) * 0.1
                        loss_fl += self.criterion(self.FL_w(specular_pre_vgg),  self.fl_specular_ref_w) * 0.1
                elif type == 'N':
                    loss_fl  = self.criterion(self.FL_n(albedo_pre_vgg), self.fl_albedo_ref_n) * 0.1
                    if not self.args.diffuseOnly:
                        loss_fl += self.criterion(self.FL_n(normal_pre_vgg), self.fl_normal_ref_n) * 0.7
                        loss_fl += self.criterion(self.FL_n(rough_pre_vgg),  self.fl_rough_ref_n) * 0.1
                        loss_fl += self.criterion(self.FL_n(specular_pre_vgg),  self.fl_specular_ref_n) * 0.1
                elif type == 'LN':
                    loss_fl  = self.criterion(self.FL_wn(albedo_pre_vgg), self.fl_albedo_ref_wn) * 0.1
                    if not self.args.diffuseOnly:
                        loss_fl += self.criterion(self.FL_wn(normal_pre_vgg), self.fl_normal_ref_wn) * 0.7
                        loss_fl += self.criterion(self.FL_wn(rough_pre_vgg),  self.fl_rough_ref_wn) * 0.1
                        loss_fl += self.criterion(self.FL_wn(specular_pre_vgg),  self.fl_specular_ref_wn) * 0.1
                else:
                    print('Latent type wrong!')
                    exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            # if self.args.loss_weight[2] > -eps:
                # # pixel loss
                # loss_lpips  = self.LPIPS.forward(albedo_pre_pix, self.albedo_ref_pix).sum() * 0.1
                # if not self.args.diffuseOnly:
                #     loss_lpips += self.LPIPS.forward(normal_pre_pix, self.normal_ref_pix).sum() * 0.7
                #     loss_lpips += self.LPIPS.forward(rough_pre_pix,  self.rough_ref_pix).sum() * 0.1
                #     loss_lpips += self.LPIPS.forward(specular_pre_pix,  self.specular_ref_pix).sum() * 0.1

                # if self.args.loss_weight[2] > eps:
                #     loss_lpips *= self.args.loss_weight[2]
                #     loss += loss_lpips
                # losses[2] = loss_lpips.item()

        else:
            if self.args.jittering:
                rendered_pre_pix, rendered_pre_pix_jitter = self.eval_render_jitter(textures_pre, light)
                rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix_jitter,True)
            else:
                rendered_pre_pix = self.eval_render(textures_pre, light)
                rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix, False)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl = self.criterion(rendered_pre_pix, self.rendered_ref_pix)
                # print('loss_pl:', loss_pl)
                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

            if self.args.loss_weight[1] > -eps:
                # feature loss
                if self.args.jittering:
                    loss_fl = self.criterion(self.SL(rendered_pre_vgg), self.sl_rendered_ref)
                else:
                    if type == 'L':
                        loss_fl = self.criterion(self.FL_w(rendered_pre_vgg), self.fl_rendered_ref_w)
                    elif type == 'N':
                        loss_fl = self.criterion(self.FL_n(rendered_pre_vgg), self.fl_rendered_ref_n)
                    elif type == 'LN':
                        loss_fl = self.criterion(self.FL_wn(rendered_pre_vgg), self.fl_rendered_ref_wn)
                    else:
                        print('Latent type wrong!')
                        exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            if self.args.loss_weight[2] > -eps:
                # pixel loss
                loss_lpips = self.LPIPS.forward(rendered_pre_pix, self.rendered_ref_pix).sum()

                if self.args.loss_weight[2] > eps:
                    loss_lpips *= self.args.loss_weight[2]
                    loss += loss_lpips
                losses[2] = loss_lpips.item()

            #if self.args.loss_weight[3] > -eps:
            #    # pixel loss
            #    loss_tex  = self.criterion(self.textures_init[:,0:3,:,:], textures_pre[:,0:3,:,:]) * 0.4
            #    loss_tex += self.criterion(self.textures_init[:,3:5,:,:], textures_pre[:,3:5,:,:]) * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,5,:,:],   textures_pre[:,5,:,:])   * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,6:9,:,:], textures_pre[:,6:9,:,:]) * 0.4

            #    lap = th.norm(self.Laplacian(self.textures_init - textures_pre).flatten())
            #    # print(lap.item())
            #    loss_tex += lap * 5e-5

            #    # print('loss_tex:', loss_tex)
            #    if self.args.loss_weight[3] > eps:
            #        loss_tex *= self.args.loss_weight[3]
            #        loss += loss_tex
            #    losses[3] = loss_tex.item()
        # print('loss_total:', loss)

        return loss, losses

    def evalMasked(self, textures_pre, light, type, epoch):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        if self.args.embed_tex:
            albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
            albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
            self.eval_texture_vgg(textures_pre)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                #loss_pl  = (albedo_pre_pix - self.albedo_ref_pix)

                pixNum = torch.sum(self.mask)
                #loss_pl  = torch.sum(self.criterion(albedo_pre_pix, self.albedo_ref_pix) * self.mask) / (3*pixNum)
                loss_pl  = torch.sum(self.criterion(albedo_pre_pix * self.mask, self.albedo_ref_pix * self.mask) ) / (3*pixNum)
                if not self.args.diffuseOnly:
                    loss_pl += torch.sum(self.criterion(normal_pre_pix, self.normal_ref_pix) * self.mask) / (3*pixNum)
                    loss_pl += torch.sum(self.criterion(rough_pre_pix, self.rough_ref_pix) * self.mask) / (pixNum)
                    loss_pl += torch.sum(self.criterion(specular_pre_pix, self.specular_ref_pix) * self.mask) / (3*pixNum)

                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

                #if self.args.alignPixMean and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixMean and (epoch >= 1000):
                    pixNum2 = torch.sum(1-self.mask)
                    pixNumAll = pixNum + pixNum2
                    #loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum), torch.sum(albedo_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum) , torch.sum(albedo_pre_pix) / (3*pixNumAll) )
                    if not self.args.diffuseOnly:
                        #loss_mean += self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix * (1-self.mask)) / (3*pixNum2) )
                        loss_mean += self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix) / (3*pixNumAll) )
                        #loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix * (1-self.mask)) / (pixNum2) )
                        loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix) / pixNumAll )
                        #loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix * (1-self.mask)) / (3*pixNum2) )
                        loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix ) / (3*pixNumAll) )
                    loss_mean *= self.args.loss_weight[1]
                    # if not self.args.diffuseOnly:
                    #     loss_mean *= 0.01
                    loss += loss_mean
                    losses[1] = loss_mean.item()

                #if self.args.alignPixStd and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixStd and (epoch >= 1000):
                    loss_std = 0
                    for c in range(3):
                        #std1 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        std1 = self.std_albedo_list[c]
                        #std2 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        std2 = torch.std(albedo_pre_pix[:,c,:,:] )
                        loss_std += self.criterion(std1, std2)
                        # if not self.args.diffuseOnly:
                        #     #std1 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        #     std1 = self.std_normal_list[c]
                        #     std2 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        #     loss_std += self.criterion(std1, std2)
                        #     #std1 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        #     std1 = self.std_specular_list[c]
                        #     std2 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        #     loss_std += self.criterion(std1, std2)
                        #     if c == 0:
                        #         #std1 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        #         std1 = self.std_rough_list[c]
                        #         std2 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        #         loss_std += self.criterion(std1, std2)
                    loss_std *= self.args.loss_weight[2]
                    # if not self.args.diffuseOnly:
                    #     loss_std *= 0.01
                    loss += loss_std
                    losses[2] = loss_std.item()

                if self.args.alignVGG and (epoch >= 1000):
                    denom = th.sum(th.ones_like(self.fml_albedo_ref_wn))
                    # loss_vgg = th.sum(self.criterion(self.FML_wn(albedo_pre_vgg, self.mask), self.fml_albedo_ref_wn) ) / denom
                    # if not self.args.diffuseOnly:
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(normal_pre_vgg, self.mask), self.fml_normal_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(rough_pre_vgg, self.mask), self.fml_rough_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(specular_pre_vgg, self.mask), self.fml_specular_ref_wn) ) / denom
                    full_mask = th.ones_like(self.mask)
                    #fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, full_mask)
                    fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, 1-self.mask)
                    denom2 = th.sum(th.ones_like(fml_albedo_pre_wn))
                    loss_vgg = self.criterion(th.sum(fml_albedo_pre_wn) / denom2, th.sum(self.fml_albedo_ref_wn) / denom)
                    if not self.args.diffuseOnly:
                        # fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, full_mask)
                        fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, 1-self.mask)
                        loss_vgg += self.criterion(th.sum(fml_normal_pre_wn) / denom2, th.sum(self.fml_normal_ref_wn) / denom)
                        # fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, full_mask)
                        fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, 1-self.mask)
                        loss_vgg += self.criterion(th.sum(fml_rough_pre_wn) / denom2, th.sum(self.fml_rough_ref_wn) / denom)
                        # fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, full_mask)
                        fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, 1-self.mask)
                        loss_vgg += self.criterion(th.sum(fml_specular_pre_wn) / denom2, th.sum(self.fml_specular_ref_wn) / denom)

                    loss_vgg *= self.args.loss_weight[3]
                    loss += loss_vgg
                    losses[3] = loss_vgg.item()


        return loss, losses

    def evalMasked2(self, textures_pre, light, type, epoch):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        if self.args.embed_tex:
            albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
            albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
            self.eval_texture_vgg(textures_pre)
            maskFull = torch.ones_like(self.mask)
            pixNumFull = torch.sum(maskFull)
            if self.args.loss_weight[0] > -eps:
                # pixel loss
                #loss_pl  = (albedo_pre_pix - self.albedo_ref_pix)

                pixNum = torch.sum(self.mask)
                #loss_pl  = torch.sum(self.criterion(albedo_pre_pix, self.albedo_ref_pix) * self.mask) / (3*pixNum)
                loss_pl  = torch.sum(self.criterion(albedo_pre_pix * maskFull, self.albedo_ref_pix * maskFull) ) / (3*pixNumFull)
                loss_pl += torch.sum(self.criterion(normal_pre_pix * self.mask, self.normal_ref_pix * self.mask) ) / (3*pixNum)
                loss_pl += torch.sum(self.criterion(rough_pre_pix * self.mask, self.rough_ref_pix * self.mask) ) / (pixNum)
                loss_pl += torch.sum(self.criterion(specular_pre_pix* self.mask, self.specular_ref_pix* self.mask) ) / (3*pixNum)

                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

                #if self.args.alignPixMean and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixMean and (epoch >= 100):
                    pixNum2 = torch.sum(1-self.mask)
                    pixNumAll = pixNum + pixNum2
                    #loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum), torch.sum(albedo_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    #loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum) , torch.sum(albedo_pre_pix) / (3*pixNumAll) )
                    #loss_mean += self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    loss_mean = self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix) / (3*pixNumAll) )
                    #loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix * (1-self.mask)) / (pixNum2) )
                    loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix) / pixNumAll )
                    #loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix ) / (3*pixNumAll) )
                    loss_mean *= self.args.loss_weight[1]
                    # if not self.args.diffuseOnly:
                    #     loss_mean *= 0.01
                    loss += loss_mean
                    losses[1] = loss_mean.item()

                #if self.args.alignPixStd and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixStd and (epoch >= 100):
                    loss_std = 0
                    # for c in range(3):
                    #     #std1 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #     # std1 = self.std_albedo_list[c]
                    #     # #std2 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #     # std2 = torch.std(albedo_pre_pix[:,c,:,:] )
                    #     # loss_std += self.criterion(std1, std2)
                    #     #std1 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #     std1 = self.std_normal_list[c]
                    #     std2 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #     loss_std += self.criterion(std1, std2)
                    #     #std1 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #     std1 = self.std_specular_list[c]
                    #     std2 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #     loss_std += self.criterion(std1, std2)
                    #     if c == 0:
                    #         #std1 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #         std1 = self.std_rough_list[c]
                    #         std2 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #         loss_std += self.criterion(std1, std2)
                    # loss_std *= self.args.loss_weight[2]
                    # # if not self.args.diffuseOnly:
                    # #     loss_std *= 0.01
                    # loss += loss_std
                    # losses[2] = loss_std.item()

                if self.args.alignVGG and (epoch >= 100):
                    denom = th.sum(th.ones_like(self.fml_albedo_ref_wn))
                    # loss_vgg = th.sum(self.criterion(self.FML_wn(albedo_pre_vgg, self.mask), self.fml_albedo_ref_wn) ) / denom
                    # if not self.args.diffuseOnly:
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(normal_pre_vgg, self.mask), self.fml_normal_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(rough_pre_vgg, self.mask), self.fml_rough_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(specular_pre_vgg, self.mask), self.fml_specular_ref_wn) ) / denom
                    full_mask = th.ones_like(self.mask)
                    #fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, full_mask)
                    fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, 1-self.mask)
                    denom2 = th.sum(th.ones_like(fml_albedo_pre_wn))
                    # loss_vgg = self.criterion(th.sum(fml_albedo_pre_wn) / denom2, th.sum(self.fml_albedo_ref_wn) / denom)

                    # fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, full_mask)
                    fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, 1-self.mask)
                    loss_vgg = self.criterion(th.sum(fml_normal_pre_wn) / denom2, th.sum(self.fml_normal_ref_wn) / denom)
                    # fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, full_mask)
                    fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, 1-self.mask)
                    loss_vgg += self.criterion(th.sum(fml_rough_pre_wn) / denom2, th.sum(self.fml_rough_ref_wn) / denom)
                    # fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, full_mask)
                    fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, 1-self.mask)
                    loss_vgg += self.criterion(th.sum(fml_specular_pre_wn) / denom2, th.sum(self.fml_specular_ref_wn) / denom)

                    loss_vgg *= self.args.loss_weight[3]
                    loss += loss_vgg
                    losses[3] = loss_vgg.item()


        return loss, losses

class LossesMaps:
    def __init__(self, args, textures_ref):
        self.args = args
        self.textures_ref = textures_ref # partial target maps, basecolor only
        self.criterion = th.nn.MSELoss().cuda()

        self.precompute()

    def precompute(self):

        self.FL_w = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_w)
        for p in self.FL_w.parameters():
            p.requires_grad = False

        self.FL_n = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_n)
        for p in self.FL_n.parameters():
            p.requires_grad = False

        self.FL_wn = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
        for p in self.FL_wn.parameters():
            p.requires_grad = False

        self.SL = StyleLoss()
        for p in self.SL.parameters():
            p.requires_grad = False

        self.LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        self.Laplacian = kornia.filters.Laplacian(3, normalized=False)

        self.basecolor_ref_pix, self.basecolor_ref_vgg = self.eval_basecolor_vgg(self.textures_ref, isConvert=True)

        self.fl_basecolor_ref_w = self.FL_w(self.basecolor_ref_vgg)

        self.fl_basecolor_ref_n = self.FL_n(self.basecolor_ref_vgg)

        self.fl_basecolor_ref_wn = self.FL_wn(self.basecolor_ref_vgg)

    def eval_basecolor_vgg(self, textures, isConvert=True):
        if isConvert:
            albedo, normal, rough, specular = tex2map(textures) # (x + 1) / 2) ** 2.2
            basecolor = (albedo + specular) + torch.sqrt((albedo + specular)**2 - 0.16 * albedo)
            basecolor = basecolor.clamp(eps,1) ** (1/2.2)
            #basecolor_vgg = normalize_vgg19(basecolor[0,:], False).unsqueeze(0)
        else:
            basecolor = textures.unsqueeze(0) # 3 x 256 x 256
        basecolor_vgg = normalize_vgg19(basecolor[0,:], False).unsqueeze(0)

        return basecolor, basecolor_vgg

    def evalBasecolor(self, textures_pre, type, epoch):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        basecolor_pre_pix, basecolor_pre_vgg = self.eval_basecolor_vgg(textures_pre)

        if self.args.loss_weight[0] > -eps:
            # pixel loss
            loss_pl = self.criterion(basecolor_pre_pix, self.basecolor_ref_pix)
            #loss_pl  = self.criterion(albedo_pre_pix, self.albedo_ref_pix)
            #loss_pl += self.criterion(normal_pre_pix, self.normal_ref_pix)
            #loss_pl += self.criterion(rough_pre_pix,  self.rough_ref_pix)
            #loss_pl += self.criterion(specular_pre_pix,  self.specular_ref_pix)

            if self.args.loss_weight[0] > eps:
                loss_pl *= self.args.loss_weight[0]
                loss += loss_pl
            losses[0] = loss_pl.item()


        if self.args.loss_weight[1] > -eps:
            # feature loss
            if type == 'L':
                loss_fl  = self.criterion(self.FL_w(basecolor_pre_vgg), self.fl_basecolor_ref_w)
                # loss_fl  = self.criterion(self.FL_w(albedo_pre_vgg), self.fl_albedo_ref_w) * 0.1
                # loss_fl += self.criterion(self.FL_w(normal_pre_vgg), self.fl_normal_ref_w) * 0.7
                # loss_fl += self.criterion(self.FL_w(rough_pre_vgg),  self.fl_rough_ref_w) * 0.1
                # loss_fl += self.criterion(self.FL_w(specular_pre_vgg),  self.fl_specular_ref_w) * 0.1
            elif type == 'N':
                loss_fl  = self.criterion(self.FL_n(basecolor_pre_vgg), self.fl_basecolor_ref_n)
                # loss_fl  = self.criterion(self.FL_n(albedo_pre_vgg), self.fl_albedo_ref_n) * 0.1
                # loss_fl += self.criterion(self.FL_n(normal_pre_vgg), self.fl_normal_ref_n) * 0.7
                # loss_fl += self.criterion(self.FL_n(rough_pre_vgg),  self.fl_rough_ref_n) * 0.1
                # loss_fl += self.criterion(self.FL_n(specular_pre_vgg),  self.fl_specular_ref_n) * 0.1
            elif type == 'LN':
                loss_fl  = self.criterion(self.FL_wn(basecolor_pre_vgg), self.fl_basecolor_ref_wn)
                # loss_fl  = self.criterion(self.FL_wn(albedo_pre_vgg), self.fl_albedo_ref_wn) * 0.1
                # loss_fl += self.criterion(self.FL_wn(normal_pre_vgg), self.fl_normal_ref_wn) * 0.7
                # loss_fl += self.criterion(self.FL_wn(rough_pre_vgg),  self.fl_rough_ref_wn) * 0.1
                # loss_fl += self.criterion(self.FL_wn(specular_pre_vgg),  self.fl_specular_ref_wn) * 0.1
            else:
                print('Latent type wrong!')
                exit()

            if self.args.loss_weight[1] > eps:
                loss_fl *= self.args.loss_weight[1]
                loss += loss_fl
            losses[1] = loss_fl.item()

        if self.args.loss_weight[2] > -eps:
            # pixel loss
            loss_lpips  = self.LPIPS.forward(basecolor_pre_pix, self.basecolor_ref_pix).sum()
            # loss_lpips  = self.LPIPS.forward(albedo_pre_pix, self.albedo_ref_pix).sum() * 0.1
            # loss_lpips += self.LPIPS.forward(normal_pre_pix, self.normal_ref_pix).sum() * 0.7
            # loss_lpips += self.LPIPS.forward(rough_pre_pix,  self.rough_ref_pix).sum() * 0.1
            # loss_lpips += self.LPIPS.forward(specular_pre_pix,  self.specular_ref_pix).sum() * 0.1

            if self.args.loss_weight[2] > eps:
                loss_lpips *= self.args.loss_weight[2]
                loss += loss_lpips
            losses[2] = loss_lpips.item()

        return loss, losses

class LossesHallucination:
    def __init__(self, args, textures_ref):
        self.args = args
        self.textures_ref = textures_ref
        self.criterion = th.nn.MSELoss().to(args.device)
        self.precompute()

    def precompute(self):

        # self.FL_w = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_w)
        # for p in self.FL_w.parameters():
        #     p.requires_grad = False

        # self.FL_n = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_n)
        # for p in self.FL_n.parameters():
        #     p.requires_grad = False

        # self.FL_wn = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
        # for p in self.FL_wn.parameters():
        #     p.requires_grad = False

        # self.albedo_ref_pix, self.normal_ref_pix, self.rough_ref_pix, self.specular_ref_pix, \
        # self.albedo_ref_vgg, self.normal_ref_vgg, self.rough_ref_vgg, self.specular_ref_vgg = \
        # self.eval_texture_vgg(self.textures_ref)
        self.albedo_ref_pix, self.normal_ref_pix, self.rough_ref_pix, self.specular_ref_pix = \
        self.eval_texture(self.textures_ref)

        # self.fl_albedo_ref_w   = self.FL_w(self.albedo_ref_vgg)
        # self.fl_normal_ref_w   = self.FL_w(self.normal_ref_vgg)
        # self.fl_rough_ref_w    = self.FL_w(self.rough_ref_vgg)
        # self.fl_specular_ref_w = self.FL_w(self.specular_ref_vgg)

        # self.fl_albedo_ref_n   = self.FL_n(self.albedo_ref_vgg)
        # self.fl_normal_ref_n   = self.FL_n(self.normal_ref_vgg)
        # self.fl_rough_ref_n    = self.FL_n(self.rough_ref_vgg)
        # self.fl_specular_ref_n = self.FL_n(self.specular_ref_vgg)

        # self.fl_albedo_ref_wn   = self.FL_wn(self.albedo_ref_vgg)
        # self.fl_normal_ref_wn   = self.FL_wn(self.normal_ref_vgg)
        # self.fl_rough_ref_wn    = self.FL_wn(self.rough_ref_vgg)
        # self.fl_specular_ref_wn = self.FL_wn(self.specular_ref_vgg)

    def eval_texture_vgg(self, tex):
        # albedo = ((tex[:,0:3,:,:].clamp(-1,1) + 1) / 2) ** 2.2
        # albedo    = albedo.clamp(eps,1) ** (1/2.2)
        albedo = ((tex[:,0:3,:,:].clamp(-1,1) + 1) / 2)

        normal_x  = tex[:,3,:,:].clamp(-1,1)
        normal_y  = tex[:,4,:,:].clamp(-1,1)
        normal_xy = (normal_x**2 + normal_y**2).clamp(min=0, max=1-eps)
        normal_z  = (1 - normal_xy).sqrt()
        normal    = th.stack((normal_x, normal_y, normal_z), 1)
        normal    = normal.div(normal.norm(2.0, 1, keepdim=True))
        normal    = (normal+1)/2

        # rough = ((tex[:,5,:,:].clamp(-0.3,1) + 1) / 2) ** 2.2
        # rough = rough.clamp(min=eps).unsqueeze(1).expand(-1,3,-1,-1)
        # rough     = rough.clamp(eps,1) ** (1/2.2)
        rough = ((tex[:,5,:,:].clamp(-0.3,1) + 1) / 2).unsqueeze(1).expand(-1, 3, -1, -1)

        # specular = ((tex[:,6:9,:,:].clamp(-1,1) + 1) / 2) ** 2.2
        # specular  = specular.clamp(eps,1) ** (1/2.2)
        specular = ((tex[:,6:9,:,:].clamp(-1,1) + 1) / 2)

        albedo_vgg = normalize_vgg19(albedo[0,:], False).unsqueeze(0)
        normal_vgg = normalize_vgg19(normal[0,:], False).unsqueeze(0)
        rough_vgg  = normalize_vgg19(rough[0,:], False).unsqueeze(0)
        specular_vgg  = normalize_vgg19(specular[0,:], False).unsqueeze(0)

        return albedo, normal, rough, specular, albedo_vgg, normal_vgg, rough_vgg, specular_vgg

    def eval_texture(self, tex):
        # albedo = ((tex[:,0:3,:,:].clamp(-1,1) + 1) / 2) ** 2.2
        # albedo    = albedo.clamp(eps,1) ** (1/2.2)
        albedo = ((tex[:,0:3,:,:].clamp(-1,1) + 1) / 2)

        normal_x  = tex[:,3,:,:].clamp(-1,1)
        normal_y  = tex[:,4,:,:].clamp(-1,1)
        normal_xy = (normal_x**2 + normal_y**2).clamp(min=0, max=1-eps)
        normal_z  = (1 - normal_xy).sqrt()
        normal    = th.stack((normal_x, normal_y, normal_z), 1)
        normal    = normal.div(normal.norm(2.0, 1, keepdim=True))
        normal    = (normal+1)/2

        # rough = ((tex[:,5,:,:].clamp(-0.3,1) + 1) / 2) ** 2.2
        # rough = rough.clamp(min=eps).unsqueeze(1).expand(-1,3,-1,-1)
        # rough     = rough.clamp(eps,1) ** (1/2.2)
        rough = ((tex[:,5,:,:].clamp(-0.3,1) + 1) / 2).unsqueeze(1).expand(-1, 3, -1, -1)

        # specular = ((tex[:,6:9,:,:].clamp(-1,1) + 1) / 2) ** 2.2
        # specular  = specular.clamp(eps,1) ** (1/2.2)
        specular = ((tex[:,6:9,:,:].clamp(-1,1) + 1) / 2)

        return albedo, normal, rough, specular

    def eval(self, textures_pre, type):
        loss = 0
        losses = np.array([0,0]).astype(np.float32)

        # albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
        # albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
        # self.eval_texture_vgg(textures_pre)
        albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix = \
        self.eval_texture(textures_pre)

        if self.args.loss_weight[0] > -eps:
            # pixel loss
            loss_pl  = self.criterion(albedo_pre_pix, self.albedo_ref_pix)
            if self.args.metallicPath is not None:
                loss_pl += self.criterion(specular_pre_pix,  self.specular_ref_pix)

            if self.args.loss_weight[0] > eps:
                loss_pl *= self.args.loss_weight[0]
                loss += loss_pl
            losses[0] = loss_pl.item()


        # if self.args.loss_weight[1] > -eps:
        #     # feature loss
        #     if type == 'L':
        #         loss_fl  = self.criterion(self.FL_w(albedo_pre_vgg), self.fl_albedo_ref_w) * 0.1
        #         if self.args.metallicPath is not None:
        #             loss_fl += self.criterion(self.FL_w(specular_pre_vgg),  self.fl_specular_ref_w) * 0.1
        #     elif type == 'N':
        #         loss_fl  = self.criterion(self.FL_n(albedo_pre_vgg), self.fl_albedo_ref_n) * 0.1
        #         if self.args.metallicPath is not None:
        #             loss_fl += self.criterion(self.FL_n(specular_pre_vgg),  self.fl_specular_ref_n) * 0.1
        #     # elif type == 'LN':
        #     #     loss_fl  = self.criterion(self.FL_wn(albedo_pre_vgg), self.fl_albedo_ref_wn) * 0.1
        #     #     if self.args.metallicPath is not None:
        #     #         loss_fl += self.criterion(self.FL_wn(specular_pre_vgg),  self.fl_specular_ref_wn) * 0.1
        #     else:
        #         print('Latent type wrong!')
        #         exit()

        #     if self.args.loss_weight[1] > eps:
        #         loss_fl *= self.args.loss_weight[1]
        #         loss += loss_fl
        #     losses[1] = loss_fl.item()

        return loss, losses

class LossesPPL: # loss for per-pixel lighting
    def __init__(self, args, textures_init, textures_ref, rendered_ref, res, size, lp, cp, oriMeans=None):
        self.args = args
        self.textures_init = textures_init
        self.textures_ref = textures_ref
        self.rendered_ref = rendered_ref
        self.res = res
        self.size = size
        self.lp = lp
        self.cp = cp
        self.oriMeans = oriMeans

        if args.applyMask or args.applyMask2:
            self.criterion = th.nn.MSELoss(reduction='none').cuda()
            png = Image.open(args.mask_path)
            if png.height != 256:
                png = png.resize((png.width // png.height * 256, 256))
            png = gyPIL2Array(png)
            self.mask = torch.from_numpy(png).unsqueeze(0).unsqueeze(1).cuda()

            # # Fill in Ref unseen pixels with average color of seen pixels
            # avgColor = torch.sum(self.textures_ref * self.mask, dim=(3, 2), keepdim=True) / torch.sum(self.mask)
            # self.textures_ref = torch.ones_like(self.textures_ref) * avgColor * (1-self.mask) + self.textures_ref * self.mask

            if args.findInit:
                initMap = torch.sum(self.textures_ref * self.mask, dim=(3, 2), keepdim=True) / torch.sum(self.mask)
                self.textures_ref = torch.ones_like(self.textures_ref) * initMap
                self.mask = torch.ones_like(self.textures_ref[:, :3, :, :])
        else:
            self.criterion = th.nn.MSELoss().cuda()
            self.mask = None

        self.precompute()

    def precompute(self):

        self.FL_w = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_w)
        for p in self.FL_w.parameters():
            p.requires_grad = False

        self.FL_n = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_n)
        for p in self.FL_n.parameters():
            p.requires_grad = False

        self.FL_wn = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
        for p in self.FL_wn.parameters():
            p.requires_grad = False

        self.SL = StyleLoss()
        for p in self.SL.parameters():
            p.requires_grad = False

        #self.LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        #self.Laplacian = kornia.filters.Laplacian(3, normalized=False)

        if self.args.embed_tex:
            self.albedo_ref_pix, self.normal_ref_pix, self.rough_ref_pix, self.specular_ref_pix, \
            self.albedo_ref_vgg, self.normal_ref_vgg, self.rough_ref_vgg, self.specular_ref_vgg = \
            self.eval_texture_vgg(self.textures_ref)

            # if self.mask is not None:
            #     print(self.albedo_ref_pix.shape, self.mask.shape)
            #     self.albedo_ref_pix = self.albedo_ref_pix * self.mask
            #     self.normal_ref_pix = self.normal_ref_pix * self.mask
            #     self.rough_ref_pix = self.rough_ref_pix * self.mask
            #     self.specular_ref_pix = self.specular_ref_pix * self.mask

            self.fl_albedo_ref_w   = self.FL_w(self.albedo_ref_vgg)
            self.fl_normal_ref_w   = self.FL_w(self.normal_ref_vgg)
            self.fl_rough_ref_w    = self.FL_w(self.rough_ref_vgg)
            self.fl_specular_ref_w = self.FL_w(self.specular_ref_vgg)

            self.fl_albedo_ref_n   = self.FL_n(self.albedo_ref_vgg)
            self.fl_normal_ref_n   = self.FL_n(self.normal_ref_vgg)
            self.fl_rough_ref_n    = self.FL_n(self.rough_ref_vgg)
            self.fl_specular_ref_n = self.FL_n(self.specular_ref_vgg)

            self.fl_albedo_ref_wn   = self.FL_wn(self.albedo_ref_vgg)
            self.fl_normal_ref_wn   = self.FL_wn(self.normal_ref_vgg)
            self.fl_rough_ref_wn    = self.FL_wn(self.rough_ref_vgg)
            self.fl_specular_ref_wn = self.FL_wn(self.specular_ref_vgg)

            if (self.args.applyMask or self.args.applyMask2) and self.args.alignPixStd:
                if self.args.alignVGG:
                    self.FML_wn = FeatureMaskLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
                    for p in self.FML_wn.parameters():
                        p.requires_grad = False
                    self.fml_albedo_ref_wn   = self.FML_wn(self.albedo_ref_vgg, self.mask)
                    self.fml_normal_ref_wn   = self.FML_wn(self.normal_ref_vgg, self.mask)
                    self.fml_rough_ref_wn    = self.FML_wn(self.rough_ref_vgg, self.mask)
                    self.fml_specular_ref_wn = self.FML_wn(self.specular_ref_vgg, self.mask)

                self.std_albedo_list = []
                if not self.args.diffuseOnly or self.args.applyMask2:
                    self.std_normal_list = []
                    self.std_rough_list = []
                    self.std_specular_list = []
                for c in range(3):
                    self.std_albedo_list.append(torch.std(torch.masked_select(self.albedo_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                    if not self.args.diffuseOnly:
                        self.std_normal_list.append(torch.std(torch.masked_select(self.normal_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                        self.std_specular_list.append(torch.std(torch.masked_select(self.specular_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                        if c == 0:
                            self.std_rough_list.append(torch.std(torch.masked_select(self.rough_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )

        else:
            self.rendered_ref_pix = self.rendered_ref
            self.rendered_ref_vgg = self.eval_render_vgg(self.rendered_ref, self.args.jittering)

            if self.args.jittering:
                self.sl_rendered_ref = self.SL(self.rendered_ref_vgg)
            else:
                self.fl_rendered_ref_w = self.FL_w(self.rendered_ref_vgg)
                self.fl_rendered_ref_n = self.FL_n(self.rendered_ref_vgg)
                self.fl_rendered_ref_wn = self.FL_wn(self.rendered_ref_vgg)

    def eval_texture_vgg(self, textures):
        albedo, normal, rough, specular = tex2map(textures)
        albedo    = albedo.clamp(eps,1) ** (1/2.2)
        normal    = (normal+1)/2
        rough     = rough.clamp(eps,1) ** (1/2.2)
        specular  = specular.clamp(eps,1) ** (1/2.2)

        albedo_vgg = normalize_vgg19(albedo[0,:].cpu(), False).unsqueeze(0).cuda()
        normal_vgg = normalize_vgg19(normal[0,:].cpu(), False).unsqueeze(0).cuda()
        rough_vgg  = normalize_vgg19(rough[0,:].cpu(), False).unsqueeze(0).cuda()
        specular_vgg  = normalize_vgg19(specular[0,:].cpu(), False).unsqueeze(0).cuda()

        return albedo, normal, rough, specular, albedo_vgg, normal_vgg, rough_vgg, specular_vgg

    def eval_render_vgg(self, rendered, isGram):
        rendered_tmp = rendered.clone()
        # for i in range(self.args.num_render_used):
        #     rendered_tmp[i,:] = normalize_vgg19(rendered[i,:], isGram)
        rendered_tmp[0,:] = normalize_vgg19(rendered[0,:], isGram)
        return rendered_tmp

    def eval_render_jitter(self, textures, li):

        renderOBJ = Microfacet(res=self.res, size=self.size)
        rendered = th.zeros(self.args.num_render_used, 3, self.res, self.res).cuda()
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:]
            rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())

        renderOBJ_jitter = Microfacet(res=self.res, size=self.size)
        rendered_jitter = th.zeros(self.args.num_render_used, 3, self.res, self.res).cuda()
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:] + np.random.randn(*self.lp[i,:].shape) * 0.1
            rendered_jitter[i,:] = renderOBJ_jitter.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())

        return rendered.clamp(eps,1)**(1/2.2), rendered_jitter.clamp(eps,1)**(1/2.2)

    def eval_render(self, textures, envmap, bbox, planeNormal):

        renderOBJ = MicrofacetPlaneCrop(self.res, bbox, planeNormal)
        rendered = th.zeros(1, 3, self.res, self.res).cuda()
        for i in range(1):
            #lp_this = self.lp[i,:]
            #rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())
            rendered[i,:] = renderOBJ.eval(textures, envmap)
        return rendered.clamp(eps,1)**(1/2.2)


    def eval(self, textures_pre, envmap, type, epoch, bbox, planeNormal):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        if self.args.embed_tex:
            albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
            albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
            self.eval_texture_vgg(textures_pre)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl  = self.criterion(albedo_pre_pix, self.albedo_ref_pix)
                if not self.args.diffuseOnly:
                    loss_pl += self.criterion(normal_pre_pix, self.normal_ref_pix)
                    loss_pl += self.criterion(rough_pre_pix,  self.rough_ref_pix)
                    loss_pl += self.criterion(specular_pre_pix,  self.specular_ref_pix)

                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()


            if self.args.loss_weight[1] > -eps:
                # feature loss
                if type == 'L':
                    loss_fl  = self.criterion(self.FL_w(albedo_pre_vgg), self.fl_albedo_ref_w) * 0.1
                    if not self.args.diffuseOnly:
                        loss_fl += self.criterion(self.FL_w(normal_pre_vgg), self.fl_normal_ref_w) * 0.7
                        loss_fl += self.criterion(self.FL_w(rough_pre_vgg),  self.fl_rough_ref_w) * 0.1
                        loss_fl += self.criterion(self.FL_w(specular_pre_vgg),  self.fl_specular_ref_w) * 0.1
                elif type == 'N':
                    loss_fl  = self.criterion(self.FL_n(albedo_pre_vgg), self.fl_albedo_ref_n) * 0.1
                    if not self.args.diffuseOnly:
                        loss_fl += self.criterion(self.FL_n(normal_pre_vgg), self.fl_normal_ref_n) * 0.7
                        loss_fl += self.criterion(self.FL_n(rough_pre_vgg),  self.fl_rough_ref_n) * 0.1
                        loss_fl += self.criterion(self.FL_n(specular_pre_vgg),  self.fl_specular_ref_n) * 0.1
                elif type == 'LN':
                    loss_fl  = self.criterion(self.FL_wn(albedo_pre_vgg), self.fl_albedo_ref_wn) * 0.1
                    if not self.args.diffuseOnly:
                        loss_fl += self.criterion(self.FL_wn(normal_pre_vgg), self.fl_normal_ref_wn) * 0.7
                        loss_fl += self.criterion(self.FL_wn(rough_pre_vgg),  self.fl_rough_ref_wn) * 0.1
                        loss_fl += self.criterion(self.FL_wn(specular_pre_vgg),  self.fl_specular_ref_wn) * 0.1
                else:
                    print('Latent type wrong!')
                    exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            # if self.args.loss_weight[2] > -eps:
                # # pixel loss
                # loss_lpips  = self.LPIPS.forward(albedo_pre_pix, self.albedo_ref_pix).sum() * 0.1
                # if not self.args.diffuseOnly:
                #     loss_lpips += self.LPIPS.forward(normal_pre_pix, self.normal_ref_pix).sum() * 0.7
                #     loss_lpips += self.LPIPS.forward(rough_pre_pix,  self.rough_ref_pix).sum() * 0.1
                #     loss_lpips += self.LPIPS.forward(specular_pre_pix,  self.specular_ref_pix).sum() * 0.1

                # if self.args.loss_weight[2] > eps:
                #     loss_lpips *= self.args.loss_weight[2]
                #     loss += loss_lpips
                # losses[2] = loss_lpips.item()

        else:
            if self.args.jittering:
                rendered_pre_pix, rendered_pre_pix_jitter = self.eval_render_jitter(textures_pre, light)
                rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix_jitter,True)
            else:
                rendered_pre_pix = self.eval_render(textures_pre, envmap, bbox, planeNormal)
                rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix, False)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl = self.criterion(rendered_pre_pix, self.rendered_ref_pix)
                # print('loss_pl:', loss_pl)
                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

            if self.args.loss_weight[1] > -eps:
                # feature loss
                if self.args.jittering:
                    loss_fl = self.criterion(self.SL(rendered_pre_vgg), self.sl_rendered_ref)
                else:
                    if type == 'L':
                        loss_fl = self.criterion(self.FL_w(rendered_pre_vgg), self.fl_rendered_ref_w)
                    elif type == 'N':
                        loss_fl = self.criterion(self.FL_n(rendered_pre_vgg), self.fl_rendered_ref_n)
                    elif type == 'LN':
                        loss_fl = self.criterion(self.FL_wn(rendered_pre_vgg), self.fl_rendered_ref_wn)
                    else:
                        print('Latent type wrong!')
                        exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            if self.args.loss_weight[2] > -eps:
                # pixel loss
                loss_lpips = self.LPIPS.forward(rendered_pre_pix, self.rendered_ref_pix).sum()

                if self.args.loss_weight[2] > eps:
                    loss_lpips *= self.args.loss_weight[2]
                    loss += loss_lpips
                losses[2] = loss_lpips.item()

            #if self.args.loss_weight[3] > -eps:
            #    # pixel loss
            #    loss_tex  = self.criterion(self.textures_init[:,0:3,:,:], textures_pre[:,0:3,:,:]) * 0.4
            #    loss_tex += self.criterion(self.textures_init[:,3:5,:,:], textures_pre[:,3:5,:,:]) * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,5,:,:],   textures_pre[:,5,:,:])   * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,6:9,:,:], textures_pre[:,6:9,:,:]) * 0.4

            #    lap = th.norm(self.Laplacian(self.textures_init - textures_pre).flatten())
            #    # print(lap.item())
            #    loss_tex += lap * 5e-5

            #    # print('loss_tex:', loss_tex)
            #    if self.args.loss_weight[3] > eps:
            #        loss_tex *= self.args.loss_weight[3]
            #        loss += loss_tex
            #    losses[3] = loss_tex.item()
            if self.oriMeans is not None: # Output albedo and rough of GAN are in gamma space, apply 2.2 to used in the renderer
                albedo_pre_mean = th.mean( ((textures_pre[:,0:3,:,:].clamp(-1,1) + 1) / 2) ** 2.2, dim=(0, 2, 3))
                rough_pre_mean = th.mean( ((textures_pre[:,5,:,:].clamp(-0.3,1) + 1) / 2) ** 2.2 )
                loss_alb_mean = self.criterion(albedo_pre_mean, th.from_numpy(self.oriMeans['albedo']).cuda() )
                loss_rou_mean = self.criterion(rough_pre_mean, th.ones_like(rough_pre_mean) * self.oriMeans['rough'] )
                loss_reg = loss_alb_mean + loss_rou_mean
                loss_reg *= 10
                loss += loss_reg
                losses[3] = loss_reg.item()

        # print('loss_total:', loss)

        return loss, losses

    def evalMultiCrop(self, textures_pre, envmapList, type, epoch, bboxList, planeNormalList):
        lossALL = 0
        lossesALL = np.array([0,0,0,0]).astype(np.float32)

        assert(len(envmapList)==len(bboxList) and len(envmapList)==len(planeNormalList) )
        nCrop = len(envmapList)
        for nc in range(nCrop):
            loss = 0
            losses = np.array([0,0,0,0]).astype(np.float32)
            envmap, bbox, planeNormal = envmapList[nc], bboxList[nc], planeNormalList[nc]
            rendered_pre_pix = self.eval_render(textures_pre, envmap, bbox, planeNormal)
            rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix, False)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl = self.criterion(rendered_pre_pix, self.rendered_ref_pix)
                # print('loss_pl:', loss_pl)
                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

            if self.args.loss_weight[1] > -eps:
                # feature loss
                if self.args.jittering:
                    loss_fl = self.criterion(self.SL(rendered_pre_vgg), self.sl_rendered_ref)
                else:
                    if type == 'L':
                        loss_fl = self.criterion(self.FL_w(rendered_pre_vgg), self.fl_rendered_ref_w)
                    elif type == 'N':
                        loss_fl = self.criterion(self.FL_n(rendered_pre_vgg), self.fl_rendered_ref_n)
                    elif type == 'LN':
                        loss_fl = self.criterion(self.FL_wn(rendered_pre_vgg), self.fl_rendered_ref_wn)
                    else:
                        print('Latent type wrong!')
                        exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            if self.args.loss_weight[2] > -eps:
                # pixel loss
                loss_lpips = self.LPIPS.forward(rendered_pre_pix, self.rendered_ref_pix).sum()

                if self.args.loss_weight[2] > eps:
                    loss_lpips *= self.args.loss_weight[2]
                    loss += loss_lpips
                losses[2] = loss_lpips.item()

            #if self.args.loss_weight[3] > -eps:
            #    # pixel loss
            #    loss_tex  = self.criterion(self.textures_init[:,0:3,:,:], textures_pre[:,0:3,:,:]) * 0.4
            #    loss_tex += self.criterion(self.textures_init[:,3:5,:,:], textures_pre[:,3:5,:,:]) * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,5,:,:],   textures_pre[:,5,:,:])   * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,6:9,:,:], textures_pre[:,6:9,:,:]) * 0.4

            #    lap = th.norm(self.Laplacian(self.textures_init - textures_pre).flatten())
            #    # print(lap.item())
            #    loss_tex += lap * 5e-5

            #    # print('loss_tex:', loss_tex)
            #    if self.args.loss_weight[3] > eps:
            #        loss_tex *= self.args.loss_weight[3]
            #        loss += loss_tex
            #    losses[3] = loss_tex.item()
            if self.oriMeans is not None:
                albedo_pre_mean = th.mean( ((textures_pre[:,0:3,:,:].clamp(-1,1) + 1) / 2), dim=(0, 2, 3))
                rough_pre_mean = th.mean( ((textures_pre[:,5,:,:].clamp(-0.3,1) + 1) / 2) )
                loss_alb_mean = self.criterion(albedo_pre_mean, th.from_numpy(self.oriMeans['albedo']).cuda() )
                loss_rou_mean = self.criterion(rough_pre_mean, th.ones_like(rough_pre_mean) * self.oriMeans['rough'] )
                loss_reg = loss_alb_mean + loss_rou_mean
                loss_reg *= 1
                loss += loss_reg
                losses[3] = loss_reg.item()

            lossALL += loss
            lossesALL += losses
        # print('loss_total:', loss)

        return lossALL, lossesALL

    def evalMasked(self, textures_pre, light, type, epoch):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        if self.args.embed_tex:
            albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
            albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
            self.eval_texture_vgg(textures_pre)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                #loss_pl  = (albedo_pre_pix - self.albedo_ref_pix)

                pixNum = torch.sum(self.mask)
                #loss_pl  = torch.sum(self.criterion(albedo_pre_pix, self.albedo_ref_pix) * self.mask) / (3*pixNum)
                loss_pl  = torch.sum(self.criterion(albedo_pre_pix * self.mask, self.albedo_ref_pix * self.mask) ) / (3*pixNum)
                if not self.args.diffuseOnly:
                    loss_pl += torch.sum(self.criterion(normal_pre_pix, self.normal_ref_pix) * self.mask) / (3*pixNum)
                    loss_pl += torch.sum(self.criterion(rough_pre_pix, self.rough_ref_pix) * self.mask) / (pixNum)
                    loss_pl += torch.sum(self.criterion(specular_pre_pix, self.specular_ref_pix) * self.mask) / (3*pixNum)

                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

                #if self.args.alignPixMean and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixMean and (epoch >= 1000):
                    pixNum2 = torch.sum(1-self.mask)
                    pixNumAll = pixNum + pixNum2
                    #loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum), torch.sum(albedo_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum) , torch.sum(albedo_pre_pix) / (3*pixNumAll) )
                    if not self.args.diffuseOnly:
                        #loss_mean += self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix * (1-self.mask)) / (3*pixNum2) )
                        loss_mean += self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix) / (3*pixNumAll) )
                        #loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix * (1-self.mask)) / (pixNum2) )
                        loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix) / pixNumAll )
                        #loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix * (1-self.mask)) / (3*pixNum2) )
                        loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix ) / (3*pixNumAll) )
                    loss_mean *= self.args.loss_weight[1]
                    # if not self.args.diffuseOnly:
                    #     loss_mean *= 0.01
                    loss += loss_mean
                    losses[1] = loss_mean.item()

                #if self.args.alignPixStd and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixStd and (epoch >= 1000):
                    loss_std = 0
                    for c in range(3):
                        #std1 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        std1 = self.std_albedo_list[c]
                        #std2 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        std2 = torch.std(albedo_pre_pix[:,c,:,:] )
                        loss_std += self.criterion(std1, std2)
                        # if not self.args.diffuseOnly:
                        #     #std1 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        #     std1 = self.std_normal_list[c]
                        #     std2 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        #     loss_std += self.criterion(std1, std2)
                        #     #std1 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        #     std1 = self.std_specular_list[c]
                        #     std2 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        #     loss_std += self.criterion(std1, std2)
                        #     if c == 0:
                        #         #std1 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                        #         std1 = self.std_rough_list[c]
                        #         std2 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                        #         loss_std += self.criterion(std1, std2)
                    loss_std *= self.args.loss_weight[2]
                    # if not self.args.diffuseOnly:
                    #     loss_std *= 0.01
                    loss += loss_std
                    losses[2] = loss_std.item()

                if self.args.alignVGG and (epoch >= 1000):
                    denom = th.sum(th.ones_like(self.fml_albedo_ref_wn))
                    # loss_vgg = th.sum(self.criterion(self.FML_wn(albedo_pre_vgg, self.mask), self.fml_albedo_ref_wn) ) / denom
                    # if not self.args.diffuseOnly:
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(normal_pre_vgg, self.mask), self.fml_normal_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(rough_pre_vgg, self.mask), self.fml_rough_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(specular_pre_vgg, self.mask), self.fml_specular_ref_wn) ) / denom
                    full_mask = th.ones_like(self.mask)
                    #fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, full_mask)
                    fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, 1-self.mask)
                    denom2 = th.sum(th.ones_like(fml_albedo_pre_wn))
                    loss_vgg = self.criterion(th.sum(fml_albedo_pre_wn) / denom2, th.sum(self.fml_albedo_ref_wn) / denom)
                    if not self.args.diffuseOnly:
                        # fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, full_mask)
                        fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, 1-self.mask)
                        loss_vgg += self.criterion(th.sum(fml_normal_pre_wn) / denom2, th.sum(self.fml_normal_ref_wn) / denom)
                        # fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, full_mask)
                        fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, 1-self.mask)
                        loss_vgg += self.criterion(th.sum(fml_rough_pre_wn) / denom2, th.sum(self.fml_rough_ref_wn) / denom)
                        # fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, full_mask)
                        fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, 1-self.mask)
                        loss_vgg += self.criterion(th.sum(fml_specular_pre_wn) / denom2, th.sum(self.fml_specular_ref_wn) / denom)

                    loss_vgg *= self.args.loss_weight[3]
                    loss += loss_vgg
                    losses[3] = loss_vgg.item()


        return loss, losses

    def evalMasked2(self, textures_pre, light, type, epoch):
        loss = 0
        losses = np.array([0,0,0,0]).astype(np.float32)

        if self.args.embed_tex:
            albedo_pre_pix, normal_pre_pix, rough_pre_pix, specular_pre_pix, \
            albedo_pre_vgg, normal_pre_vgg, rough_pre_vgg, specular_pre_vgg = \
            self.eval_texture_vgg(textures_pre)
            maskFull = torch.ones_like(self.mask)
            pixNumFull = torch.sum(maskFull)
            if self.args.loss_weight[0] > -eps:
                # pixel loss
                #loss_pl  = (albedo_pre_pix - self.albedo_ref_pix)

                pixNum = torch.sum(self.mask)
                #loss_pl  = torch.sum(self.criterion(albedo_pre_pix, self.albedo_ref_pix) * self.mask) / (3*pixNum)
                loss_pl  = torch.sum(self.criterion(albedo_pre_pix * maskFull, self.albedo_ref_pix * maskFull) ) / (3*pixNumFull)
                loss_pl += torch.sum(self.criterion(normal_pre_pix * self.mask, self.normal_ref_pix * self.mask) ) / (3*pixNum)
                loss_pl += torch.sum(self.criterion(rough_pre_pix * self.mask, self.rough_ref_pix * self.mask) ) / (pixNum)
                loss_pl += torch.sum(self.criterion(specular_pre_pix* self.mask, self.specular_ref_pix* self.mask) ) / (3*pixNum)

                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

                #if self.args.alignPixMean and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixMean and (epoch >= 100):
                    pixNum2 = torch.sum(1-self.mask)
                    pixNumAll = pixNum + pixNum2
                    #loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum), torch.sum(albedo_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    #loss_mean = self.criterion(torch.sum(self.albedo_ref_pix * self.mask) / (3*pixNum) , torch.sum(albedo_pre_pix) / (3*pixNumAll) )
                    #loss_mean += self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    loss_mean = self.criterion(torch.sum(self.normal_ref_pix * self.mask) / (3*pixNum), torch.sum(normal_pre_pix) / (3*pixNumAll) )
                    #loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix * (1-self.mask)) / (pixNum2) )
                    loss_mean += self.criterion(torch.sum(self.rough_ref_pix * self.mask) / (pixNum), torch.sum(rough_pre_pix) / pixNumAll )
                    #loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix * (1-self.mask)) / (3*pixNum2) )
                    loss_mean += self.criterion(torch.sum(self.specular_ref_pix * self.mask) / (3*pixNum), torch.sum(specular_pre_pix ) / (3*pixNumAll) )
                    loss_mean *= self.args.loss_weight[1]
                    # if not self.args.diffuseOnly:
                    #     loss_mean *= 0.01
                    loss += loss_mean
                    losses[1] = loss_mean.item()

                #if self.args.alignPixStd and (epoch >= 1000 or not self.args.diffuseOnly):
                if self.args.alignPixStd and (epoch >= 100):
                    loss_std = 0
                    # for c in range(3):
                    #     #std1 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #     # std1 = self.std_albedo_list[c]
                    #     # #std2 = torch.std(torch.masked_select(albedo_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #     # std2 = torch.std(albedo_pre_pix[:,c,:,:] )
                    #     # loss_std += self.criterion(std1, std2)
                    #     #std1 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #     std1 = self.std_normal_list[c]
                    #     std2 = torch.std(torch.masked_select(normal_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #     loss_std += self.criterion(std1, std2)
                    #     #std1 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #     std1 = self.std_specular_list[c]
                    #     std2 = torch.std(torch.masked_select(specular_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #     loss_std += self.criterion(std1, std2)
                    #     if c == 0:
                    #         #std1 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) )
                    #         std1 = self.std_rough_list[c]
                    #         std2 = torch.std(torch.masked_select(rough_pre_pix[:,c,:,:].unsqueeze(1), self.mask < 0.5) )
                    #         loss_std += self.criterion(std1, std2)
                    # loss_std *= self.args.loss_weight[2]
                    # # if not self.args.diffuseOnly:
                    # #     loss_std *= 0.01
                    # loss += loss_std
                    # losses[2] = loss_std.item()

                if self.args.alignVGG and (epoch >= 100):
                    denom = th.sum(th.ones_like(self.fml_albedo_ref_wn))
                    # loss_vgg = th.sum(self.criterion(self.FML_wn(albedo_pre_vgg, self.mask), self.fml_albedo_ref_wn) ) / denom
                    # if not self.args.diffuseOnly:
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(normal_pre_vgg, self.mask), self.fml_normal_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(rough_pre_vgg, self.mask), self.fml_rough_ref_wn) ) / denom
                    #     loss_vgg += th.sum(self.criterion(self.FML_wn(specular_pre_vgg, self.mask), self.fml_specular_ref_wn) ) / denom
                    full_mask = th.ones_like(self.mask)
                    #fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, full_mask)
                    fml_albedo_pre_wn = self.FML_wn(albedo_pre_vgg, 1-self.mask)
                    denom2 = th.sum(th.ones_like(fml_albedo_pre_wn))
                    # loss_vgg = self.criterion(th.sum(fml_albedo_pre_wn) / denom2, th.sum(self.fml_albedo_ref_wn) / denom)

                    # fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, full_mask)
                    fml_normal_pre_wn = self.FML_wn(normal_pre_vgg, 1-self.mask)
                    loss_vgg = self.criterion(th.sum(fml_normal_pre_wn) / denom2, th.sum(self.fml_normal_ref_wn) / denom)
                    # fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, full_mask)
                    fml_rough_pre_wn = self.FML_wn(rough_pre_vgg, 1-self.mask)
                    loss_vgg += self.criterion(th.sum(fml_rough_pre_wn) / denom2, th.sum(self.fml_rough_ref_wn) / denom)
                    # fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, full_mask)
                    fml_specular_pre_wn = self.FML_wn(specular_pre_vgg, 1-self.mask)
                    loss_vgg += self.criterion(th.sum(fml_specular_pre_wn) / denom2, th.sum(self.fml_specular_ref_wn) / denom)

                    loss_vgg *= self.args.loss_weight[3]
                    loss += loss_vgg
                    losses[3] = loss_vgg.item()


        return loss, losses

class LossesPPL2: # loss for per-pixel lighting
    def __init__(self, args, textures_init, textures_ref, rendered_refList, res, size, lp, cp, oriMeans=None):
        self.args = args
        self.textures_init = textures_init
        self.textures_ref = textures_ref
        self.rendered_refList = rendered_refList
        self.res = res
        self.size = size
        self.lp = lp
        self.cp = cp
        self.oriMeans = oriMeans

        if args.applyMask or args.applyMask2:
            self.criterion = th.nn.MSELoss(reduction='none').cuda()
            png = Image.open(args.mask_path)
            if png.height != 256:
                png = png.resize((png.width // png.height * 256, 256))
            png = gyPIL2Array(png)
            self.mask = torch.from_numpy(png).unsqueeze(0).unsqueeze(1).cuda()

            # # Fill in Ref unseen pixels with average color of seen pixels
            # avgColor = torch.sum(self.textures_ref * self.mask, dim=(3, 2), keepdim=True) / torch.sum(self.mask)
            # self.textures_ref = torch.ones_like(self.textures_ref) * avgColor * (1-self.mask) + self.textures_ref * self.mask

            if args.findInit:
                initMap = torch.sum(self.textures_ref * self.mask, dim=(3, 2), keepdim=True) / torch.sum(self.mask)
                self.textures_ref = torch.ones_like(self.textures_ref) * initMap
                self.mask = torch.ones_like(self.textures_ref[:, :3, :, :])
        else:
            self.criterion = th.nn.MSELoss().cuda()
            self.mask = None

        self.precompute()

    def precompute(self):

        self.FL_w = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_w)
        for p in self.FL_w.parameters():
            p.requires_grad = False

        self.FL_n = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_n)
        for p in self.FL_n.parameters():
            p.requires_grad = False

        self.FL_wn = FeatureLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
        for p in self.FL_wn.parameters():
            p.requires_grad = False

        self.SL = StyleLoss()
        for p in self.SL.parameters():
            p.requires_grad = False

        #self.LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
        #self.Laplacian = kornia.filters.Laplacian(3, normalized=False)

        if self.args.embed_tex:
            self.albedo_ref_pix, self.normal_ref_pix, self.rough_ref_pix, self.specular_ref_pix, \
            self.albedo_ref_vgg, self.normal_ref_vgg, self.rough_ref_vgg, self.specular_ref_vgg = \
            self.eval_texture_vgg(self.textures_ref)

            # if self.mask is not None:
            #     print(self.albedo_ref_pix.shape, self.mask.shape)
            #     self.albedo_ref_pix = self.albedo_ref_pix * self.mask
            #     self.normal_ref_pix = self.normal_ref_pix * self.mask
            #     self.rough_ref_pix = self.rough_ref_pix * self.mask
            #     self.specular_ref_pix = self.specular_ref_pix * self.mask

            self.fl_albedo_ref_w   = self.FL_w(self.albedo_ref_vgg)
            self.fl_normal_ref_w   = self.FL_w(self.normal_ref_vgg)
            self.fl_rough_ref_w    = self.FL_w(self.rough_ref_vgg)
            self.fl_specular_ref_w = self.FL_w(self.specular_ref_vgg)

            self.fl_albedo_ref_n   = self.FL_n(self.albedo_ref_vgg)
            self.fl_normal_ref_n   = self.FL_n(self.normal_ref_vgg)
            self.fl_rough_ref_n    = self.FL_n(self.rough_ref_vgg)
            self.fl_specular_ref_n = self.FL_n(self.specular_ref_vgg)

            self.fl_albedo_ref_wn   = self.FL_wn(self.albedo_ref_vgg)
            self.fl_normal_ref_wn   = self.FL_wn(self.normal_ref_vgg)
            self.fl_rough_ref_wn    = self.FL_wn(self.rough_ref_vgg)
            self.fl_specular_ref_wn = self.FL_wn(self.specular_ref_vgg)

            if (self.args.applyMask or self.args.applyMask2) and self.args.alignPixStd:
                if self.args.alignVGG:
                    self.FML_wn = FeatureMaskLoss(self.args.vgg_weight_dir, self.args.vgg_layer_weight_wn)
                    for p in self.FML_wn.parameters():
                        p.requires_grad = False
                    self.fml_albedo_ref_wn   = self.FML_wn(self.albedo_ref_vgg, self.mask)
                    self.fml_normal_ref_wn   = self.FML_wn(self.normal_ref_vgg, self.mask)
                    self.fml_rough_ref_wn    = self.FML_wn(self.rough_ref_vgg, self.mask)
                    self.fml_specular_ref_wn = self.FML_wn(self.specular_ref_vgg, self.mask)

                self.std_albedo_list = []
                if not self.args.diffuseOnly or self.args.applyMask2:
                    self.std_normal_list = []
                    self.std_rough_list = []
                    self.std_specular_list = []
                for c in range(3):
                    self.std_albedo_list.append(torch.std(torch.masked_select(self.albedo_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                    if not self.args.diffuseOnly:
                        self.std_normal_list.append(torch.std(torch.masked_select(self.normal_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                        self.std_specular_list.append(torch.std(torch.masked_select(self.specular_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )
                        if c == 0:
                            self.std_rough_list.append(torch.std(torch.masked_select(self.rough_ref_pix[:,c,:,:].unsqueeze(1), self.mask > 0.5) ) )

        else:
            self.rendered_ref_pixList = self.rendered_refList
            self.sl_rendered_refList = []
            self.fl_rendered_ref_wList = []
            self.fl_rendered_ref_nList = []
            self.fl_rendered_ref_wnList = []
            for rendered_ref in self.rendered_refList:
                self.rendered_ref_vgg = self.eval_render_vgg(rendered_ref, self.args.jittering)

                if self.args.jittering:
                    self.sl_rendered_ref = self.SL(self.rendered_ref_vgg)
                    self.sl_rendered_refList.append(self.sl_rendered_ref)
                else:
                    self.fl_rendered_ref_w = self.FL_w(self.rendered_ref_vgg)
                    self.fl_rendered_ref_n = self.FL_n(self.rendered_ref_vgg)
                    self.fl_rendered_ref_wn = self.FL_wn(self.rendered_ref_vgg)
                    self.fl_rendered_ref_wList.append(self.fl_rendered_ref_w)
                    self.fl_rendered_ref_nList.append(self.fl_rendered_ref_n)
                    self.fl_rendered_ref_wnList.append(self.fl_rendered_ref_wn)

    def eval_texture_vgg(self, textures):
        albedo, normal, rough, specular = tex2map(textures)
        albedo    = albedo.clamp(eps,1) ** (1/2.2)
        normal    = (normal+1)/2
        rough     = rough.clamp(eps,1) ** (1/2.2)
        specular  = specular.clamp(eps,1) ** (1/2.2)

        albedo_vgg = normalize_vgg19(albedo[0,:].cpu(), False).unsqueeze(0).cuda()
        normal_vgg = normalize_vgg19(normal[0,:].cpu(), False).unsqueeze(0).cuda()
        rough_vgg  = normalize_vgg19(rough[0,:].cpu(), False).unsqueeze(0).cuda()
        specular_vgg  = normalize_vgg19(specular[0,:].cpu(), False).unsqueeze(0).cuda()

        return albedo, normal, rough, specular, albedo_vgg, normal_vgg, rough_vgg, specular_vgg

    def eval_render_vgg(self, rendered, isGram):
        rendered_tmp = rendered.clone()
        # for i in range(self.args.num_render_used):
        #     rendered_tmp[i,:] = normalize_vgg19(rendered[i,:], isGram)
        rendered_tmp[0,:] = normalize_vgg19(rendered[0,:], isGram)
        return rendered_tmp

    def eval_render_jitter(self, textures, li):

        renderOBJ = Microfacet(res=self.res, size=self.size)
        rendered = th.zeros(self.args.num_render_used, 3, self.res, self.res).cuda()
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:]
            rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())

        renderOBJ_jitter = Microfacet(res=self.res, size=self.size)
        rendered_jitter = th.zeros(self.args.num_render_used, 3, self.res, self.res).cuda()
        for i in range(self.args.num_render_used):
            lp_this = self.lp[i,:] + np.random.randn(*self.lp[i,:].shape) * 0.1
            rendered_jitter[i,:] = renderOBJ_jitter.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())

        return rendered.clamp(eps,1)**(1/2.2), rendered_jitter.clamp(eps,1)**(1/2.2)

    def eval_render(self, textures, envmap, bbox, planeNormal):

        renderOBJ = MicrofacetPlaneCrop(self.res, bbox, planeNormal)
        rendered = th.zeros(1, 3, self.res, self.res).cuda()
        for i in range(1):
            #lp_this = self.lp[i,:]
            #rendered[i,:] = renderOBJ.eval(textures, lightPos=lp_this, cameraPos=self.cp[i,:], light=th.from_numpy(li).cuda())
            rendered[i,:] = renderOBJ.eval(textures, envmap)
        return rendered.clamp(eps,1)**(1/2.2)

    def eval(self, textures_pre, envmapList, type, epoch, bboxList, planeNormalList):
        lossALL = 0
        lossesALL = np.array([0,0,0,0]).astype(np.float32)

        assert(len(envmapList)==len(bboxList) and len(envmapList)==len(planeNormalList) )
        nCrop = len(envmapList)
        for nc in range(nCrop):
            loss = 0
            losses = np.array([0,0,0,0]).astype(np.float32)
            envmap, bbox, planeNormal = envmapList[nc], bboxList[nc], planeNormalList[nc]
            rendered_pre_pix = self.eval_render(textures_pre, envmap, bbox, planeNormal)
            rendered_pre_vgg = self.eval_render_vgg(rendered_pre_pix, False)

            if self.args.loss_weight[0] > -eps:
                # pixel loss
                loss_pl = self.criterion(rendered_pre_pix, self.rendered_ref_pixList[nc])
                # print('loss_pl:', loss_pl)
                if self.args.loss_weight[0] > eps:
                    loss_pl *= self.args.loss_weight[0]
                    loss += loss_pl
                losses[0] = loss_pl.item()

            if self.args.loss_weight[1] > -eps:
                # feature loss
                if self.args.jittering:
                    loss_fl = self.criterion(self.SL(rendered_pre_vgg), self.sl_rendered_refList[nc])
                else:
                    if type == 'L':
                        loss_fl = self.criterion(self.FL_w(rendered_pre_vgg), self.fl_rendered_ref_wList[nc])
                    elif type == 'N':
                        loss_fl = self.criterion(self.FL_n(rendered_pre_vgg), self.fl_rendered_ref_nList[nc])
                    elif type == 'LN':
                        loss_fl = self.criterion(self.FL_wn(rendered_pre_vgg), self.fl_rendered_ref_wnList[nc])
                    else:
                        print('Latent type wrong!')
                        exit()

                if self.args.loss_weight[1] > eps:
                    loss_fl *= self.args.loss_weight[1]
                    loss += loss_fl
                losses[1] = loss_fl.item()

            if self.args.loss_weight[2] > -eps:
                # pixel loss
                loss_lpips = self.LPIPS.forward(rendered_pre_pix, self.rendered_ref_pixList[nc]).sum()

                if self.args.loss_weight[2] > eps:
                    loss_lpips *= self.args.loss_weight[2]
                    loss += loss_lpips
                losses[2] = loss_lpips.item()

            #if self.args.loss_weight[3] > -eps:
            #    # pixel loss
            #    loss_tex  = self.criterion(self.textures_init[:,0:3,:,:], textures_pre[:,0:3,:,:]) * 0.4
            #    loss_tex += self.criterion(self.textures_init[:,3:5,:,:], textures_pre[:,3:5,:,:]) * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,5,:,:],   textures_pre[:,5,:,:])   * 0.1
            #    loss_tex += self.criterion(self.textures_init[:,6:9,:,:], textures_pre[:,6:9,:,:]) * 0.4

            #    lap = th.norm(self.Laplacian(self.textures_init - textures_pre).flatten())
            #    # print(lap.item())
            #    loss_tex += lap * 5e-5

            #    # print('loss_tex:', loss_tex)
            #    if self.args.loss_weight[3] > eps:
            #        loss_tex *= self.args.loss_weight[3]
            #        loss += loss_tex
            #    losses[3] = loss_tex.item()
            if self.oriMeans is not None:
                albedo_pre_mean = th.mean( ((textures_pre[:,0:3,:,:].clamp(-1,1) + 1) / 2), dim=(0, 2, 3))
                rough_pre_mean = th.mean( ((textures_pre[:,5,:,:].clamp(-0.3,1) + 1) / 2) )
                loss_alb_mean = self.criterion(albedo_pre_mean, th.from_numpy(self.oriMeans['albedo']).cuda() )
                loss_rou_mean = self.criterion(rough_pre_mean, th.ones_like(rough_pre_mean) * self.oriMeans['rough'] )
                loss_reg = loss_alb_mean + loss_rou_mean
                loss_reg *= 1
                loss += loss_reg
                losses[3] = loss_reg.item()

            lossALL += loss
            lossesALL += losses
        # print('loss_total:', loss)

        return lossALL, lossesALL
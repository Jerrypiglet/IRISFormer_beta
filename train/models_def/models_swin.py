import sys
from pathlib import Path
import torch
import torch.nn as nn

from models_def.model_swin.swin_transformer_import import SwinTransformer
from models_def.model_swin.decode_heads_import import UPerHead
class Swin(torch.nn.Module):
    def __init__(
        self,
        opt,
        cfg_DPT, 
        out_channels, 
        if_batch_norm, 
        in_channels=3, 
        patch_size=4, 
        upscale_ratio=4, 
        # modality='al', 
        # non_negative=False
    ):
        super(Swin, self).__init__()

        self.cfg_DPT = cfg_DPT

        self.backbone = SwinTransformer(in_chans=in_channels, pretrain_img_size=(opt.cfg.DATA.im_height_padded, opt.cfg.DATA.im_width_padded), patch_size=patch_size)
        if if_batch_norm:
            norm_cfg = dict(type='SyncBN' if opt.distributed else 'BN', requires_grad=True)
        else:
            norm_cfg = None
        self.decoder = UPerHead(
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            num_classes=out_channels,
            upscale_ratio=upscale_ratio
        )


    def forward(self, x, input_dict_extra={}):
        if self.cfg_DPT.if_share_pretrained:
            backbone_output_tuple = input_dict_extra['shared_pretrained']
        else:
            backbone_output_tuple = self.backbone(x)
        x_out = self.decoder(backbone_output_tuple)
        return x_out

class SwinBRDFModel(Swin):
    def __init__(
        self, opt, modality, non_negative=False, **kwargs
    ):

        self.modality = modality
        assert modality in ['al', 'de', 'ro'], 'Invalid modality: %s'%modality
        self.out_channels = {'al': 3, 'de': 1, 'ro': 1}[modality]

        self.non_negative = non_negative

        super().__init__(opt, opt.cfg.MODEL_BRDF.DPT_baseline, self.out_channels, if_batch_norm=opt.cfg.MODEL_BRDF.DPT_baseline.if_batch_norm, **kwargs)

        self.relu = nn.ReLU(True)

    def forward(self, x, input_dict_extra={}):
        x_out = super().forward(x, input_dict_extra=input_dict_extra)

        if self.non_negative:
            x_out = self.relu(x_out)
            
        if self.modality == 'al':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
        elif self.modality == 'ro':
            x_out = torch.clamp(1.01 * torch.tanh(x_out ), -1, 1)
            x_out = torch.mean(x_out, dim=1, keepdim=True)
        elif self.modality == 'de':
            '''
            where x_out is disparity (inversed * baseline)'''
            depth = self.scale * x_out + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth

        return x_out, {}

class SwinLightModel(Swin):
    def __init__(
        self, opt, cfg_DPT, modality, SGNum, **kwargs
    ):
        self.SGNum = SGNum
        self.modality = modality
        assert modality in ['axis', 'lamb', 'weight'], 'Invalid modality: %s'%modality
        self.out_channels = {'axis': 3*SGNum, 'lamb': SGNum, 'weight': 3*SGNum}[modality]

        super().__init__(opt, cfg_DPT, in_channels=11, out_channels=self.out_channels, patch_size=opt.cfg.MODEL_LIGHT.DPT_baseline.swin.patch_size, if_batch_norm=opt.cfg.MODEL_LIGHT.DPT_baseline.if_batch_norm, **kwargs)

    def forward(self, x, input_dict_extra={}):
        x_out = super().forward(x, input_dict_extra=input_dict_extra)

        if self.modality in ['lamb']:
            x_out = torch.tanh(x_out )
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, min=0.)
        elif self.modality in ['weight']:
            # x_out = self.relu(x_out )
            x_out = torch.tanh(x_out )
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, min=0.)
        elif self.modality in ['axis']:
            bn, _, row, col = x_out.size()
            x_out = x_out.view(bn, self.SGNum, 3, row, col)
            x_out = x_out / torch.clamp(torch.sqrt(torch.sum(x_out * x_out,
                dim=2).unsqueeze(2) ), min = 1e-6).expand_as(x_out )
        else:
            assert False

        return x_out

def get_LightNet_Swin(opt, SGNum, modalities=[]):
    assert all([x in ['axis', 'lamb', 'weight'] for x in modalities])

    module_dict = {}
    for modality in modalities:
        module_dict[modality] = SwinLightModel(
            opt,
            cfg_DPT=opt.cfg.MODEL_LIGHT.DPT_baseline, 
            modality=modality, 
            SGNum=SGNum
        )

    if opt.cfg.MODEL_LIGHT.DPT_baseline.if_share_pretrained:
        module_dict['shared_pretrained'] = module_dict[modalities[0]].backbone
        for modality in modalities:
            module_dict[modality].backbone = nn.Identity()

    full_model = torch.nn.ModuleDict(module_dict)
    
    return full_model

    


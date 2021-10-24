from .models_ViT import decoder_layout_emitter_heads_indeptMLP, decoder_layout_emitter_heads

from .models_base_ViT_DPT import Transformer_Hybrid_Encoder_Decoder

class ViTLayoutObjModel(Transformer_Hybrid_Encoder_Decoder):
    def __init__(
        self, 
        opt, 
        cfg_ViT, 
        modality='lo', 
        N_layers_encoder=6, 
        N_layers_decoder=6, 
        ViT_pool = 'mean',  # ViT_pool strategy in the end: 'cls' or 'mean'
        head_names=[], 
        skip_keys=[], keep_keys=[], **kwargs
    ):  
        self.cfg_ViT = cfg_ViT
        self.modality = modality
        assert modality in ['lo', 'ob'], 'Invalid modality: %s'%modality

        self.ViT_pool = ViT_pool
        assert ViT_pool in ['mean', 'cls']

        super().__init__(opt, cfg_ViT=cfg_ViT, head_names=head_names, N_layers_encoder=N_layers_encoder, N_layers_decoder=N_layers_decoder, **kwargs)

        if opt.cfg.MODEL_ALL.ViT_baseline.if_indept_MLP_heads:
            self.heads = decoder_layout_emitter_heads_indeptMLP(opt, if_layout=True, 
            if_two_decoders=not cfg_ViT.if_share_decoder_over_heads, 
            if_layer_norm=opt.cfg.MODEL_ALL.ViT_baseline.if_indept_MLP_heads_if_layer_norm)
        else:
            self.heads = decoder_layout_emitter_heads(opt, if_layout=True, if_two_decoders=not cfg_ViT.if_share_decoder_over_heads)

    def forward(self, x, input_dict_extra={}):
        decoder_out, layers_out = super().forward(x, input_dict_extra=input_dict_extra) # can be tensor + tuple, or dicts of (tensor + tuple)

        if self.cfg_ViT.if_share_decoder_over_heads:
            x_out = decoder_out.mean(dim = 1) if self.ViT_pool == 'mean' else decoder_out[:, 0]
        else:
            x_out = {}
            for head_name in ['camera', 'layout']:
                x_out[head_name] = decoder_out[head_name].mean(dim = 1) if self.ViT_pool == 'mean' else decoder_out[head_name][:, 0]

        if self.modality == 'lo':
            x_out = self.heads(x_out)
        return x_out

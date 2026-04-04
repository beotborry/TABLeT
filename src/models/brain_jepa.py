# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import pandas as pd


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, 
                 patch_size, 
                 crop_size, 
                 pred_depth, 
                 pred_emb_dim, 
                 add_w, 
                 gradient_checkpointing, 
                 model_name='vit_base', 
                 attn_mode='flash_attn', 
                 global_pool=False, 
                 device=None, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 embed_dim=None, 
                 num_classes=1,
                 gradient_csv_path=None,
                 ):
        super(VisionTransformer, self).__init__()
        
        assert gradient_csv_path is not None

        def load_gradient():
            if ".pt" in gradient_csv_path:
                gradient = torch.from_numpy(torch.load(gradient_csv_path))
                # dtype change
                gradient = gradient.to(torch.float32)
                return gradient.unsqueeze(0)
            elif ".csv" in gradient_csv_path:
                df = pd.read_csv(gradient_csv_path, header=None)
                gradient = torch.tensor(df.values, dtype=torch.float32)
                return gradient.unsqueeze(0)

        gradient = load_gradient().to(device, non_blocking=True)
        print("gradient.shape", gradient.shape)
        from src.models.brain_jepa_helper import init_model
        self.encoder, _ = init_model(
            device=device,
            patch_size=patch_size, # 49
            crop_size=crop_size, # (450, 490)
            pred_depth=pred_depth, # 12
            pred_emb_dim=pred_emb_dim, # 384
            model_name=model_name,
            gradient_pos_embed=gradient,
            attn_mode=attn_mode,
            add_w=add_w,
            gradient_checkpointing=gradient_checkpointing)
        
        self.gradient_checkpointing = gradient_checkpointing

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(self.encoder.embed_dim)        
        
        self.head = nn.Linear(self.encoder.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        if self.global_pool:
            x = x[:, :, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            outcome = x[:, 0]

        if self.gradient_checkpointing:
            try:
                x = torch.utils.checkpoint.checkpoint(self.head, outcome, use_reentrant=False)
            except ValueError as e:
                print(1)
        else:
            x = self.head(outcome)

        return x.squeeze(1)


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=1, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, in_chans=1, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
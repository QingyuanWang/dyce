# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .patcher import *
import timm


def build_model(
    model_name: str,
    device: str | int | torch.device,
    state_dict: OrderedDict = None,
    strict_dict_load: bool = True,
    freeze: bool = True,
    **model_init_args
) -> torch.nn.Module:

    model_init_args.setdefault('pretrained', state_dict is None)
    interm_feat = model_init_args.pop('interm_feat', False)
    profiling = model_init_args.pop('profiling', 0)
    no_head = model_init_args.pop('no_head', False)

    if model_name.startswith('resnet'):
        model_name += '.tv_in1k'
        patcher = ResNetPatcher
    elif model_name.startswith('convnextv2'):
        model_name += '.fcmae_ft_in1k'
        patcher = ConvNextPatcher
    elif model_name.startswith('davit'):
        # ['davit_base', 'davit_giant', 'davit_huge', 'davit_large', 'davit_small', 'davit_tiny']
        if model_name.startswith('davit_tiny'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        patcher = DaViTPatcher
    elif model_name.startswith('swin'):
        # ['swin_base_patch4_window7_224.ms_in22k_ft_in1k', 'swin_large_patch4_window7_224.ms_in22k_ft_in1k', 'swin_small_patch4_window7_224.ms_in22k_ft_in1k', 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k']
        model_name += '_patch4_window7_224.ms_in22k_ft_in1k'
        patcher = SwinTPatcher

    elif model_name.startswith('maxvit'):
        # ['maxvit_base_tf_224.in1k', 'maxvit_large_tf_224.in1k', 'maxvit_small_tf_224.in1k',  'maxvit_tiny_tf_224.in1k']
        model_name += '_tf_224.in1k'
        patcher = MaxVitPatcher
    elif model_name.startswith('mobilenetv3'):
        # ['maxvit_base_tf_224.in1k', 'maxvit_large_tf_224.in1k', 'maxvit_small_tf_224.in1k',  'maxvit_tiny_tf_224.in1k']
        if model_name == 'mobilenetv3_large_100':
            model_name = 'mobilenetv3_large_100.ra_in1k'
        elif model_name == 'mobilenetv3_small_100':
            model_name = 'mobilenetv3_small_100.lamb_in1k'
        else:
            raise NotImplementedError
        patcher = MobileNetv3Patcher

    elif model_name.startswith('efficientnet'):
        # efficientnet_b0-b4
        if model_name == 'efficientnet_b0':
            model_name = 'efficientnet_b0.ra_in1k'
        elif model_name == 'efficientnet_b1':
            model_name = 'efficientnet_b1.ft_in1k'
        elif model_name == 'efficientnet_b2':
            model_name = 'efficientnet_b2.ra_in1k'
        elif model_name == 'efficientnet_b3':
            model_name = 'efficientnet_b3.ra2_in1k'
        elif model_name == 'efficientnet_b4':
            model_name = 'efficientnet_b4.ra2_in1k'
        else:
            raise NotImplementedError
        patcher = EfficientNetPatcher
    elif model_name.startswith('efficientvit'):
        # m0-m5,b0-b3
        model_name += '.r224_in1k'
        patcher = EfficientViTPatcher
    elif model_name.startswith('efficientformerv2'):
        # s0-2,l
        model_name += '.snap_dist_in1k'
        patcher = EfficientFormerv2Patcher
    else:
        raise NotImplementedError
    model = timm.create_model(f'{model_name}', **model_init_args).to(device)
    if interm_feat or profiling > 0 or no_head:
        model = patcher().patch(model, no_head, profiling)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=strict_dict_load)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From ViT
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

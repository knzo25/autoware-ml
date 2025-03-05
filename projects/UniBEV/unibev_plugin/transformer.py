# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmcv.cnn import (build_activation_layer, build_conv_layer,
#                      build_norm_layer, xavier_init)
# from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
#                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import BaseTransformerLayer

# from mmdet.models.utils.builder import TRANSFORMER

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        "`MultiScaleDeformableAttention` in MMCV has been moved to "
        "`mmcv.ops.multi_scale_deform_attn`, please update your MMCV"
    )
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

from mmdet3d.registry import MODELS


@MODELS.register_module(force=True)
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs
    ):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs
        )
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

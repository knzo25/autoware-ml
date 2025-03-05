# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.utils import ConfigType, OptConfigType
from mmengine import ConfigDict
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

from mmdet3d.registry import MODELS

from .transformers import DetrTransformerDecoderLayer


@MODELS.register_module(force=True)
class DetrTransformerEncoder(BaseModule):
    """Encoder of DETR.

    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        num_cp (int): Number of checkpointing blocks in encoder layer.
            Default to -1.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self, num_layers: int, layer_cfg: ConfigType, num_cp: int = -1, init_cfg: OptConfigType = None
    ) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([DetrTransformerEncoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    "If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale."
                )
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor, key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).

        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query


@MODELS.register_module(force=True)
class DetrTransformerDecoder(BaseModule):
    """Decoder of DETR.

    Args:
        num_layers (int): Number of decoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        post_norm_cfg (:obj:`ConfigDict` or dict, optional): Config of the
            post normalization layer. Defaults to `LN`.
        return_intermediate (bool, optional): Whether to return outputs of
            intermediate layers. Defaults to `True`,
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        num_layers: int,
        layer_cfg: ConfigType,
        post_norm_cfg: OptConfigType = dict(type="LN"),
        return_intermediate: bool = True,
        init_cfg: Union[dict, ConfigDict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([DetrTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        key_padding_mask: Tensor,
        **kwargs
    ) -> Tensor:
        """Forward function of decoder
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor): The input key, has shape (bs, num_keys, dim).
            value (Tensor): The input value with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).

        Returns:
            Tensor: The forwarded results will have shape
            (num_decoder_layers, bs, num_queries, dim) if
            `return_intermediate` is `True` else (1, bs, num_queries, dim).
        """
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                **kwargs
            )
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
        query = self.post_norm(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query.unsqueeze(0)


@MODELS.register_module(force=True)
class DetrTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        self_attn_cfg: OptConfigType = dict(embed_dims=256, num_heads=8, dropout=0.0),
        ffn_cfg: OptConfigType = dict(
            embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)
        ),
        norm_cfg: OptConfigType = dict(type="LN"),
        init_cfg: OptConfigType = None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if "batch_first" not in self.self_attn_cfg:
            self.self_attn_cfg["batch_first"] = True
        else:
            assert (
                self.self_attn_cfg["batch_first"] is True
            ), "First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag."

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)]
        self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor, key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query

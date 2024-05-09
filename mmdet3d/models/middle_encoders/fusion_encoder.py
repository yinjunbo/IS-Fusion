
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius

import cv2 as cv
import math
import mmcv
import random
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import ext_loader
from ...models import builder
from ...models.builder import FUSION_LAYERS
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

from mmdet3d.models.middle_encoders.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning, to_2tuple)


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

class EmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv2d(input_channel, num_pos_feats, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_pos_feats, num_pos_feats, kernel_size=3, padding=1, bias=False))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class Instane2SceneAtt(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, query_scene, bs, bev_size, attn_mask=None):
        """
        :param query: B C N
        :param query_pos: B N 2
        :return:
        """

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        query2 = self.multihead_attn(query=query, key=key, value=key, attn_mask=attn_mask)[0]

        query = query + self.dropout(query2)
        query = self.norm(query).permute(1, 2, 0)

        query_ins = query.reshape(bs, query.shape[1], bev_size, bev_size)
        attention_weights = torch.matmul(query_scene, query_ins.transpose(2, 3))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attended_query_ins = torch.matmul(attention_weights, query_ins)

        return_feats = query_scene + attended_query_ins

        return return_feats


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n-1) == 0) and n != 0

        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.python_ops_for_test = False

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        if not self.python_ops_for_test:
            output = MultiScaleDeformableAttnFunction_fp32.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        else:
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output, sampling_locations, attention_weights

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_posembed=None, cross_posembed=None):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_posembed = None
        self.cross_posembed = None
        if self_posembed is not None:
            self.self_posembed = self_posembed
        if cross_posembed is not None:
            self.cross_posembed = cross_posembed

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, bev_pos=None,
                src_padding_mask=None, self_attn_mask=None):


        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        # q = k = tgt
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=self_attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, use_look_forward_twice=False, d_model=256, high_dim_query_update=False, no_sine_embed=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        self.use_look_forward_twice = use_look_forward_twice

        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            # if self.no_sine_embed:
            #     self.ref_point_head = MLP(2, d_model, d_model, 3)
            # else:
            #     self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

            self.query_pos_embed = MLP(2 * d_model, d_model, d_model, 2)
            self.key_pos_embed = MLP(2 * d_model, d_model, d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)


    def forward(self, tgt, reference_points, src, src_spatial_shapes,
                src_level_start_index, src_valid_ratios,
                bev_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []

        query_sine_embed = gen_sineembed_for_position(reference_points) # bs, nq, 256*2
        query_pos = self.query_pos_embed(query_sine_embed)

        for lid, layer in enumerate(self.layers):

            output = layer(output, query_pos, reference_points[:, :, None], src, src_spatial_shapes, src_level_start_index, bev_pos,
                           src_padding_mask, self_attn_mask=attn_mask)

            if self.return_intermediate:
                intermediate.append(output)

                if self.use_look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points



class InsContextAtt(nn.Module):
    def __init__(self,
                 num_layers=1,
                 embed_dims=128,
                 bev_size=180,
                 n_points=16,
                 ):
        super(InsContextAtt, self).__init__()

        self.bev_size = bev_size
        self.num_feature_levels = 1
        dropout = 0.1
        decoder_layer = DeformableTransformerDecoderLayer(d_model=embed_dims, d_ffn=embed_dims, dropout=dropout, n_levels=self.num_feature_levels, n_points=n_points)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.query_pos_embed = PositionEmbeddingLearned(2, embed_dims)
        self.key_pos_embed = PositionEmbeddingLearned(2, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, query_feats, query_pos, bev_pos, scene_feats=None, **kwargs):

        scene_feats = scene_feats.permute(0, 1, 3, 2)

        bev_pos = bev_pos/self.bev_size
        key_pos = self.key_pos_embed(bev_pos).permute(0, 2, 1)
        srcs = [scene_feats]
        pos_embeds = [key_pos]
        output = query_feats.transpose(1, 2)
        reference_points = query_pos/self.bev_size
        query_pos_embed = self.query_pos_embed(reference_points).permute(0, 2, 1)

        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        src = src_flatten + lvl_pos_embed_flatten
        for lid, layer in enumerate(self.layers):

            assert reference_points.shape[-1] == 2
            output = layer(output, query_pos_embed, reference_points[:, :, None], src, spatial_shapes,
                           level_start_index)

        return output.transpose(1, 2)


@FUSION_LAYERS.register_module()
class ISFusionEncoder(BaseModule):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,  num_points_in_pillar=10, embed_dims=256, num_classes=10, **kwargs):

        super(ISFusionEncoder, self).__init__()


        self.num_points_in_pillar = num_points_in_pillar
        self.bev_size = kwargs.get('bev_size', 180)
        self.num_views = kwargs.get('num_views', 6)
        region_shape =  kwargs.get('region_shape', None)
        grid_size = kwargs.get('grid_size', None)
        region_drop_info = kwargs.get('region_drop_info', None)

        # self.instance_fusion_mode = True  #kwargs.get('instance_fusion', False)

        self.random_noise = 1.0

        self.embed_dims = embed_dims
        self.conv_fusion = ConvModule(
            self.embed_dims*3,
            self.embed_dims//2,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        self.get_regions = nn.ModuleList()
        self.grid2region_att = nn.ModuleList()
        for l in range(len(region_shape)):
            this_embed_dim = self.embed_dims//2*(l+1)
            region_metas = dict(
                type='SSTInputLayerV2',
                window_shape=region_shape[l],
                sparse_shape=grid_size[l],
                shuffle_voxels=True,
                drop_info=region_drop_info[l],
                pos_temperature=1000,
                normalize_pos=False,
                pos_embed=this_embed_dim,
            )
            grid2region_att = dict(
                type='SSTv2',
                d_model=[this_embed_dim,] * 4,
                nhead=[8, ] * 4,
                num_blocks=1,
                dim_feedforward=[this_embed_dim, ] * 4,
                output_shape=grid_size[l][:2],
                in_channel=self.embed_dims//2 if l==0 else None,
            )
            self.get_regions.append(builder.build_middle_encoder(region_metas))
            self.grid2region_att.append(builder.build_backbone(grid2region_att))

        # instance fusion
        self.instance_num = kwargs.get('instance_num', 200)
        self.nms_kernel_size = 3

        def create_2D_grid(x_size, y_size):
            meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
            # NOTE: modified
            batch_x, batch_y = torch.meshgrid(
                *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
            )
            batch_x = batch_x + 0.5
            batch_y = batch_y + 0.5
            coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
            coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
            return coord_base

        self.bev_pos = create_2D_grid(self.bev_size, self.bev_size)

        self.conv_ins = ConvModule(
            embed_dims//2,
            embed_dims//2,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        self.conv_scene = ConvModule(
            embed_dims//2,
            embed_dims//2,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        self.conv_heatmap = ConvModule(
            embed_dims//2,
            embed_dims//2,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        self.heatmap_head_1 = ConvModule(
            embed_dims//2,
            embed_dims//4,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        self.heatmap_head_2 = ConvModule(
            embed_dims//4,
            embed_dims//4,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        self.heatmap_head_3 = nn.Conv2d(embed_dims//4, num_classes, kernel_size=3, stride=1, padding=1)

        self.instance_att = InsContextAtt(num_layers=2, embed_dims=embed_dims//2, bev_size=self.bev_size)
        self.instance_to_scene_att = Instane2SceneAtt(d_model=embed_dims//2)

    def img_point_sampling(self, reference_voxel, mlvl_feats, num_cam=6, batch_size=4, **kwargs):  # from UVTR

        img_aug_matrix = kwargs.get('img_aug_matrix', None)
        lidar_aug_matrix = kwargs.get('lidar_aug_matrix', None)
        lidar2image = kwargs.get('lidar2img', None)
        image_size = kwargs['img_metas'][0]['input_shape']

        # Transfer to Point cloud range with X,Y,Z
        mask = []
        reference_voxel_cam = []

        for b in range(batch_size):
            cur_coords = reference_voxel[b].reshape(-1, 3)[:, :3].clone()
            cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b] if not isinstance(lidar_aug_matrix, list) else lidar_aug_matrix[0][b]
            cur_lidar2image = lidar2image[b] if not isinstance(lidar2image, list) else lidar2image[0][b]

            # inverse aug for pseudo points
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)  # cur_coords: [3, N]
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

            if self.random_noise is not None and self.training:
                seed = np.random.rand()
                if seed > 0.5:
                    cur_coords += random.uniform(-self.random_noise, self.random_noise)

            # get 2d coords
            dist = cur_coords[:, 2, :].clone()
            this_mask = (dist > 1e-5)

            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)


            cur_coords[..., 0] /= image_size[1]
            cur_coords[..., 1] /= image_size[0]
            cur_coords = (cur_coords - 0.5) * 2  # to [-1, +1]

            this_mask = (this_mask & (cur_coords[..., 0] > -1.0)
                         & (cur_coords[..., 0] < 1.0)
                         & (cur_coords[..., 1] > -1.0)
                         & (cur_coords[..., 1] < 1.0)
                         )

            mask.append(this_mask)
            reference_voxel_cam.append(cur_coords)

        # sample img features
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            _, C, H, W = feat.size()
            feat = feat.view(batch_size, num_cam, C, H, W).transpose(0, 1)
            multi_cam_feats = []
            for b in range(batch_size):
                for k in range(num_cam):
                    reference_points_cam_lvl = reference_voxel_cam[b][k].view(-1, 1, 2)
                    sampled_feat = F.grid_sample(feat[k][b].unsqueeze(dim=0), reference_points_cam_lvl.unsqueeze(dim=0))  # feat: [24,256,32,88]; reference_points_cam_lvl: [24, num_query, 1, 2]
                    sampled_feat = sampled_feat.view(1, C, -1, 1).squeeze(0).squeeze(-1)
                    multi_cam_feats.append(sampled_feat)

            multi_cam_feats = [torch.stack(multi_cam_feats[idx:idx+num_cam]).sum(0) for idx in range(0, len(multi_cam_feats), num_cam)]
            if not len(reference_voxel[0].shape) == 3:
                multi_cam_feats = torch.cat(multi_cam_feats, dim=1)
            sampled_feats.append(multi_cam_feats)

        if not len(reference_voxel[0].shape) == 3:
            sampled_feats = torch.stack(sampled_feats).sum(0).transpose(0, 1)

        return reference_voxel_cam, mask, sampled_feats

    def img_fv_to_bev(self, mlvl_feats, bs, **kwargs):

        pts_metas = kwargs['pts_metas']
        pillars = pts_metas['pillars'][..., :3]
        pillar_coors = pts_metas['pillar_coors']
        num_points_in_pillar = pts_metas['pillars'].shape[1]
        ref_3d = []
        pillar_coors_list = []

        for i in range(bs):
            this_idx = pillar_coors[:, 0]==i
            this_coors = pillar_coors[this_idx]
            pillar_coors_list.append(this_coors)
            ref_3d.append(pillars[this_idx])

        reference_points_cam_stack, ref_mask_stack, ref_img_pillar_cam = self.img_point_sampling(
            ref_3d, mlvl_feats, self.num_views, bs, **kwargs)

        decorated_img_feat = torch.zeros([bs, self.embed_dims, self.bev_size, self.bev_size]).type_as(mlvl_feats[0])
        for b in range(bs):
            this_pillar_coors = pillar_coors_list[b]
            output = ref_img_pillar_cam[0][b].reshape(self.embed_dims, -1, num_points_in_pillar)
            decorated_img_feat[b, :, this_pillar_coors[:, 2].long(), this_pillar_coors[:, 3].long()] = output.sum(dim=2)

        return decorated_img_feat

    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord


    def instance_fusion(self, bev_feats, scene_feats, bs, **kwargs):

        bev_pos = self.bev_pos.repeat(bs, 1, 1).to(bev_feats.device)
        out = bev_feats.permute(0, 1, 3, 2).contiguous()

        ins_heatmap = self.conv_heatmap(out.clone().detach())
        ins_heatmap=self.heatmap_head_1(ins_heatmap)
        ins_heatmap=self.heatmap_head_2(ins_heatmap)
        ins_heatmap=self.heatmap_head_3(ins_heatmap)

        heatmap = ins_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.num_views == 6:
            local_max[
            :,
            8,
            ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[
            :,
            9,
            ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.num_views == 5:  # for Pedestrian & Cyclist in Waymo
            local_max[
            :,
            1,
            ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[
            :,
            2,
            ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(bs, heatmap.shape[1], -1)

        instance_num = self.instance_num
        top_proposals = heatmap.view(bs, -1).argsort(dim=-1, descending=True)[..., : instance_num]

        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
                                   dim=1)

        query_pos_new = torch.zeros_like(query_pos)
        query_pos_new[..., 0] = query_pos[..., 1]
        query_pos_new[..., 1] = query_pos[..., 0]

        x_scene = self.conv_scene(bev_feats.permute(0, 1, 3, 2))
        x_scene_flatten = x_scene.view(bs, x_scene.shape[1], -1)
        x_ins = x_scene_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, x_scene.shape[1], -1),
                                                  dim=-1)

        x_ins = self.instance_att(x_ins, query_pos_new, bev_pos, scene_feats=x_scene, **kwargs)

        bev_feats = self.conv_ins(bev_feats).flatten(2, 3)
        return_features = self.instance_to_scene_att(bev_feats, x_ins, scene_feats, bs, self.bev_size)

        return return_features, ins_heatmap


    @auto_fp16()
    def forward(self,
                img_mlvl_feats,
                lidar_feats,
                bs,
                **kwargs):


        img_bev_feats = self.img_fv_to_bev([img_mlvl_feats[1]], bs, **kwargs)

        kwargs.update(dict(img_bev_feats=img_bev_feats))
        kwargs.update(dict(lidar_feats=lidar_feats))

        bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats], dim=1))

        grid_features = bev_feats.flatten(2, 3).permute(0, 2, 1).reshape(-1, bev_feats.shape[1])
        bev_coords = self.create_dense_coord(self.bev_size, self.bev_size, bs).type_as(grid_features).int()
        this_coords = []
        for k in range(bs):
            this_coord = bev_coords[k].reshape(4, -1).transpose(1, 0)
            this_coords.append(this_coord)
        grid_coords = torch.cat(this_coords, dim=0)

        pts_backbone = kwargs.get('pts_backbone', None)

        ins_hm = None
        return_feats = [] 
        for i in range(len(self.get_regions)):
            x = self.get_regions[i](grid_features, grid_coords, bs)
            x = self.grid2region_att[i](x)

            if i == 0:
                x[0], ins_hm = self.instance_fusion(bev_feats, x[0], bs, **kwargs)

            grid_features, grid_coords, this_feat = pts_backbone(x, 'stage{}'.format(i+1))
            return_feats.append(this_feat)

        return return_feats, ins_hm






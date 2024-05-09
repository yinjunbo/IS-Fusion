# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES
import torch

@BACKBONES.register_module()  # original
class SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(SECOND, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')



    def forward(self, x, **kwargs):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):

            x = self.blocks[i](x)

            outs.append(x)

        return tuple(outs)


@BACKBONES.register_module()  # sst mode
class SECONDV2(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(SECONDV2, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            if layer_strides[i] == 2:
                self.ds_layer = nn.Sequential(build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                    build_norm_layer(norm_cfg, out_channels[i])[1],
                    nn.ReLU(inplace=True),)
                block = []
            else:
                block = [
                    build_conv_layer(
                        conv_cfg,
                        in_filters[i],
                        out_channels[i],
                        3,
                        stride=layer_strides[i],
                        padding=1),
                    build_norm_layer(norm_cfg, out_channels[i])[1],
                    nn.ReLU(inplace=True),
                ]

            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)


        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')


    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        # batch_idx =  torch.zeros_like(batch_x)
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord


    def forward(self, x, stage=None):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        if stage=="stage1":
            x = self.blocks[0](x[0])

            return_features = self.ds_layer(x)
            N, C, H, W = return_features.shape
            return_features = return_features.reshape(N, C, -1).permute(0, 2, 1).reshape(-1, C)
            bev_coords = self.create_dense_coord(H, W, N).type_as(return_features).int()
            this_coords = []
            for k in range(N):
                this_coord = bev_coords[k]
                this_coord = this_coord.reshape(4, -1).transpose(1, 0)
                this_coords.append(this_coord)

            return_coords = torch.cat(this_coords, dim=0)

            return return_features, return_coords, x


        elif stage=="stage2":
            x = self.blocks[1](x[0])
            return None, None, x

        else:
            x_1 = self.blocks[0](x)
            outs.append(x_1)
            x_2 = self.ds_layer(x_1)
            x_2 = self.blocks[1](x_2)
            outs.append(x_2)
            return tuple(outs)



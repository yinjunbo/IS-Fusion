from mmdet.models import BACKBONES

import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer, ConvModule
from mmdet3d.models.sst.sst_basic_block_v2 import BasicShiftBlockV2


@BACKBONES.register_module()
class SSTv2(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[],
        nhead=[],
        num_blocks=6,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        debug=True,
        in_channel=None,
        checkpoint_blocks=[],
        layer_cfg=dict(),
        ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )

        self.block_list = nn.ModuleList(block_list)
            
        self._reset_parameters()

        self.output_shape = output_shape

        self.debug = debug


    def forward(self, voxel_info, **kwargs):
        '''
        '''
        num_shifts = 2 
        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

        batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        voxel_feat = voxel_info['voxel_feats']
        ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)

        for i, block in enumerate(self.block_list):   # sst block
            output = block(output, pos_embed_list, ind_dict_list,
                           padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)

        output = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)

        output_list = []
        output_list.append(output)

        return output_list

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name and 'tau' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas
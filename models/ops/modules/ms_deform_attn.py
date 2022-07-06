# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

# from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, n_frame=4,
                 mode='encoder', use_pytroch_deform=False, attention_vis=False):
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
        # # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        # if not _is_power_of_2(_d_per_head):
        #     warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
        #                   "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_frame = n_frame
        self.use_pytroch_deform = use_pytroch_deform
        self.mode = mode
        assert self.mode in ['encoder', 'decoder']
        self.attention_vis = attention_vis

        # decoder samples over all frames
        offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets = nn.ModuleList([offsets for _ in range(self.n_frame)])
        weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.ModuleList([weights for _ in range(self.n_frame)])

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.sampling_offsets:
            constant_(layer.weight.data, 0.)
        # nhead: initial offsets uniformly distribute on <nhead> directions
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # [n_heads, 2]
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])\
            .view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            for layer in self.sampling_offsets:
                layer.bias = nn.Parameter(grid_init.view(-1))
        for layer in self.attention_weights:
            constant_(layer.weight.data, 0.)
            constant_(layer.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, T1, Length_{query}, C)
        :param reference_points            (N, T1, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
        :param input_flatten               (N, T2, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, T2, \sum_{l=0}^{L-1} H_l \cdot W_l, C), True for padding elements, False for non-padding elements

        :return output                     (N, T1, Length_{query}, C)
        """
        N, T1, Len_q, _ = query.shape
        N, T2, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask, float(0))
        # (N, T2, Len_in, n_heads, C//n_heads)
        value = value.view(N, T2, Len_in, self.n_heads, self.d_model // self.n_heads)
        # (N, T1, Len_q, n_heads, n_levels, n_points, T2)
        # attention_weights = self.multi_frame_attention_weights(query, value)

        # (N, T1, Len_q, n_heads, n_levels, n_points, T2, 2)
        # sampling_locations = self.multi_frame_sampling_locations(query, reference_points, value, input_spatial_shapes)

        # reference_points(N, T, Len_q, n_levels, 2), normalizer: (n_levels, 2)
        normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)

        outputs, vis_sampling_locations, vis_attention_weights = [], [], []
        for t1 in range(T1):
            # not current pose prediction
            if t1 < self.n_frame:
                # (N, Len_q, n_heads, n_levels, n_points, T2)
                # attention_weights_t1 = attention_weights[:, t1]
                att_weights_t1 = []
                # for t2 in range(T2):
                for i in range(-1, 2, 1):  # sample among neighboring frames, i = -1, 0, 1
                    t2 = t1 + i
                    if t2 < 0 or t2 >= self.n_frame:
                        continue

                    # [N, T1, Len_q, n_heads, n_levels, n_points]
                    att_weights = self.attention_weights[t2](query[:, t1]) \
                        .view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
                    att_weights_t1.append(att_weights)
                # [N, Len_q, n_heads, n_levels, n_points, T2]
                att_weights_t1 = torch.stack(att_weights_t1, dim=-1)
                # [N, Len_q, n_heads, n_levels, n_points, T2]
                att_weights_t1 = F.softmax(att_weights_t1.flatten(-3), -1) \
                    .view(N, Len_q, self.n_heads, self.n_levels, self.n_points, -1)

                outputs_t1, vis_sampling_locations_t1 = [], []
                # for t2 in range(T2):  # sample over all temporal frames
                count = 0
                for i in range(-1, 2, 1):  # sample over neighboring frames, i = -1, 0, 1
                    t2 = t1 + i  # t2 = t1-1, t1, t1+1
                    if t2 < 0 or t2 >= self.n_frame:
                        continue
                    # print('frame {}, sample on frame {}'.format(t1, t2))

                    # (N, Len_q, n_heads, n_levels, n_points, 2)
                    sampling_offsets_t2 = self.sampling_offsets[t2](query[:, t1]) \
                        .view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
                    sampling_offsets_t2 = sampling_offsets_t2 / normalizer[None, None, None, :, None, :]
                    sampling_locations_t2 = reference_points[:, t1, :, None, :, None, :] + sampling_offsets_t2

                    if self.attention_vis:
                        # (N, Len_q, n_heads, n_levels, n_points, 2)
                        vis_sampling_locations_t1.append(sampling_locations_t2.detach())

                    # output (N, Len_q, C)
                    if self.use_pytroch_deform:
                        output = ms_deform_attn_core_pytorch(
                            value[:, t2], input_spatial_shapes,
                            sampling_locations_t2, att_weights_t1[..., count]
                        )
                    else:
                        output = MSDeformAttnFunction.apply(
                            value[:, t2].contiguous(), input_spatial_shapes, input_level_start_index,
                            sampling_locations_t2, att_weights_t1[..., count].contiguous(), self.im2col_step
                        )
                    count += 1
                    outputs_t1.append(output.contiguous())
            else:
                # future pose prediction will sample on all frames
                # (N, Len_q, n_heads, n_levels, n_points, T2)
                # attention_weights_t1 = attention_weights[:, t1]
                att_weights_t1 = []
                for t2 in range(T2):
                    # [N, T1, Len_q, n_heads, n_levels, n_points]
                    att_weights = self.attention_weights[t2](query[:, t1]) \
                        .view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
                    att_weights_t1.append(att_weights)
                # [N, Len_q, n_heads, n_levels, n_points, T2]
                att_weights_t1 = torch.stack(att_weights_t1, dim=-1)
                # [N, Len_q, n_heads, n_levels, n_points, T2]
                att_weights_t1 = F.softmax(att_weights_t1.flatten(-3), -1) \
                    .view(N, Len_q, self.n_heads, self.n_levels, self.n_points, -1)

                outputs_t1, vis_sampling_locations_t1 = [], []
                for t2 in range(T2):  # sample over all temporal frames
                    # (N, Len_q, n_heads, n_levels, n_points, 2)
                    sampling_offsets_t2 = self.sampling_offsets[t2](query[:, t1]) \
                        .view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
                    sampling_offsets_t2 = sampling_offsets_t2 / normalizer[None, None, None, :, None, :]
                    sampling_locations_t2 = reference_points[:, t1, :, None, :, None, :] + sampling_offsets_t2

                    if self.attention_vis:
                        # (N, Len_q, n_heads, n_levels, n_points, 2)
                        vis_sampling_locations_t1.append(sampling_locations_t2.detach())

                    # output (N, Len_q, C)
                    if self.use_pytroch_deform:
                        output = ms_deform_attn_core_pytorch(
                            value[:, t2], input_spatial_shapes,
                            sampling_locations_t2, att_weights_t1[..., t2]
                        )
                    else:
                        output = MSDeformAttnFunction.apply(
                            value[:, t2].contiguous(), input_spatial_shapes, input_level_start_index,
                            sampling_locations_t2, att_weights_t1[..., t2].contiguous(), self.im2col_step
                        )
                    outputs_t1.append(output.contiguous())
            # outputs_t1 (N, Len_q, C, T2) -> (N, Len_q, C)
            outputs_t1 = torch.stack(outputs_t1, -1).sum(-1)
            outputs.append(outputs_t1)

            if self.attention_vis:
                vis_sampling_locations_t1 = torch.stack(vis_sampling_locations_t1, dim=-2)
                # (N, Len_q, n_heads, n_levels, n_points, T2 or 2 or 3, 2)
                vis_sampling_locations.append(vis_sampling_locations_t1.detach())
                # (N, Len_q, n_heads, n_levels, n_points, T2 or 2 or 3)
                vis_attention_weights.append(att_weights_t1.detach())

        # outputs (N, T, Len_q, C)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.output_proj(outputs)

        if self.attention_vis:
            # (N, Len_q, T1, n_heads, n_levels, n_points, T2, 2)
            return outputs, (vis_sampling_locations, vis_attention_weights)
        else:
            return outputs


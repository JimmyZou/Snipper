# ------------------------------------------------------------------------
# # Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 n_frame=4, n_future_frame=2, use_pytroch_deform=False, num_keypoints=15):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_frame = n_frame
        self.n_future_frame = n_future_frame
        self.num_keypoints = num_keypoints

        # self.memory_proj1 = nn.Conv2d(self.num_keypoints, num_keypoints, kernel_size=1)
        # self.memory_proj2 = nn.Conv2d(self.d_model - self.num_keypoints,
        #                               self.d_model - self.nhead * self.num_keypoints,
        #                               kernel_size=1)

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, n_frame,
                                                          use_pytroch_deform)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, n_frame,
                                                          use_pytroch_deform)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.temporal_embed = nn.Parameter(torch.Tensor(self.n_frame + self.n_future_frame, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        # mask [bs, c, t, h, w], assume a sequence of t masks is the same
        _, _, _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, 0, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed):
        # print('transformer')
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(3).permute(0, 2, 3, 1)  # (bs, t, hw, c)
            mask = mask.flatten(3).permute(0, 2, 3, 1)  # (bs, t, hw, c)
            pos_embed = pos_embed.flatten(3).permute(0, 2, 3, 1)  # (bs, t, hw, c)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 2)  # (bs, t, \sum_lvl hw, c)
        mask_flatten = torch.cat(mask_flatten, 2)  # (bs, t, \sum_lvl hw, c)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)  # (bs, t, \sum_lvl hw, c)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # (lvl, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  # (lvl)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # (bs, lvl, 2)
        # print(spatial_shapes, level_start_index, valid_ratios)
        # print(src_flatten.shape, mask_flatten.shape, lvl_pos_embed_flatten.shape)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten, mask_flatten, self.n_frame)
        # print(memory.shape)
        # prepare input for decoder
        bs, _, _, c = memory.shape
        tmp = memory.split([H_ * W_ for (H_, W_) in spatial_shapes], dim=2)
        # [(bs, t, hw, c)] --> [(bs, t, h, w, c)]
        out_memory = [tmp[lid_].reshape(bs, self.n_frame, H_, W_, c) for lid_, (H_, W_) in enumerate(spatial_shapes)]

        # t = self.n_frame
        # heatmaps, input_memory = [], []
        # for i in range(len(out_memory)):
        #     item = out_memory[i].flatten(0, 1).permute(0, 3, 1, 2)  # (bs*t, h, w, c) --> (bs*t, c, h, w)
        #     h, w = item.shape[2:4]
        #     # (bs*t, num_joints, h, w) -- > (bs, t, h, w, 1, num_joints)
        #     heatmap = self.memory_proj1(item[:, :self.num_keypoints, :, :]).permute(0, 2, 3, 1)\
        #         .reshape(bs, t, h, w, self.num_keypoints).unsqueeze(dim=4)
        #     heatmaps.append(heatmap)  # [(bs, t, h, w, num_joints)]
        #
        #     # (bs*t, c, h, w) -- > (bs, t, h, w, c)
        #     feat_memory = self.memory_proj2(item[:, self.num_keypoints:, :, :]).permute(0, 2, 3, 1)\
        #         .reshape(bs, t, h, w, self.d_model - self.nhead * self.num_keypoints).unsqueeze(dim=4)
        #     # (bs, t, h, w, nhead, c/nhead - num_joints)
        #     feat_memory = feat_memory.view(bs, t, h, w, self.nhead, c // self.nhead - self.num_keypoints)
        #     # (bs, t, h, w, nhead, num_joints)
        #     _heatmap = heatmap.repeat(1, 1, 1, 1, self.nhead, 1)
        #     # (bs, t, h, w, nhead, c/nhead)
        #     feat_memory = torch.cat([_heatmap, feat_memory], dim=-1)
        #     # (bs, t, h, w, nhead, c/nhead) --> (bs, t, h, w, c) --> (bs, t, hw, c)
        #     feat_memory = feat_memory.flatten(4, 5).flatten(2, 3)
        #     input_memory.append(feat_memory)
        # input_memory = torch.cat(input_memory, dim=2)

        input_memory = memory
        heatmaps = []
        t = self.n_frame
        for i in range(len(out_memory)):
            item = out_memory[i]  # (bs, t, h, w, c)
            h, w = item.shape[2:4]
            # [(bs, t, h, w, nhead, c//nhead)]
            item = item.view(bs, t, h, w, self.nhead, c // self.nhead)
            heatmap = item[..., 0:self.num_keypoints]  # [(bs, t, h, w, nhead, num_joints)]
            heatmaps.append(heatmap)

        t = self.n_frame + self.n_future_frame
        n_query = query_embed.shape[0] // t
        query_pos, query_obj = torch.split(query_embed, c, dim=-1)  # (t*n_query, c)
        query_pos = query_pos.reshape(t, n_query, c).unsqueeze(0).expand(bs, -1, -1, -1)  # (bs, t, n_query, c)
        query_pos = query_pos + self.temporal_embed.view(1, t, 1, c)
        query_obj = query_obj.reshape(t, n_query, c).unsqueeze(0).expand(bs, -1, -1, -1)  # (bs, t, n_query, c)
        # (bs, t, num_queries, 2)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        # decoder
        hs, inter_references, inter_att_data = self.decoder(query_obj, reference_points, input_memory,
                                                            spatial_shapes, level_start_index, valid_ratios,
                                                            query_pos, mask_flatten)
        # print(hs.shape)
        inter_references_out = inter_references
        return hs, heatmaps, init_reference_out, inter_references_out, inter_att_data


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, n_frame=4, use_pytroch_deform=False):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, n_frame, 'encoder', use_pytroch_deform)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                              src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # valid_ratio (b, lvl, 3), spatial_shapes (lvl, 3)
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)  # [b, HW]
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [b, HW, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [b, \sum_lvl HW, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [b, \sum_lvl HW, lvl, 2]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, n_frame=1):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # [b, T, \sum_lvl HW, lvl, 2]
        reference_points = reference_points.unsqueeze(1).expand(-1, n_frame, -1, -1, -1)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, n_frame=4, use_pytroch_deform=False):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, n_frame,
                                       'decoder', use_pytroch_deform, True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        bs, t, lq, c = tgt.shape
        # self attention
        tgt = tgt.view(bs, t*lq, c)
        query_pos = query_pos.view(bs, t*lq, c)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt = tgt.view(bs, t, lq, c)
        query_pos = query_pos.view(bs, t, lq, c)
        tgt2, atten_data = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                                           src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, atten_data


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.root_embed = None
        self.class_embed = None

    def forward(self, query_obj, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = query_obj  # (bs, t, n_query, c)

        intermediate = []
        intermediate_reference_points = []
        intermediate_att_data = []
        for lid, layer in enumerate(self.layers):
            # reference_points (bs, t, n_query, 2), src_valid_ratios (bs, lvl, 2)
            # reference_points_input (bs, t, n_query, lvl, 2)
            reference_points_input = reference_points[:, :, :, None, :] * src_valid_ratios[:, None, None, :, :]
            # output (bs, t, L_q, c)
            output, atten_data = layer(output, query_pos, reference_points_input, src,
                                       src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative refinement
            if self.root_embed is not None:
                tmp = self.root_embed[lid](output)[..., 0:2]  # [bs, t, n_query, 4] (x, y, vis, d)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_att_data.append(atten_data)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), intermediate_att_data

        return output, reference_points, intermediate_att_data


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        # two_stage=args.two_stage,
        # two_stage_num_proposals=args.num_queries,
        n_frame=args.num_frames,
        n_future_frame=args.num_future_frames,
        use_pytroch_deform=args.use_pytorch_deform,
        num_keypoints=args.num_kpts
    )



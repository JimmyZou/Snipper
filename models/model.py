# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deforamble_transformer
import copy
from datasets.hybrid_dataloader import ROOTJOINTCONT


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SnipperDeformable(nn.Module):
    def __init__(self, backbone, transformer, num_queries, num_feature_levels,
                 num_frames, num_future_frames, num_keypoints, aux_loss=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.num_frames = num_frames
        self.num_future_frames = num_future_frames
        self.num_keypoints = num_keypoints
        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
            # print(num_backbone_outs, num_feature_levels)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        # print(self.input_proj)

        self.query_embed = nn.Embedding(num_queries * (num_frames + num_future_frames), transformer.d_model * 2)
        self.class_embed = nn.Linear(transformer.d_model, 2)

        self.root_embed = MLP(transformer.d_model, transformer.d_model, 4, 1)
        self.joint_embed = nn.ModuleList([
            MLP(transformer.d_model, transformer.d_model, 4, 1) for _ in range((num_keypoints - 1))])

        self.class_embed = nn.ModuleList([self.class_embed for _ in range(transformer.decoder.num_layers)])
        self.root_embed = nn.ModuleList([self.root_embed for _ in range(transformer.decoder.num_layers)])
        self.joint_embed = nn.ModuleList([self.joint_embed for _ in range(transformer.decoder.num_layers)])

        self.transformer.decoder.root_embed = self.root_embed
        self.transformer.decoder.class_embed = self.class_embed

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (no_person or is_person) for all queries.
                                Shape = [bs x num_queries x (num_frames + num_future_frames) x 2]
               - "pred_kepts2d": predicted 2D pose (u, v, visibility)
                                shape = [bs x num_queries x (num_frames + num_future_frames) x num_kepts x 3]
               - "pred_depth": predicted depth
                                shape = [bs x num_queries x (num_frames + num_future_frames) x num_kepts x 1]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
            # print(l, srcs[-1].shape, masks[-1].shape)

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                # print(l, src.shape, mask.shape, pos_l.shape)

        # reshape from [b*t, c, h, w] to [b, c, t, h, w]
        for l in range(self.num_feature_levels):
            # print('original', l, srcs[l].shape, masks[l].shape, pos[l].shape)
            n, c, h, w = srcs[l].shape
            # [b*t, c, h, w] -> [b, t, c, h, w] -> [b, c, t, h, w]
            srcs[l] = srcs[l].reshape(n // self.num_frames, self.num_frames, c, h, w).transpose(1, 2)
            # [b*t, h, w] -> [b, t, 1, h, w] -> [b, t, c, h, w] -> [b, c, t, h, w]
            masks[l] = masks[l].reshape(n // self.num_frames, self.num_frames, 1, h, w).\
                repeat(1, 1, c, 1, 1).transpose(1, 2)
            # [b*t, c, h, w] -> [b, t, c, h, w] -> [b, c, t, h, w]
            pos[l] = pos[l].reshape(n // self.num_frames, self.num_frames, c, h, w).transpose(1, 2)
            # print('after reshaping', l, srcs[l].shape, masks[l].shape, pos[l].shape)

        # [num_queries * num_frames, hidden_dims * 2]
        query_embeds = self.query_embed.weight
        # hs: [num_dec_layers, bs, t, num_queries, d_model]
        # heatmaps: [(bs, t, h, w, num_joints)]
        hs, heatmaps, init_reference, inter_references, inter_att_data = \
            self.transformer(srcs, masks, pos, query_embeds)
        # print(hs.shape, init_reference.shape, inter_references.shape)
        num_dec_layers, bs, t, _, c = hs.shape

        outputs_classes, outputs_roots, outputs_joints = [], [], []
        for l in range(num_dec_layers):
            # [bs, num_queries, t, 2]
            outputs_class = self.class_embed[l](hs[l])
            outputs_classes.append(outputs_class.transpose(1, 2))
            # print(outputs_classes[-1].shape)

            # [bs, t, num_queries, num_frame, 2]
            if l == 0:
                reference = init_reference
            else:
                reference = inter_references[l - 1]
            reference = inverse_sigmoid(reference)
            # [bs, t, num_queries, 1, 4], (x, y, vis, d)
            tmp = self.root_embed[l](hs[l]).view(bs, t, self.num_queries, 1, 4)
            tmp[..., :2] += reference[:, :, :, None, :]
            outputs_root = tmp.sigmoid()
            outputs_roots.append(outputs_root.transpose(1, 2))  # [bs, num_queries, t, 1, 4]

            # # [bs, t, num_queries, num_kpts - 1, 4] -> [bs, num_queries, t, num_kpts - 1, 4], (u, v, vis, d)
            # outputs_joint = self.joint_embed[l](hs[l])\
            #     .reshape(bs, t, self.num_queries, self.num_keypoints - 1, 4)
            # outputs_joints.append(outputs_joint.transpose(1, 2))
            # # print(outputs_roots[-1].shape, outputs_joints[-1].shape)

            # [bs, t, num_queries, num_kpts - 1, 4] -> [bs, num_queries, t, num_kpts - 1, 4], (x, y, vis, d)
            outputs_joint = [self.joint_embed[l][i](hs[l]).reshape(bs, t, self.num_queries, 1, 4)
                             for i in range(self.num_keypoints - 1)]
            outputs_joint = torch.cat(outputs_joint, dim=3)

            outputs_joints.append(outputs_joint.transpose(1, 2))

        outputs_classes = torch.stack(outputs_classes)  # [num_dec_layers, bs, num_queries, t, 2]
        outputs_roots = torch.stack(outputs_roots)  # [num_dec_layers, bs, num_queries, t, 1, 4]
        outputs_joints = torch.stack(outputs_joints)  # [num_dec_layers, bs, num_queries, t, num_kpts - 1, 4]
        # [num_dec_layers, bs, num_queries, t, num_kpts, 4]
        outputs_kpts = torch.cat([outputs_roots, outputs_joints], dim=-2)

        out = {
            # bs x num_queries x (num_frames + num_future_frames) x 2
            'pred_logits': outputs_classes[-1],
            # bs x num_queries x (num_frames + num_future_frames) x num_kepts x 3
            'pred_kpts2d': outputs_kpts[-1, ..., 0:3],
            # bs x num_queries x (num_frames + num_future_frames) x num_kepts x 1
            'pred_depth': outputs_kpts[-1, ..., 3:4],
            # [(bs, t, h, w, nhead, num_joints)]
            'heatmaps': heatmaps,
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_classes, outputs_kpts)
        return out, (init_reference, inter_references, inter_att_data)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_classes, outputs_kpts):
        aux_outputs = []
        n = outputs_classes[:-1].shape[0]
        for i in range(n):
            out_inter = {
                # bs x num_queries x (num_frames + num_future_frames) x 2
                'pred_logits': outputs_classes[:-1][i],
                # bs x num_queries x (num_frames + num_future_frames) x num_kepts x 3
                'pred_kpts2d': outputs_kpts[:-1][i, ..., 0:3],
                # bs x num_queries x (num_frames + num_future_frames) x num_kepts x 1
                'pred_depth': outputs_kpts[:-1][i, ..., 3:4],
            }
            aux_outputs.append(out_inter)
        return aux_outputs


class SetCriterion(nn.Module):
    """ This class computes the loss for VisTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, losses, eos_coef, weight_dict, cont_weights):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the loss names to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        self.eos_coef = eos_coef
        self.weight_dict = weight_dict
        empty_weight = torch.ones(2)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.eps = 10e-6
        self.register_buffer('cont_weights', cont_weights)  # [1, 1, num_joints, 1]

    def loss_is_human(self, outputs, targets, indices, num_traj):
        """
        Classification loss (NLL)
        0 is not human, 1 is human
        """
        src_logits = outputs["pred_logits"]  # b x n_query x t x 2
        # print(src_logits)

        # b*m x T
        kpts2d = torch.cat([t['kpts2d'][indices[i][1]] for i, t in enumerate(targets)], dim=0)
        tgt_vis = (kpts2d[:, :, :, 2].sum(dim=2) > 0).long()

        idx = self._get_src_permutation_idx(indices)
        # b x n_query x t
        target_classes = torch.full(src_logits.shape[:3], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx[0], idx[1], :] = tgt_vis

        loss_is_human = F.cross_entropy(src_logits.flatten(1, 2).transpose(1, 2), target_classes.flatten(1, 2),
                                        self.empty_weight, reduction='none')
        losses = {'loss_is_human': loss_is_human.mean()}
        return losses

    def loss_root(self, outputs, targets, indices, num_traj):
        """
        root joint loss: l1 position, l2 visibility
        """
        src_idx = self._get_src_permutation_idx(indices)
        src_root_kpts2d = outputs["pred_kpts2d"][src_idx[0], src_idx[1], :, :1, :]  # b*m x T x 1 x 3
        src_root_depth = outputs["pred_depth"][src_idx[0], src_idx[1], :, :1, :]  # b*m x T x 1 x 1

        # print(src_root_kpts2d.size())

        tgt_root_kpts2d = torch.cat([t['kpts2d'][indices[i][1]] for i, t in enumerate(targets)], dim=0)[:, :, :1, :]
        tgt_root_vis = tgt_root_kpts2d[..., 2:3]   # b*m x T x 1 x 1

        _tgt_root_depth = torch.cat([t['depth'][indices[i][1]] for i, t in enumerate(targets)], dim=0)[:, :, :1, :]
        tgt_root_depth_exist = _tgt_root_depth[..., 1:2]
        tgt_root_depth = _tgt_root_depth[..., 0:1]

        # average over time and joints
        error_root = tgt_root_vis * F.l1_loss(src_root_kpts2d[..., 0:2], tgt_root_kpts2d[..., 0:2], reduction='none')
        # error_root = vis * F.mse_loss(src_root_kpts2d[..., 0:2], tgt_root_kpts2d[..., 0:2], reduction='none')
        loss_root_joint = error_root.sum(dim=(-2, -3)) / (tgt_root_vis.sum(dim=(-2, -3)) + self.eps)  # b*m x 2

        # root depth
        error_root_depth = tgt_root_depth_exist * F.l1_loss(tgt_root_depth, src_root_depth, reduction='none')
        loss_root_depth = error_root_depth.sum(dim=(-2, -3)) / (tgt_root_depth_exist.sum(dim=(-2, -3)) + self.eps)  # b*m x 1

        error_root_vis = F.mse_loss(src_root_kpts2d[..., 2:3], tgt_root_vis, reduction='none')
        loss_root_vis = error_root_vis.mean(dim=(-2, -3))  # b*m x 1

        # print('root loss', loss_root_joint.size(), loss_root_vis.size())

        losses = {}
        losses['loss_root'] = loss_root_joint.sum() / num_traj
        losses['loss_root_depth'] = loss_root_depth.sum() / num_traj
        losses['loss_root_vis'] = loss_root_vis.sum() / num_traj
        # print(losses['loss_root'], losses['loss_root_depth'], losses['loss_root_vis'])
        return losses

    def loss_joint(self, outputs, targets, indices, num_traj):
        """
        joint loss: l1, joint = root + joint displacement
        """
        src_idx = self._get_src_permutation_idx(indices)

        tgt_kpts = torch.cat([t['kpts2d'][indices[i][1]] for i, t in enumerate(targets)], dim=0)
        tgt_joint = tgt_kpts[:, :, 1:, 0:2]
        tgt_joint_vis = tgt_kpts[:, :, 1:, 2:3]
        _tgt_joint_depth = torch.cat([t['depth'][indices[i][1]] for i, t in enumerate(targets)], dim=0)[:, :, 1:, :]
        tgt_joint_depth = _tgt_joint_depth[..., 0:1]
        tgt_joint_depth_exist = _tgt_joint_depth[..., 1:2]

        # root + displacement
        max_depth = targets[0]['max_depth']
        src_kpts2d = outputs["pred_kpts2d"][src_idx[0], src_idx[1]]  # b*m x T x num_joints x 3
        src_joint_vis = src_kpts2d[:, :, 1:, 2:3]
        src_joint = src_kpts2d[:, :, 1:, 0:2] + src_kpts2d[:, :, :1, 0:2]
        src_kpts_depth = outputs["pred_depth"][src_idx[0], src_idx[1]]  # b*m x T x num_joints x 1
        src_joint_depth = src_kpts_depth[:, :, :1, :] + src_kpts_depth[:, :, 1:, :] / max_depth

        error_joint = tgt_joint_vis * F.l1_loss(src_joint, tgt_joint, reduction='none')
        # error_root_joint = vis * F.mse_loss(src_joint, tgt_joint, reduction='none')
        loss_joint = error_joint.sum(dim=(-2, -3)) / (tgt_joint_vis.sum(dim=(-2, -3)) + self.eps)  # b*m x 2

        error_joint_depth = tgt_joint_depth_exist * F.l1_loss(src_joint_depth, tgt_joint_depth, reduction='none')
        loss_joint_depth = error_joint_depth.sum(dim=(-2, -3)) / (tgt_joint_depth_exist.sum(dim=(-2, -3)) + self.eps)  # b*m x 2

        error_joint_vis = F.mse_loss(src_joint_vis, tgt_joint_vis, reduction='none')
        loss_joint_vis = error_joint_vis.mean(dim=(-2, -3))

        losses = {}
        losses['loss_joint'] = loss_joint.sum() / num_traj
        losses['loss_joint_depth'] = loss_joint_depth.sum() / num_traj
        losses['loss_joint_vis'] = loss_joint_vis.sum() / num_traj
        # print(losses['loss_joint'], losses['loss_joint_depth'], losses['loss_joint_vis'])
        return losses

    def loss_joint_displace(self, outputs, targets, indices, num_traj):
        """
        joint displacement loss: l1 displacement, l2 visibility
        """
        src_idx = self._get_src_permutation_idx(indices)

        tgt_kpts = torch.cat([t['kpts2d'][indices[i][1]] for i, t in enumerate(targets)], dim=0)
        tgt_joint_disp = tgt_kpts[:, :, 1:, 0:2] - tgt_kpts[:, :, :1, 0:2]  # b*m x T x num_joints-1 x 2
        tgt_joint_vis = tgt_kpts[:, :, 1:, 2:3]  # b*m x T x num_joints-1 x 1
        tgt_root_vis = tgt_kpts[:, :, :1, 2:3]  # b*m x T x 1 x 1
        joint_vis = tgt_joint_vis * tgt_root_vis  # if root is not visible, do not consider displacement loss

        _tgt_kpts_depth = torch.cat([t['depth'][indices[i][1]] for i, t in enumerate(targets)], dim=0)
        tgt_kpts_depth = _tgt_kpts_depth[..., 0:1]
        tgt_joint_depth_disp = tgt_kpts_depth[:, :, 1:, :] - tgt_kpts_depth[:, :, 0:1, :]
        tgt_joint_depth_exist = _tgt_kpts_depth[:, :, 1:, 1:2] * _tgt_kpts_depth[:, :, :1, 1:2]

        # root + displacement
        src_joint_disp = outputs["pred_kpts2d"][src_idx[0], src_idx[1], :, 1:, 0:2]  # b*m x T x num_joints-1 x 2
        src_joint_depth_disp = outputs["pred_depth"][src_idx[0], src_idx[1], :, 1:, :]  # b*m x T x num_joints-1 x 1

        # average over time and joints
        error_joint_disp = joint_vis * F.l1_loss(src_joint_disp, tgt_joint_disp, reduction='none')
        # error_joint_disp = vis * F.mse_loss(src_joint_disp[..., 0:2], tgt_joint_disp[..., 0:2], reduction='none')
        loss_joint_disp = error_joint_disp.sum(dim=(-2, -3)) / (joint_vis.sum(dim=(-2, -3)) + self.eps)

        error_joint_depth_disp = tgt_joint_depth_exist * F.l1_loss(src_joint_depth_disp, tgt_joint_depth_disp, reduction='none')
        loss_joint_depth_disp = error_joint_depth_disp.sum(dim=(-2, -3)) / (tgt_joint_depth_exist.sum(dim=(-2, -3)) + self.eps)

        # print('joint disp loss', loss_joint_disp.size(), loss_joint_vis.size())

        losses = {}
        losses['loss_joint_disp'] = loss_joint_disp.sum() / num_traj
        losses['loss_joint_depth_disp'] = loss_joint_depth_disp.sum() / num_traj
        # print(losses['loss_joint_disp'], losses['loss_joint_depth_disp'])
        return losses

    def loss_joint_cont(self, outputs, targets, indices, num_traj):
        """
        joint continuity loss: l2
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_joint_vis = torch.cat([t['kpts2d'][indices[i][1]] for i, t in enumerate(targets)], dim=0)[:, :, :, 2:3]

        max_depth = targets[0]['max_depth']
        src_depth = outputs["pred_depth"][src_idx[0], src_idx[1]]  # b*m x T x num_joints x 1
        src_depth[:, :, 1:, :] = src_depth[:, :, :1, :] + src_depth[:, :, 1:, :] / max_depth

        src_joint = outputs["pred_kpts2d"][src_idx[0], src_idx[1]][..., 0:2]  # b*m x T x num_joints x 3

        kepts = torch.cat([src_joint, src_depth], dim=-1)  # b*m x T x num_joints x 3
        kepts[:, :, 1:] = kepts[:, :, 1:] - kepts[:, :, :1].detach()

        cont_vis = tgt_joint_vis[:, 1:] * tgt_joint_vis[:, :-1]
        # b*m x (T-1) x num_joints x 2
        error_cont = self.cont_weights * cont_vis * \
                     F.mse_loss(kepts[:, 1:, :, :], kepts[:, :-1, :, :], reduction='none')
        loss_joint_cont = error_cont.sum(dim=(-2, -3)) / (cont_vis.sum(dim=(-2, -3)) + self.eps)  # b*m x 2

        # print('joint cont loss', loss_joint_cont.size())

        losses = {}
        losses['loss_cont'] = loss_joint_cont.sum() / num_traj
        return losses

    def loss_heatmap(self, outputs, targets, indices, num_traj):
        # [(bs, t, h, w, nhead, num_joints)]
        heatmaps = outputs['heatmaps']
        spatial_value = [item.shape[1:4] for item in heatmaps]
        target_heatmaps = self.generate_heatmap(targets, spatial_value, heatmaps[0].device)

        heatmap_loss = 0
        for i in range(len(heatmaps)):
            heatmap = heatmaps[i]
            nhead = heatmap.shape[4]
            # [bs, t, h, w, num_joints] -> [bs, t, h, w, 1, num_joints]
            target = target_heatmaps[i].unsqueeze(dim=4).repeat(1, 1, 1, 1, nhead, 1)
            error_heatmap = F.mse_loss(target, heatmap, reduction='sum')
            heatmap_loss += error_heatmap / nhead

        losses = {}
        losses['loss_heatmap'] = heatmap_loss
        return losses

    def generate_heatmap(self, targets, spatial_value, device):
        # [(bs, t, h, w, num_joints)]
        multiscale_heatmaps = []
        scales = len(spatial_value)
        bs = len(targets)

        for s in range(scales):
            t, h, w = spatial_value[s]
            kernel_size = max(h // 10 + h // 10 % 2 - 1, w // 10 + w // 10 % 2 - 1)
            bs_heatmaps = []
            for i in range(bs):
                bs_heatmaps_t = []
                kpts2d = torch.clone(targets[i]['kpts2d'])  # [n, t, num_joints, 3]
                kpts2d[..., 0] = kpts2d[..., 0] * w
                kpts2d[..., 1] = kpts2d[..., 1] * h
                for j in range(t):
                    kpts2d_t = kpts2d[:, j]  # [n, num_joints, 3]
                    heatmaps = []
                    for k in range(kpts2d_t.shape[1]):
                        kpt = kpts2d_t[:, k, :]  # [n, 3]
                        kpt = kpt[kpt[:, 2] > 0, 0:2]  # [n, 2]
                        kpt = kpt.long()
                        valid = (kpt[:, 0] >= 0) & (kpt[:, 0] < w) & (kpt[:, 1] >= 0) & (kpt[:, 1] < h)
                        kpt = kpt[valid]
                        heatmap = torch.zeros(h, w, device=device)
                        heatmap[kpt[:, 1], kpt[:, 0]] = 1
                        heatmaps.append(heatmap)
                    heatmaps = torch.stack(heatmaps, dim=0)  # [num_joints, h, w]
                    bs_heatmaps_t.append(heatmaps)
                bs_heatmaps_t = torch.stack(bs_heatmaps_t, dim=1)  # [num_joints, t, h, w]
                bs_heatmaps_t = TF.gaussian_blur(bs_heatmaps_t, kernel_size=[kernel_size, kernel_size])
                bs_heatmaps.append(bs_heatmaps_t)
            bs_heatmaps = torch.stack(bs_heatmaps, dim=0)  # [bs, num_joints, t, h, w]
            multiscale_heatmaps.append(bs_heatmaps.permute(0, 2, 3, 4, 1))  # [bs, t, h, w, num_joints]
        return multiscale_heatmaps

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_traj, **kwargs):
        loss_map = {
            'is_human': self.loss_is_human,
            'root': self.loss_root,
            'joint_disp': self.loss_joint_displace,
            'joint': self.loss_joint,
            'joint_cont': self.loss_joint_cont,
            'heatmap': self.loss_heatmap,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_traj, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices: [(pred_id, target_id)], len(indices) = batchsize
        indices = self.matcher(outputs, targets)

        # Compute the average number of target persons accross all nodes, for normalization purposes
        num_traj = sum([len(t["traj_ids"]) for t in targets])
        num_traj = torch.as_tensor([num_traj], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_traj)
        num_traj = torch.clamp(num_traj / get_world_size(), min=1).item()
        # print(num_traj)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_traj))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                _indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'heatmap':
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, _indices, num_traj, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses, indices


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, targets, indices):
        """
        Convert outputs and targets for evaluation
        """
        eps = 10e-6
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # print(outputs["pred_logits"])
        results = []
        for i in range(bs):
            target_img_size = targets[i]['input_size'].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
            tgt_bbxes = targets[i]['bbxes']  # m x T x 4

            human_prob = outputs["pred_logits"][i].softmax(-1)[..., 1]
            max_depth = targets[i]['max_depth']

            _tgt_kpts2d = torch.clone(targets[i]['kpts2d'])  # m x T x num_joints x 3
            tgt_kpts2d = _tgt_kpts2d[..., 0:2] * target_img_size  # scale to original image size
            tgt_kpts2d_vis = _tgt_kpts2d[..., 2:3]
            tgt_depth = torch.clone(targets[i]['depth'])  # m x T x num_joints x 3
            tgt_depth[..., 0] = max_depth * tgt_depth[..., 0]  # scale to original depth

            _out_kepts_depth = outputs["pred_depth"][i]  # n x T x num_kpts x 1
            # root + displacement
            _out_kepts_depth[:, :, 1:, :] = _out_kepts_depth[:, :, 0:1, :] + _out_kepts_depth[:, :, 1:, :] / max_depth
            out_kepts_depth = max_depth * _out_kepts_depth  # scale to original depth

            out_score = outputs["pred_kpts2d"][i, :, :, :, 2:3]  # n x T x num_kpts x 1
            out_kepts2d = outputs["pred_kpts2d"][i, :, :, :, 0:2]  # n x T x num_kpts x 2
            # root + displacement
            out_kepts2d[:, :, 1:, :] = out_kepts2d[:, :, :1, :] + out_kepts2d[:, :, 1:, :]
            out_kepts2d = out_kepts2d * target_img_size  # scale to original image size

            tgt_track_ids = targets[i]['track_ids']  # [m, T]
            traj_ids = targets[i]['traj_ids']  # [m]
            tgt_bbxes_head = targets[i]['bbxes_head']  # m x T x 4
            inv_trans = targets[i]['inv_trans']
            dataset_name = targets[i]['dataset']
            input_size = targets[i]['input_size']
            results.append(
                {
                    'human_score': human_prob,  # [n]
                    'pred_kpt_scores': out_score,  # [n, T, num_joints, 1]
                    'pred_kpts': out_kepts2d,  # [n, T, num_kpts, 2]
                    'pred_depth': out_kepts_depth,  # [n, T, num_kpts, 1]
                    'gt_kpts': tgt_kpts2d,  # [m, T, num_kpts, 2]
                    'gt_kpts_vis': tgt_kpts2d_vis,  # [m, T, num_kpts, 1]
                    'gt_depth': tgt_depth,  # [m, T, num_kpts, 2]
                    'bbxes': tgt_bbxes,  # [m, T, 4]
                    'gt_bbxes_head': tgt_bbxes_head,  # [m, T, 4]
                    'gt_track_ids': tgt_track_ids,  # [m, T]
                    'gt_traj_ids': traj_ids,
                    'indices': indices[i],  # [src_idx, tgt_idx]
                    'inv_trans': inv_trans,  # [2, 3]
                    'filenames': targets[i]['filenames'],
                    'video_name': targets[i]['video_name'],
                    'frame_indices': targets[i]['frame_indices'],
                    'dataset': dataset_name,
                    # 'heatmaps': [heatmap[i] for heatmap in outputs['heatmaps']],  # [(t, h, w, nhead, num_joints)]
                    'image_id': targets[i]['image_id'],
                    'input_size': input_size,  # (w, h)
                    'cam_intr': targets[i]['cam_intr'],  # [4] for evaluation of jta or mupots
                    'gt_pose3d': targets[i]['kpts3d'],  # [m, T, num_kpts, 3]  for evaluation of jta or mupots
                }
            )
        return results


def build_model(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)

    model = SnipperDeformable(
        backbone,
        transformer,
        args.num_queries,
        args.num_feature_levels,
        args.num_frames,
        args.num_future_frames,
        args.num_kpts,
        args.aux_loss
    )
    model.to(device)

    matcher = build_matcher(args)

    losses = ['is_human', 'root', 'joint', 'joint_disp', 'joint_cont', 'heatmap']
    if args.max_depth == -1:
        args.root_depth_loss_coef = 0
        args.joint_disp_depth_loss_coef = 0
        args.joint_depth_loss_coef = 0

    weight_dict = {
        'loss_is_human': args.is_human_loss_coef,

        'loss_root': args.root_loss_coef,
        'loss_root_vis': args.root_vis_loss_coef,
        'loss_root_depth': args.root_depth_loss_coef,

        'loss_joint_disp': args.joint_disp_loss_coef,
        'loss_joint_depth_disp': args.joint_disp_depth_loss_coef,

        'loss_joint': args.joint_loss_coef,
        'loss_joint_vis': args.joint_vis_loss_coef,
        'loss_joint_depth': args.joint_depth_loss_coef,

        'loss_cont': args.cont_loss_coef,

        'loss_heatmap': args.heatmap_loss_coef
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # [1, 1, num_joints, 1]
    cont_weights = torch.from_numpy(ROOTJOINTCONT).float().unsqueeze(1).unsqueeze(0).unsqueeze(0)
    criterion = SetCriterion(matcher=matcher, losses=losses, eos_coef=args.eos_coef,
                             weight_dict=weight_dict, cont_weights=cont_weights)
    criterion.to(device)
    postprocessors = PostProcess()

    return model, criterion, postprocessors


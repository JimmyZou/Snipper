# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_is_human: float = 1, cost_root: float = 1, cost_root_vis: float = 1,
                 cost_joint: float = 1, cost_joint_vis: float = 1, cost_joint_depth: float = 1,
                 cost_root_depth: float = 1):
        """
        Creates the matcher
        """
        super().__init__()
        self.cost_is_human = cost_is_human
        self.cost_root = cost_root
        self.cost_root_vis = cost_root_vis
        self.cost_joint = cost_joint
        self.cost_joint_vis = cost_joint_vis
        self.cost_joint_depth = cost_joint_depth
        self.cost_root_depth = cost_root_depth
        self.eps = 10e-6

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, 2] with the classification logits
                 "pred_kpts2d": Tensor of dim [batch_size, num_queries, num_kpts, 3] with the predicted 2D pose
                 "pred_depth": Tensor of dim [batch_size, num_queries, num_kpts, 1] with the predicted depth

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "kpts2d": [batch_size] Tensors of dim [num_queries, num_kpts, 3] with the predicted 2D pose
                 "depth": [batch_size] Tensors of dim [num_queries, num_kpts, 1] with the predicted depth

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        # match human trajectory
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for i in range(bs):
            # target 2D pose
            tgt_kpts2d = torch.clone(targets[i]['kpts2d']).unsqueeze(dim=0)  # 1 x m x T x num_kpts x 3
            tgt_kpts2d_root = tgt_kpts2d[:, :, :, :1, :]  # 1 x m x T x 1 x 3
            tgt_kpts2d_joint = tgt_kpts2d[:, :, :, 1:, 0:2]  # 1 x m x T x (num_kpts-1) x 2
            joint_visib = tgt_kpts2d[:, :, :, 1:, 2:3]   # 1 x m x T x (num_kpts-1) x 1

            # target depth
            tgt_kpts_depth = torch.clone(targets[i]['depth']).unsqueeze(dim=0)  # 1 x m x T x num_kpts x 2
            tgt_kpts_root_depth = tgt_kpts_depth[:, :, :, :1, 0:1]  # 1 x m x T x 1 x 1
            tgt_kpts_root_depth_exist = tgt_kpts_depth[:, :, :, :1, 1:2]  # 1 x m x T x 1 x 1
            tgt_kpts_joint_depth = tgt_kpts_depth[:, :, :, 1:, 0:1]  # 1 x m x T x (num_kpts-1) x 1
            tgt_kpts_joint_depth_exist = tgt_kpts_depth[:, :, :, 1:, 1:2]  # 1 x m x T x (num_kpts-1) x 1

            max_depth = targets[i]['max_depth']
            out_kepts_depth = torch.clone(outputs["pred_depth"][i]).unsqueeze(dim=1)  # n x 1 x T x num_kpts x 1
            out_kepts_root_depth = out_kepts_depth[:, :, :, :1, :]  # n x 1 x T x 1 x 1
            out_kepts_joint_depth = out_kepts_depth[:, :, :, 1:, :]  # n x 1 x T x (num_kpts-1) x 1
            # root + displacement
            out_kepts_joint_depth = out_kepts_root_depth + out_kepts_joint_depth / max_depth

            out_prob = outputs["pred_logits"][i].softmax(-1)  # n x T x 2
            out_kepts2d = torch.clone(outputs["pred_kpts2d"][i]).unsqueeze(dim=1)  # n x 1 x T x num_kpts x 3
            out_kepts2d_root = out_kepts2d[:, :, :, :1, :]  # n x 1 x T x 1 x 3
            _out_kepts2d_joint = out_kepts2d[:, :, :, 1:, :]  # n x 1 x T x (num_kpts-1) x 3
            out_kepts2d_joint_vis = _out_kepts2d_joint[..., 2:3]  # n x 1 x T x (num_kpts-1) x 1
            # root + displacement
            out_kepts2d_joint = _out_kepts2d_joint[..., 0:2] + out_kepts2d_root[..., 0:2]

            # cost of human probability
            prob = out_prob[:, :, 1].unsqueeze(dim=1)  # n x 1 x T
            vis = (joint_visib.sum((-2, -1)) > 0).float()  # 1 x m x T
            class_cost = -1 * (prob * vis).sum(-1) / (vis.sum(-1) + self.eps)  # [n, m]

            # cost of joint  [n, m]
            # n x m x T x (num_kpts-1) x 2
            joint_cost = joint_visib * (out_kepts2d_joint - tgt_kpts2d_joint)
            # joint_cost = (joint_cost / (tgt_scale + self.eps)).abs()
            # joint_cost = joint_cost.pow(2)
            joint_cost = joint_cost.abs()
            joint_cost = joint_cost.sum((-1, -2, -3)) / (joint_visib.sum((-1, -2, -3)) + self.eps)
            joint_visib_cost = (out_kepts2d_joint_vis - joint_visib).pow(2).mean((-1, -2, -3))

            # n x m x T x (num_kpts-1) x 1
            joint_depth_cost = tgt_kpts_joint_depth_exist * (out_kepts_joint_depth - tgt_kpts_joint_depth)
            joint_depth_cost = joint_depth_cost.abs()
            joint_depth_cost = joint_depth_cost.sum((-1, -2, -3)) / (tgt_kpts_joint_depth_exist.sum((-1, -2, -3)) + self.eps)

            # cost of root [n, m]
            root_visib = tgt_kpts2d_root[..., 2:3]
            # n x m x T x 1 x 2
            root_cost = root_visib * (out_kepts2d_root[..., 0:2] - tgt_kpts2d_root[..., 0:2])
            # root_cost = root_cost.pow(2)
            root_cost = root_cost.abs()
            root_cost = root_cost.sum((-1, -2, -3)) / (root_visib.sum((-1, -2, -3)) + self.eps)
            root_visib_cost = (out_kepts2d_root[..., 2:3] - root_visib).pow(2).mean((-1, -2, -3))

            # n x m x T x 1 x 1
            root_depth_cost = tgt_kpts_root_depth_exist * (out_kepts_root_depth - tgt_kpts_root_depth)
            root_depth_cost = root_depth_cost.abs()
            root_depth_cost = root_depth_cost.sum((-1, -2, -3)) / (tgt_kpts_root_depth_exist.sum((-1, -2, -3)) + self.eps)

            # print(class_cost.size(), joint_cost.size(), root_cost.size(),
            #       root_depth_cost.size(), joint_depth_cost.size())

            cost = self.cost_is_human * class_cost + \
                   self.cost_root * root_cost + \
                   self.cost_root_vis * root_visib_cost + \
                   self.cost_root_depth * root_depth_cost + \
                   self.cost_joint * joint_cost + \
                   self.cost_joint_vis * joint_visib_cost + \
                   self.cost_joint_depth * joint_depth_cost

            out_i, tgt_i = linear_sum_assignment(cost.cpu())

            if tgt_i == [] or out_i == []:
                indices.append((torch.tensor([], dtype=torch.long, device=out_prob.device),
                                torch.tensor([], dtype=torch.long, device=out_prob.device)))
            else:
                index_i = torch.from_numpy(out_i).to(dtype=torch.long, device=out_prob.device)
                index_j = torch.from_numpy(tgt_i).to(dtype=torch.long, device=out_prob.device)
                indices.append((index_i, index_j))
        return indices


def build_matcher(args):
    if args.max_depth == -1:
        args.set_cost_root_depth = 0
        args.set_cost_joint_depth = 0
    return HungarianMatcher(
        cost_is_human=args.set_cost_is_human,

        cost_root=args.set_cost_root,
        cost_root_vis=args.set_cost_root_vis,
        cost_root_depth=args.set_cost_root_depth,

        cost_joint=args.set_cost_joint,
        cost_joint_vis=args.set_cost_joint_vis,
        cost_joint_depth=args.set_cost_joint_depth,
    )


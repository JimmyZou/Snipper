import numpy as np
import torch


def eval_pose3d(key, results, start_t, end_t):
    """Evaluate pose 3d."""
    # 'human_score': human_prob,  # [n]
    # 'pred_kpt_scores': out_score,  # [n, T, num_joints, 1]
    # 'pred_kpts': out_kepts2d,  # [n, T, num_kpts, 2]
    # 'pred_depth': out_kepts_depth,  # [n, T, num_kpts, 1]
    # 'gt_kpts': tgt_kpts2d,  # [m, T, num_kpts, 2]
    # 'gt_kpts_vis': tgt_kpts2d_vis,  # [m, T, num_kpts, 1]
    # 'gt_pose3d': tgt_kpts3d,  # [m, T, num_kpts, 3]
    # 'bbxes': tgt_bbxes,  # [m, T, 4]
    # 'gt_track_ids': tgt_track_ids,  # [m, T]
    # 'max_depth': max_depth,  # [] scalar
    # 'cam_intr': cam_intr,  # [4]
    # 'indices': indices[i],  # [src_idx, tgt_idx]

    eval_results = []
    bs = len(results)
    for i in range(bs):
        dataset_name = results[i]['dataset']
        print(dataset_name)
        if dataset_name == 'mupots' or dataset_name == 'jta' or dataset_name == 'panoptic':
            pass
        else:
            continue

        pred_track_ids = results[i]['human_score'] > 0.5  # [n, T]
        # traj_ids = np.where(np.sum(track_ids, axis=-1) > 0)[0]

        for t in range(start_t, end_t):
            exist_gts = (results[i]['gt_track_ids'][:, t] > 0) & \
                        (results[i]['gt_kpts_vis'][:, t].sum(dim=(-1, -2)) > 0)
            if exist_gts.sum() == 0:
                eval_results.append(torch.zeros(0).float().cpu())  # no annotated persons, append empty tensor
                continue
            gt_pose3d = results[i]['gt_pose3d'][exist_gts, t]  # xyz [m, num_kpts, 3]
            gt_vis = results[i]['gt_kpts_vis'][exist_gts, t]  # [m, num_kpts, 1]
            # gt_kpts = results[i]['gt_kpts'][exist_gts, t]  # [m, num_kpts, 2]
            # bboxes = results[i]['bbxes'][exist_gts, t]  # [m, 4]

            exist_preds = pred_track_ids[:, t]
            if exist_preds.sum() == 0:
                eval_results.append(torch.zeros(0).float().cpu())  # no predicted persons, append empty tensor
                continue
            _pred_kpts = results[i]['pred_kpts'][exist_preds, t]  # [n, num_kpts, 2]
            cam_intr = results[i]['cam_intr']  # [4]
            inv_trans = results[i]['inv_trans']
            pred_kpts = transform_pts_torch(_pred_kpts, inv_trans)

            pred_depth = results[i]['pred_depth'][exist_preds, t]  # [n, num_kpts, 1]
            pred_pose3d = unprojection_torch(pred_kpts, pred_depth, cam_intr)  # [n, num_kpts, 3]

            src_idx, tgt_idx = matcher_pose3d(gt_pose3d.cpu(), gt_vis.cpu(), pred_pose3d.cpu())
            if tgt_idx.shape[0] == 0:
                eval_results.append(torch.zeros(0).float().cpu())  # no predicted persons, append empty tensor
                continue

            mpjpe = compute_mpjpe(gt_pose3d[tgt_idx].cpu(), gt_vis[tgt_idx].cpu(), pred_pose3d[src_idx].cpu(), key)
            eval_results.append(mpjpe.cpu())

    if len(eval_results) == 0:
        eval_results.append(torch.zeros(0).float().cpu())  # no annotated persons, append empty tensor
    eval_results = torch.cat(eval_results, dim=0)
    return eval_results


def matcher_pose3d(gt_pose3d, gt_vis, pred_pose3d, cost_joint=1, cost_root=5):
    from scipy.optimize import linear_sum_assignment
    gt_pose3d = gt_pose3d.unsqueeze(dim=0)  # gt_pose3d [1, m, num_joints, 3]
    gt_vis = gt_vis.unsqueeze(dim=0)  # gt_vis [1, m, num_joints, 1]
    pred_pose3d = pred_pose3d.unsqueeze(dim=1)  # pred_pose3d [n, 1, num_joints, 3]
    eps = 10-6

    # cost of human kepts root [n, m]
    pose_cost = gt_vis * (gt_pose3d - pred_pose3d)  # n x m x num_joints x 3
    pose_cost = pose_cost.pow(2).sum(-1).sqrt()  # n x m x num_joints
    pose_cost[:, :, :1] = pose_cost[:, :, :1] * cost_root
    pose_cost[:, :, 1:] = pose_cost[:, :, 1:] * cost_joint

    cost = pose_cost.sum(-1) / (gt_vis.sum((-1, -2)) + eps)
    # print(cost)
    out_i, tgt_i = linear_sum_assignment(cost.cpu())

    if tgt_i == [] or out_i == []:
        out_i = torch.tensor([], dtype=torch.long, device=gt_pose3d.device)
        tgt_i = torch.tensor([], dtype=torch.long, device=gt_pose3d.device)
    else:
        out_i = torch.from_numpy(out_i).to(dtype=torch.long, device=gt_pose3d.device)
        tgt_i = torch.from_numpy(tgt_i).to(dtype=torch.long, device=gt_pose3d.device)
    return out_i, tgt_i


def eval_kpts2d_pckh(key, results, start_t, end_t):
    # 'human_score': human_prob,  # [n]
    # 'pred_kpt_scores': out_score,  # [n, T, num_joints, 1]
    # 'pred_kpts': out_kepts2d,  # [n, T, num_kpts, 2]
    # 'gt_kpts': tgt_kpts2d,  # [m, T, num_kpts, 2]
    # 'gt_kpts_vis': tgt_kpts2d_vis,  # [m, T, num_kpts, 1]
    # 'bbxes': tgt_bbxes,  # [m, T, 4]
    # 'gt_bbxes_head': tgt_bbxes_head,  # [m, T, 4]
    # 'gt_track_ids': tgt_track_ids,  # [m, T]
    # 'gt_traj_ids': traj_ids,
    # 'indices': indices[i],  # [src_idx, tgt_idx]
    # 'inv_trans': inv_trans,  # [2, 3]
    # 'filenames': targets[i]['filenames'],
    # 'video_name': targets[i]['video_name'],
    # 'frame_indices': targets[i]['frame_indices']

    # print(results[0]['filenames'][0])
    # print(results[0]['gt_kpts_vis'].sum((-1, -2)))
    pckh = []
    bs = len(results)
    for i in range(bs):
        if results[i]['dataset'] != 'posetrack':
            continue
        # print(results[i]['filenames'][0])
        gt_track_ids = results[i]['gt_track_ids']
        gt_traj_ids = results[i]['gt_traj_ids']

        if gt_traj_ids.shape[0] == 0:
            # no annotations
            continue

        # src_idx, tgt_idx = match_pose2d(results[i]['gt_kpts'], results[i]['gt_kpts_vis'],
        #                                 results[i]['gt_bbxes_head'], results[i]['pred_kpts'],
        #                                 results[i]['pred_kpt_scores'])
        src_idx, tgt_idx = results[i]['indices']
        # print(src_idx, tgt_idx)
        inv_trans = results[i]['inv_trans']
        for t in range(start_t, end_t):
            exist_gt_person = (gt_track_ids[:, t] > 0) & \
                              (results[i]['gt_kpts_vis'][:, t].sum(dim=(-1, -2)) > 0)
            # print(exist_gt_person)
            if exist_gt_person.sum() == 0:
                # print(t, 'skip')
                continue

            # print(t, 'compute')
            _gt_kpts = results[i]['gt_kpts'][tgt_idx[exist_gt_person], t]  # [m, num_kpts, 2]
            _gt_kpts_vis = results[i]['gt_kpts_vis'][tgt_idx[exist_gt_person], t]  # [m, num_kpts, 1]
            _gt_bbxes_head = results[i]['gt_bbxes_head'][tgt_idx[exist_gt_person], t]  # [m, 4]
            _pred_kpts = results[i]['pred_kpts'][src_idx[exist_gt_person], t]  # [m, num_kpts, 2]

            # src_idx, tgt_idx = match_pckh(results[i]['gt_kpts'][exist_gt_person, t:t + 1],
            #                               results[i]['gt_kpts_vis'][exist_gt_person, t:t + 1],
            #                               results[i]['gt_bbxes_head'][exist_gt_person, t:t + 1],
            #                               results[i]['pred_kpts'])
            # _gt_kpts = results[i]['gt_kpts'][exist_gt_person, t]  # [m, num_kpts, 2]
            # _gt_kpts_vis = results[i]['gt_kpts_vis'][exist_gt_person, t]  # [m, num_kpts, 1]
            # _gt_bbxes_head = results[i]['gt_bbxes_head'][exist_gt_person, t]  # [m, 4]
            # _pred_kpts = results[i]['pred_kpts'][src_idx, t]  # [m, num_kpts, 2]

            # transform
            _gt_kpts = transform_pts_torch(_gt_kpts, inv_trans)
            _pred_kpts = transform_pts_torch(_pred_kpts, inv_trans)
            _head_size = 0.6 * torch.sqrt(_gt_bbxes_head[:, 2] ** 2 + _gt_bbxes_head[:, 3] ** 2)  # [m]

            for p in range(_gt_kpts.shape[0]):
                vis = _gt_kpts_vis[p, :, 0]  # [num_joints]
                error = torch.norm(_gt_kpts[p] - _pred_kpts[p], dim=-1)

                if key == 'pckh_root':
                    pck = (error[:1][vis[:1] > 0]) < (0.5 * _head_size[p])
                elif key == 'pckh_joint':
                    pck = (error[1:][vis[1:] > 0]) < (0.5 * _head_size[p])
                else:
                    raise ValueError('key: {} errors.'.format(key))
            pckh.append(pck.flatten().float().cpu())
    if len(pckh) == 0:
        return None
    pckh = torch.cat(pckh, dim=0)
    return pckh


def transform_pts_torch(pts, trans):
    # pts: [n, num_joints, 2]
    # trans: [2, 3]
    pts = torch.cat([pts, torch.ones_like(pts)[..., 0:1]], dim=-1)
    trans_pts = torch.matmul(pts, trans.T)
    return trans_pts


def compute_mpjpe(gt_pose3d, gt_kpts_vis, pred_pose3d, key):
    # predict_count = pred_pose3d.shape[0]

    if key == 'mpjpe_joint':
        _pred_pose3d = pred_pose3d[:, :, :]  # [m, num_kpts, 3]
        _gt_pose3d = gt_pose3d[:, :, :]  # [m, num_kpts, 3]
        _kpts_vis = gt_kpts_vis[:, :, 0]  # [m, num_kpts]

        dis = (_pred_pose3d - _gt_pose3d).pow(2).sum(dim=-1).sqrt()  # [m, num_kpts]
        mpjpe = dis[_kpts_vis > 0]
        return mpjpe
    elif key == 'mpjpe_root':
        valid = gt_kpts_vis[:, 0, 0] > 0
        _pred_pose3d = pred_pose3d[valid, :1, :]  # [m, 1, 3], root
        _gt_pose3d = gt_pose3d[valid, :1, :]  # [m, 1, 3], root
        _kpts_vis = gt_kpts_vis[valid, :1, 0]  # [m, 1]

        dis = (_pred_pose3d - _gt_pose3d).pow(2).sum(dim=-1).sqrt()  # [m, 1]
        mpjpe = dis[_kpts_vis > 0]
        return mpjpe
    elif key == 'pel_mpjpe_joint':
        _pred_pose3d = pred_pose3d  # [m, num_kpts, 3], exclude root joint
        _gt_pose3d = gt_pose3d  # [m, num_kpts, 3], exclude root joint

        _kpts_vis = gt_kpts_vis[:, 1:, 0]  # [m, num_kpts-1]
        joint_pred_pose3d = _pred_pose3d[:, 1:, :] - _pred_pose3d[:, :1, :]
        joint_gt_pose3d = _gt_pose3d[:, 1:, :] - _gt_pose3d[:, :1, :]

        dis = (joint_pred_pose3d - joint_gt_pose3d).pow(2).sum(dim=-1).sqrt()  # [m, num_kpts-1]
        pel_mpjpe = dis[_kpts_vis > 0]
        return pel_mpjpe
    else:
        assert ValueError('key: {} errors.'.format(key))


def unprojection_torch(pred_kpts, pred_depth, cam_intr):
    fx, fy, cx, cy = cam_intr
    z = pred_depth[..., 0]
    x = (pred_kpts[..., 0] - cx) / fx * z
    y = (pred_kpts[..., 1] - cy) / fy * z
    pose3d = torch.stack([x, y, z], dim=-1)
    return pose3d


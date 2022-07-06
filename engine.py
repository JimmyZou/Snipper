"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""

import math
import os
import sys
from typing import Iterable
import pickle
import collections

import torch

import util.misc as utils
from eval_utils import eval_pose3d, transform_pts_torch


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    count = 0
    print_freq = 10
    for samples, _targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device=device)
        # targets = [{k: v.to(device=device) for k, v in t.items() if not isinstance(v, list)} for t in targets]
        targets = []
        for t in _targets:
            tmp = {}
            for k, v in t.items():
                if k in ['kpts2d', 'depth', 'bbxes', 'track_ids', 'traj_ids', 'input_size', 'inv_trans']:
                    tmp[k] = v.to(device=device)
                else:
                    tmp[k] = v
            targets.append(tmp)

        optimizer.zero_grad()

        # float32 backward
        outputs, _ = model(samples)
        loss_dict, _ = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        # print(loss_dict_reduced_unscaled)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        # print(loss_dict_reduced_scaled)
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # print(losses_reduced_scaled)
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, save_vis, epoch,
             seq_l, future_seq_l, final_evaluation=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    save_data = collections.defaultdict(list)
    save_data_coco = collections.defaultdict(list)

    pose3d = {'mpjpe_root': list(), 'mpjpe_joint': list(), 'pel_mpjpe_joint': list(), '3dpck': list()}
    pose3d_future = {'mpjpe_root': list(), 'mpjpe_joint': list(), 'pel_mpjpe_joint': list(), '3dpck': list()}

    # test
    count = 0
    for samples, _targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, list)} for t in targets]
        targets = []
        for t in _targets:
            tmp = {}
            for k, v in t.items():
                if k in ['kpts2d', 'depth', 'bbxes', 'track_ids', 'traj_ids', 'input_size', 'inv_trans']:
                    tmp[k] = v.to(device=device)
                else:
                    tmp[k] = v
            targets.append(tmp)

        outputs, att_vis_data = model(samples)
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        results = postprocessors(outputs, targets, indices)

        count += 1
        if save_vis and count < 10 and output_dir:
            visualize_eval_kepts_pred(samples, targets, results,
                                      seq_l, 0, output_dir='{}/vis_results_{:03d}'.format(output_dir, epoch),
                                      epoch=epoch)
            # save_decoder_att_data(samples, targets, results, att_vis_data, output_dir, epoch)

        if save_vis and output_dir:
            save_results_for_evaluation(save_data, results, targets, 0, seq_l)
            save_results_for_evaluation_coco(save_data_coco, results, targets, 0, seq_l)

        if output_dir and final_evaluation:
            for i in range(len(results)):
                dataset_name = results[i]['dataset']
                if dataset_name == 'posetrack':
                    tmp = targets[i]['filenames'][0].split('/')
                    filename, frame_idx = tmp[-2], tmp[-1].split('.')[0]
                elif dataset_name == 'coco':
                    filename, frame_idx = '{:06d}'.format(targets[i]['image_id']), 0
                elif dataset_name == 'mupots':
                    tmp = targets[i]['filenames'][0].split('/')
                    filename = tmp[0]
                    frame_idx = tmp[1].split('_')[-1].split('.')[0]
                elif dataset_name == 'jta':
                    tmp = targets[i]['filenames'][0].split('/')
                    filename = tmp[0]
                    frame_idx = tmp[1].split('.')[0]
                elif dataset_name == 'panoptic':
                    filename = targets[i]['filenames'][0]
                    frame_idx = '{:08d}'.format(targets[i]['frame_indices'][0])
                else:
                    print('cannot find {}-{}'.format(dataset_name, targets[i]['filenames'][0]))
                    filename, frame_idx = 'missing', '0000'

                results_np = {}
                for k, v in results[i].items():
                    if k == 'indices':
                        results_np[k] = [v[0].cpu().numpy(), v[1].cpu().numpy()]
                    elif k == 'filenames':
                        results_np[k] = v
                    elif k in ['heatmaps', 'video_name', 'frame_indices', 'dataset', 'image_id']:
                        continue
                    else:
                        results_np[k] = v.cpu().numpy()
                # '{}/eval_results_{:03d}'.format(output_dir, epoch)
                save_dir = "{}/eval_results_{:03d}/{}_{}.pkl".format(output_dir, epoch, filename, frame_idx)
                with open(save_dir, 'wb') as f:
                    pickle.dump(results_np, f)

        for key in pose3d.keys():
            if 'mpjpe' in key:
                mpjpe = eval_pose3d(key, results, 0, seq_l)
                pose3d[key].append(mpjpe)

                if future_seq_l > 0:
                    mpjpe = eval_pose3d(key, results, seq_l, seq_l + future_seq_l)
                    pose3d_future[key].append(mpjpe)
            if '3dpck' in key:
                pel_mpjpe = eval_pose3d('pel_mpjpe_joint', results, 0, seq_l)
                pose3d[key].append((pel_mpjpe < 0.15).float())

                if future_seq_l > 0:
                    pel_mpjpe = eval_pose3d('pel_mpjpe_joint', results, seq_l, seq_l + future_seq_l)
                    pose3d_future[key].append((pel_mpjpe < 0.15).float())

    stat = {}
    for k in pose3d.keys():
        if 'mpjpe' in k:
            stat[k + '_current'] = torch.mean(1000 * torch.cat(pose3d[k], dim=0)).item()
        elif 'pck' in k:
            stat[k + '_current'] = torch.mean(torch.cat(pose3d[k], dim=0)).item()
        else:
            pass
    for k in pose3d_future.keys():
        if len(pose3d_future[k]) == 0:
            continue
        if 'mpjpe' in k:
            stat[k + '_future'] = torch.mean(1000 * torch.cat(pose3d_future[k], dim=0)).item()
        elif 'pck' in k:
            stat[k + '_future'] = torch.mean(torch.cat(pose3d_future[k], dim=0)).item()
        else:
            pass
    return stat, save_data, save_data_coco


def visualize_eval_kepts_pred(samples, targets, results, seq_l, future_seq_l, output_dir, epoch):
    # 'human_score': human_prob,  # [n]
    # 'pred_kpt_scores': out_score,  # [n, T, num_joints, 1]
    # 'pred_kpts': out_kepts2d,  # [n, T, num_kpts, 2]
    # 'gt_kpts': tgt_kpts2d,  # [m, T, num_kpts, 2]
    # 'gt_kpts_vis': tgt_kpts2d_vis,  # [m, T, num_kpts, 1]
    # 'bbxes': tgt_bbxes,  # [m, T, 4]
    # 'gt_bbxes_head': tgt_bbxes_head,  # [m, T, 4]
    # 'gt_track_ids': tgt_track_ids,  # [m, T]
    # 'indices': indices[i],  # [src_idx, tgt_idx]
    # 'inv_trans': targets[i]['inv_tran'],

    import numpy as np
    from datasets.data_preprocess.dataset_util import posetrack_visualization
    from datasets.hybrid_dataloader import SKELETONS

    _, num_joints = results[0]['pred_kpts'].shape[1:3]

    imgs, _ = samples.decompose()
    b, c, h, w = imgs.shape
    imgs = imgs.reshape(b // seq_l, seq_l, c, h, w)
    imgs = imgs.cpu().numpy()
    imgs = (imgs * 255).astype(np.uint8)

    bs = len(results)
    for i in range(bs):
        if i > 0:
            break

        # print(targets[i]['filenames'][0])
        gt_track_ids = results[i]['gt_track_ids']
        gt_traj_ids = results[i]['gt_traj_ids']
        dataset_name = results[i]['dataset']

        if gt_traj_ids.shape[0] == 0:
            # no annotations
            continue

        # src_idx, tgt_idx = match_pose2d(results[i]['gt_kpts'], results[i]['gt_kpts_vis'],
        #                                 results[i]['gt_bbxes_head'], results[i]['pred_kpts'],
        #                                 results[i]['pred_kpt_scores'])
        src_idx, tgt_idx = results[i]['indices']
        # exist_person = (results[i]['human_score'] > 0.5).cpu().numpy()
        # n = np.sum(exist_person).astype(np.int32)
        # exist_pid = np.where(exist_person == 1)[0]
        # print('epoch {:3d}: {}_{}.jpg - number of persons {}'.format(epoch, filename, frame_idx, n))

        for j in range(seq_l):
            exist_gt_person = (gt_track_ids[:, j] > 0) & \
                              (results[i]['gt_kpts_vis'][:, j].sum(dim=(-1, -2)) > 0)
            # print(exist_gt_person)
            if exist_gt_person.sum() == 0:
                # print(t, 'skip')
                continue

            exist_person = src_idx[exist_gt_person].cpu().numpy()
            exist_pid = gt_traj_ids[exist_gt_person].cpu().numpy()

            # exist_person = (results[i]['human_score'][:, j]).cpu().numpy() > 0.5
            # exist_pid = np.arange(exist_person.shape[0])[exist_person]

            if dataset_name == 'posetrack':
                tmp = targets[i]['filenames'][j].split('/')
                filename, frame_idx = tmp[-2], tmp[-1].split('.')[0]
            elif dataset_name == 'coco':
                filename, frame_idx = '{:06d}'.format(targets[i]['image_id']), 0
            elif dataset_name == 'mupots':
                tmp = targets[i]['filenames'][j].split('/')
                filename = tmp[0]
                frame_idx = tmp[1].split('_')[-1].split('.')[0]
            elif dataset_name == 'jta':
                tmp = targets[i]['filenames'][j].split('/')
                filename = tmp[0]
                frame_idx = tmp[1].split('.')[0]
            elif dataset_name == 'panoptic':
                filename = targets[i]['filenames'][j]
                frame_idx = '{:08d}'.format(targets[i]['frame_indices'][j])
            else:
                print('cannot find {}-{}'.format(dataset_name, targets[i]['filenames'][j]))
                filename, frame_idx = 'missing', '0000'

            kpts2d = results[i]['pred_kpts'][exist_person, j].cpu().numpy()
            kpt_scores = results[i]['pred_kpt_scores'][exist_person, j].cpu().numpy()
            vis = kpt_scores > 0.1
            kpts = np.concatenate([kpts2d, vis], axis=-1)  # [m, num_joints, 3]

            # save_dir = "{}/{}_{}_epoch{:03d}_kpts.npy".format(output_dir, filename, frame_idx, epoch)
            # np.save(save_dir, np.concatenate([kpts, kpt_scores], axis=-1))
            # if j == 0:
            #     with open('{}/{}_{}_epoch{:03d}.pkl'.format(output_dir, filename, frame_idx, epoch), 'wb') as f:
            #         pickle.dump(results[i], f)

            img = np.transpose(imgs[i, j], [1, 2, 0])  # [h, w, c]
            save_dir = "{}/{}_{}_epoch{:03d}.jpg".format(output_dir, filename, frame_idx, epoch)
            print(save_dir)
            posetrack_visualization(img[np.newaxis], kpts[np.newaxis], [exist_pid], 'eval', SKELETONS, save_dir)

        # future frame data
        for j in range(seq_l, seq_l + future_seq_l):
            exist_gt_person = (gt_track_ids[:, j] > 0) & \
                              (results[i]['gt_kpts_vis'][:, j].sum(dim=(-1, -2)) > 0)
            # print(exist_gt_person)
            if exist_gt_person.sum() == 0:
                # print(t, 'skip')
                continue

            exist_person = src_idx[exist_gt_person].cpu().numpy()
            # exist_pid = gt_traj_ids[exist_gt_person].cpu().numpy()
            if dataset_name == 'posetrack':
                tmp = targets[i]['filenames'][j].split('/')
                filename, frame_idx = tmp[-2], tmp[-1].split('.')[0]
            elif dataset_name == 'coco':
                filename, frame_idx = '{:06d}'.format(targets[i]['image_id']), 0
            elif dataset_name == 'mupots':
                tmp = targets[i]['filenames'][j].split('/')
                filename = tmp[0]
                frame_idx = tmp[1].split('_')[-1].split('.')[0]
            elif dataset_name == 'jta':
                tmp = targets[i]['filenames'][j].split('/')
                filename = tmp[0]
                frame_idx = tmp[1].split('.')[0]
            elif dataset_name == 'panoptic':
                filename = targets[i]['filenames'][j]
                frame_idx = '{:08d}'.format(targets[i]['frame_indices'][j])
            else:
                print('cannot find {}-{}'.format(dataset_name, targets[i]['filenames'][j]))
                filename, frame_idx = 'missing', '0000'

            kpts2d = results[i]['pred_kpts'][exist_person, j].cpu().numpy()
            kpt_scores = results[i]['pred_kpt_scores'][exist_person, j].cpu().numpy()
            vis = kpt_scores > 0.1
            kpts = np.concatenate([kpts2d, vis], axis=-1)  # [m, num_joints, 3]

            save_dir = "{}/{}_{}_epoch{:03d}_kpts_future.npy".format(output_dir, filename, frame_idx, epoch)
            np.save(save_dir, np.concatenate([kpts, kpt_scores], axis=-1))
            print(save_dir)


def save_results_for_evaluation(save_data, results, targets, start_t, end_t, post_processing=False):
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

    bs = len(results)
    for i in range(bs):
        if results[i]['dataset'] != 'posetrack':
            continue
        # print(results[i]['filenames'][0])
        gt_track_ids = results[i]['gt_track_ids']  # [n, T]
        gt_traj_ids = results[i]['gt_traj_ids']  # [n]

        if gt_traj_ids.shape[0] == 0:
            # no annotations
            continue

        # src_idx, tgt_idx = match_pose2d(results[i]['gt_kpts'], results[i]['gt_kpts_vis'],
        #                                 results[i]['gt_bbxes_head'], results[i]['pred_kpts'],
        #                                 results[i]['pred_kpt_scores'])
        src_idx, tgt_idx = results[i]['indices']
        # print(src_idx, tgt_idx)
        # print(results[i]['indices'])
        inv_trans = results[i]['inv_trans']

        for t in range(start_t, end_t):
            exist_gt_person = (gt_track_ids[:, t] > 0) & (results[i]['gt_kpts_vis'][:, t].sum(dim=(-1, -2)) > 0)
            # print(exist_gt_person)
            if exist_gt_person.sum() == 0:
                # print(t, 'skip')
                continue

            # print(t, 'compute')
            _gt_kpts = results[i]['gt_kpts'][tgt_idx[exist_gt_person], t]  # [m, num_kpts, 2]
            _gt_kpts_vis = results[i]['gt_kpts_vis'][tgt_idx[exist_gt_person], t]  # [m, num_kpts, 1]
            _gt_bbxes_head = results[i]['gt_bbxes_head'][tgt_idx[exist_gt_person], t]  # [m, 4]
            _pred_kpts = results[i]['pred_kpts'][src_idx[exist_gt_person], t]  # [m, num_kpts, 2]
            _pred_kpt_scores = results[i]['pred_kpt_scores'][src_idx[exist_gt_person], t]  # [m, num_kpts, 1]

            # src_idx, tgt_idx = match_pckh(results[i]['gt_kpts'][exist_gt_person, t:t+1],
            #                               results[i]['gt_kpts_vis'][exist_gt_person, t:t+1],
            #                               results[i]['gt_bbxes_head'][exist_gt_person, t:t+1],
            #                               results[i]['pred_kpts'])
            # # print(results[i]['indices'])
            # # print(src_idx, tgt_idx)
            # _gt_kpts = results[i]['gt_kpts'][exist_gt_person, t]  # [m, num_kpts, 2]
            # _gt_kpts_vis = results[i]['gt_kpts_vis'][exist_gt_person, t]  # [m, num_kpts, 1]
            # _gt_bbxes_head = results[i]['gt_bbxes_head'][exist_gt_person, t]  # [m, 4]
            # _pred_kpts = results[i]['pred_kpts'][src_idx, t]  # [m, num_kpts, 2]
            # _pred_kpt_scores = results[i]['pred_kpt_scores'][src_idx, t]  # [m, num_kpts, 1]

            # transform
            _gt_kpts = transform_pts_torch(_gt_kpts, inv_trans)
            _pred_kpts = transform_pts_torch(_pred_kpts, inv_trans)

            # error = torch.mean((_gt_kpts - _pred_kpts) ** 2)
            # print(error)
            # if error > 0.1:
            #     # fn, filename, frame_idx, indice, max_valid_gap
            #     print('error', error)
            #     print(_gt_kpts, _pred_kpts)
            #     print('\n')

            sample_result = {}
            sample_result['video_name'] = results[i]['video_name']
            sample_result['filename'] = results[i]['filenames'][t]
            sample_result['index'] = results[i]['frame_indices'][t]

            sample_result['pred_kpts'] = _pred_kpts.cpu().numpy()  # [n, num_joints, 2]
            sample_result['pred_kpt_scores'] = _pred_kpt_scores.cpu().numpy()  # [n, num_joints, 1]
            sample_result['traj_ids'] = gt_traj_ids[tgt_idx[exist_gt_person]].cpu().numpy()

            sample_result['gt_kpts'] = _gt_kpts.cpu().numpy()
            sample_result['gt_kpt_scores'] = _gt_kpts_vis.cpu().numpy()
            sample_result['gt_bbxes_head'] = _gt_bbxes_head.cpu().numpy()

            save_data[sample_result['video_name']].append(sample_result)


def save_results_for_evaluation_coco(save_data, results, targets, start_t, end_t, post_processing=False):
    # 'human_score': human_prob,  # [n, T]
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

    bs = len(results)
    for i in range(bs):
        if results[i]['dataset'] != 'coco':
            continue

        src_idx, tgt_idx = results[i]['indices']

        image_id = results[i]['image_id']
        inv_trans = results[i]['inv_trans']
        human_score = results[i]['human_score'][:, 0]
        _pred_kpts = results[i]['pred_kpts'][:, 0]  # [m, num_kpts, 2]
        _pred_kpt_scores = results[i]['pred_kpt_scores'][:, 0]  # [m, num_kpts, 1]

        _gt_kpts = results[i]['gt_kpts'][:, 0]  # [m, num_kpts, 2]
        _gt_kpts_vis = results[i]['gt_kpts_vis'][:, 0]  # [m, num_kpts, 1]

        exist_person = human_score > 0.5
        _pred_kpts = _pred_kpts[exist_person]
        _pred_kpt_scores = _pred_kpt_scores[exist_person]
        human_score = human_score[exist_person]

        # src_idx, tgt_idx = results[i]['indices']
        # _pred_kpts = _pred_kpts[src_idx]
        # _pred_kpt_scores = _pred_kpt_scores[src_idx]
        # human_score = human_score[src_idx]
        # _gt_kpts = _gt_kpts[tgt_idx]
        # _gt_kpts_vis = _gt_kpts_vis[tgt_idx]

        _pred_kpts = transform_pts_torch(_pred_kpts, inv_trans)
        _gt_kpts = transform_pts_torch(_gt_kpts, inv_trans)

        pred_kpts2d = torch.cat([_pred_kpts, _pred_kpt_scores], dim=-1)
        gt_kpts2d = torch.cat([_gt_kpts, _gt_kpts_vis], dim=-1)

        # human_score = human_score[src_idx]
        # pred_kpts2d = pred_kpts2d[src_idx]
        # gt_kpts2d = gt_kpts2d[tgt_idx]

        # idx = torch.where(human_score > 0.5)[0]
        # # print('idx', idx)
        # error = torch.mean((gt_kpts2d - pred_kpts2d[idx]) ** 2)
        # print(error)
        # if error > 0.1:
        #     # fn, filename, frame_idx, indice, max_valid_gap
        #     print('error', error)
        #     print(gt_kpts2d, pred_kpts2d)
        #     print('\n')

        save_data[image_id].append([human_score.cpu().numpy(),
                                    pred_kpts2d.cpu().numpy(),
                                    gt_kpts2d.cpu().numpy(),
                                    (src_idx.cpu().numpy(), tgt_idx.cpu().numpy())])


import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import random
import imageio
from tqdm import tqdm


# helper functions
def get_aug_config(img_shape, input_shape):
    rot = 0
    bb_c_x = img_shape[0] * 0.5
    bb_c_y = img_shape[1] * 0.5
    # bbx_length = max(img_shape[0], img_shape[1])
    # bb_width = bbx_length
    # bb_height = bbx_length
    bbx_scale = max(img_shape[0] / input_shape[1], img_shape[1] / input_shape[0])
    bb_width = input_shape[1] * bbx_scale
    bb_height = input_shape[0] * bbx_scale
    bbx = [bb_c_x, bb_c_y, bb_width, bb_height]

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height,
                                    input_shape[1], input_shape[0], rot, inv=False)

    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height,
                                        input_shape[1], input_shape[0], rot, inv=True)

    return trans, inv_trans


def generate_patch_image(cvimg, do_flip, trans, input_shape=(256, 256)):
    img = cvimg.copy()
    if do_flip:
        img = img[:, ::-1, :]

    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32) / 255
    return img_patch


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, rot, inv=False):
    src_w = src_width
    src_h = src_height
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d, trans):
    num_pts = pt_2d.shape[-1]
    src_pt = np.concatenate([pt_2d, np.ones_like(pt_2d)[0:1]], axis=0)
    dst_pt = np.dot(trans, src_pt.reshape(3, -1)).reshape(2, -1, num_pts)
    return dst_pt[0:2]


def transform_pts_np(pts, trans):
    # pts: [n, num_joints, 2]
    # trans: [2, 3]
    pts = np.concatenate([pts, np.ones_like(pts)[..., 0:1]], axis=-1)
    trans_pts = np.dot(pts, trans.T)
    return trans_pts


def compute_match_cost(pre_frame_data, cur_frame_data, h, w, max_depth):
    # pre_frame_data: [m, k, 4], cur_frame_data [n, k, 4]
    match_cost = pre_frame_data[:, np.newaxis] - cur_frame_data[np.newaxis, :]  # [m, n, k, 4]
    match_cost[..., 0] = match_cost[..., 0] / w
    match_cost[..., 1] = match_cost[..., 1] / h
    match_cost[..., 2] = match_cost[..., 2] / max_depth
    match_cost[..., 3] = 0.1 * match_cost[..., 3]  # kpt score
    match_cost = np.sum(match_cost**2, axis=(-1, -2))
    return match_cost


def bbox_2d_padded(pose, h_inc_perc=0.15, w_inc_perc=0.1):
    """
    pose: [num_joints, 4]
    """
    vis = pose[:, 3]
    if np.sum(vis > 0) < 2:
        bbx = [0, 0, 0, 0]
        return bbx

    kp = pose[vis > 0, 0:2]
    x_min = np.min(kp[:, 0])
    y_min = np.min(kp[:, 1])
    x_max = np.max(kp[:, 0])
    y_max = np.max(kp[:, 1])

    width = x_max - x_min
    height = y_max - y_min

    inc_h = (height * h_inc_perc) / 2
    inc_w = (width * w_inc_perc) / 2

    x_min = x_min - inc_w
    x_max = x_max + inc_w
    y_min = y_min - inc_h
    y_max = y_max + inc_h
    width = x_max - x_min
    height = y_max - y_min

    bbx = [int(x_min), int(y_min), int(width), int(height)]
    return bbx


def get_all_samples(args):
    gap = args.seq_gap
    num_frames = args.num_frames
    num_future_frames = args.num_future_frames
    if num_frames == 1:
        # select frame t, t+gap, t+gap*2, ...
        skip = gap
    else:
        # select snippets [t, t+gap, t+gap*2, t+gap*3], [t+gap*3, t+gap*4, t+gap*5, t+gap*6]
        skip = gap * (num_frames - 1)

    data_dir = args.data_dir
    all_files = os.listdir(data_dir)
    seq_length = len(all_files)

    all_samples = []
    h, w = args.input_height, args.input_width
    input_shape = (h, w)
    frame_indices = []
    for idx in range(0, seq_length - skip, skip):
        frame_indices.append(idx)
        filenames = []
        imgs = []
        for t in range(num_frames):
            filename = all_files[idx + gap * t]
            filenames.append(filename)
            img = cv2.cvtColor(
                cv2.imread('{}/{}'.format(data_dir, filename)), cv2.COLOR_BGR2RGB)
            imgs.append(img)

        imgs = np.stack(imgs, axis=0)
        img_height, img_width, img_channels = imgs.shape[1:]

        trans_imgs = []  # transform input image to size (h, w)
        trans, inv_trans = get_aug_config((img_width, img_height), input_shape)
        for t in range(num_frames):
            img = imgs[t]
            img_patch = generate_patch_image(img, False, trans, input_shape)
            trans_imgs.append(img_patch)
            # plt.figure()
            # plt.imshow(img_patch)
            # plt.show()

        imgs = torch.from_numpy(np.concatenate(trans_imgs, axis=2)).float().permute(2, 0, 1)  # [T*3, H, W]

        samples = {}
        samples['input_size'] = torch.from_numpy(np.array([w, h])).float()  # [2]
        samples['img_size'] = torch.from_numpy(np.array([img_width, img_height])).float()  # [2]
        samples['imgs'] = imgs
        samples['filenames'] = filenames
        samples['inv_trans'] = torch.from_numpy(inv_trans).float()
        all_samples.append(samples)
    return all_samples, frame_indices, all_files


def associate_snippets(results, frame_indices, all_filenames, args):
    gap = args.seq_gap
    num_frames = args.num_frames
    num_future_frames = args.num_future_frames
    max_depth = args.max_depth
    all_frames_results = {}
    max_pid = 0
    for snippet_idx, result in enumerate(results):
        # print(snippet_idx, result['filenames'])

        pred_human = result['human_score'] > 0.5  # [n_query, T]
        exist_preds = np.sum(pred_human, axis=1) > 0

        pred_human = pred_human[exist_preds]  # [n, T]
        pred_kpt_scores = result['pred_kpt_scores'][exist_preds]  # [n, T, num_kpts, 1]
        pred_kpts = result['pred_kpts'][exist_preds]  # [n, T, num_kpts, 2]
        pred_depth = result['pred_depth'][exist_preds]  # [n, T, num_kpts, 1]
        inv_trans = result['inv_trans']

        if snippet_idx == 0:
            n = pred_human.shape[0]
            seq_pids = np.arange(n)  # [0, 1, 2, ...]
            max_pid = max_pid + n

            for t in range(num_frames):
                filename = result['filenames'][t]
                # print(filename, all_filenames[frame_indices[snippet_idx] + t * gap])
                assert filename == all_filenames[frame_indices[snippet_idx] + t * gap]
                frame_idx = frame_indices[snippet_idx] + t * gap

                _exist_person = pred_human[:, t]
                _pred_kpts = pred_kpts[_exist_person, t]  # [m, num_kpts, 2]
                _pred_kpt_scores = pred_kpt_scores[_exist_person, t]  # [m, num_kpts, 1]
                _pred_depth = pred_depth[_exist_person, t]  # [m, num_kpts, 1]
                _pred_kpts = transform_pts_np(_pred_kpts, inv_trans)

                # [n, num_kpts, 4]
                frame_data = np.concatenate([_pred_kpts, _pred_depth, _pred_kpt_scores], axis=-1)
                frame_data[:, 0, :] = (frame_data[:, 9, :] + frame_data[:, 10, :]) / 2
                frame_pids = seq_pids[_exist_person]
                all_frames_results[frame_idx] = (frame_pids, frame_data)
                # print(frame_idx, frame_pids, frame_data.shape)
        else:
            # match over one overlapped frame
            # pre_frame_data [n1, num_kpts, 4], cur_frame_data [n2, num_kpts, 4]
            frame_idx = frame_indices[snippet_idx] + 0 * gap
            if num_frames > 1:
                # match with overlapped frame between two snippets
                pre_frame_pids, pre_frame_data = all_frames_results[frame_idx]
            else:
                # match with previous frame when T=1
                pre_frame_pids, pre_frame_data = all_frames_results[frame_idx - gap]

            cur_exist_pred = pred_human[:, 0]
            _pred_kpts = pred_kpts[cur_exist_pred, 0]  # [m, num_kpts, 2]
            _pred_kpt_scores = pred_kpt_scores[cur_exist_pred, 0]  # [m, num_kpts, 1]
            _pred_depth = pred_depth[cur_exist_pred, 0]  # [m, num_kpts, 1]
            _pred_kpts = transform_pts_np(_pred_kpts, inv_trans)
            cur_frame_data = np.concatenate([_pred_kpts, _pred_depth, _pred_kpt_scores], axis=-1)  # [n, num_kpts, 4]
            cur_frame_data[:, 0, :] = (cur_frame_data[:, 9, :] + cur_frame_data[:, 10, :]) / 2

            if cur_frame_data.shape[0] == 0 or pre_frame_data.shape[0] == 0:
                # cur_exist_pred is all False || cur_exist_pred is all new person
                seq_pids = np.zeros(cur_exist_pred.shape[0], dtype=np.int32) - 1
                miss_num_person = np.sum(seq_pids == -1)
                seq_pids[seq_pids == -1] = np.arange(0, miss_num_person) + max_pid
                max_pid = max_pid + miss_num_person
                cur2pre_idx = np.zeros([0])
                # print(seq_pids)
            else:
                w, h = result['img_size']
                match = compute_match_cost(pre_frame_data, cur_frame_data, h, w, max_depth)  # [m, n]
                # print(match)
                pre2cur_idx = np.argmin(match, axis=1)  # may repeat
                # print(pre2cur_idx)
                mask = np.full(match.shape, np.inf)
                mask[np.arange(pre2cur_idx.shape[0]), pre2cur_idx] = 1
                match = match * mask
                # print(mask)
                # print(match)
                cur_no_match = np.sum(mask != np.inf, axis=0) == 0
                # print('cur_no_match', cur_no_match)
                cur2pre_idx = np.argmin(match, axis=0)
                cur2pre_idx[cur_no_match] = -1
                # print('pre_frame_pids', pre_frame_pids)
                # print('cur2pre_idx', cur2pre_idx)
                # assign pid from pre to cur
                cur_frame_pids = np.zeros(cur2pre_idx.shape[0], dtype=np.int32) - 1
                for i in range(cur2pre_idx.shape[0]):
                    if cur2pre_idx[i] == -1:
                        # no matched from previous snippet
                        cur_frame_pids[i] = max_pid
                        max_pid += 1
                    else:
                        # has one matched person from previous snippet
                        cur_frame_pids[i] = pre_frame_pids[cur2pre_idx[i]]
                # print('cur_frame_pids', cur_frame_pids)
                # assign pid to the whole sequence
                seq_pids = np.zeros(cur_exist_pred.shape[0], dtype=np.int32) - 1
                seq_pids[cur_exist_pred] = cur_frame_pids
                miss_num_person = np.sum(seq_pids == -1)
                seq_pids[seq_pids == -1] = np.arange(0, miss_num_person) + max_pid
                max_pid = max_pid + miss_num_person
                # print(cur_exist_pred, seq_pids, miss_num_person, max_pid)

            # collect all frames in the snippet
            for t in range(num_frames):
                filename = result['filenames'][t]
                # print(filename, all_filenames[frame_indices[snippet_idx] + t * gap])
                assert filename == all_filenames[frame_indices[snippet_idx] + t * gap]
                frame_idx = frame_indices[snippet_idx] + t * gap

                _exist_person = pred_human[:, t]
                _pred_kpts = pred_kpts[_exist_person, t]  # [m, num_kpts, 2]
                _pred_kpt_scores = pred_kpt_scores[_exist_person, t]  # [m, num_kpts, 1]
                _pred_depth = pred_depth[_exist_person, t]  # [m, num_kpts, 1]
                _pred_kpts = transform_pts_np(_pred_kpts, inv_trans)

                # [n, num_kpts, 4]
                frame_data = np.concatenate([_pred_kpts, _pred_depth, _pred_kpt_scores], axis=-1)
                frame_data[:, 0, :] = (frame_data[:, 9, :] + frame_data[:, 10, :]) / 2

                # average matched poses
                if t == 0 and cur2pre_idx.shape[0] > 0 and num_frames > 1:
                    valid = cur2pre_idx != -1
                    cur_idx = np.arange(cur2pre_idx.shape[0])[valid]
                    pre_idx = cur2pre_idx[valid]
                    pre_pose = pre_frame_data[pre_idx]
                    cur_pose = frame_data[cur_idx]
                    # scores
                    pre_score = pre_pose[:, :, 3:4]
                    cur_score = cur_pose[:, :, 3:4]
                    frame_data[cur_idx, :, 3:4] = (pre_score + cur_score) / 2
                    # pose2d + depth
                    frame_data[cur_idx, :, 0:3] = \
                        (pre_score * pre_pose[:, :, 0:3] + cur_score * cur_pose[:, :, 0:3]) / (pre_score + cur_score)

                # [n, num_kpts, 6]
                frame_pids = seq_pids[_exist_person]
                all_frames_results[frame_idx] = (frame_pids, frame_data)
                # print(frame_idx, frame_pids, frame_data.shape)
    return all_frames_results, max_pid


def save_visual_results(all_frames_results, all_filenames, data_dir, save_dir, max_pid, max_depth, gap):
    SKELETONS = [
        (0, 9),  # root -> left_hip
        (0, 10),  # root -> right_hip
        (0, 2),  # root -> head_bottom
        (2, 3),  # head_bottom -> left_shoulder
        (2, 4),  # head_bottom -> right_shoulder
        (2, 1),  # head_bottom -> nose
        (3, 5),  # left_shoulder -> left_elbow
        (5, 7),  # left_elbow -> left_wrist
        (4, 6),  # right_shoulder -> right_elbow
        (6, 8),  # right_elbow -> right_wrist
        (9, 11),  # left_hip -> left_knee
        (11, 13),  # left_knee -> left_ankle
        (10, 12),  # right_hip -> right_knee
        (12, 14),  # right_knee -> right_ankle
    ]

    random.seed(13)
    pid_count = max_pid
    all_pids = np.arange(pid_count)
    cmap = plt.get_cmap('rainbow')
    pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]
    random.shuffle(pid_colors)
    pid_colors_opencv = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

    # sks_colors = [cmap(i) for i in np.linspace(0, 1, len(SKELETONS) + 2)]
    # sks_colors_opencv = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in sks_colors]

    print('save track2d visual results')
    if not os.path.exists('{}/track2d'.format(save_dir)):
        os.mkdir('{}/track2d'.format(save_dir))

    h, w = None, None
    for frame_idx in tqdm(all_frames_results.keys()):
        filename = all_filenames[frame_idx]
        img = cv2.imread('{}/{}'.format(data_dir, filename))
        h, w, _ = img.shape
        pids, poses = all_frames_results[frame_idx]
        for i in range(poses.shape[0]):
            pid = pids[i]
            pid_idx = np.where(all_pids == pid)[0][0]
            pose = poses[i]  # 2d pose + depth
            # pose[:, 3] = pose[:, 3] > 0.1
            for l, (j1, j2) in enumerate(SKELETONS):
                joint1 = pose[j1]
                joint2 = pose[j2]
                if joint1[3] > 0 and joint2[3] > 0:
                    t = 4
                    r = 8
                    cv2.line(img,
                             (int(joint1[0]), int(joint1[1])),
                             (int(joint2[0]), int(joint2[1])),
                             color=tuple(pid_colors_opencv[pid_idx]),
                             # color=tuple(sks_colors[l]),
                             thickness=t)
                    cv2.circle(
                        img,
                        thickness=-1,
                        center=(int(joint1[0]), int(joint1[1])),
                        radius=r,
                        color=tuple(pid_colors_opencv[pid_idx]),
                    )
                    cv2.circle(
                        img,
                        thickness=-1,
                        center=(int(joint2[0]), int(joint2[1])),
                        radius=r,
                        color=tuple(pid_colors_opencv[pid_idx]),
                    )
            bbx = bbox_2d_padded(pose, 0.3, 0.3)
            bbx_thick = 3
            cv2.line(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1]),
                     color=tuple(pid_colors_opencv[pid_idx]), thickness=bbx_thick)
            cv2.line(img, (bbx[0], bbx[1]), (bbx[0], bbx[1] + bbx[3]),
                     color=tuple(pid_colors_opencv[pid_idx]), thickness=bbx_thick)
            cv2.line(img, (bbx[0] + bbx[2], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]),
                     color=tuple(pid_colors_opencv[pid_idx]), thickness=bbx_thick)
            cv2.line(img, (bbx[0], bbx[1] + bbx[3]), (bbx[0] + bbx[2], bbx[1] + bbx[3]),
                     color=tuple(pid_colors_opencv[pid_idx]), thickness=bbx_thick)

            cv2.putText(img, '{:02d}'.format(pid), (bbx[0] + bbx[2] // 3, bbx[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=tuple(pid_colors_opencv[pid_idx]), thickness=bbx_thick)
        frame_name = filename.split('.')[0]
        cv2.imwrite('{}/track2d/{}_track.jpg'.format(save_dir, frame_name), img)

    # draw 3D image
    print('save track3d visual results')
    if not os.path.exists('{}/track3d'.format(save_dir)):
        os.mkdir('{}/track3d'.format(save_dir))

    for frame_idx in tqdm(all_frames_results.keys()):
        pids, poses = all_frames_results[frame_idx]
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        for p, pid in enumerate(pids):
            pid_idx = np.where(all_pids == pid)[0][0]
            # draw pose
            kpt_3d = poses[p]
            for l in range(len(SKELETONS)):
                i1 = SKELETONS[l][0]
                i2 = SKELETONS[l][1]
                if kpt_3d[i1, 3] > 0 and kpt_3d[i2, 3] > 0:
                    x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
                    y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
                    z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])
                    # ax.plot(x, z, -y, color=sks_colors[l], linewidth=10, alpha=1)
                    ax.plot(x, z, -y, color=pid_colors[pid_idx], linewidth=4, alpha=1)
                    ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], color=pid_colors[pid_idx],
                               marker='o', s=6)
                    ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], color=pid_colors[pid_idx],
                               marker='o', s=6)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim([0, w])
        ax.set_ylim([2, max_depth])
        ax.set_zlim([-h, 0])
        # ax.legend()

        filename = all_filenames[frame_idx]
        frame_name = filename.split('.')[0]
        ax.view_init(10, -90)  # view_init(elev, azim)
        plt.savefig('{}/track3d/{}_track3d.jpg'.format(save_dir, frame_name), bbox_inches='tight')
        ax.view_init(70, -90)  # view_init(elev, azim)
        plt.savefig('{}/track3d/{}_track3d_topdown.jpg'.format(save_dir, frame_name), bbox_inches='tight')
        # plt.close()
        # plt.show()

    # draw trajectory
    print('save trajectory visual results')
    end_frame_idx = max(list(all_frames_results.keys()))
    start_frame_idx = min(list(all_frames_results.keys()))
    _frame_length = end_frame_idx - start_frame_idx
    for _frame_idx in range(start_frame_idx, end_frame_idx, _frame_length):
        _end_frame_idx = _frame_idx + _frame_length
        # _start_frame_idx = copy.copy(_frame_idx)
        # print(_end_frame_idx, start_frame_idx)

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        is_draw_pose = np.zeros([pid_count])
        for frame_idx in range(_end_frame_idx - gap, start_frame_idx - gap, -gap):
            # print(frame_idx)
            pids, poses3d = all_frames_results[frame_idx]
            for p, pid in enumerate(pids):
                pid_idx = np.where(all_pids == pid)[0][0]
                if is_draw_pose[pid_idx] == 0:
                    is_draw_pose[pid_idx] = 1
                    # draw pose
                    kpt_3d = poses3d[p]
                    for l in range(len(SKELETONS)):
                        i1 = SKELETONS[l][0]
                        i2 = SKELETONS[l][1]
                        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
                        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
                        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])
                        ax.plot(x, z, -y, color=pid_colors[pid_idx], linewidth=4, alpha=1)
                        ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], color=pid_colors[pid_idx], marker='o',
                                   s=6)
                        ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], color=pid_colors[pid_idx], marker='o',
                                   s=6)

            # draw trajectory
            if frame_idx == end_frame_idx - gap:
                continue

            pre_pids, pre_poses3d = all_frames_results[frame_idx + gap]
            pids, poses3d = all_frames_results[frame_idx]

            for p, pid in enumerate(pids):
                if pid not in pre_pids:
                    continue

                pid_idx = np.where(all_pids == pid)[0][0]
                color = pid_colors[pid_idx]

                pose3d = poses3d[p]  # [num_joints, 3]
                pre_pose3d = pre_poses3d[pre_pids == pid][0]  # [num_joints, 3]
                # print(pose3d.shape, pre_pose3d.shape)

                for j in range(pose3d.shape[0]):
                    x = np.array([pose3d[j, 0], pre_pose3d[j, 0]])
                    y = np.array([pose3d[j, 1], pre_pose3d[j, 1]])
                    z = np.array([pose3d[j, 2], pre_pose3d[j, 2]])
                    ax.plot(x, z, -y, color=pid_colors[pid_idx], linewidth=1, alpha=1)

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Z Label')
        # ax.set_zlabel('Y Label')
        # ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim([0, w])
        ax.set_ylim([2, max_depth])
        ax.set_zlim([-h, 0])
        # ax.legend()

        filename = all_filenames[end_frame_idx]
        frame_name = filename.split('.')[0]
        ax.view_init(20, -80)  # view_init(elev, azim)
        plt.savefig('{}/track3d/{}_trajectory3d.jpg'.format(save_dir, frame_name), bbox_inches='tight')
        ax.view_init(70, -90)  # view_init(elev, azim)
        plt.savefig('{}/track3d/{}_trajectory3d_topdown.jpg'.format(save_dir, frame_name), bbox_inches='tight')
        # plt.close()


def save_as_videos(save_dir, all_frames_idx, all_filenames):
    # static image consists of first frame, middle frame and last frame multi-person pose tracking results
    # and also the trajectory in 3D space
    n = len(all_frames_idx)
    frame_name = all_filenames[all_frames_idx[0]].split('.')[0]  # first frame
    img1 = cv2.imread('{}/track2d/{}_track.jpg'.format(save_dir, frame_name))
    frame_name = all_filenames[all_frames_idx[n // 2]].split('.')[0]  # intermediate frame
    img2 = cv2.imread('{}/track2d/{}_track.jpg'.format(save_dir, frame_name))
    frame_name = all_filenames[all_frames_idx[-1]].split('.')[0]  # last frame
    img3 = cv2.imread('{}/track2d/{}_track.jpg'.format(save_dir, frame_name))

    img_height, img_width, _ = img1.shape
    tgt_h, tgt_w = 540, 960
    trans, inv_trans = get_aug_config((img_width, img_height), (tgt_h, tgt_w))
    img1 = cv2.warpAffine(img1, trans, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
    img2 = cv2.warpAffine(img2, trans, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
    img3 = cv2.warpAffine(img3, trans, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)

    frame_name = all_filenames[all_frames_idx[-1]].split('.')[0]  # first frame
    img4 = cv2.imread('{}/track3d/{}_trajectory3d.jpg'.format(save_dir, frame_name))
    img5 = cv2.imread('{}/track3d/{}_trajectory3d_topdown.jpg'.format(save_dir, frame_name))

    static_frame = np.zeros([540 * 3, 960 + 1560 + 1560, 3], dtype=np.uint8) + 255
    static_frame[0: 540, 0:960] = img1
    static_frame[540: 540 + 540, 0:960] = img2
    static_frame[540 + 540: 540 + 540 + 540, 0:960] = img3
    static_frame[30:1590, 960: 960 + 1560] = img4
    static_frame[30:1590, 960 + 1560: 960 + 1560 + 1560] = img5
    static_frame = cv2.resize(static_frame, (2040, 810))

    cv2.putText(static_frame, 'Frame {}'.format(all_frames_idx[0]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color=(0, 0, 255), thickness=2)
    cv2.putText(static_frame, 'Frame {}'.format(all_frames_idx[n//2]), (10, 270 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color=(0, 0, 255), thickness=2)
    cv2.putText(static_frame, 'Frame {}'.format(all_frames_idx[-1]), (10, 270 + 270 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color=(0, 0, 255), thickness=2)

    cv2.putText(static_frame, 'Trajectory (camera view)', (650, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 0, 255), thickness=2)
    cv2.putText(static_frame, 'Trajectory (top-down view)', (1450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 0, 255), thickness=2)

    cv2.imwrite('{}/static_img.jpg'.format(save_dir), static_frame)

    # save as gif
    frames = []
    for frame_idx in all_frames_idx:
        frame_name = all_filenames[frame_idx].split('.')[0]
        frame1 = cv2.imread('{}/track2d/{}_track.jpg'.format(save_dir, frame_name))  # 540 960
        frame1 = cv2.warpAffine(frame1, trans, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)

        frame2 = cv2.imread('{}/track3d/{}_track3d.jpg'.format(save_dir, frame_name))  # 1560 1560

        frame = np.zeros([810 + 1080, 960 + 1080, 3], dtype=np.uint8) + 255
        frame[0:810, :] = static_frame
        frame[810 + 270: 810 + 270 + 540, 0:960] = frame1
        frame[810: 810 + 1080, 960: 960 + 1080] = cv2.resize(frame2, (1080, 1080))

        cv2.putText(frame, '2D pose', (400, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    color=(0, 0, 255), thickness=2)
        cv2.putText(frame, '3D pose', (1400, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    color=(0, 0, 255), thickness=2)

        # frames.append(frame[810:, :, :])
        frames.append(frame[:, :, ::-1])

    imageio.mimsave('{}/pose_tracking.gif'.format(save_dir), frames, fps=5)
    print('pose tracking gif saving as {}/pose_tracking.gif'.format(save_dir))


def visualize_heatmaps(heatmaps, img, save_dir, filename):
    # heatmap: [h, w, num_joints]
    # fin = cv2.addWeighted(heatmap_img, 0.7, img, 0.3, 0)
    kpts_name = ['root', 'nose', 'neck', 'left_shoulder', 'right_shoulder',
                 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    img = (img * 255).astype(np.uint8)[:, :, ::-1]  # rgb -> bgr
    h, w, _ = img.shape
    num_joints = heatmaps.shape[-1]
    for j in range(num_joints):
        heatmap = heatmaps[:, :, j]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.resize(heatmap, (w, h))

        heatmap_img = np.zeros([h, w, 3], dtype=np.uint8)  # bgr
        heatmap_img[:, :, 2] = heatmap
        overlap_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

        frame_name = filename.split('.')[0]
        cv2.imwrite('{}/heatmaps/{}_{}.jpg'.format(save_dir, frame_name, kpts_name[j]), overlap_img)


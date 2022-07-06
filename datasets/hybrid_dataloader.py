import collections

import torch
from torch.utils.data import Dataset
import cv2
import pickle
import json
import copy
import random
import os
import numpy as np
from datasets.data_preprocess.dataset_util import posetrack_visualization, panoptic_visualization
from datasets.transforms import get_aug_config, generate_patch_image, trans_point2d, get_aug_config_coco

# JOINT15 ['root'='neck', 'nose/head_top', 'neck', 'left_shoulder', 'right_shoulder',
# #          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
# #          'right_knee', 'left_ankle', 'right_ankle']
JTA2JOINT15 = [2, 1, 2, 8, 4, 9, 5, 10, 6, 19, 16, 20, 17, 21, 18]
POSETRACK2JOINT15 = [2, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
ROOTJOINTCONT = np.array([0, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2, 0.1, 0.1])
FLIPJOINTS = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
SKELETONS = [
        (0, 9),  # root -> left_hip
        (0, 10),  # root -> right_hip
        (0, 2),  # root -> neck
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
JOINT152POSETRACK = [2, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
COCO2JOINT15 = [2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# COCO ['root' 0, 'neck' 1, 'nose' 2, 'left_eye' 3, 'right_eye' 4, 'left_ear' 5, 'right_ear' 6,
#       'left_shoulder' 7, 'right_shoulder' 8, 'left_elbow' 9, 'right_elbow' 10, 'left_wrist' 11,
#       'right_wrist' 12, 'left_hip' 13, 'right_hip' 14, 'left_knee' 15,
#       'right_knee' 16, 'left_ankle'17, 'right_ankle' 18]
JOINT152COCO = [0, 2, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


class HybridData(Dataset):
    def __init__(
            self,
            posetrack_dir='posetrack2018/',
            seq_length=4,
            future_seq_length=2,
            seq_max_gap=4,
            seq_min_gap=4,
            mode='train',
            input_shape=(600, 800),  # (h, w)
            num_joints=15,
            vis=False,
            coco_data_dir='coco/',
            muco_data_dir='muco/',
            jta_data_dir='jta/',
            panoptic_data_dir='panoptic/',
            max_depth=15,
            use_posetrack=1,
            use_coco=1,
            use_muco=1,
            use_jta=1,
            use_panoptic=0,
            panoptic_protocol=1,
    ):
        self.seq_l = seq_length
        self.future_seq_l = future_seq_length
        self.mode = mode
        self.seq_max_gap = seq_max_gap
        self.seq_min_gap = 1 if seq_length == 1 else seq_min_gap
        self.mode = mode
        self.input_shape = input_shape
        self.max_depth = max_depth

        self.posetrack_dir = posetrack_dir
        self.use_posetrack = use_posetrack
        self.coco_data_dir = coco_data_dir
        self.use_coco = use_coco
        self.muco_data_dir = muco_data_dir
        self.use_muco = use_muco
        self.jta_data_dir = jta_data_dir
        self.use_jta = use_jta
        self.panoptic_data_dir = panoptic_data_dir
        self.use_panoptic = use_panoptic
        self.panoptic_protocol = panoptic_protocol

        self.all_seqs, self.posetrack_data, self.coco_data, self.muco_data, self.mupots_data, self.panoptic_data = \
            self.get_labelled_seq()
        self.num_joints = num_joints
        self.vis = vis

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx):
        # print(self.all_seqs[idx][0])
        if self.all_seqs[idx][0] == 'posetrack':
            imgs, targets = self.get_posetrack(self.all_seqs[idx])
        elif self.all_seqs[idx][0] == 'coco':
            imgs, targets = self.get_coco(self.all_seqs[idx])
        elif self.all_seqs[idx][0] == 'muco':
            imgs, targets = self.get_muco(self.all_seqs[idx])
        elif self.all_seqs[idx][0] == 'mupots':
            imgs, targets = self.get_mupots(self.all_seqs[idx])
        elif self.all_seqs[idx][0] == 'jta':
            imgs, targets = self.get_jta(self.all_seqs[idx])
        elif self.all_seqs[idx][0] == 'panoptic':
            imgs, targets = self.get_panoptic(self.all_seqs[idx])
        else:
            raise ValueError('dataset name unknown {}'.format(self.all_seqs[idx][0]))
        return imgs, targets

    def get_posetrack(self, sample):
        _, fn, filename, frame_idx, indice, max_valid_gap, augmentation = sample
        if self.mode == 'train':
            gap = np.random.randint(self.seq_min_gap, max_valid_gap + 1)
        else:
            gap = 4
        # print(fn, gap)

        # collect datasets
        imgs, kpts2d, bbxes, track_ids, bboxes_head, filenames, frame_indices = [], [], [], [], [], [], []
        for j in range(self.seq_l + self.future_seq_l):
            datum = self.posetrack_data[fn][indice + j * gap]
            filenames.append(datum['filename'])
            frame_indices.append(indice + j * gap)

            if j < self.seq_l:
                img = cv2.cvtColor(cv2.imread('{}/{}'.format(self.posetrack_dir, datum['filename'])), cv2.COLOR_BGR2RGB)
                imgs.append(img)

            if datum['kpts2d'] == []:
                kpts2d.append(np.array([]).reshape(-1, self.num_joints, 3))  # [n, num_joints, 3]
                bbxes.append(np.array([]).reshape(-1, 4))  # [n, 4]
                track_ids.append(np.array([], dtype=np.int32).reshape(-1))  # [n]
            else:
                kpts2d.append(copy.copy(datum['kpts2d'][:, POSETRACK2JOINT15, :]))  # [n, num_joints, 3]
                bbxes.append(copy.copy(datum['bboxes']))  # [n, 4]
                track_ids.append(copy.copy(datum['track_id']))  # [n]
                # print(kpts2d[-1].shape, bbxes[-1].shape, track_ids[-1].shape)

            # if self.mode == 'val':
            if 'bboxes_head' not in datum.keys():
                bboxes_head.append(np.zeros([kpts2d[-1].shape[0], 4]))  # [n, 4]
            else:
                if datum['bboxes_head'] == []:
                    bboxes_head.append(np.zeros([kpts2d[-1].shape[0], 4]))  # [n, 4]
                else:
                    bboxes_head.append(copy.copy(datum['bboxes_head']))  # [n, 4]
            # print(j, track_ids[-1], bboxes_head[-1])

        imgs = np.stack(imgs, axis=0)
        _, img_height, img_width, img_channels = imgs.shape
        img_shape = (img_width, img_height)
        # print(track_ids, bboxes_head)

        if self.vis:
            # print(fn, filename, frame_idx, indice, max_valid_gap, gap)
            posetrack_visualization(imgs, kpts2d, track_ids, 'org', SKELETONS)

        # 1. get augmentation params and apply for the whole sequence of images
        rot, do_flip, color_scale, bbx, trans, inv_trans = \
            get_aug_config(img_shape, self.input_shape, augmentation)
        # do_flip = True
        # print(rot, do_flip, color_scale, bbx)

        aug_imgs, aug_kpts2d, aug_track_ids = [], [], []
        for i in range(self.seq_l + self.future_seq_l):
            # 2. perform datasets augmentation (flip, rot, color scale)
            if i < self.seq_l:
                img = imgs[i]
                img_patch = generate_patch_image(img, do_flip, trans, self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                aug_imgs.append(img_patch)

            # 3. apply Affine Transform on keypoints
            kpt2d = np.transpose(kpts2d[i], [2, 0, 1])  # [3, n, 18]
            if do_flip:
                kpt2d[0] = img_width - kpt2d[0] - 1
                kpt2d = kpt2d[:, :, FLIPJOINTS]
            kpt2d[0:2] = trans_point2d(kpt2d[0:2], trans)
            # kpt2d[2] *= (
            #         (kpt2d[0] >= 0) &
            #         (kpt2d[0] < self.input_shape[1]) &
            #         (kpt2d[1] >= 0) &
            #         (kpt2d[1] < self.input_shape[0])
            # )
            kpt2d = np.transpose(kpt2d, [1, 2, 0])  # [n, num_joints, 3]

            # normalize
            kpt2d[..., 0] = kpt2d[..., 0] / self.input_shape[1]
            kpt2d[..., 1] = kpt2d[..., 1] / self.input_shape[0]

            aug_kpts2d.append(kpt2d)
            aug_track_ids.append(track_ids[i])
        # print(aug_track_ids[0][-1])
        # print(aug_kpts2d[0][-1])

        all_ids = set(np.concatenate(aug_track_ids[0:self.seq_l], axis=0))
        if len(all_ids) == 0:
            max_id = -1
        else:
            max_id = max(all_ids)
        # print(all_ids)

        # filter out invalid target future poses
        for i in range(self.seq_l, self.seq_l + self.future_seq_l):
            valid_target_id = np.array([pid in all_ids for pid in aug_track_ids[i]], dtype=np.bool)
            aug_kpts2d[i] = aug_kpts2d[i][valid_target_id]
            aug_track_ids[i] = aug_track_ids[i][valid_target_id]
            bboxes_head[i] = bboxes_head[i][valid_target_id]

        kpts2d = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, self.num_joints, 3])
        track_ids = np.zeros([max_id + 1, self.seq_l + self.future_seq_l], dtype=np.int32)
        bboxes_head_np = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, 4])

        # print(aug_track_ids, bboxes_head)
        for i in range(self.seq_l + self.future_seq_l):
            track_ids[aug_track_ids[i], i] = 1  # one-hot: exist in the frame or not
            kpts2d[aug_track_ids[i], i] = aug_kpts2d[i]
            if self.mode == 'val':
                bboxes_head_np[aug_track_ids[i], i] = bboxes_head[i]
        # print(track_ids, bboxes_head_np)

        # some persons will be missing in all frames
        exist_id = np.sum(track_ids, axis=1) > 0
        traj_id = np.where(exist_id)[0]
        track_ids = track_ids[exist_id]
        kpts2d = kpts2d[exist_id]
        bboxes_head_np = bboxes_head_np[exist_id]
        bboxes = []
        for i in range(self.seq_l + self.future_seq_l):
            bboxes.append(self.bbox_2d_padded(kpts2d[:, i]))
        bboxes = np.stack(bboxes, axis=1)  # [n, T, 4]
        # print(track_ids, traj_id)
        # print(traj_id == aug_track_ids[0][-1])
        # print(kpts2d[traj_id == aug_track_ids[0][-1]])

        # collect data
        imgs = torch.from_numpy(np.concatenate(aug_imgs, axis=2)).float().permute(2, 0, 1)  # [T*3, H, W]

        targets = {}
        targets['kpts2d'] = torch.from_numpy(kpts2d).float()  # [n, T, K, 3]
        targets['depth'] = torch.from_numpy(np.zeros_like(kpts2d)[..., 0:2]).float()  # [n, T, num_joints, 2]
        targets['bbxes'] = torch.from_numpy(bboxes).float()  # [n, T, 4]
        targets['track_ids'] = torch.from_numpy(track_ids)  # [n, T]
        targets['traj_ids'] = torch.from_numpy(traj_id)  # [n]
        targets['max_depth'] = torch.from_numpy(np.array(self.max_depth)).float()  # for evaluation
        # print(targets['track_ids'], targets['traj_ids'])

        if True:  # self.mode == 'val':
            # for validation
            targets['input_size'] = torch.from_numpy(np.array([self.input_shape[1], self.input_shape[0]]))
            targets['inv_trans'] = torch.from_numpy(inv_trans).float()  # [2, 3]
            targets['bbxes_head'] = torch.from_numpy(bboxes_head_np).float()  # [n, T, 4]
            targets['index'] = indice
            targets['video_name'] = fn
            targets['filenames'] = filenames
            targets['frame_indices'] = frame_indices
            targets['dataset'] = 'posetrack'
            targets['image_id'] = 0

            targets['cam_intr'] = torch.tensor([0]).float()
            targets['kpts3d'] = torch.tensor([0]).float()

        if self.vis:
            imgs = imgs.reshape(self.seq_l, 3, self.input_shape[0], self.input_shape[1]).permute(0, 2, 3, 1).numpy()
            # load future frames
            for j in range(self.seq_l, self.seq_l + self.future_seq_l):
                datum = self.posetrack_data[fn][indice + j * gap]
                # print('{}/{}, {}x{}'.format(self.data_dir, datum['filename'], datum['height'], datum['width']))
                img = cv2.cvtColor(cv2.imread('{}/{}'.format(self.posetrack_dir, datum['filename'])), cv2.COLOR_BGR2RGB)
                # augmentation
                img_patch = generate_patch_image(img, do_flip, trans, self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                # print(imgs.shape, img_patch.shape)
                imgs = np.concatenate([imgs, img_patch[np.newaxis]], axis=0)
            imgs = (imgs * 255).astype(np.uint8)  # [n, h, w, c]

            # prepare kpts for each frame
            img_size = targets['input_size'].numpy().reshape(1, 1, 2)
            _kpts2d, _track_ids = [], []
            for i in range(self.seq_l + self.future_seq_l):
                exist_person = targets['track_ids'][:, i].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                _track_ids.append(exist_pids)

                kpts2d = targets['kpts2d'][exist_person, i].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                _kpts2d.append(kpts2d)

            posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug', SKELETONS)

        return imgs, targets

    def get_coco(self, sample):
        _, filename, idx, augmentation = sample
        # keys: 'filename', 'bboxes', 'kpts2d', 'width', 'height'
        datum = self.coco_data[idx]
        assert filename == datum['filename']
        kpts2d = datum['kpts2d'][:, COCO2JOINT15, :]
        num_person = kpts2d.shape[0]
        traj_id = np.arange(num_person, dtype=np.int32)
        track_ids = np.ones(num_person, dtype=np.int32)

        img = cv2.cvtColor(cv2.imread('{}/{}'.format(self.coco_data_dir, filename)), cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channels = img.shape
        img_shape = (img_width, img_height)

        aug_imgs, aug_kpts2d, aug_track_ids, aug_bbxes = [], [], [], []
        # 1. get augmentation params and apply for the whole sequence of images
        rot_list, do_flip, color_scale, bbxes_list, trans_list, inv_trans_list = \
            get_aug_config_coco(img_shape, self.input_shape, self.seq_l + self.future_seq_l, augmentation)
        # print(rot_list, do_flip, color_scale)
        for t in range(self.seq_l + self.future_seq_l):
            # 2. perform datasets augmentation (flip, rot, color scale)
            if t < self.seq_l:
                img_patch = generate_patch_image(img, do_flip, trans_list[t], self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                aug_imgs.append(img_patch)

            # 3. apply Affine Transform on keypoints
            kpt2d = np.transpose(kpts2d.copy(), [2, 0, 1])  # [3, n, num_joints]
            if do_flip:
                kpt2d[0] = img_width - kpt2d[0] - 1
                kpt2d = kpt2d[:, :, FLIPJOINTS]
            kpt2d[0:2] = trans_point2d(kpt2d[0:2], trans_list[t])
            # kpt2d[2] *= (
            #         (kpt2d[0] >= 0) &
            #         (kpt2d[0] < self.input_shape[1]) &
            #         (kpt2d[1] >= 0) &
            #         (kpt2d[1] < self.input_shape[0])
            # )
            kpt2d = np.transpose(kpt2d, [1, 2, 0])  # [n, num_joints, 3]
            bboxes = self.bbox_2d_padded(kpt2d)

            # normalize
            kpt2d[..., 0] = kpt2d[..., 0] / self.input_shape[1]
            kpt2d[..., 1] = kpt2d[..., 1] / self.input_shape[0]

            aug_kpts2d.append(kpt2d)
            aug_track_ids.append(track_ids)
            aug_bbxes.append(bboxes)

        kpts2d = np.stack(aug_kpts2d, axis=1)  # [n, T, num_joints, 3]
        track_ids = np.stack(aug_track_ids, axis=1)  # [n, T]
        bboxes = np.stack(aug_bbxes, axis=1)  # [n, T, 4]

        exist_traj = np.sum(kpts2d[:, :, :, 2], axis=(1, 2)) > (self.seq_l + self.future_seq_l)
        kpts2d = kpts2d[exist_traj]
        track_ids = track_ids[exist_traj]
        bboxes = bboxes[exist_traj]
        traj_id = traj_id[exist_traj]

        # collect data
        imgs = torch.from_numpy(np.concatenate(aug_imgs, axis=2)).float().permute(2, 0, 1)  # [T*3, H, W]

        targets = {}
        targets['kpts2d'] = torch.from_numpy(kpts2d).float()  # [n, T, num_joints, 3]
        targets['depth'] = torch.from_numpy(np.zeros_like(kpts2d)[..., 0:2]).float()  # [n, T, num_joints, 2]
        targets['bbxes'] = torch.from_numpy(bboxes).float()  # [n, T, 4]
        targets['track_ids'] = torch.from_numpy(track_ids)  # [n, T]
        targets['traj_ids'] = torch.from_numpy(traj_id)  # [n]
        targets['input_size'] = torch.from_numpy(np.array([self.input_shape[1], self.input_shape[0]]))
        targets['max_depth'] = torch.from_numpy(np.array(self.max_depth)).float()  # for evaluation

        if True: # self.mode == 'val':
            targets['dataset'] = 'coco'
            targets['image_id'] = int(filename.split('/')[-1].split('.')[0])
            targets['inv_trans'] = torch.from_numpy(inv_trans_list[0]).float()

            targets['bbxes_head'] = torch.tensor([0]).float()
            targets['index'] = 0
            targets['video_name'] = 0
            targets['filenames'] = [filename]
            targets['frame_indices'] = 0

            targets['cam_intr'] = torch.tensor([0]).float()
            targets['kpts3d'] = torch.tensor([0]).float()

        if self.vis:
            imgs = imgs.reshape(self.seq_l, 3, self.input_shape[0], self.input_shape[1]).permute(0, 2, 3, 1).numpy()
            # load future frames
            for j in range(self.seq_l, self.seq_l + self.future_seq_l):
                # augmentation
                img_patch = generate_patch_image(img, do_flip, trans_list[j], self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                # print(imgs.shape, img_patch.shape)
                imgs = np.concatenate([imgs, img_patch[np.newaxis]], axis=0)
            imgs = (imgs * 255).astype(np.uint8)  # [n, h, w, c]

            # prepare kpts for each frame
            img_size = targets['input_size'].numpy().reshape(1, 1, 2)
            _kpts2d, _track_ids = [], []
            for i in range(self.seq_l + self.future_seq_l):
                exist_person = targets['track_ids'][:, i].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                _track_ids.append(exist_pids)

                kpts2d = targets['kpts2d'][exist_person, i].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                _kpts2d.append(kpts2d)

            posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug', SKELETONS)

            if True:  # self.mode == 'val':
                # val
                img = cv2.cvtColor(cv2.imread('{}/{}'.format(self.coco_data_dir, filename)), cv2.COLOR_BGR2RGB)

                img_size = targets['input_size'].numpy().reshape(1, 1, 2)
                exist_person = targets['track_ids'][:, 0].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                kpts2d = targets['kpts2d'][exist_person, 0].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                inv_trans = targets['inv_trans'].numpy()

                trans_kpts2d = trans_point2d(
                    np.transpose(kpts2d[:, :, 0:2], [2, 0, 1]),
                    inv_trans
                )
                trans_kpts2d = np.transpose(trans_kpts2d, [1, 2, 0])
                trans_kpts2d = np.concatenate([trans_kpts2d, kpts2d[..., 2:3]], axis=-1)

                if do_flip:
                    print(trans_kpts2d.shape)
                    trans_kpts2d[..., 0] = img.shape[1] - trans_kpts2d[..., 0] - 1
                    trans_kpts2d = trans_kpts2d[:, FLIPJOINTS, :]

                imgs = img[np.newaxis]
                _kpts2d = trans_kpts2d[np.newaxis]
                _track_ids = [exist_pids]
                posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug_trans', SKELETONS)

        return imgs, targets

    def get_muco(self, sample):
        _, filename, img_id, augmentation = sample
        # keys: 'filename', 'bbox', 'kpts2d', 'width', 'height', 'kpts3d', 'f', 'c'
        datum = self.muco_data[img_id]
        assert filename == datum['file_name']
        kpts2d = datum['kpts2d']
        depth = datum['kpts3d'][:, :, 2] / 1000  # mm to m
        num_person = kpts2d.shape[0]
        traj_id = np.arange(num_person, dtype=np.int32)
        track_ids = np.ones(num_person, dtype=np.int32)

        img = cv2.cvtColor(cv2.imread('{}/{}'.format(self.muco_data_dir, filename)), cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channels = img.shape
        img_shape = (img_width, img_height)

        aug_imgs, aug_kpts2d, aug_track_ids, aug_bbxes, aug_depth = [], [], [], [], []
        # 1. get augmentation params and apply for the whole sequence of images
        rot_list, do_flip, color_scale, bbxes_list, trans_list, inv_trans_list = \
            get_aug_config_coco(img_shape, self.input_shape, self.seq_l + self.future_seq_l, augmentation)
        # print(rot_list, do_flip, color_scale)
        for t in range(self.seq_l + self.future_seq_l):
            # 2. perform datasets augmentation (flip, rot, color scale)
            if t < self.seq_l:
                img_patch = generate_patch_image(img, do_flip, trans_list[t], self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                aug_imgs.append(img_patch)

            # 3. apply Affine Transform on keypoints
            kpt2d = np.transpose(kpts2d.copy(), [2, 0, 1])  # [3, n, num_joints]
            depth_t = depth.copy()
            if do_flip:
                kpt2d[0] = img_width - kpt2d[0] - 1
                kpt2d = kpt2d[:, :, FLIPJOINTS]
                depth_t = depth_t[:, FLIPJOINTS]
            kpt2d[0:2] = trans_point2d(kpt2d[0:2], trans_list[t])
            # kpt2d[2] *= (
            #         (kpt2d[0] >= 0) &
            #         (kpt2d[0] < self.input_shape[1]) &
            #         (kpt2d[1] >= 0) &
            #         (kpt2d[1] < self.input_shape[0])
            # )
            kpt2d = np.transpose(kpt2d, [1, 2, 0])  # [n, num_joints, 3]
            bboxes = self.bbox_2d_padded(kpt2d)

            # normalize
            kpt2d[..., 0] = kpt2d[..., 0] / self.input_shape[1]
            kpt2d[..., 1] = kpt2d[..., 1] / self.input_shape[0]

            depth_t = depth_t / self.max_depth

            aug_kpts2d.append(kpt2d)
            aug_track_ids.append(track_ids)
            aug_bbxes.append(bboxes)
            aug_depth.append(depth_t)

        kpts2d = np.stack(aug_kpts2d, axis=1)  # [n, T, num_joints, 3]
        track_ids = np.stack(aug_track_ids, axis=1)  # [n, T]
        bboxes = np.stack(aug_bbxes, axis=1)  # [n, T, 4]
        depths = np.stack(aug_depth, axis=1)  # [n, T, num_joints]

        exist_traj = np.sum(kpts2d[:, :, :, 2], axis=(1, 2)) > (self.seq_l + self.future_seq_l)
        kpts2d = kpts2d[exist_traj]
        track_ids = track_ids[exist_traj]
        bboxes = bboxes[exist_traj]
        traj_id = traj_id[exist_traj]
        depths = depths[exist_traj]
        depths = np.stack([depths, np.ones_like(depths)], axis=-1)  # [n, T, num_joints, 2]

        # collect data
        imgs = torch.from_numpy(np.concatenate(aug_imgs, axis=2)).float().permute(2, 0, 1)  # [T*3, H, W]

        targets = {}
        targets['kpts2d'] = torch.from_numpy(kpts2d).float()  # [n, T, num_joints, 3]
        targets['depth'] = torch.from_numpy(depths).float()  # [n, T, num_joints, 2]
        targets['bbxes'] = torch.from_numpy(bboxes).float()  # [n, T, 4]
        targets['track_ids'] = torch.from_numpy(track_ids)  # [n, T]
        targets['traj_ids'] = torch.from_numpy(traj_id)  # [n]
        targets['input_size'] = torch.from_numpy(np.array([self.input_shape[1], self.input_shape[0]]))
        targets['max_depth'] = torch.from_numpy(np.array(self.max_depth)).float()  # for evaluation

        if True:  # self.mode == 'val':
            targets['dataset'] = 'muco'
            targets['image_id'] = img_id
            targets['inv_trans'] = torch.from_numpy(inv_trans_list[0]).float()

            targets['bbxes_head'] = torch.tensor([0]).float()
            targets['index'] = 0
            targets['video_name'] = 0
            targets['filenames'] = [filename]
            targets['frame_indices'] = 0

            targets['cam_intr'] = torch.tensor([0]).float()
            targets['kpts3d'] = torch.tensor([0]).float()

        if self.vis:
            imgs = imgs.reshape(self.seq_l, 3, self.input_shape[0], self.input_shape[1]).permute(0, 2, 3, 1).numpy()
            # load future frames
            for j in range(self.seq_l, self.seq_l + self.future_seq_l):
                # augmentation
                img_patch = generate_patch_image(img, do_flip, trans_list[j], self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                # print(imgs.shape, img_patch.shape)
                imgs = np.concatenate([imgs, img_patch[np.newaxis]], axis=0)
            imgs = (imgs * 255).astype(np.uint8)  # [n, h, w, c]

            # prepare kpts for each frame
            img_size = targets['input_size'].numpy().reshape(1, 1, 2)
            _kpts2d, _track_ids = [], []
            for i in range(self.seq_l + self.future_seq_l):
                exist_person = targets['track_ids'][:, i].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                _track_ids.append(exist_pids)

                kpts2d = targets['kpts2d'][exist_person, i].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                _kpts2d.append(kpts2d)

            posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug', SKELETONS)

            if True:  # self.mode == 'val':
                # val
                img = cv2.cvtColor(cv2.imread('{}/{}'.format(self.muco_data_dir, filename)), cv2.COLOR_BGR2RGB)

                img_size = targets['input_size'].numpy().reshape(1, 1, 2)
                exist_person = targets['track_ids'][:, 0].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                kpts2d = targets['kpts2d'][exist_person, 0].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                inv_trans = targets['inv_trans'].numpy()

                trans_kpts2d = trans_point2d(
                    np.transpose(kpts2d[:, :, 0:2], [2, 0, 1]),
                    inv_trans
                )
                trans_kpts2d = np.transpose(trans_kpts2d, [1, 2, 0])
                trans_kpts2d = np.concatenate([trans_kpts2d, kpts2d[..., 2:3]], axis=-1)

                if do_flip:
                    print(trans_kpts2d.shape)
                    trans_kpts2d[..., 0] = img.shape[1] - trans_kpts2d[..., 0] - 1
                    trans_kpts2d = trans_kpts2d[:, FLIPJOINTS, :]

                imgs = img[np.newaxis]
                _kpts2d = trans_kpts2d[np.newaxis]
                _track_ids = [exist_pids]
                posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug_trans', SKELETONS)

                # draw 3D
                pid_count = kpts2d.shape[0]
                all_pids = np.arange(pid_count)
                depths = targets['depth'][exist_person, 0].numpy() * self.max_depth
                if do_flip:
                    depths = depths[:, FLIPJOINTS, :]
                poses3d = np.concatenate([trans_kpts2d[..., 0:2], depths[..., 0:1]], axis=-1)

                import matplotlib.pyplot as plt
                cmap = plt.get_cmap('rainbow')
                pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]

                fig = plt.figure(figsize=(20, 20))
                ax = fig.add_subplot(111, projection='3d')
                # print(frame_idx)
                for p, pid in enumerate(all_pids):
                    pid_idx = np.where(all_pids == pid)[0][0]
                    # draw pose
                    kpt_3d = poses3d[p]
                    for l in range(len(SKELETONS)):
                        i1 = SKELETONS[l][0]
                        i2 = SKELETONS[l][1]
                        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
                        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
                        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])
                        ax.plot(x, z, -y, color=pid_colors[pid_idx], linewidth=3, alpha=1)
                        ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], color=pid_colors[pid_idx],
                                   marker='o', s=5)
                        ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], color=pid_colors[pid_idx],
                                   marker='o', s=5)

                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Z Label')
                # ax.set_zlabel('Y Label')
                # ax.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xlim([0, img.shape[1]])
                ax.set_ylim([0, self.max_depth])
                ax.set_zlim([-img.shape[0], 0])
                # ax.legend()

                ax.view_init(20, -80)  # view_init(elev, azim)
                plt.savefig('../vis/pose3d.jpg', bbox_inches='tight')
                ax.view_init(80, -90)  # view_init(elev, azim)
                plt.savefig('../vis/pose3d_topdown.jpg', bbox_inches='tight')

        return imgs, targets

    def get_mupots(self, sample):
        _, filename, img_id, augmentation = sample
        assert filename == self.mupots_data[img_id]['file_name']
        gap = (self.seq_min_gap + self.seq_max_gap) // 2 + 1

        # collect datasets
        cam_intr = np.array([0, 0, 0, 0])
        imgs, kpts2d, kpts3d, occs, track_ids, filenames = [], [], [], [], [], []
        for i in range(self.seq_l + self.future_seq_l):
            datum = self.mupots_data[img_id + i * gap]
            _filename = datum['file_name']  # TS11/img_000517.jpg
            filenames.append(_filename)
            if i == 0:
                cam_intr = np.array(datum['intrinsic'])
            if i < self.seq_l:
                img = cv2.cvtColor(
                    cv2.imread('{}/MuPoTS-3D_images/{}'.format(self.muco_data_dir, _filename)), cv2.COLOR_BGR2RGB)
                imgs.append(img)

            track_id = datum['track_ids']
            if track_id.shape[0] == 0:
                # no annotation for this frame
                kpt2d = np.array([]).reshape(-1, self.num_joints, 3)
                kpt3d = np.array([]).reshape(-1, self.num_joints, 3)
            else:
                kpt2d = datum['kpts2d']
                kpt3d = datum['kpts3d'] / 1000
            # print(kpt2d.shape)

            track_ids.append(track_id)
            kpts2d.append(kpt2d)
            kpts3d.append(kpt3d)

        imgs = np.stack(imgs, axis=0)
        img_height, img_width, img_channels = imgs.shape[1:]

        # 1. get augmentation params and apply for the whole sequence of images
        rot, do_flip, color_scale, bbx, trans, inv_trans = \
            get_aug_config((img_width, img_height), self.input_shape, augmentation)
        # print(rot, do_flip, color_scale, bbx)
        # print(trans)

        aug_imgs, aug_kpts2d, aug_kpts3d, aug_track_ids, aug_depth, aug_bbxes = [], [], [], [], [], []
        for i in range(self.seq_l + self.future_seq_l):
            if i < self.seq_l:
                # 2. crop patch from img and perform datasets augmentation (flip, rot, color scale)
                img = imgs[i]
                img_patch = generate_patch_image(img, do_flip, trans, self.input_shape)

                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                aug_imgs.append(img_patch)

            # 3. apply flip and affine transform on keypoints
            kpt2d = kpts2d[i].copy()
            kpt3d = kpts3d[i].copy()
            kpt2d = np.transpose(kpt2d, [2, 0, 1])  # [3, n, num_joints]
            if do_flip:
                kpt2d[0] = img_width - kpt2d[0] - 1
                kpt2d = kpt2d[:, :, FLIPJOINTS]  # [3, n, num_joints]
                kpt3d = kpt3d[:, FLIPJOINTS, :]  # [n, num_joints, 3]
            kpt2d[0:2] = trans_point2d(kpt2d[0:2], trans)
            # kpt2d[2] *= (
            #         (kpt2d[0] >= 0) &
            #         (kpt2d[0] < self.input_shape[1]) &
            #         (kpt2d[1] >= 0) &
            #         (kpt2d[1] < self.input_shape[0])
            # )
            kpt2d = np.transpose(kpt2d, [1, 2, 0])  # [n, num_joints, 3]
            bboxes = self.bbox_2d_padded(kpt2d)

            # normalize
            kpt2d[..., 0] = kpt2d[..., 0] / self.input_shape[1]
            kpt2d[..., 1] = kpt2d[..., 1] / self.input_shape[0]

            depth = kpt3d[:, :, 2] / self.max_depth  # [n, num_joints]
            depths = np.stack([depth, np.ones_like(depth)], axis=-1)  # [n, num_joints, 2]

            aug_kpts2d.append(kpt2d)
            aug_kpts3d.append(kpt3d)  # [n, num_joints, 3]
            aug_track_ids.append(track_ids[i])
            aug_bbxes.append(bboxes)
            aug_depth.append(depths)

        all_ids = set(np.concatenate(aug_track_ids[0:self.seq_l], axis=0))
        if len(all_ids) == 0:
            max_id = -1
        else:
            max_id = max(all_ids)

        # filter out invalid target future poses
        for i in range(self.seq_l, self.seq_l + self.future_seq_l):
            valid_target_id = np.array([pid in all_ids for pid in aug_track_ids[i]], dtype=np.bool)
            aug_kpts2d[i] = aug_kpts2d[i][valid_target_id]
            aug_kpts3d[i] = aug_kpts3d[i][valid_target_id]
            aug_depth[i] = aug_depth[i][valid_target_id]
            aug_track_ids[i] = aug_track_ids[i][valid_target_id]
            aug_bbxes[i] = aug_bbxes[i][valid_target_id]

        kpts2d = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, self.num_joints, 3])
        kpts3d = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, self.num_joints, 3])
        depths = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, self.num_joints, 2])
        track_ids = np.zeros([max_id + 1, self.seq_l + self.future_seq_l], dtype=np.int32)
        bbxes = np.ones([max_id + 1, self.seq_l + self.future_seq_l, 4])

        for i in range(self.seq_l + self.future_seq_l):
            track_ids[aug_track_ids[i], i] = 1  # one-hot: exist in the frame or not
            kpts2d[aug_track_ids[i], i] = aug_kpts2d[i]
            kpts3d[aug_track_ids[i], i] = aug_kpts3d[i]
            depths[aug_track_ids[i], i] = aug_depth[i]
            bbxes[aug_track_ids[i], i] = aug_bbxes[i]

        # some persons will be missing in all frames
        exist_id = np.sum(track_ids, axis=1) > 0
        traj_id = np.where(exist_id)[0]
        track_ids = track_ids[exist_id]
        kpts2d = kpts2d[exist_id]
        kpts3d = kpts3d[exist_id]
        depths = depths[exist_id]
        bbxes = bbxes[exist_id]

        # collect data
        imgs = torch.from_numpy(np.concatenate(aug_imgs, axis=2)).float().permute(2, 0, 1)  # [T*3, H, W]

        targets = {}
        targets['kpts2d'] = torch.from_numpy(kpts2d).float()  # [n, T, num_joints, 3]
        targets['depth'] = torch.from_numpy(depths).float()  # [n, T, num_joints, 2]
        targets['bbxes'] = torch.from_numpy(bbxes).float()  # [n, T, 4]
        targets['track_ids'] = torch.from_numpy(track_ids)  # [n, T]
        targets['traj_ids'] = torch.from_numpy(traj_id)  # [n]
        targets['input_size'] = torch.from_numpy(np.array([self.input_shape[1], self.input_shape[0]]))
        targets['max_depth'] = torch.from_numpy(np.array(self.max_depth)).float()  # for evaluation

        if True: # self.mode == 'val':
            targets['dataset'] = 'mupots'
            targets['image_id'] = img_id
            targets['inv_trans'] = torch.from_numpy(inv_trans).float()

            targets['bbxes_head'] = torch.tensor([0]).float()
            targets['index'] = 0
            targets['video_name'] = 0
            targets['filenames'] = filenames
            targets['frame_indices'] = 0

            targets['cam_intr'] =torch.from_numpy(cam_intr).float()  # for evaluation
            targets['kpts3d'] = torch.from_numpy(kpts3d).float()  # [n, T, num_joints, 3] for evaluation mpjpe

        if self.vis:
            imgs = imgs.reshape(self.seq_l, 3, self.input_shape[0], self.input_shape[1]).permute(0, 2, 3, 1).numpy()
            # load future frames
            for j in range(self.seq_l, self.seq_l + self.future_seq_l):
                datum = self.mupots_data[img_id + j * gap]
                filename = datum['file_name']  # TS11/img_000517.jpg
                img = cv2.cvtColor(
                    cv2.imread('{}/MuPoTS-3D_images/{}'.format(self.muco_data_dir, filename)), cv2.COLOR_BGR2RGB)
                # augmentation
                img_patch = generate_patch_image(img, do_flip, trans, self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                # print(imgs.shape, img_patch.shape)
                imgs = np.concatenate([imgs, img_patch[np.newaxis]], axis=0)
            imgs = (imgs * 255).astype(np.uint8)  # [n, h, w, c]

            # prepare kpts for each frame
            img_size = targets['input_size'].numpy().reshape(1, 1, 2)
            _kpts2d, _track_ids = [], []
            for i in range(self.seq_l + self.future_seq_l):
                exist_person = targets['track_ids'][:, i].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                _track_ids.append(exist_pids)

                kpts2d = targets['kpts2d'][exist_person, i].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                _kpts2d.append(kpts2d)

            posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug', SKELETONS)

            if True:  # self.mode == 'val':
                # val
                filename = targets['filenames'][0]
                img = cv2.cvtColor(
                    cv2.imread('{}/MuPoTS-3D_images/{}'.format(self.muco_data_dir, filename)), cv2.COLOR_BGR2RGB)

                img_size = targets['input_size'].numpy().reshape(1, 1, 2)
                exist_person = targets['track_ids'][:, 0].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                kpts2d = targets['kpts2d'][exist_person, 0].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                inv_trans = targets['inv_trans'].numpy()

                trans_kpts2d = trans_point2d(
                    np.transpose(kpts2d[:, :, 0:2], [2, 0, 1]),
                    inv_trans
                )
                trans_kpts2d = np.transpose(trans_kpts2d, [1, 2, 0])
                trans_kpts2d = np.concatenate([trans_kpts2d, kpts2d[..., 2:3]], axis=-1)

                if do_flip:
                    print(trans_kpts2d.shape)
                    trans_kpts2d[..., 0] = img.shape[1] - trans_kpts2d[..., 0] - 1
                    trans_kpts2d = trans_kpts2d[:, FLIPJOINTS, :]

                imgs = img[np.newaxis]
                _kpts2d = trans_kpts2d[np.newaxis]
                _track_ids = [exist_pids]
                posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug_trans', SKELETONS)

                # draw 3D
                pid_count = kpts2d.shape[0]
                all_pids = np.arange(pid_count)
                depths = targets['depth'][exist_person, 0].numpy() * self.max_depth
                if do_flip:
                    depths = depths[:, FLIPJOINTS, :]
                poses3d = np.concatenate([trans_kpts2d[..., 0:2], depths[..., 0:1]], axis=-1)

                import matplotlib.pyplot as plt
                cmap = plt.get_cmap('rainbow')
                pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]

                fig = plt.figure(figsize=(20, 20))
                ax = fig.add_subplot(111, projection='3d')
                # print(frame_idx)
                for p, pid in enumerate(all_pids):
                    pid_idx = np.where(all_pids == pid)[0][0]
                    # draw pose
                    kpt_3d = poses3d[p]
                    for l in range(len(SKELETONS)):
                        i1 = SKELETONS[l][0]
                        i2 = SKELETONS[l][1]
                        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
                        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
                        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])
                        ax.plot(x, z, -y, color=pid_colors[pid_idx], linewidth=3, alpha=1)
                        ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], color=pid_colors[pid_idx],
                                   marker='o', s=5)
                        ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], color=pid_colors[pid_idx],
                                   marker='o', s=5)

                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Z Label')
                # ax.set_zlabel('Y Label')
                # ax.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xlim([0, img.shape[1]])
                ax.set_ylim([0, self.max_depth + 2])
                ax.set_zlim([-img.shape[0], 0])
                # ax.legend()

                ax.view_init(20, -80)  # view_init(elev, azim)
                plt.savefig('../vis/pose3d.jpg', bbox_inches='tight')
                ax.view_init(70, -90)  # view_init(elev, azim)
                plt.savefig('../vis/pose3d_topdown.jpg', bbox_inches='tight')

        return imgs, targets

    def get_jta(self, sample):
        _, seq, img_idx, subset, augmentation = sample
        gap = (self.seq_min_gap + self.seq_max_gap) // 2
        seq_idx = img_idx + gap * np.arange(self.seq_l + self.future_seq_l)
        # print(seq, seq_idx)

        # collect datasets
        imgs, kpts2d, kpts3d, occs, track_ids, filenames = [], [], [], [], [], []
        for idx, i in enumerate(seq_idx):
            filenames.append('{}/{:03d}.jpg'.format(seq, i))
            if idx < self.seq_l:
                img = cv2.cvtColor(
                    cv2.imread('{}/images_half/{}/{}/{:03d}.jpg'.format(self.jta_data_dir, subset, seq, i)),
                    cv2.COLOR_BGR2RGB)
                # img = np.load('{}/images_np/{}/{}/{:03d}.npy'.format(self.jta_data_dir, subset, seq, i))
                imgs.append(img)

            # collect poses for current and future frames
            fname = '{}/ann_split/{}/{}/{:03d}.json'.format(self.jta_data_dir, subset, seq, i)
            with open(fname, 'r') as json_file:
                ann = json.load(json_file)

            # print(list(ann.keys()))
            track_id = np.array(list(ann.keys()), dtype=np.int32)
            if track_id.shape[0] == 0:
                # no annotation for this frame
                kpt2d, kpt3d, occ = [], [], []
            else:
                kpt2d, kpt3d, occ = zip(*list(ann.values()))
            kpt2d = np.reshape(np.array(kpt2d), [-1, 22, 2])[:, JTA2JOINT15, :] / 2
            kpt3d = np.reshape(np.array(kpt3d), [-1, 22, 3])[:, JTA2JOINT15, :]
            occ = np.reshape(np.array(occ), [-1, 22, 2])[:, JTA2JOINT15, :]

            # filter out occluded person
            vis_person = np.sum(occ[:, :, 0], axis=-1) < self.num_joints * 0.75  # [n]
            # print('{} / {}'.format(vis_person.shape[0], np.sum(vis_person)))

            track_ids.append(track_id[vis_person])
            kpts2d.append(kpt2d[vis_person])
            kpts3d.append(kpt3d[vis_person])
            occs.append(occ[vis_person])

        imgs = np.stack(imgs, axis=0)
        img_height, img_width, img_channels = imgs.shape[1:]
        # assert img_height == self.input_shape[0]
        # assert img_width == self.input_shape[1]
        # print(img_height, img_width)

        # 1. get augmentation params and apply for the whole sequence of images
        rot, do_flip, color_scale, bbx, trans, inv_trans = \
            get_aug_config((img_width, img_height), self.input_shape, augmentation)
        # print(rot, do_flip, color_scale, bbx)
        # print(trans)

        aug_imgs, aug_kpts2d, aug_track_ids, aug_kpts3d, aug_occs, aug_bbxes, aug_depth = [], [], [], [], [], [], []
        for i in range(self.seq_l + self.future_seq_l):
            if i < self.seq_l:
                # 2. crop patch from img and perform datasets augmentation (flip, rot, color scale)
                img = imgs[i]
                img_patch = generate_patch_image(img, do_flip, trans, self.input_shape)

                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                aug_imgs.append(img_patch)

            # 3. apply flip and affine transform on keypoints
            kpt2d = kpts2d[i].copy()
            kpt3d = kpts3d[i].copy()
            kpt2d = np.transpose(kpt2d, [2, 0, 1])
            # append confidence [3, n, num_joints]
            kpt2d = np.concatenate([kpt2d, np.ones([1, kpt2d.shape[1], kpt2d.shape[2]])], axis=0)
            if do_flip:
                kpt2d[0] = img_width - kpt2d[0] - 1
                kpt2d = kpt2d[:, :, FLIPJOINTS]  # [3, n, num_joints]
                kpt3d = kpt3d[:, FLIPJOINTS, :]  # [n, num_joints, 3]
            kpt2d[0:2] = trans_point2d(kpt2d[0:2], trans)
            # kpt2d[2] *= (
            #         (kpt2d[0] >= 0) &
            #         (kpt2d[0] < self.input_shape[1]) &
            #         (kpt2d[1] >= 0) &
            #         (kpt2d[1] < self.input_shape[0])
            # )
            kpt2d = np.transpose(kpt2d, [1, 2, 0])  # [n, num_joints, 3]
            bbxes = self.bbox_2d_padded(kpt2d)
            valid_kpts = kpt2d[:, 0, 2] > 0
            area = bbxes[:, 2] * bbxes[:, 3]
            valid_area = area > 10
            valid_depth = kpt3d[:, 0, 2] < self.max_depth
            valid = valid_depth & valid_kpts & valid_area

            kpt2d = kpt2d[valid]
            bbxes = bbxes[valid]
            kpt3d = kpt3d[valid]
            # print('{} / {}'.format(valid.shape[0], np.sum(valid)))

            # update occlusion
            occ = occs[i][valid]  # [n, num_joints, 2]
            if self.mode == 'test':
                # ~occ & ~self-occ & out of scope
                # kpt2d[..., 2] = (occ[..., 0] == 0) & (occ[..., 1] == 0) & (kpt2d[..., 2] == 1)
                # kpt2d[..., 2] = (occ[..., 0] == 0) & (kpt2d[..., 2] == 1)
                kpt2d[..., 2] = (kpt2d[..., 2] == 1)
            else:
                # occ & out of scope
                # kpt2d[..., 2] = (occ[..., 0] == 0) & (kpt2d[..., 2] == 1)
                kpt2d[..., 2] = (kpt2d[..., 2] == 1)

            # normalize
            kpt2d[..., 0] = kpt2d[..., 0] / self.input_shape[1]
            kpt2d[..., 1] = kpt2d[..., 1] / self.input_shape[0]

            depth = kpt3d[..., 2].copy() / self.max_depth
            depths = np.stack([depth, np.ones_like(depth)], axis=-1)  # [n, num_joints, 2]

            # update track id
            track_id = track_ids[i][valid]

            # collect datasets
            aug_kpts2d.append(kpt2d)  # [n, num_joints, 3]
            aug_kpts3d.append(kpt3d)  # [n, num_joints, 3]
            aug_depth.append(depths)  # [n, num_joints, 2]
            aug_track_ids.append(track_id)  # [n]
            aug_occs.append(occ)  # [n, num_joints, 2]
            aug_bbxes.append(bbxes)  # [n, 4]
            # print('before aggregate', i, track_id)

        # generate human trajectory
        # aug_track_ids: [(1,2,3,4), (2,3,4,6), ..., (3,4,6)]
        # print(aug_track_ids)
        all_ids = set(np.concatenate(aug_track_ids[0:self.seq_l], axis=0))
        if len(all_ids) == 0:
            max_id = -1
        else:
            max_id = max(all_ids)

        # filter out invalid target future poses
        for i in range(self.seq_l, self.seq_l + self.future_seq_l):
            valid_target_id = np.array([pid in all_ids for pid in aug_track_ids[i]], dtype=np.bool)
            aug_kpts2d[i] = aug_kpts2d[i][valid_target_id]
            aug_kpts3d[i] = aug_kpts3d[i][valid_target_id]
            aug_depth[i] = aug_depth[i][valid_target_id]
            aug_track_ids[i] = aug_track_ids[i][valid_target_id]
            aug_occs[i] = aug_occs[i][valid_target_id]
            aug_bbxes[i] = aug_bbxes[i][valid_target_id]

        kpts2d = np.zeros([max_id+1, self.seq_l + self.future_seq_l, self.num_joints, 3])
        kpts3d = np.zeros([max_id+1, self.seq_l + self.future_seq_l, self.num_joints, 3])
        depths = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, self.num_joints, 2])
        track_ids = np.zeros([max_id+1, self.seq_l + self.future_seq_l], dtype=np.int32)
        occs = np.ones([max_id+1, self.seq_l + self.future_seq_l, self.num_joints, 2], dtype=np.int32)
        bbxes = np.ones([max_id+1, self.seq_l + self.future_seq_l, 4])

        for i in range(self.seq_l + self.future_seq_l):
            track_ids[aug_track_ids[i], i] = 1  # one-hot: exist in the frame or not
            occs[aug_track_ids[i], i] = aug_occs[i]
            kpts2d[aug_track_ids[i], i] = aug_kpts2d[i]
            kpts3d[aug_track_ids[i], i] = aug_kpts3d[i]
            depths[aug_track_ids[i], i] = aug_depth[i]
            bbxes[aug_track_ids[i], i] = aug_bbxes[i]

        # some persons will be missing in all frames
        exist_id = np.sum(track_ids, axis=1) > 0
        traj_id = np.where(exist_id)[0]
        track_ids = track_ids[exist_id]
        occs = occs[exist_id]
        kpts2d = kpts2d[exist_id]
        kpts3d = kpts3d[exist_id]
        depths = depths[exist_id]
        bbxes = bbxes[exist_id]

        # collect data
        imgs = torch.from_numpy(np.concatenate(aug_imgs, axis=2)).float().permute(2, 0, 1)  # [T*3, H, W]

        targets = {}
        targets['kpts2d'] = torch.from_numpy(kpts2d).float()   # [n, T, 1, 3]
        targets['depth'] = torch.from_numpy(depths).float()  # [n, T, num_joints, 2]
        targets['traj_ids'] = torch.from_numpy(traj_id)  # [n]
        targets['track_ids'] = torch.from_numpy(track_ids)  # [n, T]
        targets['bbxes'] = torch.from_numpy(bbxes).float()  # [n, T, 4] for evaluation
        targets['input_size'] = torch.from_numpy(np.array([self.input_shape[1], self.input_shape[0]])).float()  # [2]
        targets['max_depth'] = torch.from_numpy(np.array(self.max_depth)).float()  # for evaluation

        if True:  # self.mode == 'test':
            # for evaluation
            targets['dataset'] = 'jta'
            targets['image_id'] = img_idx
            targets['inv_trans'] = torch.from_numpy(inv_trans).float()

            targets['bbxes_head'] = torch.tensor([0]).float()
            targets['index'] = 0
            targets['video_name'] = 0
            targets['frame_indices'] = 0
            targets['filenames'] = filenames

            cam_intr = np.array([1158, 1158, 960, 540]) / 2  # half of original resolution
            targets['cam_intr'] =torch.from_numpy(cam_intr).float()  # for evaluation
            # targets['occs'] = torch.from_numpy(occs).float()  # [n, T, num_joints, 2] for debug
            targets['kpts3d'] = torch.from_numpy(kpts3d).float()  # [n, T, num_joints, 3]

        if self.vis:
            imgs = imgs.reshape(self.seq_l, 3, self.input_shape[0], self.input_shape[1]).permute(0, 2, 3, 1).numpy()
            # load future frames
            for j in range(self.seq_l, self.seq_l + self.future_seq_l):
                frame_idx = img_idx + j * gap
                img = cv2.cvtColor(
                    cv2.imread('{}/images_half/{}/{}/{:03d}.jpg'.format(self.jta_data_dir, subset, seq, frame_idx)),
                    cv2.COLOR_BGR2RGB)
                # augmentation
                img_patch = generate_patch_image(img, do_flip, trans, self.input_shape)
                for j in range(img_channels):
                    img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
                # print(imgs.shape, img_patch.shape)
                imgs = np.concatenate([imgs, img_patch[np.newaxis]], axis=0)
            imgs = (imgs * 255).astype(np.uint8)  # [n, h, w, c]

            # prepare kpts for each frame
            img_size = targets['input_size'].numpy().reshape(1, 1, 2)
            _kpts2d, _track_ids = [], []
            for i in range(self.seq_l + self.future_seq_l):
                exist_person = targets['track_ids'][:, i].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                _track_ids.append(exist_pids)

                kpts2d = targets['kpts2d'][exist_person, i].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                _kpts2d.append(kpts2d)

            posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug', SKELETONS)

            if True:  # self.mode == 'val':
                # val
                seq, seq_idx = targets['filenames'][0].split('/')
                print(seq, seq_idx)
                img = cv2.cvtColor(
                    cv2.imread('{}/images_half/{}/{}/{}'.format(self.jta_data_dir, subset, seq, seq_idx)),
                    cv2.COLOR_BGR2RGB)

                img_size = targets['input_size'].numpy().reshape(1, 1, 2)
                exist_person = targets['track_ids'][:, 0].numpy() > 0
                exist_pids = targets['traj_ids'][exist_person].numpy()
                kpts2d = targets['kpts2d'][exist_person, 0].numpy()
                kpts2d[..., 0:2] = kpts2d[..., 0:2] * img_size
                inv_trans = targets['inv_trans'].numpy()

                trans_kpts2d = trans_point2d(
                    np.transpose(kpts2d[:, :, 0:2], [2, 0, 1]),
                    inv_trans
                )
                trans_kpts2d = np.transpose(trans_kpts2d, [1, 2, 0])
                trans_kpts2d = np.concatenate([trans_kpts2d, kpts2d[..., 2:3]], axis=-1)

                if do_flip:
                    print(trans_kpts2d.shape)
                    trans_kpts2d[..., 0] = img.shape[1] - trans_kpts2d[..., 0] - 1
                    trans_kpts2d = trans_kpts2d[:, FLIPJOINTS, :]

                imgs = img[np.newaxis]
                _kpts2d = trans_kpts2d[np.newaxis]
                _track_ids = [exist_pids]
                posetrack_visualization(imgs, _kpts2d, _track_ids, 'aug_trans', SKELETONS)

                # draw 3D
                pid_count = kpts2d.shape[0]
                all_pids = np.arange(pid_count)
                depths = targets['depth'][exist_person, 0].numpy() * self.max_depth
                if do_flip:
                    depths = depths[:, FLIPJOINTS, :]
                poses3d = np.concatenate([trans_kpts2d[..., 0:2], depths[..., 0:1]], axis=-1)

                import matplotlib.pyplot as plt
                cmap = plt.get_cmap('rainbow')
                pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]

                fig = plt.figure(figsize=(20, 20))
                ax = fig.add_subplot(111, projection='3d')
                # print(frame_idx)
                for p, pid in enumerate(all_pids):
                    pid_idx = np.where(all_pids == pid)[0][0]
                    # draw pose
                    kpt_3d = poses3d[p]
                    for l in range(len(SKELETONS)):
                        i1 = SKELETONS[l][0]
                        i2 = SKELETONS[l][1]
                        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
                        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
                        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])
                        ax.plot(x, z, -y, color=pid_colors[pid_idx], linewidth=3, alpha=1)
                        ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], color=pid_colors[pid_idx],
                                   marker='o', s=5)
                        ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], color=pid_colors[pid_idx],
                                   marker='o', s=5)

                # ax.set_xlabel('X Label')
                # ax.set_ylabel('Z Label')
                # ax.set_zlabel('Y Label')
                # ax.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xlim([0, img.shape[1]])
                ax.set_ylim([0, self.max_depth + 2])
                ax.set_zlim([-img.shape[0], 0])
                # ax.legend()

                ax.view_init(20, -80)  # view_init(elev, azim)
                plt.savefig('../vis/pose3d.jpg', bbox_inches='tight')
                ax.view_init(70, -90)  # view_init(elev, azim)
                plt.savefig('../vis/pose3d_topdown.jpg', bbox_inches='tight')

        return imgs, targets

    def get_panoptic(self, sample):
        _, seq_name, cam_idx, frame_idx, index = sample
        # print(seq_name, cam_idx, frame_idx)
        cam = self.panoptic_data['{}-cam{:02d}'.format(seq_name, cam_idx)]
        cam_intr = cam['intr'] * 0.5
        cam_dist = cam['distCoef']

        if self.mode == 'train':
            gap = np.random.randint(self.seq_min_gap, self.seq_max_gap + 1)
        else:
            gap = (self.seq_min_gap + self.seq_max_gap) // 2

        imgs, kpts2d, kpts3d, track_ids, frame_indices, indices, filenames = [], [], [], [], [], [], []
        for j in range(self.seq_l + self.future_seq_l):
            # frame_idx, poses, track_ids, all_cams
            _frame_idx, pose, track_id, _ = \
                self.panoptic_data['{}-poses'.format(seq_name)][index + j * gap]
            assert (frame_idx + j * gap) == _frame_idx
            frame_indices.append(_frame_idx)
            indices.append(index + j * gap)
            filenames.append('cam{:02d}_{}'.format(cam_idx, seq_name))

            if j < self.seq_l:
                img = cv2.cvtColor(
                    cv2.imread('{}/{}/hdImgs/hd_00_{:02d}/{:08d}.jpg'.
                               format(self.panoptic_data_dir, seq_name, cam_idx, _frame_idx)),
                    cv2.COLOR_BGR2RGB)
                # img = np.load('{}/images_np/{}/{}/{:03d}.npy'.format(self.data_dir, self.mode, seq, i))
                imgs.append(img)

            if pose == []:
                kpt2d = np.array([]).reshape([0, self.num_joints, 3])
                kpt3d = np.array([]).reshape([0, self.num_joints, 3])
                track_id = np.array([]).astype(np.int64)
            else:
                # pose [n, num_joints, 3]
                cam_t = np.expand_dims(cam['t'].T, axis=0)  # [1, 1, 3]
                _pt3d = 10 * (np.dot(pose[..., 0:3], cam['R'].T) + cam_t)
                _pt2d = self.projection(_pt3d, cam_intr, cam_dist, simple_mode=False)
                kpt2d = np.concatenate([_pt2d[..., 0:2], pose[..., 3:4] > 0.1], axis=-1)  # [n, 15, 3] pose2d + visib
                kpt3d = _pt3d / 1000.  # [n, 15, 3] mm -> m

            # print(seq_name, _frame_idx, kpt2d.shape)
            kpts2d.append(kpt2d)
            kpts3d.append(kpt3d)
            track_ids.append(track_id)

        imgs = np.stack(imgs, axis=0)
        img_height, img_width, img_channels = imgs.shape[1:]
        assert img_height == self.input_shape[0]
        assert img_width == self.input_shape[1]

        # generate human trajectory
        # aug_track_ids: [(1,2,3,4), (2,3,4,6), ..., (3,4,6)]
        # aug_traj_ids: [1,2,3,4,6]
        # print(aug_track_ids)
        all_ids = set(np.concatenate(track_ids[0:self.seq_l], axis=0))
        if len(all_ids) == 0:
            max_id = -1
        else:
            max_id = max(all_ids)
        # print(track_ids, all_ids)

        # filter out invalid target future poses
        for i in range(self.seq_l, self.seq_l + self.future_seq_l):
            valid_target_id = np.array([pid in all_ids for pid in track_ids[i]], dtype=np.bool)
            kpts2d[i] = kpts2d[i][valid_target_id]
            kpts3d[i] = kpts3d[i][valid_target_id]
            track_ids[i] = track_ids[i][valid_target_id]

        tmporal_kpts2d = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, self.num_joints, 3])
        tmporal_kpts3d = np.zeros([max_id + 1, self.seq_l + self.future_seq_l, self.num_joints, 3])
        tmporal_track_ids = np.zeros([max_id + 1, self.seq_l + self.future_seq_l], dtype=np.int32)

        for i in range(self.seq_l + self.future_seq_l):
            tmporal_track_ids[track_ids[i], i] = 1  # one-hot: exist in the frame or not
            tmporal_kpts2d[track_ids[i], i] = kpts2d[i]
            tmporal_kpts3d[track_ids[i], i] = kpts3d[i]

        # some persons will be missing in all frames
        exist_id = np.sum(tmporal_track_ids, axis=1) > 0
        tmporal_traj_id = np.where(exist_id)[0]
        tmporal_track_ids = tmporal_track_ids[exist_id]
        tmporal_kpts2d = tmporal_kpts2d[exist_id]  # [n, T, num_joints, 3]
        tmporal_kpts3d = tmporal_kpts3d[exist_id]  # [n, T, num_joints, 3]
        bboxes = []
        for i in range(self.seq_l + self.future_seq_l):
            bboxes.append(self.bbox_2d_padded(tmporal_kpts2d[:, i]))
        bboxes = np.stack(bboxes, axis=1)  # [n, T, 4]

        # normalize
        imgs = imgs.astype(np.float32) / 255.
        tmporal_kpts2d[..., 0] = tmporal_kpts2d[..., 0] / self.input_shape[1]
        tmporal_kpts2d[..., 1] = tmporal_kpts2d[..., 1] / self.input_shape[0]
        tmporal_kpts3d[..., 2] = tmporal_kpts3d[..., 2] / self.max_depth
        tmporal_depth = tmporal_kpts3d[..., 2:3].copy()

        imgs = torch.from_numpy(imgs).float().permute(0, 3, 1, 2).flatten(0, 1)  # [T, H, W, 3] -> [T*3, H, W]

        targets = {}
        targets['kpts2d'] = torch.from_numpy(tmporal_kpts2d).float()  # [n, T, num_joints, 3]
        targets['depth'] = torch.from_numpy(tmporal_depth).float()  # [n, T, num_joints, 3]
        targets['traj_ids'] = torch.from_numpy(tmporal_traj_id)  # [n]
        targets['track_ids'] = torch.from_numpy(tmporal_track_ids)  # [n, T]
        targets['bbxes'] = torch.from_numpy(bboxes).float()  # [n, T, 4]
        targets['input_size'] = torch.from_numpy(np.array([self.input_shape[1], self.input_shape[0]])).float()  # [2]
        targets['max_depth'] = torch.from_numpy(np.array(self.max_depth)).float()  # for evaluation

        if True:  # self.mode == 'val':
            # for evaluation
            targets['dataset'] = 'panoptic'
            targets['seq_name'] = seq_name
            targets['cam_idx'] = cam_idx
            targets['filenames'] = filenames
            targets['indices'] = indices
            targets['image_id'] = frame_idx

            targets['bbxes_head'] = torch.tensor([0]).float()
            targets['index'] = 0
            targets['video_name'] = 0
            targets['frame_indices'] = frame_indices
            targets['inv_trans'] = torch.tensor([0]).float()

            targets['cam_intr'] = torch.from_numpy(cam_intr).float()  # for evaluation
            targets['cam_dist'] = torch.from_numpy(cam_dist).float()
            targets['kpts3d'] = torch.from_numpy(tmporal_kpts3d).float()  # [n, T, num_joints, 3]

        if self.vis:
            _imgs = np.transpose(imgs.numpy().reshape(self.seq_l, 3, img_height, img_width), [0, 2, 3, 1])
            for i in range(self.seq_l + self.future_seq_l):
                exist_person = targets['track_ids'][:, i].numpy() > 0
                kpt2d = targets['kpts2d'][exist_person, i].numpy()  # [n, num_joints, 3]
                kpt2d[..., 0] = kpt2d[..., 0] * self.input_shape[1]
                kpt2d[..., 1] = kpt2d[..., 1] * self.input_shape[0]
                kpt3d = targets['kpts3d'][exist_person, i].numpy()
                kpt3d[..., 2] = kpt3d[..., 2] * self.max_depth
                exist_ids = targets['traj_ids'][exist_person].numpy()
                traj_ids = targets['traj_ids'].numpy()

                if i < self.seq_l:
                    img = (_imgs[i] * 255).astype(np.uint8)
                else:
                    # print(self.panoptic_data.keys())
                    _frame_idx = self.panoptic_data['{}-poses'.format(seq_name)][index + i * gap][0]

                    img = cv2.cvtColor(
                        cv2.imread('{}/{}/hdImgs/hd_00_{:02d}/{:08d}.jpg'.
                                   format(self.panoptic_data_dir, seq_name, cam_idx, _frame_idx)),
                        cv2.COLOR_BGR2RGB)

                # print(kpt3d.shape)
                # intr_param = targets['cam_intr'].numpy()
                # dist_coeff = targets['cam_dist'].numpy()
                # kpt2d = np.stack([self.projection(kpt3d[i], intr_param, dist_coeff) for i in range(kpt3d.shape[0])], axis=0)

                # img, kpts2d, traj_ids, exist_ids, seq_name, frame_idx, skeletons, save_dir=None
                seq_name = targets['seq_name']
                frame_idx = targets['filenames'][i]
                panoptic_visualization(img, kpt2d, traj_ids, exist_ids, seq_name, frame_idx, SKELETONS, save_dir=None)

        return imgs, targets

    @staticmethod
    def bbox_2d_padded(kpts2d, h_inc_perc=0.15, w_inc_perc=0.15):
        """
        kpts2d: [n, num_joints, 3]
        :return: bounding box around the pose in format [x_min, y_min, width, height]
            - x_min = x of the top left corner of the bounding box
            - y_min = y of the top left corner of the bounding box
        """
        if kpts2d.shape[0] == 0:
            return np.ones([0, 4])

        bbxes = []
        for i in range(kpts2d.shape[0]):
            vis = kpts2d[i, :, 2] > 0
            if np.sum(vis) == 0:
                bbxes.append(np.asarray([1, 1, 1, 1]))
                continue

            kp = kpts2d[i, vis, 0:2]
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

            bbxes.append(np.asarray([x_min, y_min, width, height]))
        bbxes = np.stack(bbxes, axis=0)  # [n, 4]
        return bbxes

    @staticmethod
    def projection(xyz, intr_param, dist_coeff, simple_mode=False):
        # xyz: [N, 3]
        # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
        assert xyz.shape[-1] == 3
        fx, fy, cx, cy = intr_param

        if not simple_mode:
            k1, k2, p1, p2, k3 = dist_coeff
            k4, k5, k6 = 0, 0, 0

            x_p = xyz[..., 0] / xyz[..., 2]
            y_p = xyz[..., 1] / xyz[..., 2]
            r2 = x_p ** 2 + y_p ** 2

            a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
            b = b + (b == 0)
            d = a / b

            x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
            y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

            u = fx * x_pp + cx
            v = fy * y_pp + cy
            d = xyz[..., 2]

            return np.stack([u, v, d], axis=-1)
        else:
            u = xyz[..., 0] / xyz[..., 2] * fx + cx
            v = xyz[..., 1] / xyz[..., 2] * fy + cy
            d = xyz[..., 2]

            return np.stack([u, v, d], axis=-1)

    def get_labelled_seq(self):
        if self.mode == 'train':
            all_seqs = []
            count = 0

            posetrack_data = {}

            if self.use_posetrack:
                # load posetrack train dataset
                posetrack_data = pickle.load(open('{}/{}_filled.pkl'.format(self.posetrack_dir, self.mode), 'rb'))
                for fn, data_seq in posetrack_data.items():
                    start_frame_idx = int(data_seq[0]['filename'].split('/')[-1].split('.')[0])
                    end_frame_idx = int(data_seq[-1]['filename'].split('/')[-1].split('.')[0])
                    n = end_frame_idx - start_frame_idx + 1
                    # # check intermediate missing labeled frame
                    # if n != len(data_seq):
                    #     print(fn)
                    #     for datum in data_seq:
                    #         print(datum['filename'])

                    # print(fn, n, len(data_seq), data_seq[0]['filename'], data_seq[-1]['filename'])
                    for i in range(n):
                        # print(data_seq[i]['filename'])
                        filename = data_seq[i]['filename']
                        frame_idx = int(filename.split('/')[-1].split('.')[0])
                        max_valid_gap = self.seq_max_gap
                        while True:
                            if (i + max_valid_gap * (self.seq_l + self.future_seq_l)) <= n:
                                all_seqs.append(('posetrack', fn, filename, frame_idx, i, max_valid_gap, True))
                                break
                            else:
                                max_valid_gap = max_valid_gap - 1

                            if max_valid_gap == (self.seq_min_gap - 1):
                                break

                # _data = pickle.load(open('{}/val.pkl'.format(self.posetrack_dir, self.mode), 'rb'))
                # for fn, data_seq in _data.items():
                #     if fn == 'categories':
                #         continue
                #     start_frame_idx = int(data_seq[0]['filename'].split('/')[-1].split('.')[0])
                #     end_frame_idx = int(data_seq[-1]['filename'].split('/')[-1].split('.')[0])
                #     n = end_frame_idx - start_frame_idx + 1
                #     val_gap = 1 if self.seq_l == 1 else 4
                #
                #     # print(fn, n, len(data_seq), data_seq[0]['filename'], data_seq[-1]['filename'])
                #     for i in range(n):
                #         # print(data_seq[i]['filename'])
                #         filename = data_seq[i]['filename']
                #         frame_idx = int(filename.split('/')[-1].split('.')[0])
                #         is_label = data_seq[i]['is_label']
                #
                #         if (i + val_gap * (self.seq_l + self.future_seq_l)) <= n:
                #             if self.seq_l > 1:
                #                 if (i // self.seq_l) % self.seq_l == 0:
                #                     _is_label = False
                #                     for k in range(self.seq_l + self.future_seq_l):
                #                         _is_label = _is_label | data_seq[i + k * val_gap]['is_label']
                #                     if _is_label:
                #                         all_seqs.append(('posetrack', fn, filename, frame_idx, i, val_gap, False))
                #             else:
                #                 # seq_l = 1
                #                 if is_label:
                #                     all_seqs.append(('posetrack', fn, filename, frame_idx, i, val_gap, False))
                # posetrack_data.update(_data)
                print('posetrack: {}'.format(len(all_seqs) - count))
                count = len(all_seqs)

            # load coco train dataset
            coco_data = None

            if self.use_coco:
                coco_data = pickle.load(open('{}/coco_train.pkl'.format(self.coco_data_dir, self.mode), 'rb'))
                # _coco_data = pickle.load(open('{}/coco_val.pkl'.format(self.coco_data_dir, self.mode), 'rb'))
                # coco_data = coco_data + _coco_data
                for i in range(len(coco_data)):
                    filename = coco_data[i]['filename']
                    # if int(filename.split('/')[1].split('.')[0]) != 35166:
                    #     continue
                    all_seqs.append(('coco', filename, i, True))
                print('coco: {}'.format(len(all_seqs) - count))
                count = len(all_seqs)

            # load muco
            muco_data = None

            if self.use_muco:
                # depth_list = []
                muco_data = pickle.load(open('{}/MuCo-3DHP.pkl'.format(self.muco_data_dir, self.mode), 'rb'))
                for img_id in muco_data.keys():
                    filename = muco_data[img_id]['file_name']
                    all_seqs.append(('muco', filename, img_id, True))
                    # depth_list.append(muco_data[img_id]['kpts3d'][:, :, 2])
                # depth_list = np.concatenate(depth_list, axis=0)
                # print('max_depth', np.max(depth_list))
                print('muco: {}'.format(len(all_seqs) - count))
                count = len(all_seqs)

            # load mupots
            mupots_data = None

            if self.use_jta:
                # load jta
                mode = 'train'
                with open('{}/jta_all_ann_files_no_moving_camera.json'.format(self.jta_data_dir), 'r') as json_file:
                    tmp = json.load(json_file)[mode]
                seq_g = (self.seq_max_gap + self.seq_min_gap) // 2 + 1
                for seq, img_ids in tmp.items():
                    # if seq != 'seq_274':
                    #     continue
                    if self.seq_l == 1:
                        img_idx = np.arange(0, len(img_ids) - (self.seq_l + self.future_seq_l + 1) * seq_g, seq_g)
                    else:
                        img_idx = np.arange(0, len(img_ids) - (self.seq_l + self.future_seq_l + 1) * seq_g, seq_g)
                    samples = list(zip(['jta'] * len(img_idx), [seq] * len(img_idx), img_idx,
                                       [mode] * len(img_idx), [False] * len(img_idx)))
                    all_seqs += samples
                print('jta: {}'.format(len(all_seqs) - count))

            # load cmu panoptic data
            panoptic_data = None

            if self.use_panoptic:
                if self.panoptic_protocol == 1:
                    fname = '{}/panoptic_all_ann_files_protocol{}.pkl' \
                        .format(self.panoptic_data_dir, self.panoptic_protocol)
                    with open(fname, 'rb') as f:
                        # '{}-poses'.format(seq): [frame_idx, poses, track_ids, all_cams], '{}-cam{:02d}'.format(seq)
                        panoptic_data = pickle.load(f)

                    test_sequence = ['170221_haggling_b1', '170221_haggling_b2', '170221_haggling_b3',
                                     '170228_haggling_b1', '170228_haggling_b2', '170228_haggling_b3']
                    all_cams = np.array([3, 12, 23])
                    for k, v in panoptic_data.items():
                        if 'poses' in k:
                            seq_name = k.split('-')[0]
                            if seq_name in test_sequence:
                                continue

                            n = len(v) - self.seq_max_gap * (self.seq_l + self.future_seq_l)
                            print('{}: {} samples, {} cams'.format(k, n, len(all_cams)))
                            for cam_idx in all_cams:
                                for index in range(n):
                                    frame_idx = v[index][0]
                                    all_seqs.append(('panoptic', seq_name, int(cam_idx), frame_idx, index))
                elif self.panoptic_protocol == 2:
                    fname = '{}/panoptic_all_ann_files_protocol{}.pkl' \
                        .format(self.panoptic_data_dir, self.panoptic_protocol)
                    with open(fname, 'rb') as f:
                        # '{}-poses'.format(seq): [frame_idx, poses, track_ids, all_cams], '{}-cam{:02d}'.format(seq)
                        panoptic_data = pickle.load(f)

                    test_cam_idx = [16, 30]
                    for k, v in panoptic_data.items():
                        if 'poses' in k:
                            seq_name = k.split('-')[0]

                            n = len(v) - self.seq_max_gap * (self.seq_l + self.future_seq_l)
                            all_cams = v[0][-1]
                            print('{}: {} samples, {} cams'.format(k, n, len(all_cams)))
                            for cam_idx in all_cams:
                                if cam_idx in test_cam_idx:
                                    continue

                                for index in range(n):
                                    frame_idx = v[index][0]
                                    all_seqs.append(('panoptic', seq_name, int(cam_idx), frame_idx, index))
                else:
                    raise ValueError('Panoptic protocol error {}'.format(self.panoptic_protocol))
                print('cmu panoptic: {}'.format(len(all_seqs) - count))
                count = len(all_seqs)

        else:
            # val
            coco_data, posetrack_data = None, None
            all_seqs = []
            count = 0

            # coco_data = None
            # assert self.seq_l == 1
            # coco_data = pickle.load(open('{}/coco_val.pkl'.format(self.coco_data_dir, self.mode), 'rb'))
            # for i in range(len(coco_data)):
            #     filename = coco_data[i]['filename']
            #     # if filename != 'val2017/000000043816.jpg':
            #     #     continue
            #     all_seqs.append(('coco', filename, i, False))

            if self.use_posetrack:
                posetrack_data = pickle.load(open('{}/val.pkl'.format(self.posetrack_dir), 'rb'))
                for fn, data_seq in posetrack_data.items():
                    if fn == 'categories':
                        continue
                    start_frame_idx = int(data_seq[0]['filename'].split('/')[-1].split('.')[0])
                    end_frame_idx = int(data_seq[-1]['filename'].split('/')[-1].split('.')[0])
                    n = end_frame_idx - start_frame_idx + 1
                    val_gap = 1 if self.seq_l == 1 else 4

                    # print(fn, n, len(data_seq), data_seq[0]['filename'], data_seq[-1]['filename'])
                    for i in range(n):
                        # print(data_seq[i]['filename'])
                        filename = data_seq[i]['filename']
                        frame_idx = int(filename.split('/')[-1].split('.')[0])
                        is_label = data_seq[i]['is_label']

                        if (i + val_gap * (self.seq_l + self.future_seq_l)) <= n:
                            if self.seq_l > 1:
                                # all_seqs.append(('posetrack', fn, filename, frame_idx, i, val_gap))
                                if (i // self.seq_l) % self.seq_l == 0:
                                    _is_label = False
                                    for k in range(self.seq_l):
                                        _is_label = _is_label | data_seq[i + k * val_gap]['is_label']
                                    if _is_label:
                                        all_seqs.append(('posetrack', fn, filename, frame_idx, i, val_gap, False))
                            else:
                                # seq_l = 1
                                if is_label:
                                    # print(filename)
                                    # if filename != 'images/val/001735_mpii_test/000044.jpg':
                                    #     continue
                                    all_seqs.append(('posetrack', fn, filename, frame_idx, i, val_gap, False))
                print('posetrack: {}'.format(len(all_seqs) - count))
                count = len(all_seqs)

            muco_data = None

            # load mupots
            mupots_data = None

            if self.use_muco:
                mupots_data = pickle.load(open('{}/MuPoTS-3D.pkl'.format(self.muco_data_dir, self.mode), 'rb'))
                seq_g = (self.seq_max_gap + self.seq_min_gap) // 2 + 1  # every 5 frame
                img_ids = list(sorted(mupots_data.keys()))
                start_img_id = img_ids[0]
                end_img_id = img_ids[-1]
                # print(mupots_data.keys())
                for img_id in range(start_img_id, end_img_id, seq_g * (self.seq_l - 1)):
                    filename = mupots_data[img_id]['file_name']
                    if self.seq_l == 1:
                        all_seqs.append(('mupots', filename, img_id, False))
                    else:
                        seq_end_img_id = img_id + seq_g * (self.seq_l + self.future_seq_l - 1)
                        if seq_end_img_id not in mupots_data.keys():
                            continue
                        seq_end_filename = mupots_data[seq_end_img_id]['file_name']
                        # not in the same sequence
                        if filename.split('/')[0] != seq_end_filename.split('/')[0]:
                            continue

                        # print(filename, img_id)
                        all_seqs.append(('mupots', filename, img_id, False))

                all_seqs = all_seqs[0:30]
                print('mupots: {}'.format(len(all_seqs) - count))
                count = len(all_seqs)

            if self.use_jta:
                # load jta
                with open('{}/jta_all_ann_files_no_moving_camera.json'.format(self.jta_data_dir), 'r') as json_file:
                    tmp = json.load(json_file)['test']
                seq_g = (self.seq_max_gap + self.seq_min_gap) // 2 + 1  # every 5 frames
                for seq, img_ids in tmp.items():
                    # if seq != 'seq_274':
                    #     continue
                    if self.seq_l == 1:
                        img_idx = np.arange(0, len(img_ids) - (self.seq_l + self.future_seq_l + 1) * seq_g, seq_g)
                    else:
                        img_idx = np.arange(0, len(img_ids) - (self.seq_l + self.future_seq_l + 1) * seq_g,
                                            (self.seq_l - 1) * seq_g)
                    samples = list(zip(['jta'] * len(img_idx), [seq] * len(img_idx), img_idx,
                                       ['test'] * len(img_idx), [False] * len(img_idx)))
                    all_seqs += samples

            # load cmu panoptic data
            panoptic_data = None

            if self.use_panoptic:
                seq_g = (self.seq_max_gap + self.seq_min_gap) // 2
                if self.panoptic_protocol == 1:
                    fname = '{}/panoptic_all_ann_files_protocol{}.pkl' \
                        .format(self.panoptic_data_dir, self.panoptic_protocol)
                    with open(fname, 'rb') as f:
                        # '{}-poses'.format(seq): [frame_idx, poses, track_ids, all_cams], '{}-cam{:02d}'.format(seq)
                        panoptic_data = pickle.load(f)

                    test_sequence = ['170221_haggling_b1', '170221_haggling_b2', '170221_haggling_b3',
                                     '170228_haggling_b1', '170228_haggling_b2', '170228_haggling_b3']
                    all_cams = np.array([3, 12, 23])
                    for k, v in panoptic_data.items():
                        if 'poses' in k:
                            seq_name = k.split('-')[0]
                            if seq_name not in test_sequence:
                                continue

                            if self.seq_l == 1:
                                indices = np.arange(0, len(v) - (self.seq_l + self.future_seq_l + 1) * seq_g,
                                                    self.seq_l * seq_g)
                            else:
                                indices = np.arange(0, len(v) - (self.seq_l + self.future_seq_l + 1) * seq_g,
                                                    (self.seq_l - 1) * seq_g)
                            print('{}: {} samples, {} cams'.format(k, len(indices), len(all_cams)))
                            for cam_idx in all_cams:
                                for index in indices:
                                    frame_idx = v[index][0]
                                    all_seqs.append(('panoptic', seq_name, int(cam_idx), frame_idx, index))
                elif self.panoptic_protocol == 2:
                    fname = '{}/panoptic_all_ann_files_protocol{}.pkl' \
                        .format(self.panoptic_data_dir, self.panoptic_protocol)
                    with open(fname, 'rb') as f:
                        # '{}-poses'.format(seq): [frame_idx, poses, track_ids, all_cams], '{}-cam{:02d}'.format(seq)
                        panoptic_data = pickle.load(f)

                    test_cam_idx = [16, 30]
                    for k, v in panoptic_data.items():
                        if 'poses' in k:
                            seq_name = k.split('-')[0]
                            all_cams = v[0][-1]

                            if self.seq_l == 1:
                                indices = np.arange(0, len(v) - (self.seq_l + self.future_seq_l + 1) * seq_g,
                                                    self.seq_l * seq_g)
                            else:
                                indices = np.arange(0, len(v) - (self.seq_l + self.future_seq_l + 1) * seq_g,
                                                    (self.seq_l - 1) * seq_g)
                            print('{}: {} samples, {} cams'.format(k, len(indices), len(all_cams)))
                            for cam_idx in all_cams:
                                if cam_idx not in test_cam_idx:
                                    continue

                                for index in indices:
                                    frame_idx = v[index][0]
                                    all_seqs.append(('panoptic', seq_name, int(cam_idx), frame_idx, index))
                else:
                    raise ValueError('Panoptic protocol error {}'.format(self.panoptic_protocol))
                print('cmu panoptic: {}'.format(len(all_seqs) - count))
                count = len(all_seqs)

        return all_seqs, posetrack_data, coco_data, muco_data, mupots_data, panoptic_data

    @staticmethod
    def write_val_results(dataset, results, output_dir):
        # assert dataset.mode == 'val'
        categories = dataset.posetrack_data['categories']
        for video_name in results.keys():
            # print('{} starts...'.format(video_name))
            video_results = results[video_name]

            # aggregate multiple predictions
            tmp_kpts, tmp_id = collections.defaultdict(list), collections.defaultdict(list)
            for sample_result in video_results:
                # sample_result['video_name']
                # sample_result['filename']
                # sample_result['index']
                # sample_result['pred_kpts'] # [n, num_frames, num_joints, 2]
                # sample_result['traj_ids']
                # sample_result['gt_kpts']
                # sample_result['gt_bbxes_head']

                assert video_name == sample_result['video_name']
                filename = sample_result['filename']
                traj_ids = sample_result['traj_ids']  # is ordered, number is fixed for each frame
                pred_kpts = sample_result['pred_kpts']  # [n, num_frames, num_joints, 2]
                pred_kpt_scores = sample_result['pred_kpt_scores']  # [n, num_frames, num_joints, 2]
                kpts = np.concatenate([pred_kpts, pred_kpt_scores], axis=-1)
                assert traj_ids.shape[0] == pred_kpts.shape[0]
                tmp_kpts[filename].append(kpts)
                tmp_id[filename].append(traj_ids)

            saved_data = {}
            saved_data['categories'] = categories
            saved_data['images'] = []
            saved_data['annotations'] = []

            for datum in dataset.posetrack_data[video_name]:
                info = datum['info']
                saved_data['images'].append(info)

                # gt_kpts2d = np.array(datum['kpts2d'])
                # gt_track_id = datum['track_id']

                filename = datum['filename']
                if filename in tmp_kpts.keys():
                    _pred_kpts = np.stack(tmp_kpts[filename], axis=0)  # [l, n, num_joints, 3]
                    traj_ids = tmp_id[filename][0]
                    for i in range(len(traj_ids)):
                        pid = traj_ids[i]
                        score = np.mean(_pred_kpts[:, i, :, 2:3], axis=0)  # [num_joints, 1]
                        score_sum = np.sum(_pred_kpts[:, i, :, 2:3], axis=0)
                        kpts = np.sum(_pred_kpts[:, i, :, 0:2] * _pred_kpts[:, i, :, 2:3], axis=0) / \
                               (score_sum + (score_sum == 0).astype(np.float32))
                        pred_kpts = np.concatenate([kpts, score], axis=-1)  # [num_joints, 3]

                        posetrack_kpts = np.zeros([18, 3])
                        posetrack_kpts[JOINT152POSETRACK] = pred_kpts

                        # np.set_printoptions(precision=2, suppress=True)
                        # gt_kpts2d[:, 3, :] = 0  # remove head top to 0
                        # error = np.mean((posetrack_kpts[1:] - gt_kpts2d[gt_track_id == pid, 1:]) ** 2)
                        # # print(error)
                        # if error > 0.1:
                        #     # fn, filename, frame_idx, indice, max_valid_gap
                        #     print('error', error, video_name, filename)
                        #     print(posetrack_kpts[1:], gt_kpts2d[gt_track_id == pid, 1:])
                        #     print('\n')

                        keypoints = posetrack_kpts[1:].reshape(-1).tolist()  # do not need root joint
                        ann = {
                            'bbox_head': [0, 0, 0, 0],
                            'keypoints': keypoints,
                            'track_id': int(pid),
                            'image_id': info['id'],
                            'bbox': [0, 0, 0, 0],
                            'scores': [],
                            'category_id': 1,
                            'id': info['id']
                        }
                        saved_data['annotations'].append(ann)
            save_fname = '{}/{}'.format(output_dir, video_name)
            with open(save_fname, 'w') as f:
                json.dump(saved_data, f)
            print(save_fname)

    @staticmethod
    def eval_posetrack(gt_dir, pred_dir):
        from datasets.poseval_old.evaluate import evaluate_posetrack2018
        evaluate_posetrack2018(gt_dir, pred_dir, pred_dir, eval_pose=True, eval_tracking=True, save_per_seq=False)

    def write_val_results_coco(self, results, output_dir):
        # results: {image_id: [human_score, kpts2d]}
        if self.eval_coco:
            anns = []
            for image_id, v in results.items():
                human_score, kpts2d, gt_kpts2d, indices = v[0]

                for p in range(kpts2d.shape[0]):
                    kpt = kpts2d[p]
                    coco_kpt = np.zeros([19, 3])
                    coco_kpt[JOINT152COCO] = kpt

                    ann = {
                        'image_id': int(image_id),
                        'category_id': 1,
                        'keypoints': coco_kpt[2:].reshape(-1).tolist(),
                        'score': float(human_score[p]),
                    }
                    anns.append(ann)
            json.dump(anns, open('{}/coco_val2017_predictions.json'.format(output_dir), 'w'))

    def eval_coco_val_results(self, gt_dir, pred_dir):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco = COCO(gt_dir)
        coco_pred = coco.loadRes(pred_dir)
        coco_eval = COCOeval(coco, coco_pred, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        print(info_str)
        return info_str


def build_hybrid(image_set, args):
    dataset = HybridData(
        seq_length=args.num_frames,
        future_seq_length=args.num_future_frames,
        seq_max_gap=args.seq_max_gap,
        seq_min_gap=args.seq_min_gap,
        mode=image_set,
        input_shape=(args.input_height, args.input_width),
        num_joints=args.num_kpts,
        max_depth=args.max_depth,
        vis=False,
        posetrack_dir=args.posetrack_dir,
        use_posetrack=args.use_posetrack,
        coco_data_dir=args.coco_dir,
        use_coco=args.use_coco,
        muco_data_dir=args.muco_dir,
        use_muco=args.use_muco,
        jta_data_dir=args.jta_dir,
        use_jta=args.use_jta,
        panoptic_data_dir=args.panoptic_dir,
        use_panoptic=args.use_panoptic,
        panoptic_protocol=args.protocol
    )
    return dataset


if __name__ == '__main__':
    image_set = 'val'
    dataset = HybridData(
        seq_length=4,
        future_seq_length=0,
        seq_max_gap=4,
        seq_min_gap=4,
        mode=image_set,
        input_shape=(600, 800),
        num_joints=15,
        vis=True,
        posetrack_dir='C:/Users/shihaozou/Desktop/posetrack2018/',
        use_posetrack=0,
        coco_data_dir='C:/Users/shihaozou/Desktop/MSCOCO/',
        use_coco=0,
        # muco_data_dir='C:/Users/shihaozou/Desktop/muco/',
        muco_data_dir='/home/shihao/data/mupots/',
        use_muco=1,
        # jta_data_dir='C:/Users/shihaozou/Desktop/jta_dataset',
        jta_data_dir='/home/shihao/data',
        use_jta=0,
        max_depth=15,
        panoptic_data_dir='C:/Users/shihaozou/Desktop/panoptic-toolbox-master/data',
        use_panoptic=0,
        panoptic_protocol=1
    )
    # dataset = HybridData(
    #     seq_length=4,
    #     future_seq_length=2,
    #     seq_max_gap=10,
    #     seq_min_gap=10,
    #     mode=image_set,
    #     input_shape=(540, 960),
    #     num_joints=15,
    #     vis=True,
    #     posetrack_dir='C:/Users/shihaozou/Desktop/posetrack2018/',
    #     use_posetrack=0,
    #     coco_data_dir='C:/Users/shihaozou/Desktop/MSCOCO/',
    #     use_coco=0,
    #     muco_data_dir='C:/Users/shihaozou/Desktop/muco/',
    #     use_muco=0,
    #     jta_data_dir='C:/Users/shihaozou/Desktop/jta_dataset',
    #     use_jta=0,
    #     max_depth=5,
    #     panoptic_data_dir='C:/Users/shihaozou/Desktop/panoptic-toolbox-master/data',
    #     use_panoptic=1,
    #     panoptic_protocol=2
    # )
    # print(dataset.all_seqs)
    # i = np.random.randint(len(dataset))
    # print(i)
    imgs, targets = dataset[100]
    print(imgs.shape)
    print(targets['dataset'])
    print(targets['filenames'])

    imgs, targets = dataset[101]
    print(imgs.shape)
    print(targets['dataset'])
    print(targets['filenames'])






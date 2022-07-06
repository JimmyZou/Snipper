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
# from datasets.data_preprocess.dataset_util import posetrack_visualization, panoptic_visualization
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
        self.seq_min_gap = seq_min_gap
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
        imgs, targets = self.get_jta(self.all_seqs[idx])
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
            # if idx < self.seq_l:
            #     img = cv2.cvtColor(
            #         cv2.imread('{}/images_half/{}/{}/{:03d}.jpg'.format(self.jta_data_dir, subset, seq, i)),
            #         cv2.COLOR_BGR2RGB)
            #     # img = np.load('{}/images_np/{}/{}/{:03d}.npy'.format(self.jta_data_dir, subset, seq, i))
            #     imgs.append(img)

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

        # imgs = np.stack(imgs, axis=0)
        # img_height, img_width, img_channels = imgs.shape[1:]
        img_height, img_width = 1080, 1920
        # assert img_height == self.input_shape[0]
        # assert img_width == self.input_shape[1]
        # print(img_height, img_width)

        # # 1. get augmentation params and apply for the whole sequence of images
        rot, do_flip, color_scale, bbx, trans, inv_trans = \
            get_aug_config((img_width, img_height), self.input_shape, augmentation)
        # print(rot, do_flip, color_scale, bbx)
        # print(trans)

        aug_imgs, aug_kpts2d, aug_track_ids, aug_kpts3d, aug_occs, aug_bbxes, aug_depth = [], [], [], [], [], [], []
        for i in range(self.seq_l + self.future_seq_l):
            # if i < self.seq_l:
            #     # 2. crop patch from img and perform datasets augmentation (flip, rot, color scale)
            #     img = imgs[i]
            #     img_patch = generate_patch_image(img, do_flip, trans, self.input_shape)
            #
            #     for j in range(img_channels):
            #         img_patch[:, :, j] = np.clip(img_patch[:, :, j] * color_scale[j], 0, 1)
            #     aug_imgs.append(img_patch)

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
        # imgs = torch.from_numpy(np.concatenate(aug_imgs, axis=2)).float().permute(2, 0, 1)  # [T*3, H, W]
        imgs = None

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
        print(all_seqs)
        return all_seqs, None, None, None, None, None


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
        seq_length=1,
        future_seq_length=0,
        seq_max_gap=4,
        seq_min_gap=4,
        mode=image_set,
        input_shape=(540, 960),
        num_joints=15,
        vis=False,
        posetrack_dir='C:/Users/shihaozou/Desktop/posetrack2018/',
        use_posetrack=0,
        coco_data_dir='C:/Users/shihaozou/Desktop/MSCOCO/',
        use_coco=0,
        muco_data_dir='C:/Users/shihaozou/Desktop/muco/',
        use_muco=0,
        # jta_data_dir='C:/Users/shihaozou/Desktop/jta_dataset',
        jta_data_dir='/home/shihao/data',
        use_jta=1,
        max_depth=60,
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
    # imgs, targets = dataset[1000]
    # print(imgs.shape)
    # print(targets['dataset'])
    # print(targets['depth'])
    # print(targets['kpts2d'].shape)
    # print(targets['filenames'])

    for sample_idx in range(len(dataset)):
        imgs, targets = dataset[sample_idx]
        max_depth = targets['max_depth']
        target_img_size = targets['input_size'].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]

        _tgt_kpts2d = torch.clone(targets['kpts2d'])  # m x T x num_joints x 3
        tgt_kpts2d = _tgt_kpts2d[..., 0:2] * target_img_size  # scale to original image size
        tgt_kpts2d_vis = _tgt_kpts2d[..., 2:3]
        tgt_depth = torch.clone(targets['depth'])  # m x T x num_joints x 3
        tgt_depth[..., 0] = max_depth * tgt_depth[..., 0]  # scale to original depth

        tgt_track_ids = targets['track_ids']  # [m, T]
        traj_ids = targets['traj_ids']  # [m]
        input_size = targets['input_size']

        results = []
        results.append(
            {
                'gt_kpts': tgt_kpts2d,  # [m, T, num_kpts, 2]
                'gt_kpts_vis': tgt_kpts2d_vis,  # [m, T, num_kpts, 1]
                'gt_depth': tgt_depth,  # [m, T, num_kpts, 2]
                'gt_track_ids': tgt_track_ids,  # [m, T]
                'gt_traj_ids': traj_ids,
                'filenames': targets['filenames'],
                'input_size': input_size,  # (w, h)
                'cam_intr': targets['cam_intr'],  # [4] for evaluation of jta or mupots
                'gt_pose3d': targets['kpts3d'],  # [m, T, num_kpts, 3]  for evaluation of jta or mupots
            }
        )

        tmp = targets['filenames'][0].split('/')
        filename = tmp[0]
        frame_idx = tmp[1].split('.')[0]

        results_np = {}
        for k, v in results[0].items():
            if k == 'indices':
                results_np[k] = [v[0].cpu().numpy(), v[1].cpu().numpy()]
            elif k == 'filenames':
                results_np[k] = v
            elif k in ['heatmaps', 'video_name', 'frame_indices', 'dataset', 'image_id']:
                continue
            else:
                results_np[k] = v.cpu().numpy()

        output_dir = '/home/shihao/data/predictions'
        # '{}/eval_results_{:03d}'.format(output_dir, epoch)
        save_dir = "{}/{}_{}.pkl".format(output_dir, filename, frame_idx)
        with open(save_dir, 'wb') as f:
            pickle.dump(results_np, f)





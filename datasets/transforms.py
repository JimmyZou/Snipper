'''
augmentation code is adopted from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py

'''
import numpy as np
import cv2
import random
import time
import torch
import copy
import math


def get_aug_config_coco(img_shape, input_shape, seq_length, aug):
    if aug:
        do_flip = random.random() <= 0.5

        color_factor = 0.2
        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

        bbx_scale = max(img_shape[0] / input_shape[1], img_shape[1] / input_shape[0])
        bb_width = input_shape[1] * bbx_scale  #  * random.uniform(0.9, 1.1)
        bb_height = input_shape[0] * bbx_scale  #  * random.uniform(0.9, 1.1)

        center_x_scale = random.uniform(0.7, 1.3)
        center_y_scale = random.uniform(0.7, 1.3)
        bb_c_x = img_shape[0] * 0.5 * center_x_scale
        bb_c_y = img_shape[1] * 0.5 * center_y_scale
        bb_c_x_seq_gap = (bb_c_x - img_shape[0] * 0.5) / seq_length
        bb_c_y_seq_gap = (bb_c_y - img_shape[1] * 0.5) / seq_length

        rot_factor = 30
        rot = np.clip(np.random.randn(), -1.0, 1.0) * rot_factor if random.random() <= 1 else 0
        rots_seq_gap = (rot - 0) / seq_length

        rot_list, bbxes_list, trans_list, inv_trans_list = [], [], [], []
        for t in range(seq_length):
            rot_t = rots_seq_gap * (t + 1)
            bb_c_x_t = bb_c_x_seq_gap * (t + 1) + img_shape[0] * 0.5
            bb_c_y_t = bb_c_y_seq_gap * (t + 1) + img_shape[1] * 0.5
            if do_flip:
                bb_c_x_t = img_shape[0] - bb_c_x_t - 1
            bbx_t = [bb_c_x_t, bb_c_y_t, bb_width, bb_height]

            trans = gen_trans_from_patch_cv(bb_c_x_t, bb_c_y_t, bb_width, bb_height,
                                            input_shape[1], input_shape[0], rot_t, inv=False)

            inv_trans = gen_trans_from_patch_cv(bb_c_x_t, bb_c_y_t, bb_width, bb_height,
                                                input_shape[1], input_shape[0], rot_t, inv=True)
            rot_list.append(rot_t)
            bbxes_list.append(bbx_t)
            trans_list.append(trans)
            inv_trans_list.append(inv_trans)
    else:
        rot, do_flip, color_scale = 0, False, [1.0, 1.0, 1.0]
        bb_c_x = img_shape[0] * 0.5
        bb_c_y = img_shape[1] * 0.5
        # input_shape (height, width), img_shape (width, height)
        bbx_scale = max(img_shape[0] / input_shape[1], img_shape[1] / input_shape[0])
        bb_width = input_shape[1] * bbx_scale
        bb_height = input_shape[0] * bbx_scale

        # rot_list, bbxes_list, trans_list, inv_trans_list = [], [], [], []
        bbx = [bb_c_x, bb_c_y, bb_width, bb_height]

        trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height,
                                        input_shape[1], input_shape[0], rot, inv=False)

        inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height,
                                            input_shape[1], input_shape[0], rot, inv=True)
        rot_list = [rot]
        bbxes_list = [bbx]
        trans_list = [trans]
        inv_trans_list = [inv_trans]

    return rot_list, do_flip, color_scale, bbxes_list, trans_list, inv_trans_list


# helper functions
def get_aug_config(img_shape, input_shape, aug):
    if aug:
        rot_factor = 25
        rot = np.clip(np.random.randn(), -1.0, 1.0) * rot_factor if random.random() <= 0.6 else 0

        do_flip = random.random() <= 0.5

        color_factor = 0.2
        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

        center_x_scale = random.uniform(0.7, 1.3)
        center_y_scale = random.uniform(0.7, 1.3)
        bb_c_x = img_shape[0] * 0.5 * center_x_scale
        bb_c_y = img_shape[1] * 0.5 * center_y_scale
        if do_flip:
            bb_c_x = img_shape[0] - bb_c_x - 1

        # bbx_scale = random.uniform(0.6, 1.2)
        # bbx_length = max(img_shape[0], img_shape[1])
        # bb_width = bbx_length * bbx_scale
        # bb_height = bbx_length * bbx_scale

        bbx_scale = max(img_shape[0] / input_shape[1], img_shape[1] / input_shape[0])
        bb_width = input_shape[1] * bbx_scale  #  * random.uniform(0.9, 1.1)
        bb_height = input_shape[0] * bbx_scale  #  * random.uniform(0.9, 1.1)
        bbx = [bb_c_x, bb_c_y, bb_width, bb_height]

        trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height,
                                        input_shape[1], input_shape[0], rot, inv=False)

        inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height,
                                            input_shape[1], input_shape[0], rot, inv=True)
    else:
        rot, do_flip, color_scale = 0, False, [1.0, 1.0, 1.0]
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

    return rot, do_flip, color_scale, bbx, trans, inv_trans


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
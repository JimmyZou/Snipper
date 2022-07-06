import os
import sys
import argparse
import os.path as osp
import numpy as np
import torch
import pickle
from pycocotools.coco import COCO
import argparse
import sys
from glob import glob
import cv2
import json
from tqdm import tqdm

'''
PoseTrack2018 annotation file data structure
{
'images':
    [
        {
            'has_no_densepose': True, 
            'is_labeled': True, 
            'file_name': 'images/val/000342_mpii_test/000000.jpg', 
            'nframes': 100, 
            'frame_id': 10003420000, 
            'vid_id': '000342', 
            'id': 10003420000
        }
        ...
    ],

'annotations':
    [
        {
            'bbox_head': [],
            'keypoints': [],
            'track_id':0,
            'image_id':10003420000,
            'bbox':[],
            'scores': [], 
            'category_id': 1, 
            'id': 1000342000000

        }
        ...
    ],

'categories':
    [
        {
            'supercategory': 'person',
            'id': 1,
            'name': 'person',
            'keypoints': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 
                          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
                          'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
            'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
                         [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
        }
    ]
'''

'''
'keypoints': 
'nose', 0
'head_bottom', 1
'head_top', 2
'left_ear', 3
'right_ear', 4
'left_shoulder', 5
'right_shoulder', 6
'left_elbow', 7
'right_elbow', 8
'left_wrist', 9
'right_wrist', 10
'left_hip', 11
'right_hip', 12
'left_knee', 13
'right_knee', 14
'left_ankle', 15
'right_ankle' 16
'''


def posetrack_extract_train(dataset_path, out_path, subset='train'):
    lhip_idx = 11
    rhip_idx = 12

    images_dir = dataset_path
    json_dir = '{}/{}/{}'.format(dataset_path, 'annotations', subset)
    # json_files = glob('{}/{}'.format(json_dir, '*.json'))
    json_files = os.listdir(json_dir)
    # store the datasets struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    data = dict()
    for fname in tqdm(json_files):
        coco = COCO('{}/{}'.format(json_dir, fname))
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        dataseq = []

        posetrack_images = [img for img in imgs if img['is_labeled']]
        for selected_im in posetrack_images:
            filename = selected_im['file_name']
            h, w = cv2.imread(osp.join(images_dir, filename)).shape[:2]
            ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
            anns = coco.loadAnns(ann_ids)
            kpts2d = list()
            bbox = list()
            track_id = list()

            for ann in anns:
                if 'bbox' in ann:
                    np_kpts2d = np.array(ann['keypoints']).reshape(-1, 3)
                    np_kpts2d[np_kpts2d[:, -1] > 0, -1] = 1
                    if np.any((np_kpts2d[np_kpts2d[..., -1] > 0] < -100) | (np_kpts2d[np_kpts2d[..., -1] > 0] > 1e4)):
                        continue
                    annbbox = np.array([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                                        ann['bbox'][1] + ann['bbox'][3]])
                    if np.any((annbbox < -100) | (annbbox > 1e4)):
                        continue

                    # add Pelvis --> root joint
                    root = (np_kpts2d[lhip_idx:lhip_idx + 1, :] + np_kpts2d[rhip_idx:rhip_idx + 1, :]) * 0.5
                    root[:, 2] = np_kpts2d[lhip_idx:lhip_idx + 1, 2] * np_kpts2d[rhip_idx:rhip_idx + 1, 2]
                    np_kpts2d = np.concatenate([root, np_kpts2d], axis=0)

                    kpts2d.append(np_kpts2d)
                    bbox.append(annbbox)
                    track_id.append(ann['track_id'])
            if not kpts2d:
                tqdm.write(str(f'No annotations in {filename} {fname}'))
                continue

            kpts2d = np.stack(kpts2d, axis=0)
            bboxes = np.stack(bbox).astype(np.float32)
            track_id = np.stack(track_id).astype(np.int32)
            assert kpts2d.shape[0] == track_id.shape[0]
            datum = {
                'filename': filename,
                'width': w,
                'height': h,
                'bboxes': bboxes,
                'kpts2d': kpts2d,
                'track_id': track_id,
            }
            dataseq.append(datum)
        data[fname] = dataseq
    out_file = os.path.join(out_path, f'{subset}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def posetrack_fillin_train(out_path, subset='train'):
    out_file = os.path.join(out_path, f'{subset}.pkl')
    with open(out_file, 'rb') as f:
        data = pickle.load(f)

    data_filled = {}
    for fn, data_seq in data.items():
        # if fn != '001687_mpii_train.json':
        #     continue
        prev_frame_idx = None
        data_seq_filled = []
        for datum in data_seq:
            filename = datum['filename']
            cur_frame_idx = int(filename.split('/')[-1].split('.')[0])
            # print(cur_frame_idx, prev_frame_idx)
            if prev_frame_idx is None:
                prev_frame_idx = cur_frame_idx
                data_seq_filled.append(datum)
            else:
                if (prev_frame_idx + 1) == cur_frame_idx:
                    data_seq_filled.append(datum)
                else:
                    for idx in range(prev_frame_idx + 1, cur_frame_idx):
                        w = datum['width']
                        h = datum['width']
                        tmp = filename.split('/')
                        tmp_filename = '{}/{:06d}.jpg'.format('/'.join(tmp[:-1]), idx)
                        tmp_datum = {
                            'filename': tmp_filename,
                            'width': w,
                            'height': h,
                            'bboxes': [],
                            'kpts2d': [],
                            'track_id': [],
                        }
                        # print(tmp_filename)
                        data_seq_filled.append(tmp_datum)
                    data_seq_filled.append(datum)
                prev_frame_idx = cur_frame_idx
        data_filled[fn] = data_seq_filled

        # for datum in data_seq_filled:
        #     print(datum['filename'])
        # print('\n')
        # for datum in data_seq:
        #     print(datum['filename'])
    out_file = os.path.join(out_path, f'{subset}_filled.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data_filled, f)
        print('save as {}'.format(out_file))


# def posetrack_extract_val(dataset_path, out_path, subset='val'):
#     images_dir = dataset_path
#     json_dir = '{}/{}/{}'.format(dataset_path, 'annotations', subset)
#     # json_files = glob('{}/{}'.format(json_dir, '*.json'))
#     json_files = os.listdir(json_dir)
#     # store the datasets struct
#     if not os.path.isdir(out_path):
#         os.makedirs(out_path)
#
#     data_val = dict()
#     categories = None
#     w, h, = -1, -1
#     for fn in tqdm(json_files):
#         data = json.load(open('{}/{}'.format(json_dir, fn), 'r'))
#         if categories is None:
#             categories = data['categories']
#         imgs = data['images']
#         dataseq = []
#         # posetrack_images = [img for img in imgs if img['is_labeled']]
#         for ii, img in enumerate(imgs):
#             filename = img['file_name']
#             if ii == 0:
#                 h, w = cv2.imread(osp.join(images_dir, filename)).shape[:2]
#
#             datum = {
#                 'filename': filename,
#                 'width': w,
#                 'height': h,
#                 'bboxes': [],
#                 'kpts2d': [],
#                 'track_id': [],
#                 'info': img
#             }
#             dataseq.append(datum)
#         data_val[fn] = dataseq
#         # for tmp in dataseq:
#         #     print(tmp['filename'])
#         # break
#     data_val['categories'] = categories
#     # print(data_val.keys())
#     out_file = os.path.join(out_path, f'{subset}.pkl')
#     with open(out_file, 'wb') as f:
#         pickle.dump(data_val, f)


def posetrack_extract_val(dataset_path, out_path, subset='val'):
    lhip_idx = 11
    rhip_idx = 12

    images_dir = dataset_path
    json_dir = '{}/{}/{}'.format(dataset_path, 'annotations', subset)
    # json_files = glob('{}/{}'.format(json_dir, '*.json'))
    json_files = os.listdir(json_dir)
    # store the datasets struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    data = dict()
    categories = None
    for fname in tqdm(json_files):
        coco = COCO('{}/{}'.format(json_dir, fname))
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        dataseq = []

        tmp_data = json.load(open('{}/{}'.format(json_dir, fname), 'r'))
        if categories is None:
            categories = tmp_data['categories']

        h, w = -1, -1
        # posetrack_images = [img for img in imgs if img['is_labeled']]
        for ii, selected_im in enumerate(imgs):
            filename = selected_im['file_name']
            if ii == 0:
                h, w = cv2.imread(osp.join(images_dir, filename)).shape[:2]
            ann_ids = coco.getAnnIds(imgIds=selected_im['id'])
            anns = coco.loadAnns(ann_ids)
            kpts2d = []
            bboxes = []
            track_id = []
            bboxes_head = []

            info = tmp_data['images'][ii]
            assert info['file_name'] == filename

            for ann in anns:
                if 'bbox' in ann:
                    np_kpts2d = np.array(ann['keypoints']).reshape(-1, 3)
                    np_kpts2d[np_kpts2d[:, -1] > 0, -1] = 1
                    if np.any((np_kpts2d[np_kpts2d[..., -1] > 0] < -100) | (np_kpts2d[np_kpts2d[..., -1] > 0] > 1e4)):
                        continue
                    annbbox = np.array([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2],
                                        ann['bbox'][1] + ann['bbox'][3]])
                    if np.any((annbbox < -100) | (annbbox > 1e4)):
                        continue

                    # add Pelvis --> root joint
                    root = (np_kpts2d[lhip_idx:lhip_idx + 1, :] + np_kpts2d[rhip_idx:rhip_idx + 1, :]) * 0.5
                    root[:, 2] = np_kpts2d[lhip_idx:lhip_idx + 1, 2] * np_kpts2d[rhip_idx:rhip_idx + 1, 2]
                    np_kpts2d = np.concatenate([root, np_kpts2d], axis=0)

                    kpts2d.append(np_kpts2d)
                    bboxes.append(annbbox)
                    track_id.append(ann['track_id'])
                    bboxes_head.append(ann['bbox_head'])
            # if not kpts2d:
            #     tqdm.write(str(f'No annotations in {filename} {fname}'))
            #     continue

            if kpts2d:
                kpts2d = np.stack(kpts2d, axis=0)
                bboxes = np.stack(bboxes).astype(np.float32)
                track_id = np.stack(track_id).astype(np.int32)
                bboxes_head = np.stack(bboxes_head).astype(np.float32)
                assert kpts2d.shape[0] == track_id.shape[0]

            datum = {
                'filename': filename,
                'width': w,
                'height': h,
                'bboxes': bboxes,
                'bboxes_head': bboxes_head,
                'kpts2d': kpts2d,
                'track_id': track_id,
                'info': info,
                'is_label': selected_im['is_labeled']
            }
            dataseq.append(datum)
        data[fname] = dataseq
        # print(fname, len(dataseq))
    data['categories'] = categories
    out_file = os.path.join(out_path, f'{subset}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Posetrack')
    parser.add_argument('--dataset_path', type=str, default='C:/Users/shihaozou/Desktop/posetrack2018')
    parser.add_argument('--out_path', type=str, default='C:/Users/shihaozou/Desktop/posetrack2018')
    parser.add_argument('--subset', type=str, default='val')

    args = parser.parse_args()
    print(args.dataset_path, args.out_path)

    # posetrack dataset has only a small sequence within each video annotated,
    # for example frame 60 to frame 90 annotated
    posetrack_extract_train(args.dataset_path, args.out_path, 'train')
    # annotation of some frames are missing, filling empty list []
    posetrack_fillin_train(args.out_path, subset='train')

    posetrack_extract_val(args.dataset_path, args.out_path, 'val')


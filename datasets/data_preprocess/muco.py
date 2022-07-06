import argparse
import numpy as np
import pickle
import json
# from pycocotools.coco import COCO
import cv2
from tqdm import tqdm

# ('Head_top' 0, 'Thorax'1, 'R_Shoulder'2, 'R_Elbow'3, 'R_Wrist'4, 'L_Shoulder'5, 'L_Elbow'6, 'L_Wrist'7,
# 'R_Hip'8, 'R_Knee'9, 'R_Ankle'10, 'L_Hip'11, 'L_Knee'12, 'L_Ankle'13,
# 'Pelvis'14, 'Spine'15, 'Head'16, 'R_Hand'17, 'L_Hand'18, 'R_Toe'19, 'L_Toe'20)

# COCO ['root' 0, 'neck' 1, 'nose' 2, 'left_eye' 3, 'right_eye' 4, 'left_ear' 5, 'right_ear' 6,
#       'left_shoulder' 7, 'right_shoulder' 8, 'left_elbow' 9, 'right_elbow' 10, 'left_wrist' 11,
#       'right_wrist' 12, 'left_hip' 13, 'right_hip' 14, 'left_knee' 15,
#       'right_knee' 16, 'left_ankle'17, 'right_ankle' 18]
# JOINT15 ['root', 'nose', 'neck', 'left_shoulder', 'right_shoulder',
#          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
#          'right_knee', 'left_ankle', 'right_ankle']
MUCO2JOINT15 = [1, 0, 1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
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


def projection(kpts3d, cam_intr):
    # kpts3d: [N, 15, 3]
    fx, fy, cx, cy = cam_intr

    u = kpts3d[..., 0] / kpts3d[..., 2] * fx + cx
    v = kpts3d[..., 1] / kpts3d[..., 2] * fy + cy
    d = kpts3d[..., 2]
    kpts2d = np.stack([u, v, d], axis=-1)
    return kpts2d


def extract_muco_dataset(dataset_path, out_path):
    # json annotation file
    json_path = '{}/MuCo-3DHP.json'.format(dataset_path)
    json_data = json.load(open(json_path, 'r'))

    data = {}
    for img in json_data['images']:
        if 'unaugmented_set' in img['file_name']:
            continue

        data[img['id']] = img
        data[img['id']]['kpts2d'] = []
        data[img['id']]['kpts3d'] = []
        data[img['id']]['bbx'] = []
        # break
    print(len(data))

    for ann in tqdm(json_data['annotations']):
        img_id = ann['image_id']
        if img_id not in data.keys():
            continue
        # if img_id != 0:
        #     continue

        kpts2d = np.asarray(ann['keypoints_img'])[MUCO2JOINT15]
        kpts3d = np.asarray(ann['keypoints_cam'])[MUCO2JOINT15]
        vis = np.asarray(ann['keypoints_vis'])[MUCO2JOINT15]
        bbx = np.asarray(ann['bbox'])

        _kpts2d = np.concatenate([kpts2d, vis[:, np.newaxis]], axis=1)
        data[img_id]['kpts2d'].append(_kpts2d)
        data[img_id]['kpts3d'].append(kpts3d)
        data[img_id]['bbx'].append(bbx)
        # print(kpts2d.shape, kpts3d.shape, vis.shape, bbx.shape)
        # break

    for img_id in data.keys():
        # if img_id != 0:
        #     continue

        data[img_id]['kpts2d'] = np.stack(data[img_id]['kpts2d'], axis=0)
        data[img_id]['kpts3d'] = np.stack(data[img_id]['kpts3d'], axis=0)
        data[img_id]['bbx'] = np.stack(data[img_id]['bbx'], axis=0)

    # sample = data[0]
    # img = cv2.imread('{}/{}'.format(dataset_path, sample['file_name']))
    # track_ids = np.arange(sample['kpts2d'].shape[0])
    # # kpts = sample['kpts2d']
    # cam_intric = sample['f'] + sample['c']
    # _kpts = projection(sample['kpts3d'], cam_intric)
    # kpts = np.concatenate([_kpts[..., 0:2], sample['kpts2d'][..., 2:3]], axis=-1)
    # bboxes = sample['bbx']
    #
    # import matplotlib.pyplot as plt
    # # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    # cmap = plt.get_cmap('rainbow')
    # sk_colors = [cmap(i) for i in np.linspace(0, 1, len(SKELETONS) + 2)]
    # sk_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in sk_colors]
    #
    # pids = set(track_ids)
    # pid_count = len(pids)
    # cmap = plt.get_cmap('rainbow')
    # pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]
    # pid_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]
    #
    # for p, pid in enumerate(track_ids):
    #     print(pid)
    #     pid_color_idx = np.where(np.array(list(pids)) == pid)[0][0]
    #
    #     pose = kpts[p]
    #     for l, (j1, j2) in enumerate(SKELETONS):
    #         joint1 = pose[j1]
    #         joint2 = pose[j2]
    #
    #         if joint1[2] > 0 and joint2[2] > 0:
    #             cv2.line(img,
    #                      (int(joint1[0]), int(joint1[1])),
    #                      (int(joint2[0]), int(joint2[1])),
    #                      color=tuple(sk_colors[l]),
    #                      thickness=2)
    #
    #         if joint1[2] > 0:
    #             cv2.circle(
    #                 img,
    #                 thickness=-1,
    #                 center=(int(joint1[0]), int(joint1[1])),
    #                 radius=2,
    #                 # color=circle_color,
    #                 color=tuple(pid_colors[pid_color_idx])
    #             )
    #
    #         if joint2[2] > 0:
    #             cv2.circle(
    #                 img,
    #                 thickness=-1,
    #                 center=(int(joint2[0]), int(joint2[1])),
    #                 radius=2,
    #                 # color=circle_color,
    #                 color=tuple(pid_colors[pid_color_idx])
    #             )
    #
    #     bbx = bboxes[p].astype(np.int32)
    #     cv2.line(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1]),
    #              color=tuple(pid_colors[pid_color_idx]), thickness=1)
    #     cv2.line(img, (bbx[0], bbx[1]), (bbx[0], bbx[1] + bbx[3]),
    #              color=tuple(pid_colors[pid_color_idx]), thickness=1)
    #     cv2.line(img, (bbx[0] + bbx[2], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]),
    #              color=tuple(pid_colors[pid_color_idx]), thickness=1)
    #     cv2.line(img, (bbx[0], bbx[1] + bbx[3]), (bbx[0] + bbx[2], bbx[1] + bbx[3]),
    #              color=tuple(pid_colors[pid_color_idx]), thickness=1)
    #
    # plt.figure()
    # plt.imshow(img[:, :, ::-1])
    # plt.show()

    out_file = '{}/MuCo-3DHP.pkl'.format(out_path)
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)
    print('{}/MuCo-3DHP.pkl'.format(out_path), len(data))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Posetrack')
    parser.add_argument('--dataset_path', type=str, default='C:/Users/shihaozou/Desktop/muco')
    parser.add_argument('--out_path', type=str, default='C:/Users/shihaozou/Desktop/muco')

    args = parser.parse_args()
    print(args.dataset_path, args.out_path)
    data = extract_muco_dataset(args.dataset_path, args.out_path)

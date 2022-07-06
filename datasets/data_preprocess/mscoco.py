import os
import argparse
import numpy as np
import pickle
import json
# from pycocotools.coco import COCO
import cv2
from tqdm import tqdm

# COCO ['root' 0, 'neck' 1, 'nose' 2, 'left_eye' 3, 'right_eye' 4, 'left_ear' 5, 'right_ear' 6,
#       'left_shoulder' 7, 'right_shoulder' 8, 'left_elbow' 9, 'right_elbow' 10, 'left_wrist' 11,
#       'right_wrist' 12, 'left_hip' 13, 'right_hip' 14, 'left_knee' 15,
#       'right_knee' 16, 'left_ankle'17, 'right_ankle' 18]
# JOINT15 ['root', 'nose', 'neck', 'left_shoulder', 'right_shoulder',
#          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
#          'right_knee', 'left_ankle', 'right_ankle']
COCO2JOINT15 = [0, 2, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
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
JOINT152COCO = [0, 2, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


def gather_per_image(x, img_dir):
    filenames = x['filename']
    indices = np.argsort(filenames)
    x = {k: v[indices] for k,v in x.items()}
    filenames = x['filename']

    image_boxes = [[]]
    old_name = str(filenames[0])
    img_count = 0
    for i in range(len(filenames)):
        name = str(filenames[i])
        if name != old_name:
            img_count += 1
            image_boxes.append([])
        old_name = name
        image_boxes[img_count].append(i)

    data = [{} for _ in range(len(image_boxes))]
    for i in range(len(image_boxes)):
        for key in x.keys():
            data[i][key] = x[key][image_boxes[i]]
        data[i]['filename'] = data[i]['filename'][0]
        img = cv2.imread(os.path.join(img_dir, data[i]['filename']))
        h, w = img.shape[:2]
        data[i]['height'] = h
        data[i]['width'] = w
        data[i]['bboxes'][:, :2] = np.maximum(data[i]['bboxes'][:, :2], np.zeros_like(data[i]['bboxes'][:, :2]))
        data[i]['bboxes'][:, 2] = np.minimum(data[i]['bboxes'][:, 2], w * np.ones_like(data[i]['bboxes'][:, 2]))
        data[i]['bboxes'][:, 3] = np.minimum(data[i]['bboxes'][:, 3], h * np.ones_like(data[i]['bboxes'][:, 3]))
    return data


def extract_coco_dataset(dataset_path, out_path, subset='train'):
    lhip_idx = 11
    rhip_idx = 12
    lshoulder_idx = 5
    rshoulder_idx = 6

    imgname_list, bbox_list, kpts2d_list = [], [], []

    # json annotation file
    json_path = '{}/annotations/person_keypoints_{}2017.json'.format(dataset_path, subset)
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in tqdm(json_data['annotations']):
        # keypoints processing
        kpts2d = annot['keypoints']
        kpts2d = np.reshape(kpts2d, (17, 3))
        kpts2d[kpts2d[:, 2] > 0, 2] = 1
        # add neck
        neck = (kpts2d[lshoulder_idx:lshoulder_idx+1, :] + kpts2d[rshoulder_idx:rshoulder_idx+1, :]) * 0.5
        neck[0, 2] = kpts2d[lshoulder_idx, 2] * kpts2d[rshoulder_idx, 2]
        # add root
        root = (kpts2d[lhip_idx:lhip_idx+1, :] + kpts2d[rhip_idx:rhip_idx+1, :]) * 0.5
        root[0, 2] = kpts2d[lhip_idx, 2] * kpts2d[rhip_idx, 2]
        kpts2d = np.concatenate([root, neck, kpts2d], axis=0)[COCO2JOINT15]

        if np.sum(kpts2d[:, 2]) == 0:
            continue

        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = '{}2017/{}'.format(subset, img_name)

        # scale and center
        bbox = annot['bbox']

        # store data
        imgname_list.append(img_name_full)
        bbox_list.append(bbox)
        kpts2d_list.append(kpts2d)

    imgname_list = np.array(imgname_list)
    bbox_list = np.array(bbox_list)
    kpts2d_list = np.array(kpts2d_list)
    data = gather_per_image(dict(filename=imgname_list, bboxes=bbox_list, kpts2d=kpts2d_list), img_dir=dataset_path)

    out_file = os.path.join(out_path, f'coco_{subset}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)
    print(subset, len(data))
    return data


def visualize(data, idx, dataset_path):
    print(data[idx].keys())
    print(data[idx]['filename'])
    # print(data[idx]['bboxes'])
    print(data[idx]['kpts2d'].shape)
    # print(data[idx]['width'])
    # print(data[idx]['height'])

    kpts = data[idx]['kpts2d']
    track_ids = np.arange(data[idx]['kpts2d'].shape[0])
    fname = data[idx]['filename']
    img = cv2.imread('{}/{}'.format(dataset_path, fname))
    bboxes = data[idx]['bboxes']

    import matplotlib.pyplot as plt
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    sk_colors = [cmap(i) for i in np.linspace(0, 1, len(SKELETONS) + 2)]
    sk_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in sk_colors]

    pids = set(track_ids)
    pid_count = len(pids)
    cmap = plt.get_cmap('rainbow')
    pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]
    pid_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

    for p, pid in enumerate(track_ids):
        print(pid)
        pid_color_idx = np.where(np.array(list(pids)) == pid)[0][0]

        pose = kpts[p]
        for l, (j1, j2) in enumerate(SKELETONS):
            joint1 = pose[j1]
            joint2 = pose[j2]

            if joint1[2] > 0 and joint2[2] > 0:
                cv2.line(img,
                         (int(joint1[0]), int(joint1[1])),
                         (int(joint2[0]), int(joint2[1])),
                         color=tuple(sk_colors[l]),
                         thickness=2)

            if joint1[2] > 0:
                cv2.circle(
                    img,
                    thickness=-1,
                    center=(int(joint1[0]), int(joint1[1])),
                    radius=2,
                    # color=circle_color,
                    color=tuple(pid_colors[pid_color_idx])
                )

            if joint2[2] > 0:
                cv2.circle(
                    img,
                    thickness=-1,
                    center=(int(joint2[0]), int(joint2[1])),
                    radius=2,
                    # color=circle_color,
                    color=tuple(pid_colors[pid_color_idx])
                )

        bbx = bboxes[p].astype(np.int32)
        cv2.line(img, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1]),
                 color=tuple(pid_colors[pid_color_idx]), thickness=1)
        cv2.line(img, (bbx[0], bbx[1]), (bbx[0], bbx[1] + bbx[3]),
                 color=tuple(pid_colors[pid_color_idx]), thickness=1)
        cv2.line(img, (bbx[0] + bbx[2], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]),
                 color=tuple(pid_colors[pid_color_idx]), thickness=1)
        cv2.line(img, (bbx[0], bbx[1] + bbx[3]), (bbx[0] + bbx[2], bbx[1] + bbx[3]),
                 color=tuple(pid_colors[pid_color_idx]), thickness=1)

    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Posetrack')
    parser.add_argument('--dataset_path', type=str, default='C:/Users/shihaozou/Desktop/MSCOCO')
    parser.add_argument('--out_path', type=str, default='C:/Users/shihaozou/Desktop/MSCOCO')
    parser.add_argument('--subset', type=str, default='val')

    args = parser.parse_args()
    print(args.dataset_path, args.out_path)
    data_train = extract_coco_dataset(args.dataset_path, args.out_path, 'train')
    print('train',  len(data_train))
    data_val = extract_coco_dataset(args.dataset_path, args.out_path, 'val')
    print('val', len(data_val))

    # # check visual results of annotations
    # out_file = os.path.join(args.dataset_path, 'coco_train.pkl')
    # with open(out_file, 'rb') as f:
    #     data = pickle.load(f)
    #
    # print(len(data))
    # for i in [500]:
    #     visualize(data, idx=i, dataset_path=args.dataset_path)

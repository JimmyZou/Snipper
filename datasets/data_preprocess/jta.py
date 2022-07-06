import os
import cv2
import argparse
import json
import glob
import numpy as np
from dataset_util import Pose
from dataset_util import Joint


def convert_video_to_image(data_dir, out_dir_path, first_frame, img_format):
    """
    Script that splits all the videos into frames and saves them
    in a specified directory with the desired format
    """
    if not os.path.exists(out_dir_path):
        print('Create a new direction {}'.format(out_dir_path))
        os.mkdir(out_dir_path)

    for dir in os.listdir('{}/videos'.format(data_dir)):

        out_subdir_path = '{}/{}'.format(out_dir_path, dir)
        if not os.path.exists(out_subdir_path):
            os.mkdir(out_subdir_path)

        print('extracting {} set'.format(dir))
        for video in os.listdir('{}/videos/{}'.format(data_dir, dir)):
            out_seq_path = '{}/{}'.format(out_subdir_path, video.split('.')[0])
            if not os.path.exists(out_seq_path):
                os.mkdir(out_seq_path)

            # print('{}/videos/{}/{}'.format(data_dir, dir, video))
            # reader = imageio.get_reader('{}/videos/{}/{}'.format(data_dir, dir, video))
            reader = cv2.VideoCapture('{}/videos/{}/{}'.format(data_dir, dir, video))
            # fps = reader.get(cv2.CAP_PROP_FPS)
            print('extracting frames of {}'.format(video))
            success, n = True, 0
            while True:
                success, frame = reader.read()
                if not success:
                    break
                img_name = '{}/{:03d}.{}'.format(out_seq_path, n, img_format)
                cv2.imwrite(img_name, frame)
                n += 1
                # return


def get_pose(frame_data, person_id):
    """
    :param frame_data: data of the current frame
    :param person_id: person identifier
    :return: list of joints in the current frame with the required person ID
    """
    pose = [Joint(j) for j in frame_data[frame_data[:, 1] == person_id]]
    pose.sort(key=(lambda j: j.type))
    return Pose(pose)


def convert_annotation_to_coco(out_dir_path):
    """
    Script for annotation conversion (from JTA format to COCO format)
    """

    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    for dir in os.listdir('{}/annotations'.format(out_dir_path)):
        out_subdir_path = '{}/annotations/{}'.format(out_dir_path, dir)
        if not os.path.exists(out_subdir_path):
            os.mkdir(out_subdir_path)
        print('converting {} set'.format(dir))

        for anno in os.listdir(out_subdir_path):
            with open('{}/{}'.format(out_subdir_path, anno), 'r') as json_file:
                data = json.load(json_file)
                data = np.array(data)
            print('converting annotations of {}'.format(anno))

            # getting sequence number from `anno`
            sequence = None
            try:
                sequence = int(anno.split('_')[1].split('.')[0])
            except:
                print('[!] error during conversion.')
                print('\ttry using JSON files with the original nomenclature.')

            coco_dict = {
                'info': {
                    'description': 'JTA 2018 Dataset - Sequence #{}'.format(sequence),
                    'url': 'http://aimagelab.ing.unimore.it/jta',
                    'version': '1.0',
                    'year': 2018,
                    'contributor': 'AImage Lab',
                    'date_created': '2018/01/28',
                },
                'licences': [{
                    'url': 'http://creativecommons.org/licenses/by-nc/2.0',
                    'id': 2,
                    'name': 'Attribution-NonCommercial License'
                }],
                'images': [],
                'annotations': [],
                'categories': [{
                    'supercategory': 'person',
                    'id': 1,
                    'name': 'person',
                    'keypoints': Joint.NAMES,
                    'skeleton': Pose.SKELETON
                }]
            }

            for frame_number in range(0, 900):

                image_id = sequence * 1000 + (frame_number + 1)
                coco_dict['images'].append({
                    'license': 4,
                    'file_name': '{}.jpg'.format(frame_number + 1),
                    'height': 1080,
                    'width': 1920,
                    'date_captured': '2018-01-28 00:00:00',
                    'id': image_id
                })

                # NOTE: frame #0 does NOT exists: first frame is #1
                frame_data = data[data[:, 0] == frame_number + 1]  # type: np.ndarray

                for p_id in set(frame_data[:, 1]):
                    pose = get_pose(frame_data=frame_data, person_id=p_id)

                    # ignore the "invisible" poses
                    # (invisible pose = pose of which I do not see any joint)
                    if pose.invisible:
                        continue

                    annotation = pose.coco_annotation
                    annotation['image_id'] = image_id
                    annotation['id'] = image_id * 100000 + int(p_id)
                    annotation['category_id'] = 1
                    coco_dict['annotations'].append(annotation)

            out_file_path = '{}/seq_{}.coco.json'.format(out_subdir_path, sequence)
            with open(out_file_path, 'w') as f:
                json.dump(coco_dict, f)


def split_annotation(data_dir, out_dir_path):
    if not os.path.exists(out_dir_path):
        print('Create a new direction {}'.format(out_dir_path))
        os.mkdir(out_dir_path)

    for dir in os.listdir('{}/annotations'.format(data_dir)):
        out_subdir_path = '{}/ann_split/{}'.format(out_dir_path, dir)
        if not os.path.exists(out_subdir_path):
            os.mkdir(out_subdir_path)

        print('extracting {} set'.format(dir))
        for ann_fname in os.listdir('{}/annotations/{}'.format(data_dir, dir)):
            if 'coco' in ann_fname:
                continue

            out_seq_path = '{}/{}'.format(out_subdir_path, ann_fname.split('.')[0])
            if not os.path.exists(out_seq_path):
                os.mkdir(out_seq_path)

            with open('{}/annotations/{}/{}'.format(data_dir, dir, ann_fname), 'r') as json_file:
                data = json.load(json_file)
                data = np.array(data)

            print('extracting {} set, {} seq'.format(dir, ann_fname))
            for frame_number in range(0, 900):
                out_file_path = '{}/{:03d}.json'.format(out_seq_path, frame_number)
                # print(out_file_path)
                if os.path.exists(out_file_path):
                    continue

                # NOTE: frame #0 does NOT exists: first frame is #1
                frame_data = data[data[:, 0] == frame_number + 1]
                if frame_data.shape[0] == 0:
                    print('[warning] {}-{} does not have annotations'.format(ann_fname, frame_number))

                frame_dict = dict()
                for p_id in set(frame_data[:, 1]):
                    ann = frame_data[frame_data[:, 1] == p_id]

                    kepts2d = []
                    for j in range(ann.shape[0]):
                        kepts2d += [ann[j, 3], ann[j, 4]]

                    kepts3d = []
                    for j in range(ann.shape[0]):
                        kepts3d += [ann[j, 5], ann[j, 6], ann[j, 7]]

                    visib = []
                    for j in range(ann.shape[0]):
                        visib += [int(ann[j, 8]), int(ann[j, 9])]

                    # occ = np.reshape(np.array(visib), [-1, 22, 2])
                    # vis_person = np.sum(occ[:, :, 0], axis=-1) < 22  # [n]
                    # if np.sum(vis_person) == 0:
                    #     print('[warning] {}-{} all annotations are occluded.'.format(ann_fname, frame_number))

                    frame_dict[int(p_id)] = (kepts2d, kepts3d, visib)
                with open(out_file_path, 'w') as f:
                    json.dump(frame_dict, f)


def check_jta_ann(data_dir, modes=('train', 'test', 'val')):
    for mode in modes:
        for seq in os.listdir('{}/ann_split/{}'.format(data_dir, mode)):
            for ann_file in os.listdir('{}/ann_split/{}/{}'.format(data_dir, mode, seq)):
                fname = '{}/ann_split/{}/{}/{}'.format(data_dir, mode, seq, ann_file)
                with open(fname, 'r') as json_file:
                    ann = json.load(json_file)
                # if len(ann.keys()) == 0:
                #     print('remove {}'.format(fname))
                #     os.remove(fname)


def prepare_jta_dataset(data_dir, out_dir_path):
    all_files = dict()
    for dir in os.listdir('{}/ann_split'.format(data_dir)):
        all_files_dir = dict()
        for seq in os.listdir('{}/ann_split/{}'.format(data_dir, dir)):
            all_files_dir[seq] = []
            for ann_file in os.listdir('{}/ann_split/{}/{}'.format(data_dir, dir, seq)):
                frame_idx = int(ann_file.split('.')[0])
                all_files_dir[seq].append(frame_idx)
            print('{}-{}'.format(seq, len(all_files_dir[seq])))
        all_files[dir] = all_files_dir
    with open('{}/jta_all_ann_files.json'.format(out_dir_path), 'w') as f:
        json.dump(all_files, f)


def prepare_jta_dataset_fix_camera(data_dir, out_dir_path):
    moving_cam_seq = []
    with open('moving_camera_seq.txt', 'r') as f:
        for line in f.readlines():
            seq_num = line.replace('\n', '')
            moving_cam_seq.append(int(seq_num))
    # print(moving_cam_seq)

    all_files = dict()
    for dir in os.listdir('{}/ann_split'.format(data_dir)):
        all_files_dir = dict()
        for seq in os.listdir('{}/ann_split/{}'.format(data_dir, dir)):
            seq_num = int(seq.split('_')[1])
            if seq_num in moving_cam_seq:
                continue
            all_files_dir[seq] = []
            for ann_file in os.listdir('{}/ann_split/{}/{}'.format(data_dir, dir, seq)):
                frame_idx = int(ann_file.split('.')[0])
                all_files_dir[seq].append(frame_idx)
            print('{}-{}'.format(seq, len(all_files_dir[seq])))
        all_files[dir] = all_files_dir
    with open('{}/jta_all_ann_files_no_moving_camera.json'.format(out_dir_path), 'w') as f:
        json.dump(all_files, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess JTA dataset')
    parser.add_argument('--dataset_path', type=str, default='/home/shihao/data')
    parser.add_argument('--out_path', type=str, default='/home/shihao/data')

    args = parser.parse_args()
    print(args.dataset_path, args.out_path)

    # convert each video to image
    # convert_video_to_image(args.dataset_path, args.out_path, 0, 'jpg')

    # # split annotations of the whole video to each frame
    # split_annotation(args.dataset_path, args.out_path)
    # check_jta_ann(args.dataset_path, ('test',))

    # # prepare all annotated filenames
    prepare_jta_dataset(args.dataset_path, args.out_path)
    prepare_jta_dataset_fix_camera(args.dataset_path, args.out_path)

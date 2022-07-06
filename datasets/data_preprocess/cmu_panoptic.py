import numpy as np
import json
import os
import glob
import pickle
import cv2
import matplotlib.pyplot as plt


def unprojection(uvd, intr_param, dist_coeff, simple_mode=False):
    # uvd: [N, 3]
    # cam_param: (fx, fy, cx, cy)
    # dist_coeff: (k1, k2, p1, p2, k3, k4, k5, k6)
    assert uvd.shape[1] == 3
    fx, fy, cx, cy = intr_param

    if not simple_mode:
        k1, k2, p1, p2, k3 = dist_coeff
        k4, k5, k6 = 0, 0, 0

        x_pp = (uvd[:, 0] - cx) / fx
        y_pp = (uvd[:, 1] - cy) / fy
        r2 = x_pp ** 2 + y_pp ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        a = a + (a == 0)
        di = b / a

        x_p = x_pp * di
        y_p = y_pp * di

        x = uvd[:, 2] * (x_p - p2 * (y_p ** 2 + 3 * x_p ** 2) - p1 * 2 * x_p * y_p)
        y = uvd[:, 2] * (y_p - p1 * (x_p ** 2 + 3 * y_p ** 2) - p2 * 2 * x_p * y_p)
        z = uvd[:, 2]

        return np.stack([x, y, z], axis=1)
    else:
        x = uvd[:, 2] * (uvd[:, 0] - cx) / fx
        y = uvd[:, 2] * (uvd[:, 1] - cy) / fy
        z = uvd[:, 2]
        return np.stack([x, y, z], axis=1)


def projection(xyz, intr_param, dist_coeff, simple_mode=False):
    # xyz: [N, 3]
    # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param

    if not simple_mode:
        k1, k2, p1, p2, k3 = dist_coeff
        k4, k5, k6 = 0, 0, 0

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)


def get_uniform_camera_order():
    """ Returns uniformly sampled camera order as a list of tuples [(panel,node), (panel,node), ...]."""
    panel_order = [1, 19, 14, 6, 16, 9, 5, 10, 18, 15, 3, 8, 4, 20, 11, 13, 7, 2, 17, 12, 9, 5, 6, 3, 15, 2, 12, 14, 16,
                   10, 4, 13, 20, 8, 17, 19, 18, 9, 4, 6, 1, 20, 1, 11, 7, 7, 14, 15, 3, 2, 16, 13, 3, 15, 17, 9, 20,
                   19, 8, 11, 5, 8, 18, 10, 12, 19, 5, 6, 16, 12, 4, 6, 20, 13, 4, 10, 15, 12, 17, 17, 16, 1, 5, 3, 2,
                   18, 13, 16, 8, 19, 13, 11, 10, 7, 3, 2, 18, 10, 1, 17, 10, 15, 14, 4, 7, 9, 11, 7, 20, 14, 1, 12, 1,
                   6, 11, 18, 7, 8, 9, 3, 15, 19, 4, 16, 18, 1, 11, 8, 4, 10, 20, 13, 6, 16, 7, 6, 16, 17, 12, 5, 17, 4,
                   8, 20, 12, 17, 14, 2, 19, 14, 18, 15, 11, 11, 9, 9, 2, 13, 5, 15, 20, 18, 8, 3, 19, 11, 9, 2, 13, 14,
                   5, 9, 17, 9, 7, 6, 12, 16, 18, 17, 13, 15, 17, 20, 4, 2, 2, 12, 4, 1, 16, 4, 11, 1, 16, 12, 18, 9, 7,
                   20, 1, 10, 10, 19, 5, 8, 14, 8, 4, 2, 9, 20, 14, 17, 11, 3, 12, 3, 13, 6, 5, 16, 3, 5, 10, 19, 1, 11,
                   13, 17, 18, 2, 5, 14, 19, 15, 8, 8, 9, 3, 6, 16, 15, 18, 20, 4, 13, 2, 11, 20, 7, 13, 15, 18, 10, 20,
                   7, 5, 2, 15, 6, 13, 4, 17, 7, 3, 19, 19, 3, 10, 2, 12, 10, 7, 7, 12, 11, 19, 8, 9, 6, 10, 6, 15, 10,
                   11, 3, 16, 1, 5, 14, 6, 5, 13, 20, 14, 4, 18, 10, 14, 14, 1, 19, 8, 14, 19, 3, 6, 6, 3, 13, 17, 8,
                   20, 15, 18, 2, 2, 16, 5, 19, 15, 9, 12, 19, 17, 8, 9, 3, 7, 1, 12, 7, 13, 1, 14, 5, 12, 11, 2, 16, 1,
                   18, 4, 18, 10, 16, 11, 7, 5, 1, 16, 9, 4, 15, 1, 7, 10, 14, 3, 2, 17, 13, 19, 20, 15, 10, 4, 8, 16,
                   14, 5, 6, 20, 12, 5, 18, 7, 1, 8, 11, 5, 13, 1, 16, 14, 18, 12, 15, 2, 12, 3, 8, 12, 17, 8, 20, 9, 2,
                   6, 9, 6, 12, 3, 20, 15, 20, 13, 3, 14, 1, 4, 8, 6, 10, 7, 17, 13, 18, 19, 10, 20, 12, 19, 2, 15, 10,
                   8, 19, 11, 19, 11, 2, 4, 6, 2, 11, 8, 7, 18, 14, 4, 12, 14, 7, 9, 7, 11, 18, 16, 16, 17, 16, 15, 4,
                   15, 9, 17, 13, 3, 6, 17, 17, 20, 19, 11, 5, 3, 1, 18, 4, 10, 5, 9, 13, 1, 5, 9, 6, 14]
    node_order = [1, 14, 3, 15, 12, 12, 8, 6, 13, 12, 12, 17, 7, 17, 21, 17, 4, 6, 12, 18, 2, 18, 5, 4, 2, 17, 12, 10,
                  18, 8, 18, 5, 10, 10, 17, 1, 18, 7, 12, 9, 13, 5, 6, 18, 16, 9, 16, 8, 8, 10, 21, 22, 16, 16, 21, 16,
                  14, 6, 14, 11, 11, 20, 4, 22, 4, 22, 20, 19, 15, 15, 15, 12, 2, 2, 3, 3, 20, 22, 5, 9, 3, 16, 23, 22,
                  20, 8, 8, 9, 2, 16, 14, 16, 16, 14, 1, 13, 16, 12, 10, 15, 18, 6, 13, 10, 7, 10, 4, 1, 7, 21, 8, 6, 4,
                  7, 9, 10, 11, 8, 4, 6, 10, 4, 5, 6, 21, 21, 6, 6, 19, 20, 20, 20, 14, 19, 22, 22, 23, 19, 9, 15, 23,
                  23, 23, 23, 19, 2, 8, 2, 8, 19, 19, 23, 23, 19, 19, 23, 24, 24, 2, 14, 12, 2, 12, 14, 12, 2, 14, 15,
                  11, 6, 6, 21, 4, 5, 5, 4, 2, 10, 5, 10, 7, 3, 7, 9, 8, 9, 3, 7, 9, 9, 7, 2, 5, 5, 5, 5, 7, 8, 8, 4, 7,
                  11, 9, 7, 5, 3, 5, 7, 6, 8, 9, 8, 7, 8, 8, 3, 8, 7, 6, 11, 7, 2, 9, 9, 2, 11, 12, 7, 4, 6, 6, 7, 4, 4,
                  9, 18, 1, 5, 6, 5, 10, 11, 5, 9, 6, 11, 12, 1, 10, 11, 6, 9, 7, 11, 5, 1, 2, 12, 11, 11, 3, 3, 21, 11,
                  10, 2, 3, 10, 11, 19, 5, 11, 13, 12, 20, 13, 3, 5, 9, 11, 8, 4, 6, 4, 7, 12, 10, 8, 11, 19, 14, 23,
                  10, 1, 3, 12, 4, 3, 10, 9, 2, 3, 20, 4, 11, 2, 20, 20, 2, 23, 10, 3, 22, 22, 1, 12, 12, 21, 4, 22, 23,
                  22, 18, 10, 18, 22, 11, 3, 18, 13, 18, 3, 3, 13, 2, 1, 3, 20, 20, 4, 20, 14, 14, 20, 20, 14, 14, 22,
                  18, 21, 20, 22, 20, 22, 9, 22, 21, 21, 22, 21, 22, 20, 21, 21, 21, 21, 23, 17, 21, 13, 20, 13, 13, 15,
                  17, 1, 23, 23, 23, 18, 13, 16, 15, 19, 17, 17, 22, 21, 17, 14, 1, 13, 13, 14, 14, 16, 19, 17, 18, 1,
                  13, 18, 24, 19, 16, 13, 18, 18, 15, 23, 17, 14, 19, 17, 1, 19, 13, 19, 1, 15, 17, 13, 23, 13, 19, 24,
                  15, 15, 19, 15, 17, 1, 16, 24, 21, 23, 14, 24, 15, 24, 24, 1, 16, 15, 24, 1, 17, 17, 15, 24, 1, 16,
                  16, 19, 13, 15, 22, 24, 23, 17, 16, 18, 1, 24, 24, 24, 17, 24, 24, 17, 16, 24, 14, 15, 16, 15, 24, 24,
                  24, 18]

    return zip(panel_order, node_order)


def extract_pose_annotation(data_dir):
    for dir in os.listdir('{}'.format(data_dir)):
        # if dir != '170221_haggling_b2':
        #     continue

        # check 3D pose extraction
        if not os.path.exists('{}/{}/hdPose3d_stage1_coco19'.format(data_dir, dir)):
            os.system('tar -xf {}/{}/hdPose3d_stage1_coco19.tar -C {}/{}'.format(data_dir, dir, data_dir, dir))
            print('finish extract {}/{}/hdPose3d_stage1_coco19.tar'.format(data_dir, dir))
        else:
            print('exists hdPose3d_stage1_coco19.tar')


def convert_video_to_image(data_dir, img_format, protocol=1):
    """
    Script that splits all the videos into frames and saves them
    in a specified directory with the desired format
    """

    for dir in sorted(os.listdir('{}'.format(data_dir))):
        if '.pkl' in dir:
            continue
        print('----------------------------------------------')
        print(dir)
        # if dir != '170221_haggling_m1':
        #     continue

        if protocol == 1:
            # check 3D pose extraction
            if not os.path.exists('{}/{}/hdPose3d_stage1_coco19'.format(data_dir, dir)):
                print('tar -xf {}/{}/hdPose3d_stage1_coco19.tar -C {}/{}'.format(data_dir, dir, data_dir, dir))
                os.system('tar -xf {}/{}/hdPose3d_stage1_coco19.tar -C {}/{}'.format(data_dir, dir, data_dir, dir))
                print('finish extract hdPose3d_stage1_coco19.tar')
            else:
                anns = os.listdir('{}/{}/hdPose3d_stage1_coco19'.format(data_dir, dir))
                if len(anns) < 1000:
                    os.system('tar -xf {}/{}/hdPose3d_stage1_coco19.tar -C {}/{}'.format(data_dir, dir, data_dir, dir))
                    print('finish extract {}/hdPose3d_stage1_coco19.tar'.format(dir))
                else:
                    print('exists {}/hdPose3d_stage1_coco19'.format(dir))

            tmp = sorted(os.listdir('{}/{}/hdPose3d_stage1_coco19'.format(data_dir, dir)))

        elif protocol == 2:
            # check 3D pose extraction
            if not os.path.exists('{}/{}/hdPose3d_stage1'.format(data_dir, dir)):
                print('tar -xf {}/{}/hdPose3d_stage1.tar -C {}/{}'.format(data_dir, dir, data_dir, dir))
                os.system('tar -xf {}/{}/hdPose3d_stage1.tar -C {}/{}'.format(data_dir, dir, data_dir, dir))
                print('finish extract hdPose3d_stage1.tar')
            else:
                anns = os.listdir('{}/{}/hdPose3d_stage1'.format(data_dir, dir))
                if len(anns) < 1000:
                    os.system('tar -xf {}/{}/hdPose3d_stage1.tar -C {}/{}'.format(data_dir, dir, data_dir, dir))
                    print('finish extract {}/hdPose3d_stage1.tar'.format(dir))
                else:
                    print('exists {}/hdPose3d_stage1'.format(dir))
            tmp = sorted(os.listdir('{}/{}/hdPose3d_stage1'.format(data_dir, dir)))

        else:
            raise ValueError('protocol {} errors.'.format(protocol))

        if len(tmp) == 0:
            print('no annotations')
            continue
        else:
            tmp = tmp[-1]

        end_frame_idx = int(tmp.split('_')[-1].split('.')[0])
        # check images extraction
        out_path = '{}/{}/hdImgs'.format(data_dir, dir)
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for video in os.listdir('{}/{}/hdVideos'.format(data_dir, dir)):
            out_seq_path = '{}/{}'.format(out_path, video.split('.')[0])
            if not os.path.exists(out_seq_path):
                os.mkdir(out_seq_path)

            if len(os.listdir(out_seq_path)) > 1000:
                print('exists {}'.format(out_seq_path))
                continue

            # print('{}/videos/{}/{}'.format(data_dir, dir, video))
            # reader = imageio.get_reader('{}/videos/{}/{}'.format(data_dir, dir, video))
            reader = cv2.VideoCapture('{}/{}/hdVideos/{}'.format(data_dir, dir, video))
            # fps = reader.get(cv2.CAP_PROP_FPS)
            print('extracting frames of {}/hdVideos/{}, last annotated frame {}'.format(dir, video, end_frame_idx))
            success, n = True, 0
            while success and (n <= end_frame_idx):
                success, frame = reader.read()
                img_name = '{}/{:08d}.{}'.format(out_seq_path, n, img_format)
                h, w, _ = frame.shape
                half_frame = cv2.resize(frame, (int(w / 2), int(h / 2)))
                cv2.imwrite(img_name, half_frame)
                n += 1
                # return

'''
def convert_3d_keypoints_demo(data_dir, seq_name, cam_idx=3, frame_idx=1000, img_format='jpg'):
    # Load camera calibration parameters
    with open('{}/{}/calibration_{}.json'.format(data_dir, seq_name, seq_name)) as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    # Convert data into numpy arrays for convenience
    for k, cam in cameras.items():
        # cam['K'] = np.matrix(cam['K'])
        cam['intr'] = np.array([cam['K'][0][0], cam['K'][1][1], cam['K'][0][2], cam['K'][1][2]])  # (fx, fy, cx, cy)
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.array(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3, 1))
    cam = cameras[(0, cam_idx)]

    # load image
    img = cv2.imread('{}/{}/hdImgs/hd_00_{:02d}/{:08d}.{}'.format(data_dir, seq_name, cam_idx, frame_idx, img_format))

    # load 3D pose
    # skel_json_fname = '{}/{}/hdPose3d_stage1_coco19/body3DScene_{:08d}.json'.format(data_dir, seq_name, frame_idx)
    skel_json_fname = '{}/{}/hdPose3d_stage1/body3DScene_{:08d}.json'.format(data_dir, seq_name, frame_idx)
    with open(skel_json_fname) as dfile:
        bframe = json.load(dfile)
    # print(bframe)

    # protocol 1
    # cmu panoptic 'keypoints':
    # [
    #   neck 0, nose 1, heap center 2, left shoulder 3, left elbow 4,
    #   left wrist 5, left hip 6, left knee 7, left ankle 8,
    #   right shoulder 9, right elbow 10, right wrist 11,
    #   right hip 12, right knee 13, right ankle 14
    # ]
    # SKELETONS = np.array([[1, 2], [1, 4], [4, 5], [5, 6], [1, 3], [3, 7], [7, 8], [8, 9],
    #                       [3, 13], [13, 14], [14, 15], [1, 10], [10, 11], [11, 12]]) - 1
    # change to
    # ['hip center' 0, 'nose' 1, 'neck' 2, 'left_shoulder' 3, 'right_shoulder' 4,
    #  'left_elbow' 5, 'right_elbow' 6, 'left_wrist' 7, 'right_wrist' 8, 'left_hip' 9, 'right_hip' 10,
    #  'left_knee' 11, 'right_knee'12, 'left_ankle' 13, 'right_ankle' 14]
    #
    # # skeleton
    # JOINTIDX = [2, 1, 0, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

    # protocol 2
    # cmu panoptic 'keypoints':
    # [
    #   neck 0, head top 1, root 2, left shoulder 3, left elbow 4,
    #   left wrist 5, left hip 6, left knee 7, left ankle 8,
    #   right shoulder 9, right elbow 10, right wrist 11,
    #   right hip 12, right knee 13, right ankle 14
    # ]
    # change to
    # ['hip center' 0, 'nose' 1, 'neck' 2, 'left_shoulder' 3, 'right_shoulder' 4,
    #  'left_elbow' 5, 'right_elbow' 6, 'left_wrist' 7, 'right_wrist' 8, 'left_hip' 9, 'right_hip' 10,
    #  'left_knee' 11, 'right_knee'12, 'left_ankle' 13, 'right_ankle' 14]
    #
    # # skeleton

    JOINTIDX = [2, 1, 0, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
    SKELETONS = [[0, 2], [1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [0, 9],
                 [0, 10], [9, 11], [10, 12], [11, 13], [12, 14]]

    # skel = np.array(bframe['bodies'][0]['joints15']).reshape((-1, 4)).transpose()  # [15, 4]
    #
    # # Show only points detected with confidence
    # valid = skel[3:4, :] > 0.1
    # # print(pt2d.shape, valid.shape)
    #
    # # get pose 3d, cm --> mm
    # _pt3d = 10 * (np.dot(cam['R'], skel[0:3]) + cam['t']).transpose()
    # # print(_pt3d)
    # _pt2d = projection(_pt3d, cam['intr'], cam['distCoef'], simple_mode=False)
    #
    # # Project skeleton into view (this is like cv2.projectPoints)
    # # pt2d, pt3d = projectPoints(skel[0:3, :], cam['K'], cam['R'], cam['t'], cam['distCoef'])
    #
    # pt2d = _pt2d[:, 0:2] / 2
    # for j in range(pt2d.shape[0]):
    #     plt.figure()
    #     plt.imshow(img[:, :, ::-1])
    #     plt.scatter(pt2d[j, 0], pt2d[j, 1], c='r', s=20, marker='h')
    #     plt.title('{}'.format(j))
    #     plt.show()

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    sk_colors = [cmap(i) for i in np.linspace(0, 1, len(SKELETONS) + 2)]
    sk_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in sk_colors]

    pid_count = len(bframe['bodies'])
    cmap = plt.get_cmap('rainbow')
    pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]
    pid_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

    # Cycle through all detected bodies
    for pid_color_idx, body in enumerate(bframe['bodies']):
        # There are 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...]
        # where c1 ... c19 are per-joint detection confidences
        skel = np.array(body['joints15']).reshape((-1, 4)).transpose()  # [4, 19]
        pid = body['id']

        # Show only points detected with confidence
        valid = skel[3:4, :] > 0.1
        # print(pt2d.shape, valid.shape)

        # get pose 3d, cm --> mm
        _pt3d = 10 * (np.dot(cam['R'], skel[0:3]) + cam['t']).transpose()
        # print(_pt3d)
        _pt2d = projection(_pt3d, cam['intr'], cam['distCoef'], simple_mode=False)

        # Project skeleton into view (this is like cv2.projectPoints)
        # pt2d, pt3d = projectPoints(skel[0:3, :], cam['K'], cam['R'], cam['t'], cam['distCoef'])

        # pt3d = unprojection(_pt2d, cam['intr'], cam['distCoef'], simple_mode=False)
        # pt2d = projection(pt3d, cam['intr'], cam['distCoef'], simple_mode=False)
        # diff = pt3d - _pt3d
        # print(diff)
        # print('3d error:', np.mean(np.sqrt(np.sum((diff)**2, axis=-1))))

        pt2d = _pt2d[:, 0:2] / 2

        # # print(cam['K'])
        # fx, fy, cx, cy = cam['K'][0, 0], cam['K'][1, 1], cam['K'][0, 2], cam['K'][1, 2]
        # cam_intr = np.array([fx, fy, cx, cy]) / 2
        # pt2d = project(pt3d.transpose(), cam_intr)
        # print(pt2d)

        pose = np.concatenate([pt2d, valid.transpose()], axis=1)[JOINTIDX]
        for l, (j1, j2) in enumerate(SKELETONS):
            joint1 = pose[j1]
            joint2 = pose[j2]

            if joint1[2] > 0 and joint2[2] > 0:
                cv2.line(img,
                         (int(joint1[0]), int(joint1[1])),
                         (int(joint2[0]), int(joint2[1])),
                         color=tuple(sk_colors[l]),
                         thickness=3)

            if joint1[2] > 0:
                cv2.circle(
                    img,
                    thickness=-1,
                    center=(int(joint1[0]), int(joint1[1])),
                    radius=3,
                    # color=circle_color,
                    color=tuple(pid_colors[pid_color_idx])
                )

            if joint2[2] > 0:
                cv2.circle(
                    img,
                    thickness=-1,
                    center=(int(joint2[0]), int(joint2[1])),
                    radius=3,
                    # color=circle_color,
                    color=tuple(pid_colors[pid_color_idx])
                )

        # for i in range(pose.shape[0]):
        #     plt.figure()
        #     plt.imshow(img[..., ::-1])
        #     plt.scatter(pose[i, 0], pose[i, 1], s=20, marker='.', c='r')
        #     plt.axis('off')
        #     plt.title('{}'.format(i))
        #     plt.show()
        # break

    plt.figure(figsize=(15, 10))
    plt.imshow(img[..., ::-1])
    plt.axis('off')
    plt.show()
'''

def get_annotations(ann_fname, cam):
    # skeleton
    JOINTIDX = [2, 1, 0, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

    # load 3D pose
    if not os.path.exists(ann_fname):
        assert ValueError('{} cannot find.'.format(ann_fname))

    poses, track_ids = [], []
    with open(ann_fname) as dfile:
        bframe = json.load(dfile)
    for body in bframe['bodies']:
        pid = body['id']
        skel = np.array(body['joints19']).reshape((-1, 4))  # [19, 4]
        valid = skel[:, 3:4]
        pt3d = 10 * (np.dot(skel[:, 0:3], cam['R'].T) + cam['t'].T)  # cm to mm
        # 0.5 means image is resize to half size of original size
        pt2d = projection(pt3d, cam['intr'] * 0.5, cam['distCoef'], simple_mode=False)

        pose = np.concatenate([pt2d[:, 0:2], pt3d, valid], axis=-1)[JOINTIDX]  # [15, 2+3+1]
        poses.append(pose)
        track_ids.append(pid)
    if poses != []:
        poses = np.stack(poses, axis=0)
        track_ids = np.array(track_ids)
    return poses, track_ids, cam['intr'] * 0.5, cam['distCoef']


def prepare_panoptic_protocol1(data_dir, all_cams=(3, 12, 23)):
    all_files = dict()

    for dir in os.listdir('{}'.format(data_dir)):
        if 'pkl' in dir:
            continue

        if '170' not in dir:
            continue

        # Load camera calibration parameters
        with open('{}/{}/calibration_{}.json'.format(data_dir, dir, dir)) as cfile:
            calib = json.load(cfile)
        # Cameras are identified by a tuple of (panel#,node#)
        cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}
        # Convert data into numpy arrays for convenience
        for k, cam in cameras.items():
            # cam['K'] = np.matrix(cam['K'])
            cam['intr'] = np.array(
                [cam['K'][0][0], cam['K'][1][1], cam['K'][0][2], cam['K'][1][2]])  # (fx, fy, cx, cy)
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['R'] = np.array(cam['R'])
            cam['t'] = np.array(cam['t']).reshape((3, 1))

        for cam_idx in all_cams:
            cam = cameras[(0, cam_idx)]
            print('save cam infomation {}-cam{:02d}'.format(dir, cam_idx))
            all_files['{}-cam{:02d}'.format(dir, cam_idx)] = cam

        seq_all_files = []
        anns = sorted(os.listdir('{}/{}/hdPose3d_stage1_coco19'.format(data_dir, dir)))
        for ann in anns:
            # if '00002000' not in ann:
            #     continue

            ann_fname = '{}/{}/hdPose3d_stage1_coco19/{}'.format(data_dir, dir, ann)
            frame_idx = int(ann.split('_')[1].split('.')[0])

            # skeleton
            JOINTIDX = [2, 1, 0, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

            # load 3D pose
            if not os.path.exists(ann_fname):
                assert ValueError('{} cannot find.'.format(ann_fname))

            poses, track_ids = [], []
            with open(ann_fname) as dfile:
                bframe = json.load(dfile)
            for body in bframe['bodies']:
                pid = body['id']
                skel = np.array(body['joints19']).reshape((-1, 4))  # [19, 4]
                valid = skel[:, 3:4]
                pt3d = skel[:, 0:3]  # cm to mm
                pose = np.concatenate([pt3d, valid], axis=-1)[JOINTIDX]  # [15, 3+1]
                poses.append(pose)
                track_ids.append(pid)
            if poses != []:
                poses = np.stack(poses, axis=0)
                track_ids = np.array(track_ids)


            seq_all_files.append([frame_idx, poses, track_ids, all_cams])
        all_files['{}-poses'.format(dir)] = seq_all_files

        # break
        # print(all_files['val'])
    with open('{}/panoptic_all_ann_files_protocol1.pkl'.format(data_dir), 'wb') as f:
        pickle.dump(all_files, f)


def prepare_panoptic_protocol2(data_dir, all_cams):
    all_files = dict()

    for dir in os.listdir('{}'.format(data_dir)):
        if 'pkl' in dir:
            continue

        if '160' not in dir and '170' in dir:
            continue

        # Load camera calibration parameters
        with open('{}/{}/calibration_{}.json'.format(data_dir, dir, dir)) as cfile:
            calib = json.load(cfile)
        # Cameras are identified by a tuple of (panel#,node#)
        cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}
        # Convert data into numpy arrays for convenience
        for k, cam in cameras.items():
            # cam['K'] = np.matrix(cam['K'])
            cam['intr'] = np.array(
                [cam['K'][0][0], cam['K'][1][1], cam['K'][0][2], cam['K'][1][2]])  # (fx, fy, cx, cy)
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['R'] = np.array(cam['R'])
            cam['t'] = np.array(cam['t']).reshape((3, 1))

        for cam_idx in all_cams:
            cam = cameras[(0, cam_idx)]
            print('start {}-{}'.format(dir, cam_idx))
            all_files['{}-cam{:02d}'.format(dir, cam_idx)] = cam

        valid_cams = []
        for cam_idx in all_cams:
            if os.path.exists('{}/{}/hdImgs/hd_00_{:02d}'.format(data_dir, dir, cam_idx)):
                valid_cams.append(cam_idx)
        valid_cams = np.array(valid_cams)

        seq_all_files = []
        anns = sorted(os.listdir('{}/{}/hdPose3d_stage1'.format(data_dir, dir)))
        for ann in anns:
            # if '00002000' not in ann:
            #     continue

            ann_fname = '{}/{}/hdPose3d_stage1/{}'.format(data_dir, dir, ann)
            frame_idx = int(ann.split('_')[1].split('.')[0])

            # skeleton
            JOINTIDX = [2, 1, 0, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

            # load 3D pose
            if not os.path.exists(ann_fname):
                assert ValueError('{} cannot find.'.format(ann_fname))

            poses, track_ids = [], []
            with open(ann_fname) as dfile:
                bframe = json.load(dfile)
            for body in bframe['bodies']:
                pid = body['id']
                skel = np.array(body['joints15']).reshape((-1, 4))  # [19, 4]
                valid = skel[:, 3:4]
                pt3d = skel[:, 0:3]  # cm to mm
                pose = np.concatenate([pt3d, valid], axis=-1)[JOINTIDX]  # [15, 3+1]
                poses.append(pose)
                track_ids.append(pid)
            if poses != []:
                poses = np.stack(poses, axis=0)
                track_ids = np.array(track_ids)

            # poses, track_ids, cam_intr, cam_dist = get_annotations(ann_fname, cam)
            seq_all_files.append([frame_idx, poses, track_ids, valid_cams])
        all_files['{}-poses'.format(dir)] = seq_all_files

        # break
    # print(all_files['val'])
    with open('{}/panoptic_all_ann_files_protocol2.pkl'.format(data_dir), 'wb') as f:
        pickle.dump(all_files, f)


def check_ann_file(data_dir, seq='170221_haggling_b1-poses'):
    with open('{}/panoptic_all_ann_files_protocol1.pkl'.format(data_dir), 'rb') as f:
        all_files = pickle.load(f)

    seq_all_poses = all_files['{}-poses'.format(seq)]
    frame_idx, poses, track_ids, all_cams = seq_all_poses[0]

    SKELETONS = [[0, 2], [1, 2], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [0, 9],
                 [0, 10], [9, 11], [10, 12], [11, 13], [12, 14]]

    for cam_idx in all_cams:
        img_name = '{}/{}/hdImgs/hd_00_{:02d}/{:08d}.jpg'.format(data_dir, seq, cam_idx, frame_idx)
        if not os.path.exists(img_name):
            print('not exist image {}'.format(img_name))
            continue

        img = cv2.imread(img_name)
        cam = all_files['{}-cam{:02d}'.format(seq, cam_idx)]
        valid = poses[:, 3:4]
        pose3d = 10 * (np.dot(poses[:, 0:3], cam['R'].T) + cam['t'].T)  # cm to mm
        # 0.5 means image is resize to half size of original size
        pose2d = projection(pose3d, cam['intr'] * 0.5, cam['distCoef'], simple_mode=False)

        pts2d = np.concatenate([pose2d, valid > 0.1], axis=-1)

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        sk_colors = [cmap(i) for i in np.linspace(0, 1, len(SKELETONS) + 2)]
        sk_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in sk_colors]

        pid_count = pts2d.shape[0]
        cmap = plt.get_cmap('rainbow')
        pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]
        pid_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

        for pid_color_idx, pose in enumerate(pts2d):
            for l, (j1, j2) in enumerate(SKELETONS):
                joint1 = pose[j1]
                joint2 = pose[j2]

                if joint1[2] > 0 and joint2[2] > 0:
                    cv2.line(img,
                             (int(joint1[0]), int(joint1[1])),
                             (int(joint2[0]), int(joint2[1])),
                             color=tuple(sk_colors[l]),
                             thickness=3)

                if joint1[2] > 0:
                    cv2.circle(
                        img,
                        thickness=-1,
                        center=(int(joint1[0]), int(joint1[1])),
                        radius=3,
                        # color=circle_color,
                        color=tuple(pid_colors[pid_color_idx])
                    )

                if joint2[2] > 0:
                    cv2.circle(
                        img,
                        thickness=-1,
                        center=(int(joint2[0]), int(joint2[1])),
                        radius=3,
                        # color=circle_color,
                        color=tuple(pid_colors[pid_color_idx])
                    )
        plt.figure(figsize=(15, 10))
        plt.imshow(img[..., ::-1])
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    data_dir = 'C:/Users/shihaozou/Desktop/panoptic-toolbox-master/data'
    # extract_pose_annotation(data_dir)
    # convert_video_to_image(data_dir, img_format='jpg', protocol=2)
    # convert_3d_keypoints_demo(data_dir, seq_name='160422_mafia2', cam_idx=0, frame_idx=12400, img_format='jpg')

    # test_seqs = ['170221_haggling_b1', '170221_haggling_b2', '170221_haggling_b3',
    #              '170228_haggling_b1', '170228_haggling_b2', '170228_haggling_b3']
    # prepare_panoptic_protocol1(data_dir, all_cams=np.array([3, 12, 23]))

    # all_seqs = ['160422_haggling1', '160422_mafia2', '160422_ultimatum1', '160906_pizza1']
    # all_cams = np.arange(0, 31)
    # prepare_panoptic_protocol2(data_dir, all_cams=all_cams)





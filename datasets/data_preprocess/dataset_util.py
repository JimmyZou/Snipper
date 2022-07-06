import time
import cv2
import numpy as np


def panoptic_visualization(img, kpts2d, traj_ids, exist_ids, seq_name, frame_idx, skeletons, save_dir=None):
    import matplotlib.pyplot as plt
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    sk_colors = [cmap(i) for i in np.linspace(0, 1, len(skeletons) + 2)]
    sk_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in sk_colors]

    pids = set(traj_ids)
    pid_count = len(pids)
    cmap = plt.get_cmap('rainbow')
    pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]
    pid_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

    img = img[:, :, ::-1].copy()
    for p, pid in enumerate(exist_ids):
        pid_color_idx = np.where(np.array(list(pids)) == pid)[0][0]

        pose = kpts2d[p]
        for l, (j1, j2) in enumerate(skeletons):
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

    if save_dir is None:
        print("../vis/{}_{:08d}.jpg".format(seq_name, frame_idx))
        cv2.imwrite("../vis/{}_{:08d}.jpg".format(seq_name, frame_idx), img)
    else:
        # print("{}/{}_{:03d}.jpg".format(save_dir, seq, img_id))
        # cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_dir, img)


def posetrack_visualization(imgs, kpts, track_ids, fname, skeletons, save_dir=None):
    import matplotlib.pyplot as plt

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    sk_colors = [cmap(i) for i in np.linspace(0, 1, len(skeletons) + 2)]
    sk_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in sk_colors]

    pids = set(np.concatenate(track_ids, axis=0))
    pid_count = len(pids)
    cmap = plt.get_cmap('rainbow')
    pid_colors = [cmap(i) for i in np.linspace(0, 1, pid_count)]
    pid_colors = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, ::-1].copy()
        track_id = track_ids[i]

        for p, pid in enumerate(track_id):
            pid_color_idx = np.where(np.array(list(pids)) == pid)[0][0]

            pose = kpts[i][p]
            for l, (j1, j2) in enumerate(skeletons):
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


        if save_dir is None:
            print("../vis/{}_{:04d}.jpg".format(fname, i))
            cv2.imwrite("../vis/{}_{:04d}.jpg".format(fname, i), img)
        else:
            # print("{}/{}_{:03d}.jpg".format(save_dir, seq, img_id))
            # cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_dir, img)


def get_colors(number_of_colors, cmap_name='rainbow'):
    # type: (int, str) -> List[List[int]]
    """
    :param number_of_colors: number of colors you want to get
    :param cmap_name: name of the colormap you want to use
    :return: list of 'number_of_colors' colors based on the required color map ('cmap_name')
    """
    # import matplotlib.pyplot as plt
    # colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, number_of_colors))[:, :-1]*255
    # return colors.astype(int).tolist()
    colors = np.random.random((number_of_colors, 3)) * 255
    return colors.astype(int).tolist()


def get_pose(frame_data, person_id):
    # type: (np.ndarray, int) -> Pose
    """
    :param frame_data: data of the current frame
    :param person_id: person identifier
    :return: list of joints in the current frame with the required person ID
    """
    pose = [Joint(j) for j in frame_data[frame_data[:, 1] == person_id]]
    pose.sort(key=(lambda j: j.type))
    return Pose(pose)


def jta_visualization(image, frame_data, seq, img_id, hide=True, save_dir=None):
    MAX_COLORS = 42

    colors = get_colors(number_of_colors=MAX_COLORS, cmap_name='jet')

    for p_id in set(frame_data[:, 1]):
        pose = get_pose(frame_data=frame_data, person_id=p_id)

        # print(pose.invisible)
        # if the "hide" flag is set, ignore the "invisible" poses
        # (invisible pose = pose of which I do not see any joint)
        if hide and pose.invisible:
            continue

        # select pose color base on its unique identifier
        color = colors[int(p_id) % len(colors)]

        # draw pose on image
        image = pose.draw(image=image, color=color, hide=hide)

    if save_dir is None:
        print("../vis/{}_{:03d}.jpg".format(seq, img_id))
        cv2.imwrite("../vis/{}_{:03d}.jpg".format(seq, img_id),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # print("{}/{}_{:03d}.jpg".format(save_dir, seq, img_id))
        # cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_dir, image[:, :, ::-1].copy())


class Pose(list):
    """
    a Pose is a list of Joint(s) belonging to the same person.
    """

    # LIMBS = [
    #     (0, 1),  # head_top -> head_center
    #     (1, 2),  # head_center -> neck
    #     (2, 3),  # neck -> right_clavicle
    #     (3, 4),  # right_clavicle -> right_shoulder
    #     (4, 5),  # right_shoulder -> right_elbow
    #     (5, 6),  # right_elbow -> right_wrist
    #     (2, 7),  # neck -> left_clavicle
    #     (7, 8),  # left_clavicle -> left_shoulder
    #     (8, 9),  # left_shoulder -> left_elbow
    #     (9, 10),  # left_elbow -> left_wrist
    #     (2, 11),  # neck -> spine0
    #     (11, 12),  # spine0 -> spine1
    #     (12, 13),  # spine1 -> spine2
    #     (13, 14),  # spine2 -> spine3
    #     (14, 15),  # spine3 -> spine4
    #     (15, 16),  # spine4 -> right_hip
    #     (16, 17),  # right_hip -> right_knee
    #     (17, 18),  # right_knee -> right_ankle
    #     (15, 19),  # spine4 -> left_hip
    #     (19, 20),  # left_hip -> left_knee
    #     (20, 21)  # left_knee -> left_ankle
    # ]

    # JTA 'keypoints': [0  'root',
    #                   1  'head_center',
    #                   2  'head_bottom',
    #                   3  'left_shoulder',
    #                   4  'right_shoulder',
    #                   5  'left_elbow',
    #                   6  'right_elbow',
    #                   7  'left_wrist',
    #                   8  'right_wrist',
    #                   9 'left_hip',
    #                   10 'right_hip',
    #                   11 'left_knee',
    #                   12 'right_knee',
    #                   13 'left_ankle',
    #                   14 'right_ankle']

    LIMBS = [
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


    SKELETON = [[l[0] + 1, l[1] + 1] for l in LIMBS]

    def __init__(self, joints):
        # type: (List[Joint]) -> None
        super().__init__(joints)

    @property
    def invisible(self):
        # type: () -> bool
        """
        :return: True if all the joints of the pose are occluded, False otherwise
        """
        for j in self:
            if not j.occ:
                return False
        return True

    @property
    def bbox_2d(self):
        # type: () -> List[int]
        """
        :return: bounding box around the pose in format [x_min, y_min, width, height]
            - x_min = x of the top left corner of the bounding box
            - y_min = y of the top left corner of the bounding box
        """
        x_min = int(np.min([j.x2d for j in self]))
        y_min = int(np.min([j.y2d for j in self]))
        x_max = int(np.max([j.x2d for j in self]))
        y_max = int(np.max([j.y2d for j in self]))
        width = x_max - x_min
        height = y_max - y_min
        return [x_min, y_min, width, height]

    @property
    def bbox_2d_padded(self, h_inc_perc=0.15, w_inc_perc=0.1):
        x_min = int(np.min([j.x2d for j in self]))
        y_min = int(np.min([j.y2d for j in self]))
        x_max = int(np.max([j.x2d for j in self]))
        y_max = int(np.max([j.y2d for j in self]))
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

        return [x_min, y_min, width, height]

    @property
    def coco_annotation(self):
        # type: () -> Dict
        """
        :return: COCO annotation dictionary of the pose
        ==========================================================
        NOTE#1: in COCO, each keypoint is represented by its (x,y)
        2D location and a visibility flag `v` defined as:
            - `v=0` ==> not labeled (in which case x=y=0)
            - `v=1` ==> labeled but not visible
            - `v=2` ==> labeled and visible
        ==========================================================
        NOTE#2: in COCO, a keypoint is considered visible if it
        falls inside the object segment. In JTA there are no
        object segments and every keypoint is labelled, so we
        v=2 for each keypoint.
        ==========================================================
        """
        keypoints = []
        for j in self:
            keypoints += [j.x2d, j.y2d, 2]
        annotation = {
            'keypoints': keypoints,
            'num_keypoints': len(self),
            'bbox': self.bbox_2d
        }
        return annotation

    def draw(self, image, color, hide):
        # type: (np.ndarray, List[int]) -> np.ndarray
        """
        :param image: image on which to draw the pose
        :param color: color of the limbs make up the pose
        :return: image with the pose
        """
        # draw limb(s) segments
        h, w, _ = image.shape
        for (j_id_a, j_id_b) in Pose.LIMBS:
            joint_a = self[j_id_a]  # type: Joint
            joint_b = self[j_id_b]  # type: Joint
            if (joint_a.occ or joint_b.occ) and hide:
                continue
            t = 1 if joint_a.cam_distance > 25 else 2
            if joint_a.is_on_screen(h, w) and joint_b.is_on_screen(h, w):
                cv2.line(image, joint_a.pos2d, joint_b.pos2d, color=color, thickness=t)

        # draw joint(s) circles
        for joint in self:
            if joint.occ and hide:
                continue
            image = joint.draw(image)

        return image

    def __iter__(self):
        # type: () -> Iterator[Joint]
        return super().__iter__()


class Joint(object):
    """
    a Joint is a keypoint of the human body.
    """
    # # list of joint names
    # NAMES = [
    #     'head_top',
    #     'head_center',
    #     'neck',
    #     'right_clavicle',
    #     'right_shoulder',
    #     'right_elbow',
    #     'right_wrist',
    #     'left_clavicle',
    #     'left_shoulder',
    #     'left_elbow',
    #     'left_wrist',
    #     'spine0',
    #     'spine1',
    #     'spine2',
    #     'spine3',
    #     'spine4',
    #     'right_hip',
    #     'right_knee',
    #     'right_ankle',
    #     'left_hip',
    #     'left_knee',
    #     'left_ankle',
    # ]
    NAMES = [
        'root',
        'nose',
        'head_bottom',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

    def __init__(self, array):
        """
        :param array: array version of the joint
        """
        self.frame = int(array[0])
        self.person_id = int(array[1])
        self.type = int(array[2])
        self.x2d = int(array[3])
        self.y2d = int(array[4])
        self.x3d = array[5]
        self.y3d = array[6]
        self.z3d = array[7]
        self.occ = bool(array[8])  # is this joint occluded?
        self.soc = bool(array[9])  # is this joint self-occluded?

    @property
    def cam_distance(self):
        """
        :return: distance of the joint from the camera
        """
        # NOTE: camera coords = (0, 0, 0)
        return np.sqrt(self.x3d**2 + self.y3d**2 + self.z3d**2)

    def is_on_screen(self, h, w):
        # type: (int, int) -> bool
        """
        :return: True if the joint is on screen, False otherwise
        """
        return (0 <= self.x2d <= w) and (0 <= self.y2d <= h)


    @property
    def visible(self):
        # type: () -> bool
        """
        :return: True if the joint is visible, False otherwise
        """
        return not (self.occ or self.soc)


    @property
    def pos2d(self):
        # type: () -> Tuple[int, int]
        """
        :return: 2D coordinates of the joints [px]
        """
        return (self.x2d, self.y2d)


    @property
    def pos3d(self):
        # type: () -> Tuple[float, float, float]
        """
        :return: 3D coordinates of the joints [m]
        """
        return (self.x3d, self.y3d, self.z3d)


    @property
    def color(self):
        # type: () -> Tuple[int, int, int]
        """
        :return: the color with which to draw the joint;
        this color is chosen based on the visibility of the joint:
        (1) occluded joint --> RED
        (2) self-occluded joint --> ORANGE
        (2) visible joint --> GREEN
        """
        if self.occ:
            return (255, 0, 42)  # red
        elif self.soc:
            return (255, 128, 42)  # orange
        else:
            return (0, 255, 42)  # green

    @property
    def radius(self):
        # type: () -> int
        """
        :return: appropriate radius [px] for the circle that represents the joint;
        this radius is a function of the distance of the joint from the camera
        """
        radius = int(round(np.power(10, 1 - (self.cam_distance/20.0))))
        return radius if radius >= 1 else 1

    @property
    def name(self):
        # type: () -> str
        """
        :return: name of the joint (eg: 'neck', 'left_elbow', ...)
        """
        return Joint.NAMES[self.type]

    def draw(self, image):
        # type: (np.ndarray) -> np.ndarray
        """
        :param image: image on which to draw the joint
        :return: image with the joint
        """
        image = cv2.circle(
            image, thickness=-1,
            center=self.pos2d,
            radius=self.radius,
            color=self.color,
        )
        return image

    def __str__(self):
        visibility = 'visible' if self.visible else 'occluded'
        return f'{self.name}|2D:({self.x2d},{self.y2d})|3D:({self.x3d},{self.y3d},{self.z3d})|{visibility}'

    __repr__ = __str__
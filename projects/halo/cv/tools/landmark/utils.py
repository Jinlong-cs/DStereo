import cv2
import numpy as np

HAND_SKELETON_CMU = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]
)

HAND_SKELETON_COLOR_CMU = [
    [0, 0, 255],
    [0, 76, 255],
    [0, 153, 255],
    [0, 229, 255],
    [0, 255, 203],
    [0, 255, 127],
    [0, 255, 51],
    [25, 255, 0],
    [102, 255, 0],
    [178, 255, 0],
    [255, 255, 0],
    [255, 178, 0],
    [255, 102, 0],
    [255, 25, 0],
    [255, 0, 50],
    [255, 0, 127],
    [255, 0, 204],
    [229, 0, 255],
    [152, 0, 255],
    [76, 0, 255],
]

FACE_LDMK68_LINES = [
    list(range(0, 17)),  # profile
    list(range(17, 22)),  # left eyebrow
    list(range(22, 27)),  # right eyebrow
    list(range(27, 36)) + [30],  # nose
    list(range(36, 42)) + [36],  # left eye
    list(range(42, 48)) + [42],  # right eye
    list(range(48, 60)) + [48],  # outer lips
    list(range(60, 68)) + [60],  # inner lips
]


class LandmarkVis:
    def __init__(self, vis_bbox=False, vis_boundary=False, vis_pose=False):
        self._vis_bbox = vis_bbox
        self._vis_boundary = vis_boundary
        self._vis_pose = vis_pose

    def _vis_all(self, img, **kwargs):
        ldmk = kwargs["ldmk"]
        num_ldmk = kwargs["num_ldmk"]
        ldmk_type = kwargs.get("ldmk_type", None)

        if self._vis_bbox:
            bbox = kwargs["bbox"]
            img = self.vis_bbox(img, bbox)
        if self._vis_pose:
            pose = kwargs["head pose"]
            img = self.vis_pose(img, pose)
        if self._vis_boundary:
            img = self.vis_boundary(img, ldmk, num_ldmk, ldmk_type)

        img = self.vis_landmark(img, ldmk, num_ldmk)

    @classmethod
    def vis_landmark(cls, img, ldmk, num_ldmk):
        if isinstance(ldmk, list):
            ldmk = np.array(ldmk)
        ldmk = ldmk.reshape(num_ldmk, -1).round().astype(np.int32)
        for k in range(ldmk.shape[0]):
            img = cv2.circle(
                img,
                (ldmk[k, 0], ldmk[k, 1]),
                color=(0, 255, 255),
                thickness=-1,
                radius=2,
            )
        return img

    @classmethod
    def vis_bbox(cls, img, bbox):
        x1, y1, x2, y2 = [int(x) for x in bbox]
        img = cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2,
        )
        return img

    @classmethod
    def vis_pose(cls, img, pose):
        raise NotImplementedError

    @classmethod
    def vis_boundary(cls, img, ldmk, num_ldmk, ldmk_type="hand21"):
        if ldmk_type.lower() == "hand21":
            if isinstance(ldmk, list):
                ldmk = np.array(ldmk)
            ldmk = ldmk.round().astype(np.int32).reshape(num_ldmk, -1)
            img = draw_hand_skeleton(img, ldmk)
        elif ldmk_type.lower() == "face68":
            for line in FACE_LDMK68_LINES:
                for i in range(len(line) - 1):
                    pt1 = tuple(ldmk[line[i]])
                    pt2 = tuple(ldmk[line[i + 1]])
                    cv2.line(img, pt1, pt2, color=(255, 255, 0), thickness=1)
        elif ldmk_type.lower() == "body17":
            raise NotImplementedError()
        else:
            raise ValueError(f"Not supported ldmk_type: {ldmk_type}")
        return img

    def run(self, img, ldmk):
        self._vis_all(img, ldmk)


def draw_hand_skeleton(
    img,
    kps,
    draw_skeleton=True,
    drawpoint=True,
    pointcolor=(0, 0, 255),
    thickness=1,
):
    kps = kps.round().astype(np.int32)
    if draw_skeleton:
        for j in range(HAND_SKELETON_CMU.shape[0]):
            p1 = HAND_SKELETON_CMU[j, 0]
            p2 = HAND_SKELETON_CMU[j, 1]
            x1 = kps[p1, 0]
            y1 = kps[p1, 1]
            x2 = kps[p2, 0]
            y2 = kps[p2, 1]

            color = HAND_SKELETON_COLOR_CMU[j % len(HAND_SKELETON_COLOR_CMU)]
            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    if drawpoint:
        for pt in kps:
            cv2.circle(
                img,
                (pt[0], pt[1]),
                1,
                pointcolor,
                -1,
                cv2.LINE_AA,
            )

    return img

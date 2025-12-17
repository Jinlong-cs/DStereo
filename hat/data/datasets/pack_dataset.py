import copy
import glob
import logging
from math import pi
from typing import List, Optional

import cv2
import numpy as np
import torch.utils.data as data

try:
    from tat._video import AVLogLevel, av_log_set_level
    from tat.matrix.msg.reader import MSGReader, TopicChannel
    from tat.matrix.pack_sdk import PackFileReader

    av_log_set_level(AVLogLevel.FATAL)
except ImportError:
    MSGReader = None
    TopicChannel = None
    PackFileReader = None

try:
    from tat._core import ColorRange, PixFormat
    from tat.video.image import convert_image

    PIXFORMAT = {
        "bgr": PixFormat.BGR,
        "rgb": PixFormat.BGR,  # Convert to RGB later.
        "nv12": PixFormat.NV12,
        "yu12": PixFormat.YU12,
        "yuv444p": PixFormat.YUV444P,
    }
    COLORRANGE = {
        "full": ColorRange.FULL,
        "limited": ColorRange.LIMITED,
    }
except ImportError:
    convert_image = None

from hat.core.virtual_camera import FisheyeCamera, PinholeCamera
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = ["PackDataset"]

logger = logging.getLogger(__name__)


PACK_CAMERA_PARAM_KEYS = [
    "camera_x",
    "camera_y",
    "camera_z",
    "pitch",
    "roll",
    "yaw",
    "center_u",
    "center_v",
    "distort",
    "focal_u",
    "focal_v",
    "type",
    "fov",
    "version",
    "vcs",
    "image_height",
    "image_width",
]


@OBJECT_REGISTRY.register
class PackDataset(data.Dataset):
    """Dataset which gets img data from the packs.

    Args:
        pack_path: Pack path that contain multi views. Like
        "/bucket/auto_data/xxx/ADAS_xxx_$Index.pack", use $Index instead index.
        camera_view_names: Sub directory name of each view.
        start_idx: Index expected to start.
        pix_format: The pixel format, support "None", "bgr",
        "rgb"(convert to rgb by cv2), "nv12", "yu12", "yuv444p".
        color_range: The color range, support "full" and "limited".
        expect_length: Length of data expected to obtain.
        expect_timestamp: Select timestamp combination expected to obtain,
            such as[{"view1": [111, 222, 333], "view2": [444, 555, 666]}.
        camera_calib: Whether need calibration parameter.
        transform: List of transform.
        topic: Topic of timeline.
        to_buf: Whether convert img to buf.
        with_cam: Whether add cam to dataset.
        camera_calib_key: The key to save camera calib.
        crop_roi: Whether add crop_roi to dataset.
        return_odometry: Whether add odometry (the vcs2global info) to dataset,
            the format of odometry is [x, y, yaw]. To ensure consistency with
            software implementation, the odo is calculated by interpolating
            the latest odometry info.
        interpolated_odometry: Whether interpolated odometry. If True, the
            odometry will be interpolated according the speed and yaw rate.
        return_pack_start_flag: Whether add pack start flag to dataset.
        return_pack_path: Whether add pack path to dataset.
        using_odo_diagnostic_code: Using diagnostic code to get odo status.

    """

    @require_packages(
        "tat>=0.2.1", raise_msg="Please `pip3 install horizon-tat>=0.2.1`"
    )
    def __init__(
        self,
        pack_path: str,
        camera_view_names: list,
        start_idx: int = 0,
        pix_format: str = None,
        color_range: str = "full",
        expect_length: int = None,
        expect_timestamp: dict = None,
        camera_calib: bool = False,
        transforms: Optional[List] = None,
        topic: tuple = ("image", 0),
        to_buf: bool = False,
        with_cam: bool = False,
        camera_calib_key: str = "camera_calib",
        crop_roi: bool = False,
        return_odometry: bool = False,
        interpolated_odometry: bool = False,
        return_pack_start_flag: bool = False,
        return_pack_path: bool = False,
        using_odo_diagnostic_code: bool = False,
    ):
        super(PackDataset, self).__init__()
        self.pack_path = pack_path
        self.views = camera_view_names
        self.start_idx = start_idx
        self.pix_format = pix_format
        self.color_range = color_range
        self.expect_length = expect_length
        self.expect_timestamp = expect_timestamp
        self.camera_calib = camera_calib
        self.transforms = transforms
        self.topic = topic
        self.to_buf = to_buf
        self.frame_ts = None
        self._client = None
        self.with_cam = with_cam
        self.camera_calib_key = camera_calib_key
        self.crop_roi = crop_roi
        self.return_odometry = return_odometry
        self.interpolated_odometry = interpolated_odometry
        self.return_pack_start_flag = return_pack_start_flag
        self.return_pack_path = return_pack_path
        self.pre_odo_diagnostic_code = 0
        self.using_odo_diagnostic_code = using_odo_diagnostic_code

        assert (
            "$Index" in self.pack_path
        ), "Please replace pack index with $Index."
        assert self.start_idx >= 0

        if self.expect_timestamp:
            for view, _ in self.expect_timestamp.items():
                assert (
                    view in self.views
                ), f"please make sure the {view} in expect timestamp dict."

        if convert_image is None and self.pix_format is not None:
            # `pix_format` must be None when tat version lower than 0.2.1
            raise ImportError(
                "Unable to import convert_image from tat,"
                "please make sure the version horizon-tat>=0.2.1"
            )
        if convert_image is not None and self.pix_format is None:
            logger.warning(
                "`pix_format` is None, pack will return ImageFrame object"
            )

        (
            self.img_reader,
            self.param_reader,
            self.roi_reader,
            self.odo_reader,
        ) = self._get_reader()
        self.camera_params = self._get_camera_params()
        self.timelines, self.pack_frame_ids = self._build_timelines()
        self.pack_start_index, self.pack_end_index = self._get_pack_index()
        self.cameras_inst = self._get_cameras_inst()
        # Pack_flags identities data belonging to different packs. Data
        # belonging to the same pack has the same pack_flag and vice versa.
        # As all data in a pack dataset belong to the same pack, the
        # pack_flag is set to zeros for all data.
        self.pack_flag = np.zeros(len(self))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("img_reader")
        state.pop("param_reader")
        state.pop("odo_reader")
        state.pop("camera_params")
        state.pop("pack_start_index")
        state.pop("pack_end_index")
        state.pop("roi_reader")
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        (
            self.img_reader,
            self.param_reader,
            self.roi_reader,
            self.odo_reader,
        ) = self._get_reader()
        self.camera_params = self._get_camera_params()
        self.pack_start_index, self.pack_end_index = self._get_pack_index()

    def _get_cameras_inst(self):
        cameras_inst = {}
        if isinstance(self.camera_params, dict):
            for view in self.camera_params.keys():
                if "fisheye" not in view:
                    cameras_inst[view] = PinholeCamera.init_cam_param_by_dict(
                        self.camera_params[view]
                    )
                else:
                    self.camera_params[view]["distort"] = self.camera_params[
                        view
                    ]["distort"][:4]
                    cameras_inst[view] = FisheyeCamera.init_cam_param_by_dict(
                        self.camera_params[view]
                    )
        return cameras_inst

    def _get_pack_index(self):
        """Set start idx and end idx to sync frame in pack."""

        if not self.expect_timestamp:
            start_ts, end_ts = [], []
            for view in self.views:
                start_ts.append(self.timelines[view][0])
                end_ts.append(self.timelines[view][-1])

            start_ts, end_ts = max(start_ts), min(end_ts)
            ref_timeline = self.timelines[self.views[0]]
            ref_pack_frame_ids = self.pack_frame_ids[self.views[0]]
            start_index, _ = self.find_nearest_ts(
                start_ts, ref_timeline, ref_pack_frame_ids
            )
            end_index, _ = self.find_nearest_ts(
                end_ts, ref_timeline, ref_pack_frame_ids
            )
        else:
            start_index, end_index = (
                0,
                len(self.expect_timestamp[self.views[0]]) - 1,
            )
        assert end_index >= start_index
        return start_index, end_index

    def _get_reader(self):
        """Get the reader of each pack and cach it."""

        img_reader, param_reader, roi_reader = {}, {}, {}
        glob_pack_path = glob.glob(self.pack_path.replace("$Index", "*"))
        for pack_path in glob_pack_path:
            cam_reader = MSGReader(
                handle=PackFileReader(pack_path),
                topic_channel=[
                    TopicChannel("camera_default", 0),
                ],
                decode_data=True,
            )
            if not cam_reader.Read()["camera_default"]:
                logging.warning(f"The camera param of {pack_path} is None")
                continue
            proto = cam_reader.Read()["camera_default"][0].proto[0]
            fov, proto_type, yaw = (
                proto.fov,
                proto.type,
                proto.vcs.rotation[-1],
            )

            yaw %= 2 * pi
            if yaw < 0:
                yaw += 2 * pi

            # each camera's proto_type/fov/standard_yaw as follow:
            #                     | proto_type | fov    | yaw (radian) |
            # camera_front_30fov  |     0      | 30     |     0        |
            # camera_front        |     0      | 120    |     0        |
            # camera_front_left   |     0      | 100    |     pi/3     |
            # camera_rear_left    |     0      | 100    |     2*pi/3   |
            # camera_rear         |     0      | 100/60 |     pi       |
            # camera_rear_right   |     0      | 100    |     4*pi/3   |
            # camera_front_right  |     0      | 100    |     5*pi/3   |
            # fisheye_front       |     1      | >180   |     0        |
            # fisheye_left        |     1      | >180   |     pi/2     |
            # fisheye_rear        |     1      | >180   |     pi       |
            # fisheye_right       |     1      | >180   |     3*pi/2   |

            delta_fov = 5
            delta_theta = pi / 6
            fisheye_delta_theta = pi / 4
            if proto_type == 1:
                if abs(yaw - pi / 2) <= fisheye_delta_theta:
                    view = "fisheye_left"
                elif abs(yaw - pi) < fisheye_delta_theta:
                    view = "fisheye_rear"
                elif abs(yaw - 3 * pi / 2) <= fisheye_delta_theta:
                    view = "fisheye_right"
                else:
                    view = "fisheye_front"
            elif proto_type == 0:
                if abs(fov - 30) <= delta_fov:
                    view = "camera_front_30fov"
                else:
                    if abs(yaw - pi / 3) <= delta_theta:
                        view = "camera_front_left"
                    elif abs(yaw - 2 * pi / 3) < delta_theta:
                        view = "camera_rear_left"
                    elif abs(yaw - pi) <= delta_theta:
                        view = "camera_rear"
                    elif abs(yaw - 4 * pi / 3) < delta_theta:
                        view = "camera_rear_right"
                    elif abs(yaw - 5 * pi / 3) <= delta_theta:
                        view = "camera_front_right"
                    else:
                        view = "camera_front"

            param_reader[view] = cam_reader
            img_reader[view] = MSGReader(
                handle=PackFileReader(pack_path),
                topic_channel=[
                    TopicChannel("image", 0),
                ],
                decode_data=True,
            )
            roi_reader[view] = MSGReader(
                handle=PackFileReader(pack_path),
                topic_channel=[
                    TopicChannel("crop_roi", 0),
                ],
                decode_data=True,
            )

            # load odometry message, saved in camera front pack
            if view == "camera_front":
                camera_front_pack_path = pack_path

        for view in self.views:
            assert (
                view in img_reader
            ), f"Not found {view} pack, please check pack is fully prepared."

        if self.return_odometry:
            # load odometry message, default saved in camera front pack, if
            # there is a "ODO" pack in the folder, use it instead.
            odo_pack_path = glob.glob(
                self.pack_path.replace("$Index", "*").replace(
                    "/ADAS_", "/ODO#_"
                )
            )
            if len(odo_pack_path) == 0:
                odo_pack_path = camera_front_pack_path
                logging.warning(
                    "ODO pack not found, use the odometry in front pack,"
                    + "which may be inconsistant with the odometry "
                    + "applied in the software."
                )
            else:
                assert (
                    len(odo_pack_path) == 1
                ), "There should be only one ODO pack in the folder."
                odo_pack_path = odo_pack_path[0]
            odo_reader = MSGReader(
                handle=PackFileReader(odo_pack_path),
                topic_channel=[
                    TopicChannel("odometry", 0),
                ],
                decode_data=True,
            )
        else:
            odo_reader = None

        return img_reader, param_reader, roi_reader, odo_reader

    def _build_timelines(self):
        """Build timeline for reader."""
        self.timelines, self.pack_frame_ids = {}, {}

        def get_reader_timelines_and_pack_frame_ids(reader):
            reader.pack_reader.SeekByIndex(0)
            pack_frame_ids = []
            timelines = []
            while True:
                pack_frame_idx = reader.pack_reader.TellIndex()
                try:
                    frame = reader.pack_reader.Read(need_data=False)
                    if not frame.valid:
                        break
                except RuntimeError as err:
                    logging.warning(
                        "Error while reading {} pack. ".format(view)
                        + "Error {}.".format(err)
                    )
                    break
                ts = int(frame.timestamp)
                timelines.append(ts)
                pack_frame_ids.append(pack_frame_idx)
            return np.array(timelines, dtype=np.int64), pack_frame_ids

        for view, reader in self.img_reader.items():
            (
                timelines,
                pack_frame_ids,
            ) = get_reader_timelines_and_pack_frame_ids(reader)
            self.timelines[view] = timelines
            self.pack_frame_ids[view] = pack_frame_ids

        if self.return_odometry:
            (
                odo_timeline,
                odo_pack_frame_ids,
            ) = get_reader_timelines_and_pack_frame_ids(self.odo_reader)
            self.timelines["odo"] = odo_timeline
            self.pack_frame_ids["odo"] = odo_pack_frame_ids

        return self.timelines, self.pack_frame_ids

    def _get_camera_params(self):
        camera_params = {}
        for view in self.views:
            reader = self.param_reader[view]
            camera_params[view] = self._extract_camera_params(reader)
        return camera_params

    def _extract_camera_params(self, reader):
        """Extract camera params from pack."""

        msgs = reader.Read()
        data = {}
        camera = msgs["camera_default"][0]

        # convert data type
        for key in PACK_CAMERA_PARAM_KEYS:
            value = getattr(camera.proto[0], key)
            if key == "distort":
                value = list(value.param)
                if len(value) != 8:  # fisheye
                    # For fisheye cam, calib distort has 4 params,
                    # corr. to the 4 inputs of cv2.fisheye.distortPoints.
                    # While for pinhole cam, calib distort has 8 params,
                    # corr. to the 8 inputs of cv2.projectPoints.
                    # Complete fisheye distort to 8 params for 10v batch
                    value += [0.0] * (8 - len(value))
            if key == "vcs":
                vcs_value = copy.deepcopy(value)
                value = {}
                for key_ in ["rotation", "translation"]:
                    value[key_] = list(getattr(vcs_value, key_))
            data[key] = value

        return data

    def find_nearest_ts(self, timestamp, timeline, pack_frame_ids):
        ts = int(timestamp)
        idx = np.argmin(np.abs(ts - timeline))
        return pack_frame_ids[idx], timeline[idx]

    def odometry_interpolation(
        self, nearest_timestamp, target_timestamp, odo_msg
    ):
        # Interpolate odometry according speed, yaw rate and target timestamp.
        x = odo_msg.proto[0].x
        y = odo_msg.proto[0].y
        yaw = odo_msg.proto[0].yaw
        speed = odo_msg.proto[0].speed
        yaw_rate = odo_msg.proto[0].yaw_rate
        delta_time = (target_timestamp - nearest_timestamp) / 1000.0
        x = x + np.cos(yaw) * speed * delta_time
        y = y + np.sin(yaw) * speed * delta_time
        yaw = yaw + yaw_rate * delta_time
        return np.array([x, y, yaw])

    def __getitem__(self, index):
        """Get images from all views at one timestamp."""
        if self.expect_timestamp is None:
            index += self.start_idx + self.pack_start_index
        else:
            self.frame_ts = self.expect_timestamp

        images = []
        img_bufs = []
        image_name_list = []
        crop_roi = []
        for view in self.views:
            reader = self.img_reader[view]
            view_timeline = self.timelines[view]
            sub_roi_reader = self.roi_reader[view]
            if not self.frame_ts:
                self.frame_ts = view_timeline[index]
            if self.expect_timestamp:
                view_ts = int(self.frame_ts[view][index])
            else:
                view_ts = self.frame_ts
            view_pack_frame_ids = self.pack_frame_ids[view]
            target_frame_id, _ = self.find_nearest_ts(
                view_ts, view_timeline, view_pack_frame_ids
            )
            reader.SeekByIndex(target_frame_id)
            msgs = reader.Read()
            img_msg = msgs["image"][0]

            sub_roi_reader.SeekByIndex(target_frame_id)
            roi_msgs = sub_roi_reader.Read()
            if roi_msgs.get("crop_roi", None) is not None:
                roi = roi_msgs["crop_roi"][0].proto[0].roi
                crop_roi += [
                    (
                        int(roi.left),
                        int(roi.top),
                        int(roi.right),
                        int(roi.bottom),
                    )
                ]
            else:
                crop_roi += [None]
            if self.pix_format is not None:
                image = convert_image(
                    img_msg.data[0],
                    PIXFORMAT[self.pix_format],
                    COLORRANGE[self.color_range],
                ).numpy()
            else:
                image = img_msg.data[0]

            if self.pix_format == "rgb":
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            if self.to_buf:
                img_buf = cv2.imencode(".jpg", image)[1].tobytes()
                img_bufs.append(img_buf)

            images.append(image)
            image_name_list.append(str(img_msg.frame.timestamp) + ".jpg")

        data_dict = {}
        main_view_frame_ts = (
            self.frame_ts[self.views[0]][index]
            if isinstance(self.frame_ts, dict)
            else self.frame_ts
        )
        data_dict["timestamp"] = np.array(main_view_frame_ts)
        data_dict["camera_list"] = self.views
        if self.with_cam:
            data_dict["meta_info"] = {}
            data_dict["meta_info"]["src_cam"] = copy.deepcopy(
                self.cameras_inst
            )
            data_dict["meta_info"]["dst_cam"] = copy.deepcopy(
                self.cameras_inst
            )

        if self.to_buf:
            data_dict["ori_img"] = (
                images[0] if len(self.views) == 1 else images
            )
            data_dict["img_name"] = (
                image_name_list[0] if len(self.views) == 1 else image_name_list
            )
            data_dict["layout"] = "hwc"
            data_dict["img_shape"] = image.shape
            data_dict["img_id"] = index
            data_dict["img_buf"] = (
                img_bufs[0] if len(self.views) == 1 else img_bufs
            )
        else:
            data_dict["img"] = images[0] if len(self.views) == 1 else images

        if self.camera_calib:
            if len(self.views) == 1:
                data_dict[self.camera_calib_key] = self.camera_params[
                    self.views[0]
                ]
            else:
                data_dict[self.camera_calib_key] = []
                for view in self.views:
                    data_dict[self.camera_calib_key].append(
                        self.camera_params[view]
                    )
        if self.crop_roi:
            if len(self.views) == 1:
                if crop_roi[0] is not None:
                    data_dict["crop_roi"] = crop_roi[0]
            else:
                data_dict["crop_roi"] = []
                for idx in range(len(self.views)):
                    if crop_roi[idx] is not None:
                        data_dict["crop_roi"] = crop_roi[idx]
        if self.return_odometry:
            # load odo info
            odo_timeline = self.timelines["odo"]
            odo_pack_frame_ids = self.pack_frame_ids["odo"]
            # search the index of neastest timestamp and
            # calculate the odo info.
            target_frame_id, nearest_timestamp = self.find_nearest_ts(
                main_view_frame_ts, odo_timeline, odo_pack_frame_ids
            )
            self.odo_reader.SeekByIndex(target_frame_id)
            odo_msgs = self.odo_reader.Read()
            odo_msg = odo_msgs["odometry"][0]
            if self.using_odo_diagnostic_code:
                odo_diagnostic_code = odo_msg.proto[0].diagnostic_code
            else:
                odo_diagnostic_code = 0
            if odo_diagnostic_code > 0:
                odo_info = np.array(
                    [
                        0,
                        0,
                        0,
                    ]
                )
            else:
                if self.interpolated_odometry:
                    odo_info = self.odometry_interpolation(
                        nearest_timestamp, main_view_frame_ts, odo_msg
                    )
                else:
                    odo_info = np.array(
                        [
                            odo_msg.proto[0].x,
                            odo_msg.proto[0].y,
                            odo_msg.proto[0].yaw,
                        ]
                    )
            data_dict["odo_info"] = odo_info
            data_dict["odo_invaild_flag"] = (
                odo_diagnostic_code > 0 or self.pre_odo_diagnostic_code > 0
            )
            self.pre_odo_diagnostic_code = (
                odo_diagnostic_code  # update history odo code
            )
        if self.return_pack_start_flag:
            if self.expect_length is None:
                if index == self.start_idx + self.pack_start_index:
                    return_pack_start_flag = True
                else:
                    return_pack_start_flag = False
            elif index == 0:
                return_pack_start_flag = True
            else:
                return_pack_start_flag = False
            data_dict["pack_start_flag"] = return_pack_start_flag
        if self.return_pack_path:
            data_dict["pack_path"] = self.pack_path

        self.frame_ts = None

        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __len__(self):
        if self.expect_length is not None:
            return self.expect_length
        length = self.pack_end_index - self.pack_start_index + 1
        length = length - self.start_idx
        return length

    def __repr__(self):
        return "PackDataset"

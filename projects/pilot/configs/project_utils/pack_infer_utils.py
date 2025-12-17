import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Union

from hatbc.utils import _as_list

try:
    from tat.matrix.msg.reader import MSGReader, TopicChannel
except ImportError:
    MSGReader = None
    TopicChannel = None


logger = logging.getLogger(__name__)


def get_bev_pack_timestamps(
    camera_view_names: List[str],
    bev_pack_path: str,
    max_num_frame: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Get bev task timestamp combination in fillback pack.

    Args:
        camera_view_names (List[str]): The names of each views.
        bev_pack_path (str): Filback BEV pack file path.
        max_num_frame (Optional[int], optional): Max number sync frames
            to be returned. Defaults to None.

    Returns:
        Dict[str, List[str]]: Target timestamps of each views.
            e.g.
            expect_timestamp = dict(
                camera_front=[1666076602164, 1666076602231],
                camera_front_left=[1666076602165, 1666076602232],
                camera_front_right=[1666076602165, 1666076602232],
                camera_rear_left=[1666076602165, 1666076602232],
                camera_rear_right=[1666076602165, 1666076602232],
                camera_rear=[1666076602166, 1666076602232],
                camera_front_30fov=[1666076602160, 1666076602227],
            ).
    """
    reader = MSGReader(
        handle=bev_pack_path,
        topic_channel=[TopicChannel("bev_param_vec", 0)],
        decode_data=True,
    )

    sync_timestamps = defaultdict(list)
    reader.SeekByIndex(0)
    count = 0
    while True:
        reader.TellIndex()
        megs = reader.Read()
        if not megs:
            break
        megs = megs["bev_param_vec"][0].proto[0].cam_params
        for view, meg in zip(camera_view_names, megs):
            sync_timestamps[view].append(meg.time_stamp)

        count += 1
        if max_num_frame and count >= max_num_frame:
            break

    return sync_timestamps


def build_pack_datasets(
    pack_path: Union[str, List[str]],
    transforms: List,
    camera_view_names: List[str],
    bev_pack_path: Optional[Union[str, List[str]]] = None,
    max_num_frame: Optional[int] = None,
    return_odometry: Optional[bool] = False,
    interpolated_odometry: Optional[bool] = False,
    using_odo_diagnostic_code: Optional[bool] = False,
) -> List[Dict]:
    pack_path = _as_list(pack_path)
    if not bev_pack_path:
        bev_pack_path = [None] * len(pack_path)
    bev_pack_path = _as_list(bev_pack_path)
    assert len(pack_path) == len(bev_pack_path)

    datasets = []
    for _pack, _bev_pack in zip(pack_path, bev_pack_path):
        expect_timestamp = None
        if _bev_pack:
            expect_timestamp = get_bev_pack_timestamps(
                camera_view_names=camera_view_names,
                bev_pack_path=_bev_pack,
                max_num_frame=max_num_frame,
            )

        _dataset = dict(  # noqa
            type="PackDataset",
            pix_format="nv12",
            pack_path=_pack,
            camera_view_names=camera_view_names,
            expect_timestamp=expect_timestamp,
            camera_calib=True,
            transforms=deepcopy(transforms),
            with_cam=False,
            return_odometry=return_odometry,
            interpolated_odometry=interpolated_odometry,
            return_pack_start_flag=True,
            return_pack_path=True,
            expect_length=max_num_frame,
            using_odo_diagnostic_code=using_odo_diagnostic_code,
        )

        datasets.append(_dataset)

    return datasets


def update_pack_transform(
    transforms: List[Dict], sensor_module: Dict, camera_view_names: List
):
    def _update_list(input_list, input_keys, update_list, update_keys) -> list:
        """Update input_list with update_list as the input_keys.

        Args:
            input_list (list): input list for update.
            input_keys (list): input keys correspond to input list.
            update_list (list): list to update input list.
            update_keys (list): update keys correspond to update list.
            nv12_format: 设置True表示输入图像处理和回灌对齐，False表示和训练对齐

        Returns:
            list: updated list.
        """
        assert len(input_list) == len(input_keys)
        assert len(update_list) == len(update_keys)

        # list -> dict
        input_dic = {k: v for k, v in zip(input_keys, input_list)}
        update_dic = {k: v for k, v in zip(update_keys, update_list)}
        input_dic = _update_dict(input_dic, update_dic)

        # dict -> list
        ret_input_list = list(input_dic.values())
        return ret_input_list

    def _update_dict(input_dict, update_dict) -> dict:
        """Update input_dict with update_dict as the input_dict keys.

        Args:
            input_dict (dict): input list for update.
            update_dict (dict): list to update input list.

        Returns:
            dict: updated dict.
        """
        assert isinstance(input_dict, dict) and isinstance(update_dict, dict)
        for k in input_dict.keys():
            if k in update_dict:
                input_dict[k] = update_dict[k]
        return input_dict

    update_camera_names = list(sensor_module["per_view_shape"].keys())
    assert set(camera_view_names).issubset(
        set(update_camera_names)
    ), "Model camera views must be included in the pack's."
    for transform in transforms:
        if transform["type"] == "ANCConvertPackDataTo3DV":
            for key in transform["homo_gen"].keys():
                if key == "homo_transforms":
                    update_homo_transforms = get_homo_transforms(
                        [
                            sensor_module["transforms"]["ANCResize3DV"],
                            sensor_module["transforms"]["ANCCrop3DV"],
                        ],
                        update_camera_names,
                    )
                    transform["homo_gen"][key] = _update_dict(
                        transform["homo_gen"][key],
                        update_homo_transforms,
                    )
                elif key in sensor_module.keys():
                    transform["homo_gen"][key] = _update_dict(
                        transform["homo_gen"][key],
                        sensor_module[key],
                    )
        elif transform["type"] == "ANCNV12Transform3DV":
            transform["ori_size"] = [
                sensor_module["per_view_shape"][view]
                for view in camera_view_names
            ]
        elif transform["type"] in sensor_module["transforms"].keys():
            for k, v in transform.items():
                if isinstance(v, list):
                    transform[k] = _update_list(
                        transform[k],
                        camera_view_names,
                        sensor_module["transforms"][transform["type"]][k],
                        update_camera_names,
                    )
                elif isinstance(v, dict):
                    transform[k] = _update_dict(
                        transform[k],
                        sensor_module["transforms"][transform["type"]][k],
                    )
                else:  # str, int, float
                    transform[k] = sensor_module["transforms"][
                        transform["type"]
                    ][k]

    return transforms


def get_homo_transforms(
    transforms_list: List[Dict], camera_view_names: List[str]
):
    homo_transforms = {}
    for _index, view in enumerate(camera_view_names):
        view_transform = {}
        for _transform in transforms_list:
            if not _transform:
                continue
            if _transform["type"] == "ANCResize3DV":
                assert len(_transform["size"]) == len(camera_view_names)
                view_transform["Resize"] = _transform["size"][_index]
            if _transform["type"] == "ANCPad3DV":
                assert len(_transform["paddings"]) == len(camera_view_names)
                view_transform["Pad"] = _transform["paddings"][_index]
            if _transform["type"] == "ANCCrop3DV":
                assert (
                    len(_transform["top"])
                    == len(_transform["left"])
                    == len(_transform["height"])
                    == len(_transform["width"])
                    == len(camera_view_names)
                )
                view_transform["Crop"] = (
                    _transform["top"][_index],
                    _transform["left"][_index],
                    _transform["height"][_index],
                    _transform["width"][_index],
                )
        homo_transforms[view] = view_transform
    return homo_transforms

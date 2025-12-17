import copy
import logging
import math
import os
import random
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Sequence

import fsspec
import numpy as np
import torch
import yaml

from hat.data.collates.collates import collate_3d
from hat.data.samplers.dist_group_sampler import DistributedGroupSampler
from hat.utils.apply_func import _as_list, is_list_of_type
from hat.utils.config import Config
from hat.utils.seed import worker_reset_seed

try:
    from tat.matrix.msg.reader import MSGReader, TopicChannel
except ImportError:
    MSGReader = None
    TopicChannel = None

logger = logging.getLogger(__name__)


STAGE1_SEG_DICT = {
    "33_8": {
        0: 0,
        1: 1,
        2: 7,
        3: 1,
        4: 7,
        5: 7,
        6: 7,
        7: 7,
        8: 2,
        9: 7,
        10: 7,
        11: 7,
        12: 7,
        13: 7,
        14: 7,
        15: 7,
        16: 7,
        17: 7,
        18: 7,
        19: 3,
        20: 7,
        21: 7,
        22: 7,
        23: 7,
        24: 4,
        25: 5,
        26: 7,
        27: 6,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        255: 7,
    },  # side_walk_terrain
    "33_16": {
        0: 0,
        1: 1,
        2: 15,
        3: 1,
        4: 15,
        5: 15,
        6: 15,
        7: 15,
        8: 2,
        9: 7,
        10: 8,
        11: 9,
        12: 10,
        13: 11,
        14: 12,
        15: 13,
        16: 14,
        17: 15,
        18: 15,
        19: 3,
        20: 15,
        21: 15,
        22: 15,
        23: 15,
        24: 4,
        25: 5,
        26: 15,
        27: 6,
        28: 15,
        29: 15,
        30: 15,
        31: 15,
        32: 15,
        33: 15,
        255: 15,
    },
}


def get_data_dict(url):
    with fsspec.open(url, "r") as fid:
        content = fid.read()
    data_dict = yaml.safe_load(content)
    return data_dict


def clean_homogen(homogen, info):
    """To remove reduntant keys-relate in homogen.

    模型某些视角加载不到数据时，需要清理这些视角的homogen。
    Example：
    (1) 模型为11v，原始数据为7v，实际使用7v，鱼眼数据缺失，需去除
        模型homogen中鱼眼视角的key。
        10v/11v加载4v/6v数据时，6v/7v/10v/11v加载1v数据时，同理。
    (2) 模型为7v，原始数据为10v，实际使用6v，鱼眼数据为多余数据，
        窄角数据缺失，需去除模型homogen中窄角的key。
    (3) 模型为4v/6v，原始数据为10v/11v，实际使用4v/6v，仅有多余数据，
        无缺失数据，无需去除homogen中的任何key。
    (4) 某些视角的数据有问题，不参与训练，通过以下方式来去除这些视角：
        (a) 剩余视角恰好与相机模组关联，如11v数据去除窄角，组成10v数据。
            此时以下两种操作均可选：
            - 对齐camera_module_type到10v，如 X8b_X3c_isx031_X8b
                -> X8b_X3c_isx031.
            - 保留camera_module_type不变，数据属性内设置去除窄角后的
                camera_view_names
        (b) 剩余视角不能恰好关联相机模组类型，如11v去除了前视和窄角，只能
            于数据属性内设置去除前视和窄角的camera_view_names

    模型视角名称：homo_camera_view_names
    数据视角名称：data_camera_view_names
    冗余的key是通过计算模型视角名称和数据视角名称的差集来实现的。
    函数会清除homogen中下述三项中冗余的key, 以保证homo参数的合法性：
    {
        "camera_view_names",
        "per_view_shape",
        "homo_transforms",
    }
    """

    def _clean(input_data, keys):
        """Remove specific keys in input_data"""
        if isinstance(input_data, dict):
            return dict(filter(lambda x: x[0] not in keys, input_data.items()))
        elif isinstance(input_data, list):
            return list(filter(lambda x: x not in keys, input_data))
        else:
            raise TypeError("Not support type")

    data_camera_view_names = info["camera_view_names"]
    homo_camera_view_names = homogen["camera_view_names"]
    remove_keys = list(
        set(homo_camera_view_names) - set(data_camera_view_names)
    )

    clean_contents = [
        "camera_view_names",
        "per_view_shape",
        "homo_transforms",
    ]
    for clean_content in clean_contents:
        homogen[clean_content] = _clean(homogen[clean_content], remove_keys)
    return homogen


def update_dataset_with_module_info(dataset, update_info_dict):
    """update module-related infos for common bev dataset

        NOTE: this func will be used only update_info_dict not None.

    Args:
        dataset (dict): common dataset which contains all things
        update_info_dict (dict): infos used to update dataset

    Returns:
        dict: updated dataset
    """

    def _update_list(input_list, input_keys, update_list, update_keys) -> list:
        """Update input_list with update_list as the input_keys.

        Args:
            input_list (list): input list for update.
            input_keys (list): input keys correspond to input list.
            update_list (list): list to update input list.
            update_keys (list): update keys correspond to update list.

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

    assert (
        "per_view_shape" in update_info_dict
    ), "per_view_shape must in bev_transform_factory !"
    update_camera_names = list(update_info_dict["per_view_shape"].keys())
    dataset_camera_names = dataset["camera_view_names"]
    for name, update_info in update_info_dict.items():
        # update transforms
        if name == "transforms":
            for transform in dataset["transforms"]:
                if transform["type"] in update_info.keys():
                    for k, v in transform.items():
                        if isinstance(v, list):
                            transform[k] = _update_list(
                                transform[k],
                                dataset_camera_names,
                                update_info[transform["type"]][k],
                                update_camera_names,
                            )
                        elif isinstance(v, dict):
                            transform[k] = _update_dict(
                                transform[k],
                                update_info[transform["type"]][k],
                            )
                        else:  # str, int, float
                            transform[k] = update_info[transform["type"]][k]
            # update homo_transforms of homogen
            update_tranforms = [  # noqa
                trans  # noqa
                for trans in update_info_dict["transforms"].values()  # noqa
            ]
            update_homo_transforms = get_homo_transforms(
                update_tranforms, update_camera_names
            )
            homo_gen_names = ["homo_gen", "homo_gen_small"]
            for homo_gen_name in homo_gen_names:
                if (
                    homo_gen_name in dataset
                    and dataset[homo_gen_name] is not None
                ):
                    dataset[homo_gen_name]["homo_transforms"] = _update_dict(
                        dataset[homo_gen_name]["homo_transforms"],
                        update_homo_transforms,
                    )
        else:  # update other parameters
            if name in dataset:
                if isinstance(update_info, dict):
                    dataset[name] = _update_dict(dataset[name], update_info)
                elif isinstance(update_info, list):
                    dataset[name] = _update_list(
                        dataset[name],
                        dataset_camera_names,
                        update_info,
                        update_camera_names,
                    )
                else:
                    dataset[name] = update_info
            homo_gen_names = ["homo_gen", "homo_gen_small"]
            for homo_gen_name in homo_gen_names:
                if homo_gen_name in dataset and name in dataset[homo_gen_name]:
                    dataset[homo_gen_name][name] = _update_dict(
                        dataset[homo_gen_name][name], update_info
                    )
    return dataset


def get_update_bev_dataset_func(
    added_info,
    update_trans_info={},  # noqa
    allow_view_data_miss=True,
    use_stage1_loss=False,
):
    # the "update_bev_dataset" only contains the common bev_data, for the
    # added info, should add it in the bev_xxx.config, e.g. homo_offset_path,
    # homo_path
    """
    Args:
        update_trans_info (dict): Specially parameter should be update
            in dataset.yaml, e.g. filter_vcs_range in  Bev3dTargetGenerator.
            Defaults to dict().
    """
    added_info = _as_list(added_info)
    bev_transform_factory = (
        f"{os.path.dirname(__file__)}/bev_transform_factory.py"
    )
    bev_transform_factory = Config.fromfile(bev_transform_factory)

    def update_bev_dataset(dataset, info, global_sample_interval=1):
        data_camera_view_names = info.get("camera_view_names", None)
        if data_camera_view_names is None:
            camera_module_type = info.get("camera_module_type", None)
            if camera_module_type is None:
                camera_module_type = "front0820_weisen0233"
                info["camera_module_type"] = camera_module_type
            data_camera_view_names = getattr(
                bev_transform_factory, camera_module_type
            )["per_view_shape"].keys()
            info["camera_view_names"] = list(data_camera_view_names)
        if not allow_view_data_miss:
            model_camera_view_names = dataset["camera_view_names"]
            assert set(model_camera_view_names).issubset(
                set(data_camera_view_names)
            ), "Data misses in some views, which is not allowd."

        info_keys = [
            "root",
            "sync_file",
            "sync_file_lmdb",
            "img_data_path",
            "HDE_data_path",
            "fill_fake_temporal_data",
        ]
        if use_stage1_loss:
            info_keys.append("seg_data_path")
        for info_key in info_keys:
            dataset[info_key] = info[info_key] if info_key in info else None

        if len(added_info) > 0:
            for _info in added_info:
                assert isinstance(_info, str)
                value = info.get(_info, None)
                if _info in ["calib_path", "homo_path", "homo_noise"]:
                    homo_gen_names = ["homo_gen", "homo_gen_small"]
                    for homo_gen_name in homo_gen_names:
                        if (
                            homo_gen_name in dataset
                            and dataset[homo_gen_name] is not None
                        ):
                            dataset[homo_gen_name][_info] = value
                elif _info == "camera_module_type":
                    # NOTE 下面的判断同时兼顾以下四种情况:
                    # (1) dataset.yaml没有设置camera_module_type ==> 使用旧模组类型
                    # (2) dataset.yaml设置了camera_module_type, 但是给了none ==> 使用旧模组类型  # noqa
                    # (3) dataset.yaml设置了camera_module_type, 并且在bev_transform_factory里配置了相关参数 ==> 使用指定模组类型  # noqa
                    # (4) dataset.yaml设置了camera_module_type, 但是不在bev_transform_factory里面 ==> 报错  # noqa
                    if value is not None:
                        update_info_dict = getattr(
                            bev_transform_factory, value
                        )
                        dataset = update_dataset_with_module_info(
                            dataset, update_info_dict
                        )
                    else:
                        pass  # 不更新
                else:
                    dataset[_info] = value

        if len(update_trans_info) > 0:
            for trans_type, key in update_trans_info.items():
                assert isinstance(key, str) or isinstance(key, List)
                key_list = _as_list(key)
                for _key in key_list:
                    value = info.get(_key, None)
                    if value is not None:
                        for idx, transform in enumerate(dataset["transforms"]):
                            if transform["type"] == trans_type:
                                transform[_key] = value
                                dataset["transforms"][idx] = transform

        # NOTE:
        # 对无对应数据的模型视角进行homogen的clean：
        # (1) 模型的每个 view 都能有数据时，不进行清除，例如 4v/6v 使用 10v/11v 数据
        # (2) 模型存在 view 加载不出数据时，需要将不存在数据的 view 的对应 homogen 清除
        homo_gen_names = ["homo_gen", "homo_gen_small"]
        for homo_gen_name in homo_gen_names:
            if homo_gen_name in dataset and dataset[homo_gen_name] is not None:
                dataset[homo_gen_name] = clean_homogen(
                    copy.deepcopy(dataset[homo_gen_name]), info
                )

        dataset["sample_interval"] = info["sample_interval"]

        resample_dataset = dict(
            type="ResampleDataset",
            dataset=dataset,
            with_flag=True,
            resample_interval=1,  # do not use ResampleDataset now.
        )
        return resample_dataset

    return update_bev_dataset


def get_datasets(
    dataset_dict,
    location_names,
    template_dataset,
    data_dict,
    update_bev_dataset,
    homo_noise=None,
    repeat_dataset_times=1,
    global_sample_interval=1,
    repeat_dataset_config=None,
):
    """Get dataset list.

    Args:
        repeat_dataset_times: Only used during validation with `homo_noise`.
            Each origin `val` dataset is repeated multiple times and different
            `homo_noise` is generated each time to increase the stability of
            `val` result. Default to 1.
    """
    ret_datasets = []
    # to adapt fisheye_bevseg and fisheye-bev3d that donot support manage dataset with data version now.  # noqa
    if location_names is None:
        location_names = dataset_dict.keys()
    for location_name in location_names:
        for plat_name in dataset_dict[location_name]:
            for dataset_name in dataset_dict[location_name][plat_name]:
                cur_dataset = copy.deepcopy(template_dataset)
                cur_info = data_dict[location_name][plat_name][dataset_name]
                # update dataset from version_dict
                if (
                    isinstance(dataset_dict[location_name][plat_name], dict)
                    and dataset_dict[location_name][plat_name][dataset_name]
                ):
                    for cur_key in dataset_dict[location_name][plat_name][
                        dataset_name
                    ].keys():
                        assert cur_key in cur_info.keys()
                        cur_info[cur_key] = dataset_dict[location_name][
                            plat_name
                        ][dataset_name][cur_key]
                # update homo_noise config
                if homo_noise is not None:
                    cur_info = copy.deepcopy(cur_info)
                    cur_info["homo_noise"] = homo_noise
                cur_dataset = update_bev_dataset(
                    cur_dataset, cur_info, global_sample_interval
                )
                # only used during validation with `homo_noise`.
                if repeat_dataset_times > 1:
                    ret_datasets.extend(
                        [
                            copy.deepcopy(cur_dataset)
                            for _ in range(repeat_dataset_times - 1)
                        ]
                    )
                if repeat_dataset_config:
                    for k, v in repeat_dataset_config.items():
                        if k in dataset_name and v > 1:
                            ret_datasets.extend(
                                [
                                    copy.deepcopy(cur_dataset)
                                    for _ in range(v - 1)
                                ]
                            )
                ret_datasets.append(cur_dataset)
    return ret_datasets


def get_dataset_list(
    dataset_dict,
    location_names,
    template_dataset,
    data_dict,
    update_bev_dataset,
    homo_noise=None,
    repeat_dataset_times=1,
):
    """Get dataset list."""
    dataset_list = []
    for dataset_dict_item in dataset_dict:
        ret_datasets = get_datasets(
            dataset_dict=dataset_dict_item,
            location_names=location_names,
            template_dataset=template_dataset,
            data_dict=data_dict,
            update_bev_dataset=update_bev_dataset,
            homo_noise=homo_noise,
            repeat_dataset_times=repeat_dataset_times,
        )
        dataset_list.append(ret_datasets)
    return dataset_list


def get_dataloader(
    datasets,
    num_workers,
    batch_size_per_gpu,
    collate_ignore_keys=["homo_transforms"],  # noqa
    shuffle=False,
    dataloader_seed=0,
    persistent_workers=False,
):
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        collate_fn=partial(
            collate_3d,
            ignore_keys=collate_ignore_keys,
        ),
        dataset=dict(type="ConcatDataset", datasets=datasets, with_flag=True),
        batch_size=batch_size_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_reset_seed,
        persistent_workers=persistent_workers,
        sampler=dict(
            type=DistributedGroupSampler,
            samples_per_gpu=batch_size_per_gpu,
            seed=dataloader_seed,
        ),
    )
    return data_loader


def get_dataloader_list(
    datasets, num_workers, batch_size_per_gpu, shuffle, persistent_workers
):
    """Get dataloader list."""
    data_loader_list = []
    for datasets_item in datasets:
        data_loader_item = get_dataloader(
            datasets=datasets_item,
            num_workers=num_workers,
            batch_size_per_gpu=batch_size_per_gpu,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
        )
        data_loader_list.append(data_loader_item)
    return data_loader_list


def get_template_dataset(
    img_load_size,
    camera_view_names,
    per_view_shape,
    transforms,
    homo_transforms,
    spatial_resolution,
    H_persp_view_scale,
    vcs_range,
    use_distorted_offset,
    vcs_plane_heights,
    cal_homo_offset_on_gpu,
    offset_save_path,
    temporal_bev,
    length_of_clip,
    train_num_frames_per_iter,
):
    assert isinstance(transforms, list)
    homo_gen = dict(
        homo_path=None,
        calib_path=None,
        spatial_resolution=spatial_resolution,
        H_persp_view_scale=H_persp_view_scale,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        task_camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        use_distorted_offset=use_distorted_offset,
        homo_transforms=homo_transforms,
        homo_noise=None,
        vcs_plane_heights=vcs_plane_heights,
        return_offset_in_meta_info=False if cal_homo_offset_on_gpu else True,
        offset_save_path=offset_save_path,
    )
    template_dataset = dict(
        type="ANCAuto3DV",
        transforms=transforms,
        img_load_size=img_load_size,
        camera_view_names=camera_view_names,
        per_view_shape=per_view_shape,
        root=None,
        sync_file=None,
        sync_file_lmdb=None,
        sample_interval=1,
        img_data_path=None,
        seg_data_path=None,
        bev_seg_data_path=None,
        bev_3d_lmdb_path=None,
        multi_view_lmdb_path=None,
        flag_for_group=1,
        num_frames_per_iter=train_num_frames_per_iter
        if temporal_bev
        else None,
        num_max_frames=length_of_clip if temporal_bev else None,
        homo_gen=homo_gen,
    )

    return template_dataset


def get_temporal_transform(
    model_flow_mode,
    ipm_output_size,
    ipm_output_size_small,
    vcs_range,
    vcs_range_small,
    is_relative_homography,
):
    temporal_bev_size = []
    temporal_vcs_range = []
    temporal_homography_names = []
    if model_flow_mode in ["wide", "wide_small"]:
        temporal_bev_size.append(ipm_output_size)
        temporal_vcs_range.append(vcs_range)
        temporal_homography_names.append("homography_temporal")
    if model_flow_mode in ["small", "wide_small"]:
        temporal_bev_size.append(ipm_output_size_small)
        temporal_vcs_range.append(vcs_range_small)
        temporal_homography_names.append("homography_temporal_small")
    temporal_transform = dict(
        type="ANCTemporalHomo",
        bev_size=temporal_bev_size,
        vcs_range=temporal_vcs_range,
        homography_names=temporal_homography_names,
        return_relative=is_relative_homography,
    )
    return temporal_transform


def get_common_transforms(
    size,
    remap_dict,
    crop_roi,
    train_num_frames_per_iter,
    views_domain2nums,
    temporal_bev,
    temporal_transform,
):
    crop_height = [roi[3] - roi[1] for roi in crop_roi]
    crop_width = [roi[2] - roi[0] for roi in crop_roi]
    common_transforms = {
        "ANCCollect3DV": dict(
            type="ANCCollect3DV",
            load_data_types=None,
            img_idxs=list(range(train_num_frames_per_iter))
            if temporal_bev
            else [0],
            fill_fake_temporal_data=False,
            pose_idxs=list(range(train_num_frames_per_iter + 1))
            if temporal_bev
            else [0],
        ),
        "ANCResize3DV": dict(
            type="ANCResize3DV",
            size=size,
        ),
        "ANCCrop3DV": dict(
            type="ANCCrop3DV",
            height=crop_height,
            width=crop_width,
            top=[roi[1] for roi in crop_roi],
            left=[roi[0] for roi in crop_roi],
        ),
        "ANCPad3DV": dict(
            type="ANCPad3DV",
            paddings=[(0, 0, 0, 0)] * sum(views_domain2nums.values()),
            gt_seg_fill=255,
        ),
        "ANCTemporalHomo": temporal_transform,
        "ANCClassRemap": dict(
            type="ANCClassRemap",
            remap_dict=remap_dict,
        ),
        "ANCToTensor3DV": dict(type="ANCToTensor3DV", with_color_imgs=False),
        "ANCNormalize3DV": dict(type="ANCNormalize3DV", mean=128, std=128),
        "ANCPrepareDataBEV": dict(
            type="ANCPrepareDataBEV",
            views_domain2nums=views_domain2nums,
            single_frame=True,
        ),
        "ANCPrepareTempoDataBEV": dict(
            type="ANCPrepareTempoDataBEV",
            views_domain2nums=views_domain2nums,
        ),
    }
    return common_transforms


def remove_none(data_list: Sequence):
    assert isinstance(data_list, Sequence)
    return list(filter(lambda x: x is not None, data_list))


def get_bev_3d_transforms(
    bev3d_target,
    bev_common_transforms,
    load_data_types,
    task_name,
    temporal_bev=True,
    bev3d_rpy_transforms=None,
    use_occlusion_attribute=False,
    occlusion_attribute_dict=None,
    use_ignore_mask_img=False,
):
    bev3d_target = _as_list(bev3d_target)
    bev_common_transforms["ANCMultiViewTargetGenerator"][
        "occlusion_attribute"
    ] = use_occlusion_attribute
    bev_common_transforms["ANCMultiViewTargetGenerator"][
        "occlusion_attribute_dict"
    ] = occlusion_attribute_dict
    bev_common_transforms["ANCMultiViewTargetGenerator"][
        "use_ignore_mask_img"
    ] = use_ignore_mask_img
    collect_3dv = bev_common_transforms["ANCCollect3DV"]
    collect_3dv["gt_bev_3d_idx"] = 0
    collect_3dv["load_data_types"] = copy.deepcopy(load_data_types)
    bev_3d_transforms = [
        collect_3dv,
        bev_common_transforms["ANCResize3DV"],
        bev_common_transforms["ANCCrop3DV"],
        bev_common_transforms["ANCTemporalHomo"] if temporal_bev else None,
        bev_common_transforms["ANCToTensor3DV"],
        bev_common_transforms["ANCMultiViewTargetGenerator"],
        bev_common_transforms["ANCApplyMaskOnImg"]
        if use_ignore_mask_img
        else None,
        bev3d_rpy_transforms,
        *bev3d_target,
    ]
    if temporal_bev:
        bev_3d_transforms.append(
            bev_common_transforms["ANCPrepareTempoDataBEV"]
        )
        bev_3d_transforms.append(
            dict(type="ANCSetTemporalClearFlag", clr_mode="clip")
        )
        bev_3d_transforms.append(
            dict(type="AddKeys", kv={"return_latest_flag": True})
        )
        bev_3d_transforms.append(
            dict(type="AddKeys", kv={"task_name": task_name})
        )
    else:
        bev_3d_transforms.append(bev_common_transforms["ANCPrepareDataBEV"])
    return remove_none(bev_3d_transforms)


def update_content(data: Dict, **kwargs):
    update_data = copy.deepcopy(data)
    update_data.update(**kwargs)
    return update_data


def get_homo_transforms(transforms_list, camera_view_names):
    homo_transforms = {}
    for _index, view in enumerate(camera_view_names):
        view_transform = {}
        for _transform in transforms_list:
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


def convert_to_temporal_split_dataloader(
    dataloader,
    dataloader_seed=0,
):
    # 这个convert只针对train dataloader
    assert (
        dataloader["num_workers"] > 0
    ), "RankSplitDataLoader only support num_worker>0"
    assert (
        dataloader["dataset"]["type"] == "ConcatDataset"
    ), "RankSplitDataset only support ConcatDataset"

    random.seed(dataloader_seed)
    random.shuffle(dataloader["dataset"]["datasets"])
    random.seed(None)
    for d in dataloader["dataset"]["datasets"]:
        d["__lazy_build__"] = True
    dataloader["dataset"]["type"] = "ANCRankSplitE2EDataset"
    dataloader["dataset"]["sub_clip_num"] = (
        dataloader["dataset"]["datasets"][0]["dataset"]["num_max_frames"]
        // dataloader["dataset"]["datasets"][0]["dataset"][
            "num_frames_per_iter"
        ]
    )
    dataloader["dataset"]["batch_size"] = dataloader["batch_size"]
    dataloader["sampler"] = dict(
        type=torch.utils.data.DistributedSampler,
        dataset=dataloader["dataset"],
        drop_last=True,
        seed=dataloader_seed,
    )

    dataloader["type"] = "ANCRankSplitE2EDataLoader"
    dataloader["drop_last"] = True
    if "shuffle" in dataloader:
        dataloader.pop("shuffle")
    return dataloader


def convert_to_split_dataloader(
    dataloader,
    dataloader_seed=0,
    temporal_bev=True,
):
    if temporal_bev:
        return convert_to_temporal_split_dataloader(dataloader)

    assert (
        dataloader["num_workers"] > 0
    ), "RankSplitDataLoader only support num_worker>0"
    assert (
        dataloader["dataset"]["type"] == "ConcatDataset"
    ), "RankSplitDataset only support ConcatDataset"
    support_sampler_type = [
        torch.utils.data.DistributedSampler,
        DistributedGroupSampler,
    ]
    assert (
        dataloader["sampler"]["type"] in support_sampler_type
    ), "only support DistributedSampler and DistributedGroupSampler"

    random.seed(dataloader_seed)
    random.shuffle(dataloader["dataset"]["datasets"])
    random.seed(None)
    for d in dataloader["dataset"]["datasets"]:
        d["__lazy_build__"] = True
    dataloader["dataset"]["type"] = "RankSplitDataset"

    dataloader["sampler"] = dict(
        type=torch.utils.data.DistributedSampler,
        drop_last=True,
        seed=dataloader_seed,
    )

    dataloader["type"] = "RankSplitDataLoader"
    dataloader["drop_last"] = True
    return dataloader


def get_grid_quant_scale(output_max_size, input_max_size):
    max_coord = max(output_max_size, input_max_size)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


def get_block_warping(vcs_range, spatial_resolution):
    pad_top = round(abs(vcs_range[2] / spatial_resolution[0]))
    pad_bottom = round(abs(vcs_range[0] / spatial_resolution[0]))
    pad_left = round(abs(vcs_range[3] / spatial_resolution[1]))
    pad_right = round(abs(vcs_range[1] / spatial_resolution[1]))

    # warp_padding`s order is (left,right,up,bottom)
    block_warp_padding = [
        (0, 0, 0, pad_bottom),
        (0, pad_right, 0, 0),
        (pad_left, 0, 0, 0),
        (0, pad_right, 0, 0),
        (pad_left, 0, 0, 0),
        (0, 0, pad_top, 0),
    ]

    return block_warp_padding


def get_fisheye_block_warping(
    vcs_range, fisheye_warp_range, spatial_resolution
):
    pad_top = round(abs(vcs_range[2] / spatial_resolution[0]))
    pad_bottom = round(abs(vcs_range[0] / spatial_resolution[0]))
    pad_left = round(abs(vcs_range[3] / spatial_resolution[1]))
    pad_right = round(abs(vcs_range[1] / spatial_resolution[1]))

    # fisheye warp range in 4v/10v/11v fusion.
    # In general, there are the following situations:
    # 1. 10v/11v wide range::
    #     front, side and narrow view warp in the whole vcs_range(wide range).
    #     however, due to the limited visibility of the fisheye, fisheye only
    #     warp in a specific small range. Further, if warp the whole wide range
    #     will introduce noise and affect performance.
    #     In this case fisheye_warp_range < vcs_range, e.g.,
    #     vcs_range=(-153.6, -76.8, 153.6, 76.8),
    #     fisheye_warp_range = (-12.8, -12.8, 25.6, 12.8)
    # 2. 10v/11v small range::
    #     Due to the small range, the visual range of fisheye is no longer
    #     limited, and 11v can warp the entire vcs_range.
    #     In this case fisheye_warp_range is vcs_range, e.g.,
    #     vcs_range=(-12.8, -12.8, 25.6, 12.8),
    #     fisheye_warp_range = (-12.8, -12.8, 25.6, 12.8)
    # 3. 4v fisheye small range::
    #     In this case fisheye_warp_range is vcs_range, e.g.,
    #     vcs_range=(-12.8, -12.8, 25.6, 12.8),
    #     fisheye_warp_range = (-12.8, -12.8, 25.6, 12.8)
    # Usually fisheye_warp_range is aligned with vcs_range of the small range model.  # noqa

    fisheye_pad_top = round(
        abs((vcs_range[2] - fisheye_warp_range[2]) / spatial_resolution[0])
    )
    fisheye_pad_bottom = round(
        abs((vcs_range[0] - fisheye_warp_range[0]) / spatial_resolution[0])
    )
    fisheye_pad_left = round(
        abs((vcs_range[3] - fisheye_warp_range[3]) / spatial_resolution[1])
    )
    fisheye_pad_right = round(
        abs((vcs_range[1] - fisheye_warp_range[1]) / spatial_resolution[1])
    )
    fisheye_block_warp_padding = [
        (fisheye_pad_left, fisheye_pad_right, fisheye_pad_top, pad_bottom),
        (fisheye_pad_left, fisheye_pad_right, pad_top, fisheye_pad_bottom),
        (fisheye_pad_left, pad_right, fisheye_pad_top, fisheye_pad_bottom),
        (pad_left, fisheye_pad_right, fisheye_pad_top, fisheye_pad_bottom),
    ]

    return fisheye_block_warp_padding


def get_narrow_block_warping(vcs_range, spatial_resolution):
    pad_bottom = round(abs(vcs_range[0] / spatial_resolution[0]))

    narrow_block_warp_padding = [
        (0, 0, 0, pad_bottom),
    ]

    return narrow_block_warp_padding


def get_valid_head_output_cfg(
    task_head_output_cfg,
    input_size,
    vcs_range,
):
    """Get whether do RoiReize and valid head_output_cfg.

    Args:
        task_head_output_cfg (dict): output cfg of each task.
        input_size (Tuple): input size of specified stage backbone.
            e.g., input size of stage1 or stage2 backbone.
        vcs_range (Tuple): vcs_range, (bottom, right, top, left).
            e.g., (-30.0, -51.2, 72.4, 51.2)

    """
    do_roi_resize = False
    valid_head_output_cfg = []
    if task_head_output_cfg is not None:
        assert isinstance(
            task_head_output_cfg, dict
        ), "task_head_output_cfg shoule be dict"
        for task_cfg in task_head_output_cfg.values():
            strides = _as_list(task_cfg["in_stride"])
            out_sizes = task_cfg["out_size"]
            if not is_list_of_type(out_sizes, tuple):
                out_sizes = [out_sizes]
            assert len(strides) == len(
                out_sizes
            ), "stride should match out_size"
            for stride, out_size in zip(strides, out_sizes):
                vaild_task_cfg = {
                    "in_stride": stride,
                    "roi_vcs_range": task_cfg["roi_vcs_range"],
                    "out_size": out_size,
                }
                ori_out_size = (
                    input_size[0] // vaild_task_cfg["in_stride"],
                    input_size[1] // vaild_task_cfg["in_stride"],
                )
                if not (
                    vaild_task_cfg["roi_vcs_range"] == vcs_range
                    and vaild_task_cfg["out_size"] == ori_out_size
                ):
                    do_roi_resize = True
                    if vaild_task_cfg not in valid_head_output_cfg:
                        valid_head_output_cfg.append(vaild_task_cfg)

    return do_roi_resize, valid_head_output_cfg


def get_roi_resize_cfg(
    input_size, in_stride, output_size, ori_vcs_range, roi_vcs_range
):
    """Get roi_resize cfg of RoiResize module.

    Args:
        input_size (Tuple): input size of specified stage backbone.
            e.g., input size of stage1 or stage2 backbone.
        in_stride (int): in_stride of roi_resize module.
        output_size (Tuple): output size of roi_upsample module,
            (height, width).
        ori_vcs_range (Tuple): origin vcs_range, (bottom, right, top, left).
        roi_vcs_range (Tuple): roi vcs_range, (bottom, right, top, left).

    """
    input_size_h = input_size[0] // in_stride
    input_size_w = input_size[1] // in_stride
    ori_vcs_range_h = abs(ori_vcs_range[2] - ori_vcs_range[0])
    ori_vcs_range_w = abs(ori_vcs_range[3] - ori_vcs_range[1])
    roi_box_x0 = (
        (roi_vcs_range[1] - ori_vcs_range[1]) / ori_vcs_range_w * input_size_w
    )
    roi_box_x1 = (
        (roi_vcs_range[3] - ori_vcs_range[1]) / ori_vcs_range_w * input_size_w
    )
    roi_box_y0 = (
        (ori_vcs_range[2] - roi_vcs_range[2]) / ori_vcs_range_h * input_size_h
    )
    roi_box_y1 = (
        (ori_vcs_range[2] - roi_vcs_range[0]) / ori_vcs_range_h * input_size_h
    )
    roi_box = [roi_box_x0, roi_box_y0, roi_box_x1, roi_box_y1]

    # to avoid coordinate offset due to loss of precision
    # in float point calculations. e.g., x=(76.8-51.2)/153.6*192,
    # x should be 32.0, but the result is 31.99999...
    roi_box = [round(x) for x in roi_box]

    roi_resize_cfg = dict(
        in_stride=in_stride,
        output_size=output_size,
        roi_box=roi_box,
    )

    return roi_resize_cfg


def get_update_metric_func(mode, task_name, pred_patterns):
    assert mode in ["val"]

    def _val(metrics, batch, model_outs):
        assert len(metrics) == len(pred_patterns)

        for metric, pred_pattern in zip(metrics, pred_patterns):
            pred_dict = OrderedDict()
            if task_name in model_outs:
                if isinstance(model_outs[task_name], list):
                    task_outs = model_outs[task_name][0]
                else:
                    task_outs = model_outs[task_name]
                for key, pattern in pred_pattern.items():
                    per_pred_pattern = re.compile(pattern)
                    for k, v in task_outs.items():
                        if per_pred_pattern.match(k):
                            pred_dict[key] = v

                metric.update(batch[0], pred_dict)

    if mode == "val":
        return _val
    else:
        raise NotImplementedError


def save_data_to_npy(data, save_path):
    """
    if tuple in data,should not save in json,but can save in numpy
    """
    save_dir, _ = os.path.split(save_path)
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, data)

    # if data is dict, should read like that:
    # np.load(save_path, allow_pickle=True).item()


def get_warp_sizes(ipm_output_size, block_warp_padding):
    warp_sizes = [
        (
            ipm_output_size[0] - pad[2] - pad[3],
            ipm_output_size[1] - pad[0] - pad[1],
        )
        for pad in block_warp_padding
    ]

    return warp_sizes


def repeat_metric_updater_by_name(
    task_name_list, val_metric_updater, replace_key_list
):
    """Replace the save_path involved in val_metric_updater
    to ensure that the results are saved in different path.
    Return a list of val_metric_updater whose length is equal to len(task_name_list) # noqa
    The replacement rules are as follows:
    1.replace_value is None
        return None
    2.replace_value is a/b
        return a/taskname_b
    3.replace_value is a
        return taskname_a

    Args:
        task_name_list: Eval version name,used to generate new save_path.
        val_metric_updater: MetricUpdater Callback.
        replace_key_list: replace list. e.g."save_dir","name"...

    Returns: [val_metric_updater_1,val_metric_updater_2].
    """
    val_metric_updater_list = []
    for task_name in task_name_list:
        _val_metric_updater = copy.deepcopy(val_metric_updater)
        for idx, val_metrics in enumerate(_val_metric_updater["metrics"]):
            _val_metrics = copy.deepcopy(val_metrics)
            for replace_key in replace_key_list:
                # assert replace_key in _val_metrics
                if replace_key not in _val_metrics:
                    continue
                replace_value = _val_metrics[replace_key]
                if not replace_value:
                    continue
                base_dir, filename = os.path.split(replace_value)
                replace_value = os.path.join(
                    base_dir, task_name + "_" + filename
                )
                _val_metrics[replace_key] = replace_value
            _val_metric_updater["metrics"][idx] = _val_metrics
        val_metric_updater_list.append(_val_metric_updater)
    return val_metric_updater_list


def transform_bool2int(data):
    if isinstance(data, bool):
        return int(data)
    elif isinstance(data, list):
        return [transform_bool2int(d) for d in data]
    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = transform_bool2int(data[key])
        return data
    elif isinstance(data, (int, float, str)):
        return data
    else:
        raise TypeError


def reformat_compile_vcs_range(vcs_range: Sequence[float]):
    """Reformat vcs_range for compile.

    Args:
        vcs_range: (bottom, right, top, left)

    Returns:
        vcs_range_compile: (top, bottom, left, right)
    """
    vcs_range_compile = [
        float(vcs_range[2]),
        float(vcs_range[0]),
        float(vcs_range[3]),
        float(vcs_range[1]),
    ]
    return vcs_range_compile


class ViewDomainFactory(object):
    """View domain factory, mapping view domain to camera views.

    So for, we have 4 view domains, containing "front", "side", "round"
    and "narrow", each corresponding to some camera views.
    """

    _view_domain_factory = {
        "front": ["camera_front"],
        "side": [
            "camera_front_left",
            "camera_front_right",
            "camera_rear_left",
            "camera_rear_right",
            "camera_rear",
        ],
        "round": [
            "fisheye_front",
            "fisheye_rear",
            "fisheye_left",
            "fisheye_right",
        ],
        "narrow": ["camera_front_30fov"],
    }

    @staticmethod
    def get_all_view_names():
        _views = ViewDomainFactory._view_domain_factory.values()
        return [i for v in _views for i in v]

    @staticmethod
    def get_domain2views():
        return copy.deepcopy(ViewDomainFactory._view_domain_factory)

    @staticmethod
    def get_valid_camera_view_names(camera_view_names: list):
        """Check whether the camera view names list is valid.

        Args:
            camera_view_names: the list of camera view names.

        Returns:
            list: the camera view names list sorted according to the order in
                the factory.
        """
        if camera_view_names is None:
            view_domain_factory = ViewDomainFactory._view_domain_factory
            camera_view_names = (
                view_domain_factory["front"] + view_domain_factory["side"]
            )
        assert len(camera_view_names) == len(
            set(camera_view_names)
        ), "Duplicate keys are not allowed in `camera_view_names`."
        assert (
            len(camera_view_names) > 0
        ), "`camera_view_names` must contains at least one key."

        all_camera_view_names = ViewDomainFactory.get_all_view_names()
        assert set(camera_view_names).issubset(
            set(all_camera_view_names)
        ), "`camera_view_names` includes invalid keys"

        ordered_camera_view_names = []
        for view_name in all_camera_view_names:
            if view_name in camera_view_names:
                ordered_camera_view_names.append(view_name)

        return ordered_camera_view_names

    @staticmethod
    def get_views_domain2nums(camera_view_names: list):
        """Get a dict of view domains and corresponding camera view nums.

        view_domain2nums is a dict to map view domains to corresponding
        view nums, will be used to control the model structure. If the
        view nums in a domain > 0, we should add the corresponding stage1
        model to the topology structure and process the corresponding data
        in the data pipeline.
        Example:
            (a) 7v model has view domain of ["front", "side", "narrow"],
                the views_domain2nums is:
                {
                    "front": 1,
                    "side": 5,
                    "round": 0,
                    "narrow": 1,
                }
                the round view num is 0, fisheye stage1 model will not be
                added in the topology structure, fisheye data will not be
                processed.
            (b) 11v model has view domain of ["front", "side", "round",
                "narrow"], the views_domain2nums is:
                {
                    "front": 1,
                    "side": 5,
                    "round": 4,
                    "narrow": 1,
                }
                the view num of all domains > 0, all kinds of stage1 model
                will be added in the topology structure, all kinds of data
                will be processed in the data pipeline.

        Args:
            camera_view_names: the list of camera view names.

        Returns:
            dict: the dict of view domains and corresponding camera view nums.
        """
        views_domain2nums = OrderedDict()
        view_domain_factory = ViewDomainFactory._view_domain_factory
        for view_domain, domain_view_names in view_domain_factory.items():
            num_views_in_domain = sum(
                [name in domain_view_names for name in camera_view_names]
            )
            views_domain2nums[view_domain] = num_views_in_domain
        return views_domain2nums


def internal_state_dict_converter(state_dict):
    state_dict_new = deepcopy(state_dict)
    key_mapping = [
        ("fusion_upsampling.", "bev_fusion_upsampling."),
        # roi_resize
        (
            "roi_resize.resize_modules.0.",
            "bev_3d_vrumerge_roi_resize.resize_module.",
        ),
        (
            "roi_resize.resize_modules.1.",
            "bev_evevation_roi_resize.resize_module.",
        ),
        (
            "roi_resize.resize_modules.2.",
            "bev_om_roi_resize_stride2.resize_module.",
        ),
        (
            "roi_resize.resize_modules.3.",
            "bev_om_roi_resize_stride4.resize_module.",
        ),
        (
            "roi_resize.resize_modules.4.",
            "bev_arrow_roi_resize.resize_module.",
        ),
        ("roi_resize.resize_modules.5.", "junction_roi_resize.resize_module."),
        (
            "roi_resize.resize_modules.6.",
            "bev_roadmarking_roi_resize.resize_module.",
        ),
        (
            "roi_resize.resize_modules.7.",
            "bev_psd_roi_resize_stride2.resize_module.",
        ),
        (
            "roi_resize.resize_modules.8.",
            "bev_psd_roi_resize_stride8.resize_module.",
        ),
        # e2e
        ("bev_stage2_3d_vehicle_head.head.", "bev_stage2_3d_vehicle_head."),
        ("bev_stage2_3d_vrumerge_head.head.", "bev_stage2_3d_vrumerge_head."),
    ]

    for key, value in state_dict.items():
        for src, tgt in key_mapping:
            if src in key:
                new_key = key.replace(src, tgt)
                if new_key in state_dict:
                    raise ValueError(f"{key} >> {new_key}")
                state_dict_new[new_key] = deepcopy(value)
                del state_dict_new[key]
                logger.info(f"Mapping: {key} >> {new_key}")

    return state_dict_new


def get_fillback_timestamp(camera_view_names, bev_pack_path):
    """
    Get bev task timestamp combination in fillback pack.
    """
    reader = MSGReader(
        handle=bev_pack_path,
        topic_channel=[TopicChannel("bev_param_vec", 0)],
        decode_data=True,
    )

    all_timestamp = defaultdict(list)
    reader.SeekByIndex(0)
    while True:
        reader.TellIndex()
        megs = reader.Read()
        if not megs:
            break
        megs = megs["bev_param_vec"][0].proto[0].cam_params
        for view, meg in zip(camera_view_names, megs):
            all_timestamp[view].append(meg.time_stamp)

    return all_timestamp


def update_pack_transform(
    transforms,
    pack_camera_module_type,
    camera_view_names,
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

    bev_transform_factory = (
        f"{os.path.dirname(__file__)}/bev_transform_factory.py"
    )
    bev_transform_factory = Config.fromfile(bev_transform_factory)
    update_info_dict = bev_transform_factory[pack_camera_module_type]
    update_camera_names = list(update_info_dict["per_view_shape"].keys())
    assert set(camera_view_names).issubset(
        set(update_camera_names)
    ), "Model camera views must be included in the pack's."
    for transform in transforms:
        if transform["type"] == "ANCConvertPackDataTo3DV":
            for key in transform["homo_gen"].keys():
                if key == "homo_transforms":
                    update_homo_transforms = get_homo_transforms(
                        [
                            update_info_dict["transforms"]["ANCResize3DV"],
                            update_info_dict["transforms"]["ANCCrop3DV"],
                        ],
                        update_camera_names,
                    )
                    transform["homo_gen"][key] = _update_dict(
                        transform["homo_gen"][key],
                        update_homo_transforms,
                    )
                elif key in update_info_dict.keys():
                    transform["homo_gen"][key] = _update_dict(
                        transform["homo_gen"][key],
                        update_info_dict[key],
                    )
        elif transform["type"] == "ANCNV12Transform3DV":
            transform["ori_size"] = [
                update_info_dict["per_view_shape"][view]
                for view in camera_view_names
            ]
        elif transform["type"] in update_info_dict["transforms"].keys():
            for k, v in transform.items():
                if isinstance(v, list):
                    transform[k] = _update_list(
                        transform[k],
                        camera_view_names,
                        update_info_dict["transforms"][transform["type"]][k],
                        update_camera_names,
                    )
                elif isinstance(v, dict):
                    transform[k] = _update_dict(
                        transform[k],
                        update_info_dict["transforms"][transform["type"]][k],
                    )
                else:  # str, int, float
                    transform[k] = update_info_dict["transforms"][
                        transform["type"]
                    ][k]
    return transforms


def update_state_dict_task(state_dict):
    state_dict_new = copy.deepcopy(state_dict)
    # load ckpt from hat-internal
    for k, v in state_dict.items():
        if "bev_stage2_3d_e2e_vehicle_head" in k:
            _ = state_dict_new.pop(k)
            state_dict_new[
                k.replace("stage2_3d_e2e", "stage2_3d")
            ] = copy.deepcopy(v)
        if "bev_stage2_3d_e2e_vrumerge_head" in k:
            _ = state_dict_new.pop(k)
            state_dict_new[
                k.replace("stage2_3d_e2e", "stage2_3d")
            ] = copy.deepcopy(v)
    state_dict_new2 = copy.deepcopy(state_dict_new)
    for k, v in state_dict_new.items():
        if "bev_stage2_3d_vehicle_head.head." in k:
            _ = state_dict_new2.pop(k)
            state_dict_new2[k.replace("head.head.", "head.")] = copy.deepcopy(
                v
            )
        if "bev_stage2_3d_vrumerge_head.head" in k:
            _ = state_dict_new2.pop(k)
            state_dict_new2[k.replace("head.head.", "head.")] = copy.deepcopy(
                v
            )

    return state_dict_new2


def get_update_state_dict(keep_modules=None):
    def update_state_dict(state_dict, keep_modules=keep_modules):
        if keep_modules is not None:
            keep_state_dict = OrderedDict()
            for key in state_dict.keys():
                for keep_module in keep_modules:
                    if key.startswith(keep_module):
                        keep_state_dict[key] = state_dict[key]
            state_dict = keep_state_dict
        if update_state_dict_task is not None:
            state_dict = update_state_dict_task(state_dict)
        return state_dict

    return update_state_dict


def set_data_dict_value(data_dict, key, value):
    for location_name in data_dict:
        for plat_name in data_dict[location_name]:
            for dataset_name in data_dict[location_name][plat_name]:
                data_dict[location_name][plat_name][dataset_name][key] = value

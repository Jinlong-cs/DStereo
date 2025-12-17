import copy
import json
import logging
import os
from dataclasses import dataclass, fields, make_dataclass
from typing import Any, ClassVar, Dict, List, Optional

try:
    from hatbc.message import Instance
except ImportError:
    Instance = None

from hat.utils.apply_func import _as_list
from hat.utils.package_helper import require_packages
from .base import BaseData, BaseDataList

logger = logging.getLogger(__file__)


def build_task_struct(
    singleton_name,
    list_name,
    fields_desc,
    bases=(BaseData, BaseDataList),
):
    assert len(bases) == 2
    singleton_bases = _as_list(bases[0])
    list_bases = _as_list(bases[1])

    singleton_cls = make_dataclass(
        singleton_name,
        [(fd[0], fd[-1].singleton_cls) for fd in fields_desc],
        bases=tuple(singleton_bases),
    )

    list_cls = make_dataclass(
        list_name,
        [(fd[-2], fd[-1]) for fd in fields_desc]
        + [("singleton_cls", ClassVar[singleton_cls], singleton_cls)],
        bases=tuple(list_bases),
    )

    return singleton_cls, list_cls


@dataclass
class DetObject(BaseData):
    @require_packages("hatbc")
    def to_hatbc_msg(self, topic: Optional[str] = None):
        instance = Instance(topic=topic)
        field_names = fields(self)

        for f_name in field_names:
            name = f_name.name
            data = getattr(self, name)
            res = data.to_hatbc_msg(name)
            inst_msg = getattr(instance, data.instance_attr)
            if isinstance(res, list):
                inst_msg += res
            else:
                inst_msg.append(res)
        return instance


@dataclass
class DetObjects(BaseDataList):
    singleton_cls: ClassVar[BaseData] = DetObject


def reformat_to_hatbc_msg(batch_outputs: Dict[str, List], batch_data: Any):
    batch_size = None
    results = []
    for k, structs in batch_outputs.items():
        if batch_size is None:
            batch_size = len(structs)
            results = [[] for _ in range(batch_size)]

        for i, struct in enumerate(structs):
            if isinstance(struct, BaseDataList):
                results[i] += struct.to_hatbc_msg(k)
            elif isinstance(struct, BaseData):
                results[i].append(struct.to_hatbc_msg(k, instance=False))
            else:
                logger.debug("Ignore object type = {}".format(type(struct)))
    return results, batch_data


def reformat_det_to_aidi_eval(
    batch_data,
    batch_outputs,
    obj_key,
    det_task_key,
    extra_task_key=None,
    dump_obj_key=None,
    dump_extra_obj_key=None,
    custom_func=None,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, batch_obj_key = batch_data

    if custom_func is not None:
        batch_results = copy.deepcopy(batch_data)
        batch_results.update(
            dict(  # noqa
                data_info=batch_data,
                object_key=obj_key,
                detection_task_key=det_task_key,
                extra_task_key=extra_task_key,
                decoder_results=batch_outputs,
            )
        )
        batch_outputs = custom_func(batch_results)

    batch_objects = batch_outputs[obj_key]
    out_key = dump_obj_key if dump_obj_key is not None else obj_key
    rets = []
    for img_name, objects in zip(batch_data["img_name"], batch_objects):
        objects = objects.to("cpu")
        ret = {
            "image_key": img_name,
            out_key: [
                obj.to_aidi_eval()
                for obj in iter(getattr(objects, det_task_key))
            ],
        }

        if extra_task_key is not None:
            extra_update_res = []
            reorder_obj_list = False
            for i, extra_res in enumerate(
                iter(getattr(objects, extra_task_key))
            ):
                extra_dump_res = extra_res.to_aidi_eval()

                if isinstance(extra_dump_res, list):
                    # hotfix for weird results of bin det task
                    for res in extra_dump_res:
                        res["parent_box"] = ret[out_key][i]["bbox"]
                    extra_update_res += extra_dump_res
                    reorder_obj_list = True
                else:
                    ret[out_key][i].update(extra_dump_res)

                    if "attrs" in extra_dump_res:
                        if isinstance(dump_extra_obj_key, str):
                            dump_extra_obj_key = [dump_extra_obj_key]
                        extra_out_keys = (
                            dump_extra_obj_key
                            if dump_extra_obj_key is not None
                            else []
                        )
                        if len(extra_out_keys) == 0:
                            continue
                        # replace 2pe out key
                        attrs_key = list(ret[out_key][i]["attrs"].keys())
                        assert len(attrs_key) == len(extra_out_keys)
                        new_attrs = {}
                        for key, extra_out_key in zip(
                            attrs_key, extra_out_keys
                        ):
                            new_attrs[extra_out_key] = ret[out_key][i][
                                "attrs"
                            ][key]
                        ret[out_key][i]["attrs"] = new_attrs
                    else:
                        # deal with 3d
                        ret[out_key][i]["score"] *= ret[out_key][i].pop(
                            "bbox_score"
                        )
                        ret[out_key][i]["bbox_2d"] = ret[out_key][i].pop(
                            "bbox"
                        )

            if reorder_obj_list:
                ret[out_key] = extra_update_res

        rets.append(ret)

    return rets


def reformat_det_3d_to_aidi_eval(
    batch_data,
    batch_outputs,
    obj_key,
    dump_obj_key=None,
    custom_func=None,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, batch_obj_key = batch_data

    if custom_func is not None:
        batch_results = copy.deepcopy(batch_data)
        batch_results.update(
            dict(  # noqa
                data_info=batch_data,
                obj_key=obj_key,
                decoder_results=batch_outputs,
            )
        )
        batch_outputs = custom_func(batch_results)

    batch_objects = batch_outputs[obj_key]
    out_key = dump_obj_key if dump_obj_key is not None else obj_key
    rets = []
    for img_name, objects in zip(batch_data["img_name"], batch_objects):
        objects = objects.to("cpu")
        ret = {
            "image_key": img_name,
            out_key: [obj.to_aidi_eval() for obj in iter(objects)],
        }

        rets.append(ret)

    return rets


def reformat_bev_det_to_aidi_eval(
    batch_data,
    batch_outputs,
    obj_key,
    dump_obj_key=None,
    custom_func=None,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, batch_obj_key = batch_data

    if custom_func is not None:
        batch_results = copy.deepcopy(batch_data)
        batch_results.update(
            dict(  # noqa
                data_info=batch_data,
                obj_key=obj_key,
                decoder_results=batch_outputs,
            )
        )
        batch_outputs = custom_func(batch_results)

    batch_frames = batch_outputs[obj_key]
    out_key = dump_obj_key if dump_obj_key is not None else obj_key
    rets = []
    for meta, frame in zip(batch_data["meta"], batch_frames):
        meta = json.loads(meta)
        ret = {
            "timestamp": meta["timestamp"],
            "image_keys": {
                view: data["meta"]["image_key"]
                for view, data in meta["view_anno"].items()
            },
        }
        instances = frame.get_instances()
        obj_list = []
        for ins in instances:
            bbox3d_msg = ins.bbox3ds[0]
            cls_msg = ins.attributes[0]
            dim_whl = bbox3d_msg.dim.as_list()
            obj_list.append(
                dict(  # noqa
                    dimensions=[dim_whl[1], dim_whl[0], dim_whl[2]],
                    location=bbox3d_msg.loc.as_list(),
                    rotation_y=bbox3d_msg.yaw,
                    score=bbox3d_msg.score,
                    attrs=dict(  # noqa
                        sub_type=cls_msg.value,
                    ),
                )
            )
        ret.update({out_key: obj_list})
        rets.append(ret)
    return rets


def reformat_seg_to_aidi_eval(
    batch_data,
    batch_outputs,
    obj_key,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, batch_obj_key = batch_data
    batch_objects = batch_outputs[obj_key]
    rets = []
    for img_name, objects in zip(batch_data["img_name"], batch_objects):
        objects = objects.to("cpu")
        if os.path.splitext(img_name)[-1] != ".png":
            img_postfix = os.path.splitext(img_name)[-1]
            assert img_postfix in [".jpg", ".jpeg", "bmp"]
            img_name = img_name.replace(img_postfix, ".png")
        assert img_name.endswith(
            ".png"
        ), f"Image type error! expect .png file, but get {img_name}"
        ret = {
            "image_name": img_name,
            "out_img": objects.mask.numpy(),
        }

        rets.append(ret)

    return rets

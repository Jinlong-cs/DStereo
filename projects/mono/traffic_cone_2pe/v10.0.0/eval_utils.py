# Copyright (c) Horizon Robotics. All rights reserved.
import collections
import os
import uuid

from hatbc.filestream import io as fs


def reformat_tc2pe_to_aidi_eval(
    batch_data,
    batch_outputs,
    task_cfgs,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, _ = batch_data
    rets = []
    tmp_rets = {}
    obj_ids = batch_data.get("obj_id", None)
    img_names = batch_data.get("img_name", None)

    obj_key = task_cfgs[0]
    batch_objects = batch_outputs[
        0
    ].cone_bollard_classification_head_predict.tolist()

    cur_ret = collections.defaultdict(list)
    obj_rets = []
    for img_name, _, object in zip(img_names, obj_ids, batch_objects):
        res_i = {
            "obj_id": _,
            "prediction": object.index(max(object)),
            "scores": object,
        }
        cur_ret[img_name].append(res_i)
    for key in cur_ret:
        obj_rets.append({"image_key": key, "objects": cur_ret[key]})
    tmp_rets[obj_key] = obj_rets
    rets.append(tmp_rets)
    return rets


def download_hat_model_from_aidi(
    aidi_model_name: str,
    aidi_model_version: str,
    aidi_model_stage: str,
    output_dir: str = "./tmp/aidi_models",
    overwrite: bool = False,
) -> str:
    """
    Parameters
    ---------
    aidi_model_name : str
        Model name on AiDi Model, like `j3_mono_3.0_x8b_4pe_resize_day` .
    aidi_model_version : str
        Model version of the model, like `v6.0.0` .
    aidi_model_stage : str
        Model stage of the model, like `without_bn` .
    aidi_cfg : str, optional
        The path of aidi config, by default is None.
        If None, the default path is `$HOME/.olympus/config.yaml` .
    output_dir :  str, optional
        Cache dir of the model, by default `./tmp/aidi_models` .
    overwrite : bool, optional
        Whether to overlap the cached model, by default True.
    """

    f_pth_file = f"{aidi_model_name}_{aidi_model_version}_{aidi_model_stage}-checkpoint-last.pth.tar"  # noqa
    f_pth_path = os.path.join(output_dir, f_pth_file)

    if fs.exists(f_pth_path) and not overwrite:  # noqa
        return f_pth_path

    if not fs.exists(output_dir):
        fs.makedirs(output_dir)

    tmp_dir = "tmp/.tmp" + str(uuid.uuid4())
    fs.makedirs(tmp_dir)

    from aidisdk import AIDIClient

    client = AIDIClient()
    t_pth_path = client.model.download(
        tmp_dir, aidi_model_name, aidi_model_version, aidi_model_stage
    )
    if not t_pth_path:
        raise Exception(
            f"load model from aidi failed, "
            f"model_name: {aidi_model_name}, "
            f"model_version: {aidi_model_version}, "
            f"model_stage: {aidi_model_stage}",
            "pls check its existence on AiDi",
        )
    os.rename(t_pth_path, f_pth_path)

    fs.delete(tmp_dir, recursive=True)
    return f_pth_path

import torch
import yaml


def get_norm_mean_std(data_type, norm, bit_depth):
    mean = 0.0
    std = 2 ** bit_depth - 1
    if data_type == "yuv" and norm == "01":
        mean = 0.0
        std = 2 ** bit_depth - 1
    elif data_type == "yuv" and norm == "11":
        mean = 2 ** (bit_depth - 1)
        std = 2 ** (bit_depth - 1)
    elif data_type == "raw" and norm == "01":
        mean = 0.0
        std = 2 ** bit_depth - 1
    elif data_type == "raw" and norm == "11":
        mean = 2 ** (bit_depth - 1)
        std = 2 ** (bit_depth - 1)

    return mean, std


def get_datasets(dataset_yaml, det_task, time, data_type):
    # read dataset yaml and safe load
    with open(dataset_yaml, "r") as f:
        dataset_dict = yaml.safe_load(f)

    dataset_info = dataset_dict[det_task][time]
    train_datas = dataset_info["train_data"]
    val_datas = dataset_info["val_data"]
    eval_datas = dataset_info["eval_data"]
    leaderboard_id = dataset_info["leaderboard_id"]

    def list_info(datas, key):
        res = []
        for data in datas:
            res.append(data[key])
        return res

    train_lmdbs = list_info(train_datas, "%s_lmdb_path" % data_type)
    train_nums = list_info(train_datas, "sample_nums")
    train_coco_jsons = list_info(train_datas, "coco_json_path")
    val_lmdbs = list_info(val_datas, "%s_lmdb_path" % data_type)
    val_nums = list_info(val_datas, "sample_nums")
    val_coco_jsons = list_info(val_datas, "coco_json_path")
    eval_lmdbs = list_info(eval_datas, "%s_lmdb_path" % data_type)
    eval_nums = list_info(eval_datas, "sample_nums")
    eval_coco_jsons = list_info(eval_datas, "coco_json_path")

    res = {
        "train_lmdbs": train_lmdbs,
        "train_nums": train_nums,
        "train_coco_jsons": train_coco_jsons,
        "val_lmdbs": val_lmdbs,
        "val_nums": val_nums,
        "val_coco_jsons": val_coco_jsons,
        "eval_lmdbs": eval_lmdbs,
        "eval_nums": eval_nums,
        "eval_coco_jsons": eval_coco_jsons,
        "leaderboard_id": leaderboard_id,
    }
    return res


def tb_update(writer, model_outs, global_step_id, **kwargs):
    cls_loss = float(kwargs.get("losses")[0])
    reg_loss = float(kwargs.get("losses")[1])
    centerness_loss = float(kwargs.get("losses")[2])
    writer.add_scalar(
        tag="cls_loss",
        scalar_value=cls_loss,
        global_step=global_step_id,
    )
    writer.add_scalar(
        tag="reg_loss",
        scalar_value=reg_loss,
        global_step=global_step_id,
    )
    writer.add_scalar(
        tag="centerness_loss",
        scalar_value=centerness_loss,
        global_step=global_step_id,
    )


def tb_update_sd(writer, model_outs, global_step_id, **kwargs):
    pos_loss = float(kwargs.get("losses")[0])
    neg_loss = float(kwargs.get("losses")[1])
    center_loss = float(kwargs.get("losses")[2])
    writer.add_scalar(
        tag="pos_loss",
        scalar_value=pos_loss,
        global_step=global_step_id,
    )
    writer.add_scalar(
        tag="neg_loss",
        scalar_value=neg_loss,
        global_step=global_step_id,
    )
    writer.add_scalar(
        tag="center_loss",
        scalar_value=center_loss,
        global_step=global_step_id,
    )


def tensor_to_ndarray(output):
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            output[k] = v.cpu().numpy()
        else:
            output[k] = v
    return output


def reformat_aidi_eval_out(batch, output, task_name):
    output = tensor_to_ndarray(output)
    img_data = []
    assert isinstance(batch["img_name"], list)
    for i in range(len(batch["img_name"])):
        one_img_data = {}
        one_img_data["image_key"] = batch["img_name"][i][:-4] + ".jpg"
        one_img_data[task_name] = []

        bboxes = output["pred_bboxes"][i][:, :4]
        bboxes_scores = output["pred_bboxes"][i][:, 4]
        bboxes_classes = output["pred_bboxes"][i][:, 5]
        for j in range(bboxes.shape[0]):
            one_bbox = {}
            if bboxes_classes[j] == 0:
                one_bbox["bbox"] = bboxes[j].tolist()
                one_bbox["bbox_score"] = float(bboxes_scores[j])
                one_img_data[task_name].append(one_bbox)
        img_data.append(one_img_data)
    return img_data

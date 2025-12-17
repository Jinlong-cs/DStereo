import numpy as np


def is_contains(lhs_dict, rhs_dict):
    if isinstance(lhs_dict, dict) and isinstance(rhs_dict, dict):
        for key in rhs_dict:
            if key not in lhs_dict:
                return False
            if not is_contains(lhs_dict[key], rhs_dict[key]):
                return False
    else:
        return lhs_dict == rhs_dict
    return True


def parse_gts(record, config):
    gt_to_eval_category = config["gt_to_eval_category"]
    gts = {}
    # image tags filter
    cfg_image_tags = config["image_tags"]
    if cfg_image_tags is not None:
        assert not len(cfg_image_tags) > 1, "only support one composed tag"
        cfg_image_tags = cfg_image_tags[0]

    image_key = record["image_key"]
    # image tags filter
    if cfg_image_tags is not None:
        gt_image_tags = record["attrs"].get("tags", None)
        keep = True
        for key, value in cfg_image_tags.items():
            value = value if isinstance(value, list) else [value]
            if gt_image_tags[key] not in value:
                keep = False
        if not keep:
            return gts

    min_height = config["min_height"]
    min_width = config["min_width"]
    max_height = config["max_height"]
    max_width = config["max_width"]

    for obj_id, obj in enumerate(
        record.get(config["annokey_first_layer"], [])
    ):
        obj_attrs = obj["attrs"]
        category = obj_attrs[config["annokey_second_layer"]]
        if category in gt_to_eval_category:
            category = gt_to_eval_category[category]
        if category not in config["eval_categorys"]:
            continue
        ignore = False
        for ignore_dict in config["ignores"]:
            if is_contains(obj_attrs, ignore_dict):
                ignore = True
        if ignore:
            continue
        if (
            config["full_image_as_bbox"]
            and "height" in record
            and "width" in record
        ):
            bbox = [0, 0, record["width"] - 1, record["width"] - 1]
        elif "data" in obj:
            bbox = obj["data"]
        else:
            continue

        if bbox[3] - bbox[1] < (min_height if min_height is not None else 0):
            continue
        if bbox[2] - bbox[0] < (min_width if min_width is not None else 0):
            continue
        if bbox[3] - bbox[1] > (
            max_height if max_height is not None else 1000000
        ):
            continue
        if bbox[2] - bbox[0] > (
            max_width if max_width is not None else 1000000
        ):
            continue

        gt = {
            "image_key": image_key,
            "obj_id": obj_id,
            "bbox": bbox,
            "category": category,
        }
        obj_key = (image_key, obj_id)
        gts[obj_key] = gt

    return gts


def parse_gts_original(record, config):
    # gt_to_eval_category = config.get("gt_to_eval_category", {})
    gts = {}
    image_key = record["image_key"]
    min_height = config["min_height"]
    min_width = config["min_width"]
    max_height = config["max_height"]
    max_width = config["max_width"]

    for obj_id, obj in enumerate(
        record.get(config["annokey_first_layer"], [])
    ):
        obj_attrs = obj["attrs"]
        category = obj_attrs[config["annokey_second_layer"]]
        ignore = False
        for ignore_dict in config["ignores"]:
            if is_contains(obj_attrs, ignore_dict):
                ignore = True
        if ignore:
            continue
        if (
            config["full_image_as_bbox"]
            and "height" in record
            and "width" in record
        ):
            bbox = [0, 0, record["width"] - 1, record["width"] - 1]
        elif "data" in obj:
            bbox = obj["data"]
        else:
            continue

        if bbox[3] - bbox[1] < (min_height if min_height is not None else 0):
            continue
        if bbox[2] - bbox[0] < (min_width if min_width is not None else 0):
            continue
        if bbox[3] - bbox[1] > (
            max_height if max_height is not None else 1000000
        ):
            continue
        if bbox[2] - bbox[0] > (
            max_width if max_width is not None else 1000000
        ):
            continue

        gt = {
            "image_key": image_key,
            "obj_id": obj_id,
            "bbox": bbox,
            "category": category,
        }
        obj_key = (image_key, obj_id)
        gts[obj_key] = gt
    return gts


def parse_preds(record, config):
    score_process = config["score_process"]
    eval_category_map = {
        category: label_id
        for label_id, category in enumerate(config["eval_categorys"])
    }
    eval_category_to_preds = {}
    for score_id in config["pred_to_eval_category"]:
        eval_category = config["pred_to_eval_category"][score_id]
        eval_category_to_preds.setdefault(eval_category, []).append(score_id)
    if config["pred_prefix"] != "":
        pred_prefix = "%s_" % config["pred_prefix"]
    else:
        pred_prefix = ""

    preds = {}
    image_key = record["image_key"]
    for obj in record.get("objects", []):
        pred = {
            "image_key": image_key,
            "obj_id": obj["obj_id"],
        }
        pred_id = obj.get("%sprediction" % pred_prefix, None)
        if config["with_scores"]:
            scores = obj["%sscores" % pred_prefix]
            if score_process == "softmax":
                exp_scores = np.exp(scores)
                scores = exp_scores / exp_scores.sum()
                scores = scores.tolist()
            elif score_process == "raw":
                pass
            elif score_process is None:
                pass
            else:
                raise Exception("Invalid score process type")
            if pred_id is None:
                pred_id = np.argmax(scores)
            score = scores[pred_id]
            scores = list(
                map(
                    lambda category: max(
                        list(
                            map(
                                lambda score_id: scores[int(score_id)],
                                eval_category_to_preds[category],
                            )
                        )
                    ),
                    config["eval_categorys"],
                )
            )
            pred["scores"] = scores
        else:
            score = obj["%sscore" % pred_prefix]
        if pred_id is None:
            raise Exception("Prediction or scores is required")
        _pred_id = str(pred_id)
        if pred_id in config["pred_to_eval_category"]:
            pred_category = config["pred_to_eval_category"][pred_id]
        elif _pred_id in config["pred_to_eval_category"]:
            pred_category = config["pred_to_eval_category"][_pred_id]
        else:
            raise ValueError(
                "pred_id %d not defined in pred_to_eval_category" % pred_id
            )
        pred_id = eval_category_map.get(pred_category, -1)
        pred["pred_id"] = pred_id
        pred["score"] = score
        obj_key = (image_key, obj["obj_id"])
        preds[obj_key] = pred
    return preds


def parse_preds_original(record, config):
    score_process = config["score_process"]
    # eval_category_map = {
    #     category: label_id
    #     for label_id, category in enumerate(config["eval_categorys"])
    # }
    eval_category_to_preds = {}
    for score_id in config["pred_to_eval_category"]:
        eval_category = config["pred_to_eval_category"][score_id]
        eval_category_to_preds.setdefault(eval_category, []).append(score_id)

    if config["pred_prefix"] != "":
        pred_prefix = "%s_" % config["pred_prefix"]
    else:
        pred_prefix = ""

    preds = {}
    image_key = record["image_key"]
    for obj in record.get("objects", []):
        pred = {
            "image_key": image_key,
            "obj_id": obj["obj_id"],
        }
        pred_id = obj.get("%sprediction" % pred_prefix, None)
        if config["with_scores"]:
            scores = obj["%sscores" % pred_prefix]
            if score_process == "softmax":
                exp_scores = np.exp(scores)
                scores = exp_scores / exp_scores.sum()
                scores = scores.tolist()
            elif score_process == "raw":
                pass
            elif score_process is None:
                pass
            else:
                raise Exception("Invalid score process type")
            if pred_id is None:
                pred_id = np.argmax(scores)
            score = scores[pred_id]
            pred["scores"] = scores
        else:
            score = obj["score"]
        if pred_id is None:
            raise Exception("Prediction or scores is required")
        pred["pred_id"] = pred_id
        pred["score"] = score
        obj_key = (image_key, obj["obj_id"])
        preds[obj_key] = pred
    return preds

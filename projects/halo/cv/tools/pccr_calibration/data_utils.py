import json


def gather_all_infos(img_infos_file, modelbase_res_file):
    imginfos = [json.loads(line.strip("\n")) for line in open(img_infos_file)]
    modelbase_res = [
        json.loads(line.strip("\n")) for line in open(modelbase_res_file)
    ]
    all_id_infos = {}
    for img in imginfos:
        # 筛选gt
        if (
            max(img["gt_list"]["left_angle"]) > 30
            or min(img["gt_list"]["left_angle"]) < -30
        ):
            continue
        if img["gt_list"]["left_eye3d"][-1] < 300:
            continue

        img["glint_list_new"].sort(key=lambda x: x[0], reverse=False)

        idp = img["id"]
        tag_d = img["tag_id"]
        if all_id_infos.get(idp) is None:
            all_id_infos[idp] = {}
        if all_id_infos[idp].get(tag_d) is None:
            img_info = {
                "img_path": [],
                "left_angle": [],
                "left_eye3d": [],
                "pog_pixel": [],
                "glint": [],
                "pupil_boundary": [],
            }
            all_id_infos[idp][tag_d] = img_info
        all_id_infos[idp][tag_d]["img_path"].append(img["img_path"])
        all_id_infos[idp][tag_d]["left_angle"].append(
            img["gt_list"]["left_angle"]
        )
        all_id_infos[idp][tag_d]["left_eye3d"].append(
            img["gt_list"]["left_eye3d"]
        )
        all_id_infos[idp][tag_d]["pog_pixel"].append(
            img["gt_list"]["pog_pixel"]
        )
        all_id_infos[idp][tag_d]["glint"].append(img["glint_list_new"])
        all_id_infos[idp][tag_d]["pupil_boundary"].append(
            img["pupil_boundary_new"]
        )
    for res in modelbase_res:
        idp = res["id"]
        assert all_id_infos.get(idp) is not None
        all_id_infos[idp]["all_tags"] = list(all_id_infos[idp].keys())
        # res["eye_params"]["alpha_left"] *= 3.141592653589793 / 180.0
        # res["eye_params"]["beta"] *= 3.141592653589793 / 180.0
        all_id_infos[idp]["eye_params_modelbase"] = res["eye_params"]
    return all_id_infos


def padding_pbs(train_infos, max_pb_nums):
    for tag_idx, tag_info in enumerate(train_infos):
        for img_idx, pb in enumerate(tag_info["pupil_boundary"]):
            pad_nums = max_pb_nums - len(pb)
            for _ in range(pad_nums):
                train_infos[tag_idx]["pupil_boundary"][img_idx].append(
                    [-1, -1]
                )
    return train_infos

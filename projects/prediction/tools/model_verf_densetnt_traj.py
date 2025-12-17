import os
import pickle
import sys
from copy import deepcopy

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

from hat.core.traj_pred_utils import Affine2D
from hat.registry import build_from_registry
from hat.utils import Config
from hat.utils.checkpoint import (
    load_checkpoint,
    load_state_dict,
    update_state_dict_by_strip_prefix,
)
from hat.utils.qconfig_manager import QconfigMode, set_qconfig_mode

set_qconfig_mode(QconfigMode.QAT)

HAT_PATH = "/home/users/shiqi.tan/HAT"
sys.path.append(HAT_PATH)
torch.cuda.set_device(3)

print("model verf!")
# 0. Belows are parameters that the user should check.
# 新版数据生产pipeline生成的pkl数据：
# pkl_dataset_path = "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/val_167753.pkl"  # noqa: E501
pkl_dataset_path = "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/shiqi.tan/traj_dataset_pkl_path/fus_version-hdmap_dataset_version1/fus_data_val.pkl"  # noqa: E501

# 旧版数据生产pipeline生成的pkl数据：
# pkl_20230407_113327_408 = "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/shiqi.tan/badcase_train/J4243_20230407_D/20230407-113327_408/traj_process_multitask/20_0.8.4_0_0.0.1_12_70_3_70_50contextframes_dist_restriction/pickles/pickle_test_train.pkl"  # noqa: E501
# pkl_dataset_path = pkl_20230407_113327_408

pre_file_name = "/jfs-public/users/shiqi.tan/results/densetnt_task12_1_stage1_compile5out_20231019/checkpoint/densetnt_traj_release_v11.0.0_20231024_bpu2.5/"  # noqa: E501
qat_ckpt_path = pre_file_name + "qat-checkpoint-best.pth.tar"
pt_model_path = (
    pre_file_name
    + "densetnt_v11.0.0_bpu2.5_16_obstacles_c18268edd5f5a0f5fff449b17e0465a4.pt"  # noqa: E501
)
hbm_path = (
    pre_file_name
    + "densetnt_v11.0.0_bpu2.5_16_obstacles_4bd9964f803c68e6df32b3f1d8bd0a05.hbm"  # noqa: E501
)
debug_path = pre_file_name + "debug/"
simulator_path = "model_verfier_simulator"
model_name = "densetnt"
num_obs_in_hbm = 16
bpu = 2.5
workflow = {
    "pt": True,
    "simulator": False,
    "bpu": True,
}
used_sample_idx = 20
timestamp = None

# 1. Prepare
if bpu == 2.2:
    march = "bayes1"
    bpu_ip = "10.103.72.160"
    verf_files = [
        "hbdk_output0_feature__normal_head_aten_reshape_torch_native_elem_float.txt",  # noqa: E501
        "hbdk_output1_feature__normal_head_goal_score_decoder_fc_hz_conv2d_torch_native_elem_float.txt",  # noqa: E501
    ]
elif bpu == 2.5:
    march = March.BAYES
    # bpu_ip = "10.103.72.180"
    bpu_ip = "10.103.53.229"
    verf_files = [
        "hbdk_output1_feature__normal_head_hz_topk_torch_native_elem_float.txt",  # noqa: E501
        "hbdk_output0_feature__normal_head_aten_reshape_torch_native_elem_float.txt",  # noqa: E501
    ]
else:
    raise ValueError(f"Unsupported bpu version {bpu}.")

os.makedirs(debug_path, exist_ok=True)
with open(pkl_dataset_path, "rb") as f:
    dataset = pickle.load(f)
cfg_file = os.path.join(
    HAT_PATH, "projects/prediction/configs/densetnt_traj.py"
)
cfg = Config.fromfile(cfg_file)
dataloader = build_from_registry(cfg.val_dataloader)

gt_traj_len = cfg.gt_traj_len
num_goals = cfg.num_goals
polyline_seg_len = cfg.polyline_seg_len
road_feat_dim = cfg.road_feat_dim
num_ele_seg = cfg.max_num_ele_seg
max_obs_num = cfg.max_obs_num
traj_polyline_len = cfg.traj_polyline_len
traj_feat_dim = cfg.traj_feat_dim
context_frames = cfg.context_frames
num_all_poly = num_ele_seg + max_obs_num

# 2. Get model input.
if used_sample_idx is None:
    used_sample_idx = 0

if timestamp:
    for i in range(len(dataset)):
        ts = dataset[i]["seq_df"]["timestamp"].unique().tolist()
        ts = np.sort(ts)
        if ts[3] == timestamp:
            used_sample_idx = i
            print("find the correct timestamp:", i, ts[3])
            break

sample = dataset[used_sample_idx]
output = {}
output[0] = ["original", sample]
num_wrong_trans = 1
for idx, tr in enumerate(dataloader.dataset.transforms.transforms):
    try:
        sample = tr(sample)
        output[idx + 1] = [str(type(tr)), deepcopy(sample)]
    except Exception:
        print(type(tr))
        output[idx + 1] = [str(type(tr)), None]
        num_wrong_trans += 1
last_idx = len(output.keys())
model_in = dataloader.collate_fn([output[last_idx - num_wrong_trans][1]])
road_mask = model_in["struct_road_masks"].cuda()
traj_mask = model_in["struct_traj_masks"].cuda()
concat_mask = torch.cat([traj_mask, road_mask], dim=1)
attention_mask = torch.matmul(
    concat_mask[:, :, None], concat_mask[:, None, :]
)  # noqa: E501
attention_mask = attention_mask[:, None, :, :]
model_in["attention_mask"] = attention_mask.cuda()

model_in["struct_road_feats"] = model_in["struct_road_feats"].permute(
    0, 3, 1, 2
)
model_in["struct_traj_feats"] = model_in["struct_traj_feats"].permute(
    0, 3, 1, 2
)


print("valid track id: ", model_in["valid_track_ids"][0])
print("last context frame: ", sample["lcf_timestamp"])


for i in range(len(output)):
    if output[i][1] is None:
        continue
    if "seq_index" in output[i][1]:
        output[i][1]["seq_index"] = None
    if "seq_center" in output[i][1]:
        output[i][1]["seq_center"] = {
            "pos_x": output[i][1]["seq_center"].pos_x,
            "pos_y": output[i][1]["seq_center"].pos_y,
            "yaw": output[i][1]["seq_center"].yaw,
        }

# remove error transform index
bad_trans = [
    "original",
    "<class 'hat.data.transforms.traj_pred.traj_pred_obstacle.GetObstacleSafeArea'>",  # noqa: E501
]
output_pure = deepcopy(output)
keys = list(output_pure.keys())
for key in keys:
    if output_pure[key][0] in bad_trans:
        output_pure.pop(key)
ret = {"transfrom_results": output_pure}

# 3. Build qat model and infer.
deploy_model = deepcopy(cfg.model)
deploy_model["losses"] = None
deploy_model["is_int_infer_model"] = True
for _, neck in deploy_model["necks"].items():
    neck["is_int_infer_model"] = True
for _, head in deploy_model["heads"].items():
    head["training_step"] = "int_infer"
qat_model = build_from_registry(deploy_model)
qat_model.fuse_model()
qat_model.set_qconfig()

horizon.march.set_march(march)
horizon.quantization.prepare_qat(qat_model, inplace=True)

ckpt_dict = load_checkpoint(
    qat_ckpt_path,
    map_location="cpu",
    state_dict_update_func=update_state_dict_by_strip_prefix,
)
qat_model = load_state_dict(
    qat_model,
    ckpt_dict["state_dict"],
    # ignore_extra=True,
    # allow_miss=True,
    verbose=True,
)
qat_model.eval()
qat_model.cuda()

qat_model_in = dict(
    attention_mask=model_in["attention_mask"].cuda(),
    goal_coords=model_in["goal_coords"].cuda(),
    struct_road_feats=model_in["struct_road_feats"].cuda(),  # noqa: E501
    struct_traj_feats=model_in["struct_traj_feats"].cuda(),  # noqa: E501
    state_vectors=model_in["state_vectors"].cuda(),
)
output = qat_model(qat_model_in)


trajs_for_val_1 = output.predict_trajs
# goal_scores_1 = output.predict_goal_scores
goal_scores_1 = output.real_scores
print("python qat result: ")
print(trajs_for_val_1[0, 0, 0, 0], goal_scores_1[0, 0, 0, 0])

ret["qat_trajs_for_val"] = trajs_for_val_1.to("cpu").detach().numpy()
ret["qat_goal_scores"] = goal_scores_1.to("cpu").detach().numpy()

# 4. Build pt model and infer.
real_attention_mask = model_in[
    "attention_mask"
].cpu()  # torch.Size([1, 1, 544, 544])
real_goal_coords = model_in[
    "goal_coords"
].cpu()  # torch.Size([1, 2, 1, 20127])
real_struct_road_feats = model_in[
    "struct_road_feats"
].cpu()  # torch.Size([1, 7, 512, 9])
real_struct_traj_feats = model_in[
    "struct_traj_feats"
].cpu()  # torch.Size([1, 6, 32, 3])
real_state_vectors = model_in[
    "state_vectors"
].cpu()  # torch.Size([1, 6, 32, 3])


# 补零
struct_road_feats = torch.zeros(
    (num_obs_in_hbm, road_feat_dim, num_ele_seg, polyline_seg_len - 1)
)  # torch.Size([16x12x128x9])
struct_traj_feats = torch.zeros(
    (num_obs_in_hbm, traj_feat_dim, max_obs_num, traj_polyline_len)
)  # torch.Size([16x9x32x3])
attention_mask = torch.zeros(
    (num_obs_in_hbm, 1, num_all_poly, num_all_poly)
)  # torch.Size([16x1x160x160])
goal_coords = torch.zeros(
    (num_obs_in_hbm, 2, 1, num_goals)
)  # torch.Size([16x2x1x1024])
state_vectors = torch.zeros(
    (num_obs_in_hbm, 3, 1, 1)
)  # torch.Size([16x3x1x1])
struct_road_feats[: real_struct_road_feats.shape[0]] = real_struct_road_feats
struct_traj_feats[: real_struct_traj_feats.shape[0]] = real_struct_traj_feats
goal_coords[: real_goal_coords.shape[0]] = real_goal_coords
attention_mask[: real_attention_mask.shape[0]] = real_attention_mask
state_vectors[: real_state_vectors.shape[0]] = real_state_vectors

ret.update(
    {
        "input_attention_mask": attention_mask.numpy(),
        "input_goal_coords": goal_coords.numpy(),
        "input_struct_road_feats": struct_road_feats.numpy(),
        "input_struct_traj_feats": struct_traj_feats.numpy(),
        "input_state_vectors": state_vectors.numpy(),
    }
)

pt_model = torch.jit.load(pt_model_path)
if workflow["pt"]:
    pt_inputs = dict(
        attention_mask=attention_mask,
        goal_coords=goal_coords,
        struct_road_feats=struct_road_feats,
        struct_traj_feats=struct_traj_feats,
        state_vectors=state_vectors,
    )
    trajs_for_val_2, goal_scores_2 = pt_model(pt_inputs)
    print(trajs_for_val_2.shape, goal_scores_2.shape)
    print("python pt result: ")
    print(trajs_for_val_2[0, 0, 0, 0], goal_scores_2[0, 0, 0, 0])

    ret["pt_trajs_for_val_2"] = trajs_for_val_2.to("cpu").detach().numpy()
    ret["pt_goal_scores_2"] = goal_scores_2.to("cpu").detach().numpy()

    np.savetxt(
        os.path.join(debug_path + "pt_trajs_for_val.txt"),
        ret["pt_trajs_for_val_2"][:, :, 0, 0],
        fmt="%.6f",
        delimiter="\t",
    )


trajs_for_val_2 = trajs_for_val_2.to("cpu").detach().numpy()
trajs_for_val_1 = trajs_for_val_1.to("cpu").detach().numpy()
print(
    "max diff between qat and pt: "
    + str(
        np.max(
            np.abs(
                trajs_for_val_2[: trajs_for_val_1.shape[0], :, :, :]
                - trajs_for_val_1
            )
        )
    )
)


# 5.1 cal quantized data for hbm infer
def quant(input, scale, int16=False):
    input = input.numpy()
    if not int16:
        output = np.clip(np.floor((input / scale) + 0.5), -128, 127)
    else:
        output = np.clip(np.floor((input / scale) + 0.5), -32768, 32767)
    return output


int_attention_mask = quant(
    attention_mask.cpu(),
    pt_model.state_dict()["backbone.attn_quant.scale"].numpy()[0],
)
int_struct_road_feats = quant(
    struct_road_feats.cpu(),
    pt_model.state_dict()["backbone.road_quant.scale"].numpy()[0],
    int16=True,
)
int_struct_traj_feats = quant(
    struct_traj_feats.cpu(),
    pt_model.state_dict()["backbone.traj_quant.scale"].numpy()[0],
    int16=True,
)
int_goal_coords = quant(
    goal_coords.cpu(),
    pt_model.state_dict()["normal_head.goal_quant.scale"].numpy()[0],
    int16=True,
)
int_state_vectors = quant(
    state_vectors.cpu(),
    pt_model.state_dict()["traj_neck.state_vector_quant.scale"].numpy()[0],
    int16=True,
)

attention_mask_file = os.path.join(debug_path + "int_attention_mask.bin")
road_feats_file = os.path.join(debug_path + "int_struct_road_feats.bin")
traj_feats_file = os.path.join(debug_path + "int_struct_traj_feats.bin")
goal_coords_file = os.path.join(debug_path + "int_goal_coords.bin")
state_vectors_file = os.path.join(debug_path + "int_state_vectors.bin")

# hbm的维度与python端的维度不同
int_attention_mask = int_attention_mask.astype("int8").transpose(
    0, 2, 3, 1
)  # (1, 544, 544, 1)
int_struct_road_feats = int_struct_road_feats.astype("int16").transpose(
    0, 2, 3, 1
)  # (1, 512, 9, 7)
int_struct_traj_feats = int_struct_traj_feats.astype("int16").transpose(
    0, 2, 3, 1
)  # (1, 32, 3, 6)
int_goal_coords = int_goal_coords.astype("int16").transpose(
    0, 2, 3, 1
)  # (1, 1, 20127, 2)
int_state_vectors = int_state_vectors.astype("int8").transpose(
    0, 2, 3, 1
)  # (1, 1, 20127, 2)

int_attention_mask.tofile(attention_mask_file)
int_struct_road_feats.tofile(road_feats_file)
int_struct_traj_feats.tofile(traj_feats_file)
int_goal_coords.tofile(goal_coords_file)
int_state_vectors.tofile(state_vectors_file)

ret.update(
    {
        "input_quant_attention_mask": int_attention_mask,
        "input_quant_struct_road_feats": int_struct_road_feats,
        "input_quant_struct_traj_feats": int_struct_traj_feats,
        "input_quant_goal_coords": int_goal_coords,
        "input_quant_state_vectors": int_state_vectors,
    }
)


# 5.2 common commands.
hbdk_cmd = f"cd {debug_path} && hbdk-model-verifier --hbm {hbm_path} "
hbdk_cmd += f"--model-input {attention_mask_file},{goal_coords_file},{state_vectors_file},"  # noqa: E501
hbdk_cmd += f"{road_feats_file},{traj_feats_file} --model-pt {pt_model_path} "
hbdk_cmd += f"--model-name {model_name} --times 1000"
print(hbdk_cmd)

# 5.3 verifiy on simulator.
tmp_subdirs = os.listdir(debug_path)
if workflow["simulator"]:
    tmp_cmd = hbdk_cmd + " --skip-bpu"
    os.system(tmp_cmd)
    new_subdirs = os.listdir(debug_path)
    simulator_path = list(set(new_subdirs) - set(tmp_subdirs))
    assert len(simulator_path) == 1
    simulator_path = simulator_path[0]

    predict_goal_scores_path = os.path.join(
        debug_path, f"{simulator_path}/simulator_output", verf_files[1]
    )
    predict_trajs_for_val_path = os.path.join(
        debug_path, f"{simulator_path}/simulator_output", verf_files[0]
    )

    with open(predict_goal_scores_path, "r") as f:
        predict_goal_scores_infos = f.readlines()
    with open(predict_trajs_for_val_path, "r") as f:
        predict_trajs_for_val_infos = f.readlines()

    predict_goal_scores_infos = predict_goal_scores_infos[3:]
    predict_goal_scores = []
    for info in predict_goal_scores_infos:
        tmp = info.split("\n")[0].split(" ")
        tmp = [float(p) for p in tmp[:-1]]
        predict_goal_scores.append(tmp)
    predict_goal_scores = np.array(predict_goal_scores).reshape(
        -1, 1, 1, cfg.num_goals
    )

    predict_trajs_for_val_infos = predict_trajs_for_val_infos[3:]
    predict_trajs_for_val = []
    for info in predict_trajs_for_val_infos:
        tmp = info.split("\n")[0].split(" ")
        tmp = [float(p) for p in tmp[:-1]]
        predict_trajs_for_val.append(tmp)
    predict_trajs_for_val = np.array(predict_trajs_for_val).reshape(
        -1, max(cfg.k_values), cfg.gt_traj_len, 2
    )

    print("Simulator results:")
    print(predict_trajs_for_val[0, 0, 0, 0], predict_goal_scores[0, 0, 0, 0])
    ret["hbdk-verf_trajs_for_val"] = predict_trajs_for_val
    ret["hbdk-verf_goal_scores"] = predict_goal_scores


# 5.4 verifiy on bpu.
tmp_subdirs = os.listdir(debug_path)
if workflow["bpu"]:
    tmp_cmd = hbdk_cmd + f" --ip {bpu_ip}"
    os.system(tmp_cmd)
    new_subdirs = os.listdir(debug_path)
    bpu_path = list(set(new_subdirs) - set(tmp_subdirs))
    assert len(bpu_path) == 1
    bpu_path = bpu_path[0]

    predict_goal_scores_path = os.path.join(
        debug_path, f"{bpu_path}/bpu_output/", verf_files[0]
    )
    predict_trajs_for_val_path = os.path.join(
        debug_path, f"{bpu_path}/bpu_output/", verf_files[1]
    )

    with open(predict_goal_scores_path, "r") as f:
        predict_goal_scores_infos = f.readlines()
    with open(predict_trajs_for_val_path, "r") as f:
        predict_trajs_for_val_infos = f.readlines()

    predict_goal_scores_infos = predict_goal_scores_infos[3:]
    predict_goal_scores = []
    for info in predict_goal_scores_infos:
        tmp = info.split("\n")[0].split(" ")
        tmp = [float(p) for p in tmp[:-1]]
        predict_goal_scores.append(tmp)
    predict_goal_scores = np.array(predict_goal_scores).reshape(-1, 1, 1, 5)

    predict_trajs_for_val_infos = predict_trajs_for_val_infos[3:]
    predict_trajs_for_val = []
    for info in predict_trajs_for_val_infos:
        tmp = info.split("\n")[0].split(" ")
        tmp = [float(p) for p in tmp[:-1]]
        predict_trajs_for_val.append(tmp)
    predict_trajs_for_val = np.array(predict_trajs_for_val).reshape(
        -1, max(cfg.k_values), cfg.gt_traj_len, 2
    )

    print("Bpu results:")
    print(predict_trajs_for_val[0, 0, 0, 0], predict_goal_scores[0, 0, 0, 0])
    ret["bpu_hbdk-verf_trajs_for_val"] = predict_trajs_for_val
    ret["bpu_hbdk-verf_goal_scores"] = predict_goal_scores


# 6. Trans output to vcs (use_pt).
bev_origin_x = cfg.bev_origin_x
bev_origin_y = cfg.bev_origin_y
img_resolution = cfg.img_resolution


def trans_bev_coords_to_vcs_coords(
    bev_coords, bev_origin_x, bev_origin_y, img_resolution
):
    inv_resolu = 1 / img_resolution
    bev_x_offset = bev_origin_x / img_resolution
    bev_y_offset = bev_origin_y / img_resolution
    bev_x = bev_coords[:, 0]
    bev_y = bev_coords[:, 1]
    bev_yaw = bev_coords[:, 2]
    yaw_diff = np.pi
    bev_x, bev_y = Affine2D.coord_translate(
        bev_x, bev_y, bev_x_offset, bev_y_offset
    )
    bev_x, bev_y = Affine2D.coord_rotate(bev_x, bev_y, yaw_diff)
    vcs_x, vcs_y = Affine2D.coord_scale(bev_x, bev_y, inv_resolu, inv_resolu)
    vcs_yaw = bev_yaw - yaw_diff
    return np.stack([vcs_x, vcs_y, vcs_yaw], axis=-1)


# 6. Trans output to vcs (use_pt).
img_coords = np.array(model_in["valid_img_coords"], dtype=np.float32)
vcs_coords = trans_bev_coords_to_vcs_coords(
    img_coords[0], bev_origin_x, bev_origin_y, img_resolution
)

#
# (n,1,12,2)
# [n,1,i,2]

# 差分轨迹转正常轨迹
if "diff_fut_trajs" in model_in:
    predict_trajs_for_val = torch.cumsum(
        torch.tensor(predict_trajs_for_val), dim=2
    )
    trajectories = predict_trajs_for_val[
        : real_struct_road_feats.shape[0], :, :, :
    ]

all_vcs_traj = []
for vcs_c, mean in zip(vcs_coords, trajectories.to("cpu").detach().numpy()):
    num_ancs, traj_len, _ = mean.shape
    tmp_mean = mean.reshape([-1, 2])
    centric_x = tmp_mean[:, 0]
    centric_y = tmp_mean[:, 1]
    obs_vcs_x, obs_vcs_y = Affine2D.coord_rotate(
        centric_x, centric_y, -vcs_c[2]
    )
    obs_vcs_x, obs_vcs_y = Affine2D.coord_translate(
        obs_vcs_x, obs_vcs_y, -vcs_c[0], -vcs_c[1]
    )
    tmp_vcs = np.stack([obs_vcs_x, obs_vcs_y], axis=-1).reshape(
        num_ancs, traj_len, 2
    )
    all_vcs_traj.append(tmp_vcs)
all_vcs_traj = np.stack(all_vcs_traj)
ret["all_vcs_traj"] = all_vcs_traj

with open(os.path.join(debug_path, "out.pkl"), "wb") as f:
    pickle.dump(ret, f)

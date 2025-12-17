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

HAT_PATH = "/home/users/shiqi.tan/HAT"
sys.path.append(HAT_PATH)
torch.cuda.set_device(3)

print("model verf!")
# 0. Belows are parameters that the user should check.
pkl_20230407_113327_408 = "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/shiqi.tan/badcase_train/J4243_20230407_D/20230407-113327_408/traj_process_multitask/20_0.8.4_0_0.0.1_12_70_3_70_50contextframes_dist_restriction/pickles/pickle_test_train.pkl"  # noqa: E501
pkl_dataset_path = pkl_20230407_113327_408

pre_file_name = "/jfs-public/users/shiqi.tan/results/7.0.0-Experiment-vectornet_totaldata_trajpred_addodd_20230604/vectornet_release_v7.1.0_20230607_bpu2.5/"  # noqa: E501
qat_ckpt_path = pre_file_name + "qat-checkpoint-best.pth.tar"
pt_model_path = (
    pre_file_name
    + "traj_pred_vectornet_truncated_v7.1.0_bpu2.5_32_obstacles_a84d3f5e4e8f462db5ad31461253dc12.pt"  # noqa: E501
)
hbm_path = (
    pre_file_name
    + "traj_pred_vectornet_truncated_v7.1.0_bpu2.5_32_obstacles_8281cbeee988e9a4b4e30fef2118c66c.hbm"  # noqa: E501
)
debug_path = pre_file_name + "debug/"
simulator_path = "model_verfier_simulator"
model_name = "VectorNetV2"  # vectornet or BasicVectornet or VectorNetV2
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
        "hbdk_output0_feature_vectornet.traj_head.anchor_prob_conv2d_elem_float.txt",  # noqa: E501
        "hbdk_output1_feature_vectornet.traj_head.anchor_mean_var_conv2d_elem_float.txt",  # noqa: E501
    ]
elif bpu == 2.5:
    march = March.BAYES
    bpu_ip = "10.103.42.248"
    verf_files = [
        "hbdk_output0_feature__traj_head_anchor_prob_hz_conv2d_elem_float.txt",  # noqa: E501
        "hbdk_output1_feature__traj_head_anchor_mean_var_hz_conv2d_elem_float.txt",  # noqa: E501
    ]
else:
    raise ValueError(f"Unsupported bpu version {bpu}.")

os.makedirs(debug_path, exist_ok=True)
with open(pkl_dataset_path, "rb") as f:
    dataset = pickle.load(f)
cfg_file = os.path.join(HAT_PATH, "projects/prediction/configs/vectornet.py")
cfg = Config.fromfile(cfg_file)
dataloader = build_from_registry(cfg.val_dataloader)


pkl_fps = cfg.pkl_fps
context_frames = cfg.context_frames
target_freq = cfg.target_freq
max_obs_num = cfg.max_obs_num
polyline_seg_len = cfg.polyline_seg_len
max_num_ele_seg = cfg.max_num_ele_seg
traj_sample_ratio = cfg.traj_sample_ratio
traj_polyline_len = cfg.traj_polyline_len
hdm_max_num_obs = cfg.hdm_max_num_obs
road_feat_dim = cfg.road_feat_dim
traj_feat_dim = cfg.traj_feat_dim
state_vector_dim = cfg.num_state_vectors
num_all_poly = cfg.num_all_poly
traj_len = cfg.traj_len

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
for idx, tr in enumerate(dataloader.dataset.transforms.transforms):
    try:
        sample = tr(sample)
        output[idx + 1] = [str(type(tr)), deepcopy(sample)]
    except Exception:
        print(type(tr))
        output[idx + 1] = [str(type(tr)), None]
last_idx = len(output.keys())
model_in = dataloader.collate_fn([output[last_idx - 1][1]])
road_mask = model_in["struct_road_masks"].cuda()
traj_mask = model_in["struct_traj_masks"].cuda()
concat_mask = torch.cat([traj_mask, road_mask], dim=1)
attention_mask = torch.matmul(
    concat_mask[:, :, None], concat_mask[:, None, :]
)  # noqa: E501
attention_mask = attention_mask[:, None, :, :]
model_in["attention_mask"] = attention_mask.cuda()

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
qat_model = build_from_registry(cfg.model)
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
    allow_miss=True,
    ignore_extra=True,
    verbose=0,
)
qat_model.eval()
qat_model.cuda()


output = qat_model(model_in)

anchor_probs_1 = output["traj_head_anchor_probs"]
anchor_mean_var_1 = output["traj_head_anchor_mean_var"]
print("python qat result: ")
print(anchor_probs_1.shape, anchor_mean_var_1.shape)
print(anchor_probs_1[0, 0, 0, 0], anchor_mean_var_1[0, 0, 0, 0])

ret["qat_anchor_probs_1"] = anchor_probs_1.to("cpu").detach().numpy()
ret["qat_anchor_mean_var_1"] = anchor_mean_var_1.to("cpu").detach().numpy()
ret["model_anchors"] = output["traj_head_anchors"].to("cpu").detach().numpy()

# 4. Build pt model and infer.
real_road_feats = model_in["struct_road_feats"].cpu()
real_traj_feats = model_in["struct_traj_feats"].cpu()
real_attention_mask = model_in["attention_mask"].cpu()
real_state_vectors = model_in["state_vectors"].cpu()


road_feats = torch.zeros(
    (num_obs_in_hbm, road_feat_dim, max_num_ele_seg, polyline_seg_len - 1)
)
traj_feats = torch.zeros(
    (num_obs_in_hbm, traj_feat_dim, max_obs_num, traj_polyline_len)
)
attention_mask = torch.zeros((num_obs_in_hbm, 1, num_all_poly, num_all_poly))
state_vectors = torch.zeros((num_obs_in_hbm, state_vector_dim, 1, 1))
road_feats[: real_road_feats.shape[0]] = real_road_feats
traj_feats[: real_traj_feats.shape[0]] = real_traj_feats
state_vectors[: real_state_vectors.shape[0]] = real_state_vectors
attention_mask[: real_attention_mask.shape[0]] = real_attention_mask

ret.update(
    {
        "input_road_feats": road_feats.numpy(),
        "input_traj_feats": traj_feats.numpy(),
        "input_attention_mask": attention_mask.numpy(),
        "input_state_vectors": state_vectors.numpy(),
    }
)

pt_model = torch.jit.load(pt_model_path)
if workflow["pt"]:
    pt_inputs = dict(
        struct_road_feats=road_feats,
        struct_traj_feats=traj_feats,
        attention_mask=attention_mask,
        state_vectors=state_vectors,
    )
    (
        anchor_probs_2,
        anchor_mean_var_2,
    ) = pt_model(pt_inputs)
    print("python pt result: ")
    print(anchor_probs_2.shape, anchor_mean_var_2.shape)
    print(anchor_probs_2[0, 0, 0, 0], anchor_mean_var_2[0, 0, 0, 0])

    ret["pt_anchor_probs_2"] = anchor_probs_2.to("cpu").detach().numpy()
    ret["pt_anchor_mean_var_2"] = anchor_mean_var_2.to("cpu").detach().numpy()

    np.savetxt(
        os.path.join(debug_path + "pt_anchor_prob.txt"),
        ret["pt_anchor_probs_2"][:, :, 0, 0],
        fmt="%.6f",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(debug_path + "pt_anchor_mean_var.txt"),
        ret["pt_anchor_mean_var_2"][:, :, 0, 0],
        fmt="%.6f",
        delimiter="\t",
    )


# 5.1 cal quantized data for hbm infer
def quant(input, scale, int16=False):
    input = input.numpy()
    if not int16:
        output = np.clip(np.floor((input / scale) + 0.5), -128, 127)
    else:
        output = np.clip(np.floor((input / scale) + 0.5), -32768, 32767)
    return output


int_road_feats = quant(
    road_feats.cpu(),
    pt_model.state_dict()["backbone.road_quant.scale"].numpy()[0],
    int16=True,
)
int_traj_feats = quant(
    traj_feats.cpu(),
    pt_model.state_dict()["backbone.traj_quant.scale"].numpy()[0],
    int16=True,
)
int_attention_mask = quant(
    attention_mask.cpu(),
    pt_model.state_dict()["backbone.attn_quant.scale"].numpy()[0],
)
int_state_vectors = quant(
    state_vectors.cpu(),
    pt_model.state_dict()["traj_neck.state_vector_quant.scale"].numpy()[0],
)


road_feats_file = os.path.join(debug_path + "int_road_feats.bin")
traj_feats_file = os.path.join(debug_path + "int_traj_feats.bin")
attention_mask_file = os.path.join(debug_path + "int_attention_mask.bin")
state_vectors_file = os.path.join(debug_path + "int_state_vectors.bin")

int_road_feats = int_road_feats.astype("int16").transpose(0, 2, 3, 1)
int_traj_feats = int_traj_feats.astype("int16").transpose(0, 2, 3, 1)
int_attention_mask = int_attention_mask.astype("int8").transpose(0, 2, 3, 1)
int_state_vectors = int_state_vectors.astype("int8").transpose(0, 2, 3, 1)

int_road_feats.tofile(road_feats_file)
int_traj_feats.tofile(traj_feats_file)
int_attention_mask.tofile(attention_mask_file)
int_state_vectors.tofile(state_vectors_file)

ret.update(
    {
        "input_quant_road_feats": int_road_feats,
        "input_quant_traj_feats": int_traj_feats,
        "input_quant_attention_mask": int_attention_mask,
        "input_quant_state_vectors": int_state_vectors,
    }
)


# 5.2 common commands.
hbdk_cmd = f"cd {debug_path} && hbdk-model-verifier --hbm {hbm_path} "
hbdk_cmd += f"--model-input {attention_mask_file},{state_vectors_file},"
hbdk_cmd += f"{road_feats_file},{traj_feats_file} --model-pt {pt_model_path} "
hbdk_cmd += f"--model-name {model_name} --times 300"
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

    raw_prob_path = os.path.join(
        debug_path, f"{simulator_path}/simulator_output", verf_files[0]
    )
    raw_traj_path = os.path.join(
        debug_path, f"{simulator_path}/simulator_output", verf_files[1]
    )

    with open(raw_prob_path, "r") as f:
        raw_prob_infos = f.readlines()
    with open(raw_traj_path, "r") as f:
        raw_traj_infos = f.readlines()

    raw_prob_infos = raw_prob_infos[3:]
    raw_hbm_prob = []
    for raw_info in raw_prob_infos:
        raw_prob = raw_info.split("\n")[0].split(" ")
        raw_prob = [float(p) for p in raw_prob[:-1]]
        raw_hbm_prob.append(raw_prob)
    raw_hbm_prob = np.array(raw_hbm_prob).reshape(-1, cfg.anchor_num, 1, 1)

    raw_traj_infos = raw_traj_infos[3:]
    raw_hbm_meanvar = []
    for raw_info in raw_traj_infos:
        raw_meanvar = raw_info.split("\n")[0].split(" ")
        raw_meanvar = [float(p) for p in raw_meanvar[:-1]]
        raw_hbm_meanvar.append(raw_meanvar)
    num_meanvar_feats = cfg.anchor_num * cfg.traj_len * 5
    raw_hbm_meanvar = np.array(raw_hbm_meanvar).reshape(
        -1, num_meanvar_feats, 1, 1
    )

    print("Simulator results:")
    print(raw_hbm_prob.shape, raw_hbm_meanvar.shape)
    print(raw_hbm_prob[0, 0, 0, 0], raw_hbm_meanvar[0, 0, 0, 0])
    ret["hbdk-verf_probs_float"] = raw_hbm_prob
    ret["hbdk-verf_mean_var_float"] = raw_hbm_meanvar

# 5.4 verifiy on bpu.
tmp_subdirs = os.listdir(debug_path)
if workflow["bpu"]:
    tmp_cmd = hbdk_cmd + f" --ip {bpu_ip}"
    os.system(tmp_cmd)
    new_subdirs = os.listdir(debug_path)
    bpu_path = list(set(new_subdirs) - set(tmp_subdirs))
    assert len(bpu_path) == 1
    bpu_path = bpu_path[0]
    raw_bpu_prob_path = os.path.join(
        debug_path, f"{bpu_path}/bpu_output/", verf_files[0]
    )
    raw_bpu_traj_path = os.path.join(
        debug_path, f"{bpu_path}/bpu_output/", verf_files[1]
    )

    with open(raw_bpu_prob_path, "r") as f:
        raw_bpu_prob_infos = f.readlines()
    with open(raw_bpu_traj_path, "r") as f:
        raw_bpu_traj_infos = f.readlines()

    raw_bpu_prob_infos = raw_bpu_prob_infos[3:]
    raw_bpu_hbm_prob = []
    for raw_info in raw_bpu_prob_infos:
        raw_prob = raw_info.split("\n")[0].split(" ")
        raw_prob = [float(p) for p in raw_prob[:-1]]
        raw_bpu_hbm_prob.append(raw_prob)
    raw_bpu_hbm_prob = np.array(raw_bpu_hbm_prob).reshape(
        -1, cfg.anchor_num, 1, 1
    )

    raw_bpu_traj_infos = raw_bpu_traj_infos[3:]
    raw_bpu_hbm_meanvar = []
    for raw_info in raw_bpu_traj_infos:
        raw_meanvar = raw_info.split("\n")[0].split(" ")
        raw_meanvar = [float(p) for p in raw_meanvar[:-1]]
        raw_bpu_hbm_meanvar.append(raw_meanvar)
    num_meanvar_feats = cfg.anchor_num * cfg.traj_len * 5
    raw_bpu_hbm_meanvar = np.array(raw_bpu_hbm_meanvar).reshape(
        -1, num_meanvar_feats, 1, 1
    )

    print("Bpu results:")
    print(raw_bpu_hbm_prob[0, 0, 0, 0], raw_bpu_hbm_meanvar[0, 0, 0, 0])
    ret["bpu_hbdk-verf_probs_float"] = raw_bpu_hbm_prob
    ret["bpu_hbdk-verf_mean_var_float"] = raw_bpu_hbm_meanvar


# 6. Trans output to vcs (use_pt).
bev_origin_x = cfg.bev_origin_x
bev_origin_y = cfg.bev_origin_y
img_resolution = cfg.img_resolution
anchors = output["traj_head_anchors"]

if workflow["pt"]:
    meanvar = anchor_mean_var_2
elif workflow["simulator"]:
    meanvar = torch.Tensor(
        raw_hbm_meanvar.reshape(-1, cfg.anchor_num, cfg.traj_len, 5)
    )
elif workflow["bpu"]:
    meanvar = torch.Tensor(
        raw_bpu_hbm_meanvar.reshape(-1, cfg.anchor_num, cfg.traj_len, 5)
    )
else:
    meanvar = None

if meanvar is not None:
    means, scale_trils = qat_model.traj_head._extract_gaussian(
        meanvar.to(anchors.device), anchors
    )

    # TODO (shengzhe.dai): the coordinate trans function below should be
    # integrated to HAT.
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
        vcs_x, vcs_y = Affine2D.coord_scale(
            bev_x, bev_y, inv_resolu, inv_resolu
        )
        vcs_yaw = bev_yaw - yaw_diff
        return np.stack([vcs_x, vcs_y, vcs_yaw], axis=-1)

    img_coords = np.array(model_in["valid_img_coords"], dtype=np.float64)
    vcs_coords = trans_bev_coords_to_vcs_coords(
        img_coords[0], bev_origin_x, bev_origin_y, img_resolution
    )

    all_vcs = []
    for vcs_c, mean in zip(vcs_coords, means.to("cpu").detach().numpy()):
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
        all_vcs.append(tmp_vcs)
    all_vcs = np.stack(all_vcs)
    ret["all_vcs"] = all_vcs

with open(os.path.join(debug_path, "out.pkl"), "wb") as f:
    pickle.dump(ret, f)

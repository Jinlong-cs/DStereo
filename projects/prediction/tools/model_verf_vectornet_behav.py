import os
import pickle
import sys
from copy import deepcopy

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March
from torch.nn.functional import softmax

from hat.data.collates.traj_pred_collates import collate_vectornet
from hat.registry import build_from_registry
from hat.utils import Config
from hat.utils.checkpoint import (
    load_checkpoint,
    load_state_dict,
    update_state_dict_by_strip_prefix,
)

torch.cuda.set_device(3)
# from req_pkg_path import HAT_PATH
HAT_PATH = "/home/users/xu.zhu/HAT"
sys.path.append(HAT_PATH)
# if '/home/users/zepei.sun/HAT' in sys.path:
#     sys.path.remove('/home/users/zepei.sun/HAT')
# print(sys.path)

print("model verf!")
# 0. Belows are parameters that the user should check.
# pkl_dataset_path = "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/zepei.sun/SD/real_car_test_data/H3165_20230216_D/20230216-155310_790/20_0.8.4_0_0.0.1_12_70_3_70_navi/samples_dataset.pkl"  # noqa: E501
pkl_dataset_path = "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/TRAJ_DATA_2/UT1R3_20230207_D/20230207-071351_171/20_3.3.0_0_0.0.1_12_70_3_70_perrec_low/pickles/pickle_test_train.pkl"  # noqa: E501
pre_file_name = "/jfs-public/users/sien.chen/results/behav_release_v11.1.0/vectornet_behav_release_v11.1.0_20230915_bpu2.5/"  # noqa: E501
qat_ckpt_path = pre_file_name + "qat-checkpoint-best.pth.tar"
pt_model_path = (
    pre_file_name
    + "Vectornet_behav_test_v7.0.0_bpu2.5_16_obstacles_fa78e5e6e282be23e415d95e402e10e0.pt"  # noqa: E501
)
hbm_path = (
    pre_file_name
    + "Vectornet_behav_test_v7.0.0_bpu2.5_16_obstacles_50ae665e573de8983b7a1b96938d0332.hbm"  # noqa: E501
)
debug_path = pre_file_name + "debug_cpp/"
simulator_path = "model_verfier_simulator"
model_name = (
    "Vectornet_behav_fullscene"  # vectornet or BasicVectornet or VectorNetV2
)
num_obs_in_hbm = 16
bpu = 2.5
workflow = {
    "pt": True,
    "simulator": False,
    "bpu": True,
}
used_sample_idx = 20
timestamp = None
# timestamp = 1676534027101

# 1. Prepare
if bpu == 2.2:
    march = "bayes1"
    bpu_ip = "10.103.72.160"
    verf_files = [
        "hbdk_output0_feature__behav_head_lat_cls_hz_conv2d_elem_float.txt",
        "hbdk_output1_feature__behav_head_lon_cls_hz_conv2d_elem_float.txt",
    ]
elif bpu == 2.5:
    march = March.BAYES
    bpu_ip = "10.103.53.218"  # "10.103.72.180"
    verf_files = [
        "hbdk_output0_feature__behav_head_lat_cls_hz_conv2d_elem_float.txt",
        "hbdk_output1_feature__behav_head_lon_cls_hz_conv2d_elem_float.txt",
    ]
else:
    raise ValueError(f"Unsupported bpu version {bpu}.")

os.makedirs(debug_path, exist_ok=True)
with open(pkl_dataset_path, "rb") as f:
    dataset = pickle.load(f)
cfg_file = os.path.join(
    HAT_PATH, "projects/prediction/configs/vectornet_behav.py"
)
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
state_vector_dim = cfg.num_state_vectors + cfg.num_behav_state_vectors
num_all_poly = cfg.num_all_poly

# 2. Get model input.
if used_sample_idx is None:
    used_sample_idx = 0

if timestamp:
    for i in range(len(dataset)):
        a = dataset[i]["seq_df"]["timestamp"].unique().tolist()
        a = np.sort(a)
        if a[3] == timestamp:
            used_sample_idx = i
            print("find the correct timestamp:", i, a[3])
            break

sample = dataset[used_sample_idx]
output = {}
output[0] = ["original", sample]
offline_transforms = build_from_registry(cfg.offline_transforms)
for idx, tr in enumerate(offline_transforms):
    if type(tr).__name__ == "VectorNetStructuredMapServer":
        tr.shuffle = False
        print("set shuffle_road_feat as fales")
    if type(tr).__name__ == "FilterObstacles":
        tr.is_training = False
        print("set is_training as fales")
    if type(tr).__name__ == "GenStatesAndMask":
        tr.is_training = False
        print("set is_training as fales")
    try:
        sample = tr(sample)
        output[idx + 1] = [str(type(tr)), deepcopy(sample)]
    except Exception:
        print(type(tr))
        output[idx + 1] = [str(type(tr)), None]

last_idx = len(output.keys())
collate_fn = collate_vectornet
model_in = collate_fn([output[last_idx - 1][1]])
road_mask = model_in["struct_road_masks"].cuda()
traj_mask = model_in["struct_traj_masks"].cuda()
concat_mask = torch.cat([traj_mask, road_mask], dim=1)
attention_mask = torch.matmul(
    concat_mask[:, :, None], concat_mask[:, None, :]
)  # noqa: E501
attention_mask = attention_mask[:, None, :, :]
model_in["attention_mask"] = attention_mask.cuda()
int_model_in = deepcopy(model_in)

print("有效障碍物数量：", model_in["valid_track_ids"])
print("last context frame:", sample["lcf_timestamp"])


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

# error transform index: 0:sample, 有行为：20:安全区， 无行为：18：安全区
output_pure = deepcopy(output)
output_pure[0][1] = None
# output_pure.pop(18)
ret = {"transfrom_results": output_pure}

# 3. Build qat model and infer.
qat_model = build_from_registry(cfg.model)
qat_model.fuse_model()
qat_model.set_qconfig()
horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))
horizon.quantization.prepare_qat(qat_model, inplace=True)

ckpt_dict = load_checkpoint(
    qat_ckpt_path,
    map_location="cpu",
    state_dict_update_func=update_state_dict_by_strip_prefix,
)
qat_model = load_state_dict(
    qat_model,
    ckpt_dict["state_dict"],
    allow_miss=False,
    ignore_extra=False,
    verbose=0,
)
qat_model.eval()
qat_model.cuda()

output = qat_model(model_in)

lat_behav_probs_1 = output["behav_head_lat_behav_probs"].squeeze()
lon_behav_probs_1 = output["behav_head_lon_behav_probs"]
print("python qat result: ")
print("qat output: ", lat_behav_probs_1)
print("qat probs: ", softmax(lat_behav_probs_1, dim=1))


ret["qat_lat_behav_probs_1"] = lat_behav_probs_1.to("cpu").detach().numpy()
ret["qat_lon_behav_probs_1"] = lon_behav_probs_1.to("cpu").detach().numpy()


# Build quantized model and infer.
quant_model = build_from_registry(cfg.deploy_model)
quant_model.fuse_model()
quant_model.set_qconfig()
horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))
horizon.quantization.prepare_qat(quant_model, inplace=True)
quant_model = load_state_dict(
    quant_model,
    ckpt_dict["state_dict"],
    allow_miss=True,
    ignore_extra=True,
    verbose=0,
)
quant_model.eval()
quant_model = horizon.quantization.convert(quant_model, inplace=True)
int_model_in["struct_road_feats"] = model_in["struct_road_feats"].cpu()
int_model_in["struct_traj_feats"] = model_in["struct_traj_feats"].cpu()
int_model_in["attention_mask"] = model_in["attention_mask"].cpu()
int_model_in["behav_state_vectors"] = model_in["behav_state_vectors"].cpu()
int_output = quant_model(int_model_in)
print("int res:")
print("int_output:", int_output.behav_head_lat_behav_probs.squeeze())
print("int_probs: ", softmax(int_output.behav_head_lat_behav_probs.squeeze()))

# 4. Build pt model and infer.
real_road_feats = model_in["struct_road_feats"].cpu()
real_traj_feats = model_in["struct_traj_feats"].cpu()
real_attention_mask = model_in["attention_mask"].cpu()
real_state_vectors = model_in["behav_state_vectors"].cpu()

# *************HardCode**********************
# real_state_vectors[1,0,0,0]=1.34135
# real_state_vectors[1,1,0,0]=-5
# real_state_vectors[1,2,0,0]=0.172863
# *************HardCode**********************


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
    lat_behav_probs_2, lon_behav_probs_2 = pt_model(pt_inputs)
    print("python pt result: ")
    print(
        lat_behav_probs_2.shape,
        lon_behav_probs_2.shape,
    )
    print("pt probs: ", softmax(lat_behav_probs_2.squeeze(), dim=1))
    print("pt output: ", lat_behav_probs_2.squeeze())

    ret["pt_lat_behav_probs_2"] = lat_behav_probs_2.to("cpu").detach().numpy()
    ret["pt_lon_behav_probs_2"] = lon_behav_probs_2.to("cpu").detach().numpy()

    np.savetxt(
        os.path.join(debug_path + "pt_lat_behav_prob.txt"),
        ret["pt_lat_behav_probs_2"][:, :, 0, 0],
        fmt="%.6f",
        delimiter="\t",
    )  # noqa: E501
    np.savetxt(
        os.path.join(debug_path + "pt_lon_behav_prob.txt"),
        ret["pt_lon_behav_probs_2"][:, :, 0, 0],
        fmt="%.6f",
        delimiter="\t",
    )  # noqa: E501


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
# hbdk_cmd += f"--model-input {attention_mask_file},"
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

    raw_lat_behav_path = os.path.join(
        debug_path, f"{simulator_path}/simulator_output", verf_files[0]
    )
    raw_lon_behav_path = os.path.join(
        debug_path, f"{simulator_path}/simulator_output", verf_files[1]
    )

    with open(raw_lat_behav_path, "r") as f:
        raw_lat_behav_infos = f.readlines()
    with open(raw_lon_behav_path, "r") as f:
        raw_lon_behav_infos = f.readlines()

    raw_lat_behav_infos = raw_lat_behav_infos[3:]
    raw_hbm_lat_behav = []
    for raw_info in raw_lat_behav_infos:
        raw_lat_behav = raw_info.split("\n")[0].split(" ")
        raw_lat_behav = [float(p) for p in raw_lat_behav[:-1]]
        raw_hbm_lat_behav.append(raw_lat_behav)
    raw_hbm_lat_behav = np.array(raw_hbm_lat_behav).reshape(-1, 3, 1, 1)

    raw_lon_behav_infos = raw_lon_behav_infos[3:]
    raw_hbm_lon_behav = []
    for raw_info in raw_lon_behav_infos:
        raw_lon_behav = raw_info.split("\n")[0].split(" ")
        raw_lon_behav = [float(p) for p in raw_lon_behav[:-1]]
        raw_hbm_lon_behav.append(raw_lon_behav)
    raw_hbm_lon_behav = np.array(raw_hbm_lon_behav).reshape(-1, 3, 1, 1)

    print("Simulator results:")
    print(
        raw_hbm_lat_behav.shape,
        raw_hbm_lon_behav.shape,
    )
    print(
        raw_hbm_lat_behav[0, 0, 0, 0],
        raw_hbm_lon_behav[0, 0, 0, 0],
    )
    ret["hbdk-verf_lat_behav_float"] = raw_hbm_lat_behav
    ret["hbdk-verf_lon_behav_float"] = raw_hbm_lon_behav

# 5.4 verifiy on bpu.
tmp_subdirs = os.listdir(debug_path)
if workflow["bpu"]:
    tmp_cmd = hbdk_cmd + f" --ip {bpu_ip}"
    os.system(tmp_cmd)
    new_subdirs = os.listdir(debug_path)
    bpu_path = list(set(new_subdirs) - set(tmp_subdirs))
    assert len(bpu_path) == 1
    bpu_path = bpu_path[0]
    raw_bpu_lat_behav_path = os.path.join(
        debug_path, f"{bpu_path}/bpu_output/", verf_files[0]
    )
    raw_bpu_lon_behav_path = os.path.join(
        debug_path, f"{bpu_path}/bpu_output/", verf_files[1]
    )

    with open(raw_bpu_lat_behav_path, "r") as f:
        raw_bpu_lat_behav_infos = f.readlines()
    with open(raw_bpu_lon_behav_path, "r") as f:
        raw_bpu_lon_behav_infos = f.readlines()

    raw_bpu_lat_behav_infos = raw_bpu_lat_behav_infos[3:]
    raw_bpu_hbm_lat_behav = []
    for raw_info in raw_bpu_lat_behav_infos:
        raw_lat_behav = raw_info.split("\n")[0].split(" ")
        raw_lat_behav = [float(p) for p in raw_lat_behav[:-1]]
        raw_bpu_hbm_lat_behav.append(raw_lat_behav)
    raw_bpu_hbm_lat_behav = np.array(raw_bpu_hbm_lat_behav).reshape(
        -1, 3, 1, 1
    )

    raw_bpu_lon_behav_infos = raw_bpu_lon_behav_infos[3:]
    raw_bpu_hbm_lon_behav = []
    for raw_info in raw_bpu_lon_behav_infos:
        raw_lon_behav = raw_info.split("\n")[0].split(" ")
        raw_lon_behav = [float(p) for p in raw_lon_behav[:-1]]
        raw_bpu_hbm_lon_behav.append(raw_lon_behav)
    raw_bpu_hbm_lon_behav = np.array(raw_bpu_hbm_lon_behav).reshape(
        -1, 3, 1, 1
    )

    print("Bpu results:")
    print(
        raw_bpu_hbm_lat_behav[0, 0, 0, 0],
        raw_bpu_hbm_lon_behav[0, 0, 0, 0],
    )
    ret["bpu_hbdk-verf_lat_behav_float"] = raw_bpu_hbm_lat_behav
    ret["bpu_hbdk-verf_lon_behav_float"] = raw_bpu_hbm_lon_behav

with open(os.path.join(debug_path, "out.pkl"), "wb") as f:
    pickle.dump(ret, f)

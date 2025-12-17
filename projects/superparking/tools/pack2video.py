"""Draw pack infer result and save to video.

Originally written by: @weibin.han
"""

import argparse
import os
from typing import Optional

import cv2
import torch
import torch.nn.functional as F
from hatbc.filestream.bucket.client import BucketClient
from tat.matrix.msg.reader import MSGReader, TopicChannel
from tat.matrix.pack_sdk import PackFileReader
from tqdm import tqdm

from hat.registry import build_from_registry
from projects.superparking.tools.parsing.parsing_vis import (
    attach_color_to_seg,
    super_parking_ipm_color_list,
)


def psd_visualization(preds, data, postprocess):

    (
        global_features,
        global_offsets,
        global_occupancys,
        global_slot_types,
        global_directions,
        local_features,
        local_offsets,
        local_sline_angles,
        local_point_types,
    ) = preds

    global_features = global_features.cpu()
    global_offsets = global_offsets.cpu()
    global_occupancys = global_occupancys.cpu()
    global_slot_types = global_slot_types.cpu()
    global_directions = global_directions.cpu()
    local_features = local_features.cpu()
    local_offsets = local_offsets.cpu()
    local_sline_angles = local_sline_angles.cpu()
    local_point_types = local_sline_angles.cpu()
    meta_data = {}
    meta_data["ori_img"] = data["ori_img"]
    for batch_idx, features in enumerate(
        zip(
            global_features,
            global_offsets,
            global_occupancys,
            global_slot_types,
            global_directions,
            local_features,
            local_offsets,
            local_sline_angles,
            local_point_types,
        )
    ):
        img_name = data["img_name"][batch_idx]
        meta_data["pred_slots"] = []
        meta_data = postprocess.process(features, meta_data, batch_idx)
        img_name = os.path.basename(img_name)
    return meta_data["visualized_results"]


def parsing_visualization(preds, data):
    preds = preds[0]
    ori_img = data["ori_img"][0]
    ori_h, ori_w = ori_img.shape[0], ori_img.shape[1]
    mask = F.interpolate(
        preds.type(torch.float32), size=(ori_h, ori_w), mode="nearest"
    )
    mask = mask.squeeze(dim=1)

    if isinstance(ori_img, torch.Tensor):
        ori_img = ori_img.cpu().numpy()
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)

    pred_result = mask.squeeze(dim=0)
    # output for seg
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()

    img_h, img_w, _ = ori_img.shape
    pred_h, pred_w = pred_result.shape
    if pred_h != img_h or pred_w != img_w:
        pred_result = cv2.resize(
            pred_result, dsize=(img_w, img_h), interpolation=cv2.INTER_NEAREST
        )

    roi_im = ori_img.copy()
    seg_im = attach_color_to_seg(pred_result, super_parking_ipm_color_list)
    mask = pred_result >= 0
    scale = 0.4
    roi_im[mask, :] = scale * roi_im[mask, :] + (1 - scale) * seg_im[mask, :]
    return roi_im


class Inference:
    def __init__(
        self,
        pt: str,
        gpu_ids: Optional[str] = None,
        global_threshold=0.18,
        local_threshold=0.25,
    ) -> None:
        self.pt = pt
        # parse gpu_ids
        if gpu_ids and torch.cuda.is_available():
            self.gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
        else:
            self.gpu_ids = []
        self.model = self.create_pt_model()
        postprocess_cfg = dict(
            type="PSDPostprocess",
            input_size=[896, 896],
            downsample_factor_globalslot_feature=32,
            downsample_factor_localjunction_feature=4,
            threshold_globalslot_feature=global_threshold,
            threshold_localjunction_feature=local_threshold,
            topk_globalslot_feature=50,
            topk_localjunction_feature=30,
            threshold_occupancy_feature=0.5,
            threshold_junction_type_feature=0.5,
            threshold_fuse_distance=30,
            global_kernel_size=3,
            local_kernel_size=5,
            draw_flag=True,
        )
        self.postprocess = build_from_registry(postprocess_cfg)

    def create_pt_model(self):
        assert self.pt.split(".")[-1] == "pt"
        print(self.pt)
        model = torch.jit.load(self.pt)
        model.eval()
        if self.gpu_ids != "":
            model = model.cuda(self.gpu_ids[0])
        return model

    def img2batch(self, image, img_name, to_rgb=True):
        color_space = "bgr"
        if to_rgb:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"
        data = dict()
        data["img_name"] = img_name
        data["img"] = image
        data["ori_img"] = image.copy()
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        data["img_id"] = torch.Tensor([[0]])
        data["img_height"] = torch.Tensor([[image.shape[0]]])
        data["img_width"] = torch.Tensor([[image.shape[1]]])

        # transformer
        transforms_cfg = [
            dict(type="ToTensor", to_yuv=True),
            dict(type="ImageNormalize", mean=128.0, std=128.0),
        ]
        transforms = build_from_registry(transforms_cfg)
        for transform in transforms:
            data = transform(data)
        data["img"] = torch.unsqueeze(data["img"], 0).cuda(self.gpu_ids[0])
        data["img_name"] = [data["img_name"]]
        data["ori_img"] = [data["ori_img"]]
        data["layout"] = [data["layout"]]
        data["img_shape"] = torch.unsqueeze(
            torch.tensor(data["img_shape"]), 1
        ).cuda(self.gpu_ids[0])

        return data

    def deal(self, img, imgname):
        data = self.img2batch(image=img, img_name=imgname)
        data_input = {}
        data_input["img"] = data["img"]
        with torch.no_grad():
            preds = self.model(data_input)
        parsing_pred = preds[0][0]
        psd_pred = preds[1][0]
        vis_imgs_parsing = parsing_visualization(parsing_pred, data)
        vis_imgs_psd = psd_visualization(psd_pred, data, self.postprocess)
        return vis_imgs_parsing, vis_imgs_psd[0]


class PackProcesser(object):
    def __init__(
        self,
        inference=None,
        video_save_dir=None,
        fps=10,
        frame_size=(896, 896),
    ) -> None:
        super().__init__()
        self.bkt_clt = BucketClient()
        self.inference = inference
        self.video_save_dir = video_save_dir
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        self.fps = fps
        self.frame_size = frame_size

    def unpack(self, pack_path):

        pack_local_path = (
            self.bkt_clt.url_to_local(pack_path)
            if pack_path.startswith("dmpv2")
            else pack_path
        )
        reader = PackFileReader(pack_local_path)
        topic_channel = [TopicChannel("image", 0)]
        pack_name = os.path.basename(os.path.splitext(pack_local_path)[0])

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")

        parsing_video_file = os.path.join(
            self.video_save_dir, pack_name + "_parsing.avi"
        )
        parsing_writer = cv2.VideoWriter(
            parsing_video_file, fourcc, self.fps, self.frame_size
        )
        psd_video_file = os.path.join(
            self.video_save_dir, pack_name + "_psd.avi"
        )
        psd_writer = cv2.VideoWriter(
            psd_video_file, fourcc, self.fps, self.frame_size
        )

        msg_center = MSGReader(
            handle=reader, topic_channel=topic_channel, decode_data=True
        )
        for img_msg in tqdm(msg_center, total=len(msg_center)):
            timestamp = int(
                str(img_msg["image"][0].proto[0]).split("\n")[3].split(":")[-1]
            )
            img = img_msg["image"][0].data[0]
            img = cv2.resize(img, self.frame_size)
            parsing_pred, psd_pred = inference.deal(
                img=img,
                imgname="%s__%08d.jpg" % (pack_name, timestamp),
            )
            parsing_writer.write(parsing_pred)
            psd_writer.write(psd_pred)
        parsing_writer.release()
        psd_writer.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_save_dir", type=str)
    parser.add_argument("--pack_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpu_ids", type=str)
    args = parser.parse_args()
    inference = Inference(args.model_path, args.gpu_ids)
    processer = PackProcesser(
        inference=inference, video_save_dir=args.video_save_dir
    )
    processer.unpack(args.pack_path)

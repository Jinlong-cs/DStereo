# -*- coding: utf-8 -*-

import onnxruntime as rt
import numpy as np
import cv2
import copy
import math
import numpy
import tqdm
import os
from PIL import Image, ImageEnhance
import argparse
from horizon_tc_ui import HB_ONNXRuntime
import torch
import torch.nn.functional as F
import shutil


def disp2rgb(disp, disp_max, disp_min):
    mask = np.logical_or(disp > disp_max, disp < disp_min)
    disp = np.clip(disp, disp_min, disp_max)
    mat_min = disp.min()
    mat_max = disp.max()
    norm_matrix = (disp - mat_min) / (mat_max - mat_min)
    disp = 0.1 + norm_matrix * (0.9 - 0.1)
    disp *= 255
    disp = np.round(disp).astype(np.uint8)[..., None]
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    disp[mask] = (0, 0, 0)

    return disp

def postprocess(args, data):
    disp, spx, _, _ = data
    # DStereo branch
    b, _, h, w = disp.shape
    disp = F.interpolate(torch.from_numpy(disp), (h*4, w*4), mode='nearest').reshape(b, 9, h*4, w*4)
    disp = disp.numpy()
    infer_disp = np.sum(disp*spx, axis=1, keepdims=False)[0]

    return np.ascontiguousarray(infer_disp)

def predict_quant(args, sess, file, thr=0.09, maxdisp=192, color=(252,247,192), left_img_path="left.png", right_img_path="right.png", result_path="float_onnx_infer_result.png", preprocess_fn = None):
    timestamp = os.path.splitext(file)[0]
    left_input, origin_left = preprocess_fn(os.path.join(left_img_path, file))
    right_input, origin_right = preprocess_fn(os.path.join(right_img_path, file.replace("left", "right")))

    if args.save_npy:
       save_left_input = copy.deepcopy(left_input)
       save_left_input = save_left_input + 1.0
       save_left_input = save_left_input * 128.0
       if "left" in file:
          file_name = file.replace("_left", "")
       else:
          file_name = file
       file_name = os.path.basename(file_name).split(".")[0]
       save_left_input.tofile(os.path.join(args.save_npy_path, "infra1", file_name + ".npy"))

       save_right_input = copy.deepcopy(right_input)
       save_right_input = save_right_input + 1.0
       save_right_input = save_right_input * 128.0
       save_right_input.tofile(os.path.join(args.save_npy_path, "infra2", file_name + ".npy"))

    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    feed_dict = {
        input_names[0]: copy.deepcopy(left_input),
        input_names[1]: copy.deepcopy(right_input)
    }
    onnx_result = sess.run(output_names, feed_dict)
    # print(onnx_result)
    infer_disp = postprocess(args, onnx_result)
    
    # cv2.imwrite(os.path.join(result_path, timestamp + ".tiff"), infer_disp)
    view_disp = disp2rgb(infer_disp, maxdisp, 1)

    left_right = np.hstack((origin_left, origin_right))
    orgin_mask_disp = np.hstack((view_disp, view_disp))
    cv2.imwrite(os.path.join(result_path, timestamp + ".png"), np.vstack((left_right, orgin_mask_disp)))
        
    return True

def predict(args, file, thr=0.09, maxdisp=192, color=(252,247,192), left_img_path="left.png", right_img_path="right.png", preprocess_fn = None):
    timestamp = os.path.splitext(file)[0]
    left_input, origin_left = preprocess_fn(os.path.join(left_img_path, file))
    right_input, origin_right = preprocess_fn(os.path.join(right_img_path, file.replace("left", "right")))
    # dfeature_left_input  = preprocess_df(os.path.join(left_img_path, file))
    # dfeature_right_input = preprocess_df(os.path.join(right_img_path, file.replace("left", "right")))
    if args.save_npy:
       save_left_input = copy.deepcopy(left_input)
       save_left_input = save_left_input + 1.0
       save_left_input = save_left_input * 128.0
       if "left" in file:
          file_name = file.replace("_left", "")
       else:
          file_name = file
       file_name = os.path.basename(file_name).split(".")[0]
       save_left_input.tofile(os.path.join(args.save_npy_path, "infra1", file_name + ".npy"))

       save_right_input = copy.deepcopy(right_input)
       save_right_input = save_right_input + 1.0
       save_right_input = save_right_input * 128.0
       save_right_input.tofile(os.path.join(args.save_npy_path, "infra2", file_name + ".npy"))

    return True

def _bgr2nv12(image):
    image = copy.deepcopy(image)
    image = image.astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    frame_size = width * height
    nv12 = np.zeros(frame_size + frame_size // 2, dtype=np.int32)

    # Reshape BGR24 to height x width x 3
    bgr24_reshaped = image.reshape((height, width, 3)).astype(np.int32)
    b, g, r = bgr24_reshaped[..., 0], bgr24_reshaped[..., 1], bgr24_reshaped[..., 2]

    # Compute Y plane
    y_plane = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)
    nv12[:frame_size] = y_plane.flatten()

    uv_b = b[::2, ::2]
    uv_g = g[::2, ::2]
    uv_r = r[::2, ::2]
    # Compute UV plane for even rows
    uv_plane = np.zeros((height * width // 2), dtype=np.int32)

    u = ((-38 * uv_r - 74 * uv_g + 112 * uv_b + 128) >> 8) + 128
    v = ((112 * uv_r - 94 * uv_g - 18 * uv_b + 128) >> 8) + 128

    u = np.clip(u, 0, 255).astype(np.uint8).flatten()
    v = np.clip(v, 0, 255).astype(np.uint8).flatten()

    uv_plane[0::2] = u
    uv_plane[1::2] = v

    nv12[frame_size:] = uv_plane.flatten()

    return np.ascontiguousarray(nv12.astype(np.uint8))

def _nv12Toyuv444(data, height, width):
    data = copy.deepcopy(data)
    nv12_data = data.flatten()
    yuv444 = np.empty([height, width, 3], dtype=np.uint8)
    yuv444[:, :, 0] = nv12_data[:width * height].reshape(
        height, width)
    u = nv12_data[width * height::2].reshape(
        height // 2, width // 2)
    yuv444[:, :, 1] = Image.fromarray(u).resize((width, height),
                                                resample=0)
    v = nv12_data[width * height + 1::2].reshape(
        height // 2, width // 2)
    yuv444[:, :, 2] = Image.fromarray(v).resize((width, height),
                                                resample=0)
    data = yuv444.astype(np.uint8)
    # if yuv444_output_layout == "CHW":
    #     data = np.transpose(data, (2, 0, 1))
    return np.ascontiguousarray(data)

def preprocess(p, crop_width=640, crop_height=352):
    assert os.path.isfile(p), p
    # load bgr
    img = cv2.imread(p)[:,:,:3]
    h, w, c = img.shape

    # resize 
    ratio = w / h
    nh = int(crop_width / ratio)
    img_resized = cv2.resize(img, (crop_width, nh))
    assert img_resized.shape[1] == crop_width
    # padding
    if nh < crop_height:
        top_pad = (crop_height - nh) // 2
        bottom_pad = crop_height - nh - top_pad
        if top_pad > 0 or bottom_pad > 0:
            img_resized = cv2.copyMakeBorder(img_resized, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    # crop 
    elif nh > crop_height:
        top = int((nh - crop_height + 1) * 0.5)
        bottom = top + crop_height
        img_resized = img_resized[top:bottom, ...]
    assert img_resized.shape == (crop_height, crop_width, 3)
    # bgr2nv12
    img_nv12 = _bgr2nv12(img_resized)
    # nv12Toyuv444
    img_yuv444 = _nv12Toyuv444(img_nv12, crop_height, crop_width)

    # norm
    input = np.array(img_yuv444).astype(np.float32)
    input = input / 128.0 - 1.0

    # hwc --> n, c, h, w
    input = np.transpose(input, (2, 0, 1))
    input = np.ascontiguousarray(input[None, ...])

    return input, np.ascontiguousarray(img_resized)


def create_sess(onnx_path):
    sess = HB_ONNXRuntime(model_file=onnx_path)
    return sess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='onnx inference')
    parser.add_argument('--left_img', type=str, default='/docker-mount/zengpeng.sun/DStereo_calib_data_origin/test_dstereo', help='left image path')
    parser.add_argument('--right_img', type=str, default='/docker-mount/zengpeng.sun/DStereo_calib_data_origin/test_dstereo', help='right image path')
    parser.add_argument('--save_npy', action='store_true', default=True, help='save npy')
    parser.add_argument('--save_npy_path', type=str, default='/docker-mount/zengpeng.sun/DStereo_calib_data_origin/npy', help='save npy path')
    parser.add_argument('--ptq_num', type=int, default=0, help='Image Number for PTQ (default: 0)')

    args = parser.parse_args()

    images = [p for p in os.listdir(args.left_img) if "left" in p]
    
    # 处理 ptq_num 指定数量的图片
    images = images[:args.ptq_num] if args.ptq_num > 0 else images
    
    for file in tqdm.tqdm(images):
        bboxes = predict(args, file, thr=0.09, 
                         left_img_path=args.left_img, right_img_path=args.right_img, 
                         preprocess_fn=preprocess)

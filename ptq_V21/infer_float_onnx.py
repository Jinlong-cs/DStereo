      
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

from infer_float import predict_quant, preprocess, create_sess

def preprocess_quant(p, crop_width=640, crop_height=352):
    input, img_resized = preprocess(p, crop_width, crop_height)

    return input, img_resized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='onnx inference')
    parser.add_argument('--onnx_path', type=str, default='ptq_V21/DStereoV23_quantized_model.onnx', help='onnx path')
    parser.add_argument('--left_img', type=str, default='/docker-mount/zengpeng.sun/DStereo_calib_data_origin/test_dstereo', help='left image path')
    parser.add_argument('--right_img', type=str, default='/docker-mount/zengpeng.sun/DStereo_calib_data_origin/test_dstereo', help='right image path')
    parser.add_argument('--result_path', type=str, default='/docker-mount/zengpeng.sun/DStereo_calib_data_origin/result', help='result image path')
    parser.add_argument('--save_npy', action='store_true', default=False, help='save npy')
    parser.add_argument('--save_npy_path', type=str, default='/docker-mount/zengpeng.sun/DStereo_calib_data_origin/npy2', help='save npy path')

    args = parser.parse_args()
    
    if os.path.exists(args.result_path):
        shutil.rmtree(args.result_path)
    os.makedirs(args.result_path, exist_ok=True)
    sess = create_sess(args.onnx_path)

    images = [p for p in os.listdir(args.left_img) if "left" in p]

    for file in tqdm.tqdm(images):
        bboxes = predict_quant(args, sess, file, thr=0.09, 
                         left_img_path=args.left_img, right_img_path=args.right_img, 
                         result_path=args.result_path, 
                         preprocess_fn=preprocess_quant)

    
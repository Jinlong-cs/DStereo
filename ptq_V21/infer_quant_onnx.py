# -*- coding: utf-8 -*-

import numpy as np
import argparse
import tqdm
import os

from infer_origin_onnx import predict, preprocess, create_sess

def preprocess_quant(p, crop_width=640, crop_height=352):
    input, input_resized = preprocess(p, crop_width, crop_height)
    input = (input * 128) + 128

    # for quantization onnx
    input -= 128
    input = input.astype(np.int8)
    input = input.transpose(0, 2, 3, 1)

    return input, input_resized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='onnx inference')
    parser.add_argument('--onnx_path', type=str, default='float_modify.onnx', help='onnx path')
    parser.add_argument('--left_img', type=str, default='left.png', help='left image path')
    parser.add_argument('--right_img', type=str, default='right.png', help='right image path')
    parser.add_argument('--result_path', type=str, default='float_onnx_infer_result.png', help='result image path')
    args = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)
    
    sess = create_sess(args.onnx_path)
    for file in tqdm.tqdm(os.listdir(args.left_img)):
        bboxes = predict(sess, file, thr=0.09, left_img_path=args.left_img, right_img_path=args.right_img, result_path=args.result_path, preprocess_fn=preprocess_quant)
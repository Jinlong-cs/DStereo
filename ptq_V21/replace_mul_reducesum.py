import numpy as np
import os
from horizon_nn.ir import load_model, save_model, DataType
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import *
from tqdm import tqdm
import os
import cv2
import matplotlib
import glob
import sys
import yaml
import copy
import numpy as np
import multiprocessing
import torch
import torch.nn.functional as F

def result2disp(disp_unfold, spx):
    # spx = F.softmax(torch.tensor(spx), dim=1)
    # b, _, h, w = disp_unfold.shape
    # disp_unfold = F.interpolate(torch.from_numpy(disp_unfold),(h*4,w*4),mode='nearest')#.reshape(b,9,h*4,w*4)
    disp = torch.sum(disp_unfold*spx, 1, keepdim=False)[0]
    # top_pad = 2
    # left_pad = 4
    # disp = disp[top_pad:-top_pad, left_pad:-left_pad]
    return disp.cpu().numpy()

def replace_reducesum_to_gemm(origin_path, target_path):
    model = load_model(origin_path)
    pre_node = model.graph.node_mappings["/get_initdisp/Transpose_1"]
    reduce_node = model.graph.node_mappings["/get_initdisp/ReduceSum"]
    # reshape node.
    target_shape = model.graph.create_variable(
        is_param=True,
        value=np.array([48, -1], dtype=np.int64)
    )
    reshape_node = model.graph.create_node(
        op_type="Reshape",
        name="/get_initdisp/Reshape_beforeGEMM",
        inputs=[pre_node.outputs[0], target_shape],
        num_outputs=1).insert_after(pre_node)
    # transpose node.
    transpose_node = model.graph.create_node(
        op_type="Transpose",
        name="/get_initdisp/Transpose_beforeGEMM",
        inputs=[reshape_node.outputs[0]],
        attributes = {
            "perm": np.array([1,0]).astype(np.int64),
        },
        num_outputs=1).insert_after(reshape_node)
    # gemm node.
    weight = model.graph.create_variable(
        is_param=True,
        value=np.arange(48).astype(np.float32).reshape(48, 1)
    )
    bias = model.graph.create_variable(
        is_param=True,
        value=np.zeros(1).astype(np.float32)
    )
    gemm_node = model.graph.create_node(
        op_type="Gemm",
        name="/get_initdisp/GEMM",
        inputs=[transpose_node.outputs[0], weight, bias],
        attributes = {
            "alpha": 1.0,
            "beta": 1.0,
            "transA": 0,
            "transB": 0,
        },
        num_outputs=1).insert_after(transpose_node)
    # transpose node.
    transpose_node = model.graph.create_node(
        op_type="Transpose",
        name="/get_initdisp/Transpose_afterGEMM",
        inputs=[gemm_node.outputs[0]],
        attributes = {
            "perm": np.array([1,0]).astype(np.int64),
        },
        num_outputs=1).insert_after(gemm_node)
    # reshape node.
    target2_shape = model.graph.create_variable(
        is_param=True,
        value=np.array([1, 1, 88, 160], dtype=np.int64)
    )
    reshape2_node = model.graph.create_node(
        op_type="Reshape",
        name="/get_initdisp/Reshape_afterGEMM",
        inputs=[gemm_node.outputs[0], target2_shape],
        num_outputs=1).insert_after(gemm_node)
    reduce_node.replace_all_uses_with(reshape2_node)
    model.infer_shapes()
    model.check_validity()
    save_model(model, target_path)

def checker(origin_path, target_path):
    model_list = [origin_path, target_path]
    origin_infra1 = (np.random.randn(1, 3, 352, 640) * 255).astype(np.float32)
    origin_infra2 = origin_infra1
    
    for i, onnx_model_path in enumerate(model_list):
        infra1 = copy.deepcopy(origin_infra1) / 128.0 - 1.0
        infra1 = np.ascontiguousarray(infra1).astype(np.float32)
        infra2 = copy.deepcopy(origin_infra2) / 128.0 - 1.0
        infra2 = np.ascontiguousarray(infra2).astype(np.float32)

        print(f"\n************ {onnx_model_path} ************")
        sess = HB_ONNXRuntime(model_file=onnx_model_path)
        input_names = [input.name for input in sess.get_inputs()]
        output_names = [output.name for output in sess.get_outputs()]
        feed_dict = {
            input_names[0]: infra1,
            input_names[1]: infra2,
        }
        disp_unfold, spx, init_disp, init_spx = sess.run(output_names, feed_dict)
        disp_unfold = torch.from_numpy(disp_unfold)
        spx = torch.from_numpy(spx)
        outputs = result2disp(disp_unfold, spx)

        if i == 0:
            float_out = outputs
        else:       
            diff = np.abs(outputs.flatten() - float_out.flatten())  
            max_diff = np.max(diff)   
            print("depth max_diff:", max_diff)
            mean_diff = np.mean(diff)   
            print("depth mean_diff:", mean_diff)
        
if __name__ == "__main__":
    origin_path = "input/float.onnx"
    target_path = "ptq_V21/float_modify.onnx"
    replace_reducesum_to_gemm(origin_path, target_path)
    checker(origin_path, target_path)
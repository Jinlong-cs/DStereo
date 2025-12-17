import logging
import os
import pprint

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from hbdk import hbir_base
from horizon_plugin_pytorch.quantization import check_model, perf_model

from hat.engine.predictor import Predictor
from hat.utils.apply_func import _as_list
from hat.utils.checkpoint import load_state_dict
from hat.utils.logger import init_logger
from hat.utils.statistics import cal_ops
from hat.visualize.lidar_det import lidar_det_visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def trace_int_model(deploy_model, deploy_inputs, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # save quantized pth
    stage = {"state_dict": deploy_model.state_dict()}
    ckpt_file = os.path.join(output_dir, "int_infer-checkpoint-best.pth.tar")
    torch.save(stage, ckpt_file)
    logger.info(f"Save quantized deploy model checkpoint: {ckpt_file}")

    # save pt
    deploy_model.cpu()
    script_module = torch.jit.trace(
        func=deploy_model.eval(),
        example_inputs=deploy_inputs,
    )

    #  output format.
    per_tensor_anno = horizon.get_output_annotation(script_module)
    logger.info(
        "annotation str of each output tensor:\n%s"
        % pprint.pformat(per_tensor_anno)
    )

    pt_file = os.path.join(
        output_dir,
        "deploy-checkpoint-last.pt",
    )

    # may override
    torch.jit.save(script_module, pt_file)
    logger.info(f"Save last traced deploy_model checkpoint: {pt_file}")


def model_checker(deploy_model, deploy_inputs):
    deploy_model.eval()
    flag = check_model(deploy_model, deploy_inputs, advice=10)
    if flag != 0:
        raise AssertionError("Failed to pass hbdk checker")


def calops(deploy_model, deploy_inputs):
    deploy_model.eval()
    total_ops, total_params = cal_ops(
        model=deploy_model,
        inputs=deploy_inputs,
    )
    print("Params: %.6f M" % (total_params / (1000 ** 2)))
    print("FLOPs: %.6f G" % (total_ops / (1000 ** 3)))


def compile(
    deploy_model,
    deploy_inputs,
    march,
    task_name,
    int_ckpt=None,
    opt="O3",
    input_source=["ddr"],  # noqa B006
    output_layout="NHWC",
    layer_details=True,
    output_dir="./tmp_models",
):
    logger.info("=" * 50 + "BEGIN COMPILE" + "=" * 50)

    if int_ckpt is not None:
        logger.warning("Make sure ckpt is from int_infer stage")
        load_state_dict(
            deploy_model,
            path_or_dict=int_ckpt,
        )

    compile_dir = os.path.join(output_dir, "compile")
    if not os.path.exists(compile_dir):
        os.makedirs(compile_dir)

    hbm = os.path.join(compile_dir, "model.hbm")

    # have a test
    deploy_model.eval()
    deploy_model(deploy_inputs)

    # compile, perf
    # wrap dict, tensor as list
    example_inputs = tuple(_as_list(deploy_inputs))
    hbir_base.CleanUpContext()
    perf_model(
        module=deploy_model.eval(),
        example_inputs=example_inputs,
        march=march,
        name=task_name,
        out_dir=compile_dir,
        hbm=hbm,
        layer_details=layer_details,
        input_source=input_source,
        opt=opt,
        output_layout=output_layout,
    )

    logger.info("=" * 50 + "END COMPILE" + "=" * 50)


def align_bpu_validation(
    int_model,
    data_loader,
    val_batch_processor,
    val_metrics,
    val_callbacks,
    device,
    ckpt_dir,
):
    init_logger(f"{ckpt_dir}/align_bpu_validation")
    logger.info("=" * 50 + "BEGIN ALIGN BPU VALIDATION" + "=" * 50)

    device_id = list(map(int, device.split(",")))[0]
    torch.cuda.set_device(device_id)

    int_model.eval()

    predictor = Predictor(
        model=int_model,
        data_loader=data_loader,
        batch_processor=val_batch_processor,
        device=device_id,
        callbacks=_as_list(val_callbacks),
        num_epochs=0,
        log_interval=1,
    )
    predictor.val_metrics = _as_list(val_metrics)
    predictor.fit()
    logger.info("=" * 50 + "END ALIGN BPU VALIDATION" + "=" * 50)


def int_infer_viz_lidar(
    model,
    input_points,
    device,
    is_plot=True,
):
    logger.info("=" * 50 + "BEGIN LIDAR INFER" + "=" * 50)

    device_id = list(map(int, device.split(",")))[0]
    torch.cuda.set_device(device_id)
    model.cuda()

    points = np.fromfile(input_points, dtype=np.float32).reshape((-1, 4))
    points = torch.from_numpy(points).cuda()
    data = {
        "points": [points],
    }

    model.eval()
    model_out = model(data)

    lidar_det_visualize(
        points=points,
        predictions=model_out[0],
        score_thresh=0.4,
        is_plot=is_plot,
    )
    logger.info("=" * 50 + "END LIDAR INFER" + "=" * 50)

import os
import subprocess

import pytest

from hat.utils.config import Config

type_params = [
    pytest.param("fisheye_multitask"),
    # pytest.param("fisheye_laneparsing"),
    # pytest.param("fisheye_image_faile"),
    # pytest.param("ipm_multitask"),
]


patch_cfg = f"{os.path.dirname(__file__)}/configs/test_train_cfg_patch.py"
test_level = os.getenv("HAT_SP_TEST_LEVEL")


class TestPipeline:
    @pytest.mark.parametrize(["model_type"], type_params)
    def test_training(self, model_type):
        print(
            f"\n--------------------- start check {model_type} training -------------------"  # noqa
        )

        stages = ["float", "qat"]
        int_infer_stages = ["stage_two", "stage_one"]
        train_cmd_pattern = (
            "python3 tools/train.py --config {} --stage {} --pipeline-test"
            + f" --hat-sp-model-type {model_type} "
        )
        int_infer_cmd_pattern = (
            "python3 tools/train.py --config {} --stage int_infer --hat-infer-stage {} --pipeline-test"  # noqa
            + f" --hat-sp-model-type {model_type} "
        )

        cmds = [train_cmd_pattern.format(patch_cfg, stage) for stage in stages]
        int_infer_cmds = [
            int_infer_cmd_pattern.format(patch_cfg, stage)
            for stage in int_infer_stages
        ]
        cmds = cmds + int_infer_cmds
        for cmd in cmds:
            print(cmd)
            subprocess.check_call(cmd, shell=True)

        print(
            f"\n---------------------- end check {model_type} training --------------------"  # noqa
        )

    @pytest.mark.skipif(test_level == "commit", reason="daily-build only")
    @pytest.mark.parametrize(["model_type"], type_params)
    def test_infer(self, model_type):
        os.environ["HAT_SP_MODEL_TYPE"] = model_type
        print(
            f"\n--------------------- start check {model_type} compilation --------------------"  # noqa
        )

        cfg = Config.fromfile(patch_cfg)
        ckpt_dir = os.path.join(cfg["save_prefix"], model_type)

        ckpt_file = os.path.join(ckpt_dir, "qat-checkpoint-last.pth.tar")

        cmd = (
            f"python3 tools/predict.py --config {patch_cfg} --stage int_infer "
            + f"--hat-infer-model-ckpt {ckpt_file} --hat-val-only 1 "
            + f"--hat-sp-model-type {model_type} "
            + "--hat-aidi-eval 1 --pipeline-test"
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)

        print(
            f"\n---------------------- end check {model_type} compilation ---------------------"  # noqa
        )

    @pytest.mark.parametrize(["model_type"], type_params)
    def test_compile(self, model_type):
        os.environ["HAT_SP_MODEL_TYPE"] = model_type
        print(
            f"\n--------------------- start check {model_type} compilation --------------------"  # noqa
        )

        cfg = Config.fromfile(patch_cfg)
        ckpt_dir = os.path.join(cfg["save_prefix"], model_type)
        compile_input_shape = cfg["compile_shape"]

        compile_pt_file = os.path.join(
            ckpt_dir, "pipelinetest-deploy-checkpoint-last.pt"
        )
        compile_out_dir = os.path.join(ckpt_dir, "ci_test")
        compile_opt = "O3" if test_level == "daily" else "O0"
        compile_march = cfg["march"]
        hbm_name = "fisheye"
        cmd = (
            f"python3 tools/deploy/compile_standalone.py {compile_pt_file} "
            f"--output {compile_out_dir} --name {hbm_name} --input-size {compile_input_shape} "  # noqa
            f"--opt {compile_opt} --march {compile_march} --debug"
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)

        print(
            f"\n---------------------- end check {model_type} compilation ---------------------"  # noqa
        )

    @pytest.mark.parametrize(["model_type"], type_params)
    def test_hbkd_verify(self, model_type):
        os.environ["HAT_SP_MODEL_TYPE"] = model_type
        print(
            f"\n--------------------- start check {model_type} hbkd_verify --------------------"  # noqa
        )

        cfg = Config.fromfile(patch_cfg)
        ckpt_dir = os.path.join(cfg["save_prefix"], model_type)

        compile_opt = "O3" if test_level == "daily" else "O0"
        hbm_path = os.path.join(
            ckpt_dir, "ci_test", f"model_opt_{compile_opt}.hbm"
        )
        pb_path = os.path.join(
            ckpt_dir, "ci_test", f"model_opt_{compile_opt}.pt"
        )
        input_path = cfg["hbdk_input_path"]
        compile_input_shape = cfg["compile_shape"]
        yuv_shape = "x".join(compile_input_shape.split("x")[2:])

        cmd = (
            f"hbdk-model-verifier --hbm {hbm_path} --model-pt {pb_path} "
            + f"--model-input {input_path} --yuv-shape {yuv_shape} --skip-bpu"
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)

        print(
            f"\n---------------------- end check {model_type} hbkd_verify ---------------------"  # noqa
        )

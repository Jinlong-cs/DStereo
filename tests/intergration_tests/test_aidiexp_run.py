import os
import shutil
import subprocess
import uuid

import pytest

from hat.utils.config import Config
from hat.utils.package_helper import check_packages_available
from tests.utils import execute_cmd

# Optional requirement check
aidisdk_available = check_packages_available("aidisdk", raise_exception=False)


cfg = "tests/data/resnet18_with_deploy_model.py"

test_dir = "./tmp_test"

if not os.path.exists(test_dir):
    os.makedirs(test_dir, exist_ok=True)

test_cfg = os.path.join(test_dir, "test_resnet18_with_deploy_model.py")


class TestAIDILib:
    def setup(self):

        if aidisdk_available:
            tracking = True
        else:
            tracking = False

        cmd = f"""
        cp {cfg} {test_cfg}
        sed -i 's/enable_model_tracking = True/enable_model_tracking = {tracking}/g' {test_cfg}  # noqa E501
        sed -i 's/task_name = "resnet18_cls"/task_name = "tmp_resnet18_cls"/g' {test_cfg}        # noqa E501
        """

        execute_cmd(cmd)

        self.experiment_name = "HAT_INTEGRATION_TESTS-test_aidiexp_run"
        self.run_name = f"cicd_local_run_{uuid.uuid1()}"
        self.project_id = "RDS20220011"
        self.experiment_path = "dmpv2://HDLTAlgorithm/aidi_exp/HAT_CICD/"
        self.config = Config.fromfile(test_cfg)
        self.task_name = "resnet18_cls"

        if aidisdk_available:
            self.exp_args_pattern = (
                f"--experiment-name {self.experiment_name} "
                + f" --experiment-path {self.experiment_path}"
                + f" --run-name {self.run_name} --project-id {self.project_id}"
                + " --group-name {} --enable-tracking"
            )

            self.aidiexp_run_cmd_pattern = "python3 tools/aidiexp_run.py {} --script {} --script-args '{}'"  # noqa E501
        else:
            self.exp_args_pattern = None
        self.output_dir = self.config.ckpt_dir
        self.compile_dir = os.path.join(self.output_dir, "compile")
        self.pack_dir = os.path.join(self.output_dir, "pack")

    @pytest.mark.parametrize(
        ["stage"],
        [
            pytest.param("float"),
            pytest.param("calibration"),
            pytest.param("qat"),
            pytest.param("int_infer"),
        ],
    )
    def test_training(self, stage):

        script_args = f"--config {test_cfg} --stage {stage}"
        script = "tools/train.py"

        if aidisdk_available:
            exp_args = self.exp_args_pattern.format(f"training-group-{stage}")
            cmd = self.aidiexp_run_cmd_pattern.format(
                exp_args,
                script,
                script_args + " --enable-tracking",
            )
        else:
            cmd = f"python3 {script} {script_args}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    @pytest.mark.parametrize(
        ["stage"],
        [
            pytest.param("int_infer"),
        ],
    )
    def test_predict(self, stage):

        script_args = f"--config {test_cfg} --stage {stage}"
        script = "tools/predict.py"

        if aidisdk_available:
            exp_args = self.exp_args_pattern.format(f"predict-group-{stage}")
            cmd = self.aidiexp_run_cmd_pattern.format(
                exp_args,
                script,
                script_args + " --enable-tracking",
            )
        else:
            cmd = f"python3 {script} {script_args}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    def test_compile_standalone(self):

        pt_file = os.path.join(
            self.output_dir,
            "deploy-checkpoint-last.pt",
        )
        script_args = (
            f"{pt_file} --input-size 1x3x224x224 --march bayes"
            + f" --name resnet18_cls --opt O0 --output {self.compile_dir}"
        )
        script = "tools/deploy/compile_standalone.py"

        if aidisdk_available:
            exp_args = (
                self.exp_args_pattern.format("compile-group")
                + f" --output-artifact-name {self.task_name}-compile"
            )
            cmd = self.aidiexp_run_cmd_pattern.format(
                exp_args,
                script,
                script_args,
            )
        else:
            cmd = f"python3 {script} {script_args}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    def test_pack_hbm(self):

        hbm = os.path.join(self.compile_dir, "model_opt_O0.hbm")
        script_args = f"--input-hbm-list {hbm} --output {self.pack_dir}"
        script = "tools/deploy/pack_hbm.py"

        if aidisdk_available:
            exp_args = (
                self.exp_args_pattern.format("compile_pack-group")
                + f" --output-artifact-name {self.task_name}-pack"
            )
            cmd = self.aidiexp_run_cmd_pattern.format(
                exp_args,
                script,
                script_args,
            )
        else:
            cmd = f"python3 {script} {script_args}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    def teardown_class(self):
        shutil.rmtree(test_dir)

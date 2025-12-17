import json
import os
import subprocess
import time

import pytest
import yaml

from hat.utils.config import Config
from projects.pilot.configs.project_utils.enum import (
    BEVModelSetting,
    BEVModelType,
)

# from hatbc.aidi.gallery import GalleryClient
# from numeric_consistent import compare_dump_zip, compress_dir


with open(
    os.path.join(os.path.dirname(__file__), "../model_meta.yaml"), "r"
) as rf:
    model_meta = yaml.safe_load(rf)

test_level = os.getenv("HAT_PILOT_TEST_LEVEL")
assert test_level is not None

if test_level == "commit":
    params = [
        pytest.param("resize_2_rear_bayes", None),
        pytest.param("resize_2_side_bayes", None),
        pytest.param("crop_bayes", None),
        pytest.param("image_fail_segmentation", None),
    ]
else:
    params = [
        # pytest.param(
        #     "resize_2",
        #     "pilot_multitask_resize2_test_lmdb_test_v0.0.1_before",  # noqa
        # ),
        # pytest.param(
        #     "resize_4",
        #     "pilot_multitask_resize4_test_lmdb_test_v0.0.1_before",  # noqa
        # ),
        # pytest.param(
        #     "crop", "pilot_multitask_crop_test_lmdb_test_v0.0.1_before"
        # ),
        pytest.param(
            "resize_2_rear_bayes",
            "pilot5_multitask_resize2_rear_bayes_test_lmdb_test_v0.0.1_before",
        ),
        pytest.param(
            "resize_2_side_bayes",
            "pilot5_multitask_resize2_side_bayes_test_lmdb_test_v0.0.1_before",
        ),
        pytest.param(
            "crop_bayes",
            "pilot5_multitask_crop_bayes_test_lmdb_test_v0.0.1_before",
        ),
        pytest.param(
            "image_fail_segmentation", None
        ),  # not support auto-threshold eval
    ]

infer_params = [pytest.param(p.values[0]) for p in params]


@pytest.mark.skipif(test_level == "bev_dev", reason="Skip for bev dev")
class TestPipeline:
    def setup(self):
        self.setting = "test_lmdb"
        os.environ["HAT_PILOT_MODEL_SETTING"] = self.setting
        self.artifact_dir = "artifacts"
        if not os.path.exists(self.artifact_dir):
            os.mkdir(self.artifact_dir)

    @pytest.mark.parametrize(["model_type", "diff_name"], params)
    def test_training_eval(self, model_type, diff_name):
        eval_setting = "test"
        patch_cfg = (
            f"{os.path.dirname(__file__)}/configs/test_train_cfg_patch.py"
        )
        _arguments = {
            "--config": "projects/pilot/tools/production/model_workflow.py",
            "--local-executor": "simple",
            "--queue-name": "svc-aip-cpu",
            "--project-id": "PDT20220001",
            "--git-branch": "master",
            "--model-version": "v0.0.1",
            "--pipeline-name": "pilot_multitask_cicd_test",
            "--multi-train-resource": "1x2",
            "--launcher": "torch",
        }
        _actions = [
            "--enable-tracking",
            "--pipeline-test",
            "--allow-wo-inputs",
        ]

        print(
            f"\n---------------------start check model product -------------------"  # noqa
        )
        if diff_name is not None:
            _actions.extend(["--enable-eval", "--enable-auto-threshold"])
            _arguments["--multi-eval-resource"] = "1x2"
            _arguments["--multi-eval-data-setting"] = eval_setting
            _arguments["--multi-diff-eval-name"] = diff_name
            _arguments["--multi-diff-eval-name-release"] = diff_name

        _arguments["--multi-model-type"] = model_type  # noqa
        _arguments["--multi-model-setting"] = self.setting
        _arguments["--multi-train-config"] = patch_cfg

        cmd = [
            "python3",
            "-c",
            "'from hdflow.cli.execute import main, parse_args;args = parse_args();main(args)'",  # noqa
        ]
        for k, v in _arguments.items():
            cmd += [k, v]
        cmd += _actions
        cmd = " ".join(cmd)
        print(f"Run command: {cmd}")
        subprocess.check_call(cmd, shell=True)
        print(
            f"\n----------------------end check model product --------------------"  # noqa
        )

    @pytest.mark.parametrize(["model_type"], infer_params)
    def test_infer(self, model_type):
        train_cfg = model_meta[model_type]["train"]["entry"]
        last_stage = model_meta[model_type]["train"]["stages"][-1]
        infer_cfg = model_meta[model_type]["infer"]["entry"]

        ckpt_dir = Config.fromfile(train_cfg)["ckpt_dir"].as_posix()
        model_ckpt = os.path.join(
            ckpt_dir, last_stage + "-checkpoint-last.pth.tar"
        )

        print(
            f"\n---------------------start check {model_type} infer -------------------"  # noqa
        )

        cmd = (
            f"python3 tools/predict.py --config {infer_cfg} --stage int_infer"
            + f" --hat-pilot-model-name {model_type}"
            + " --hat-pilot-model-name-postfix cicd_test"
            + f" --hat-pilot-model-checkpoint {model_ckpt}"
        )
        subprocess.check_call(cmd, shell=True)

        # TODO(@yilin.xiong): Add diff check
        print(
            f"\n----------------------end check {model_type} infer --------------------"  # noqa
        )

    # @pytest.mark.skipif(test_level == "commit", reason="daily-wise only")
    # def test_numeric_consistent(self):
    #     user = "pilot.runner"
    #     password = "!Kj70Ic7feMh%ER$"
    #     group = "auto.pilot_algo.build"
    #     project = "numerical_result"
    #     dump_path = "dump_res/train_dump"
    #     # compress newest data
    #     dump_zip = "newest_dump.zip"
    #     timestrip = time.strftime("%y%m%d%H%M", time.localtime())
    #     compress_dir(dump_path, dump_zip, without_dir=True)

    #     # # download compared data from gallery
    #     old_dump_zip = "old_dump.zip"
    #     gallery_cli = GalleryClient(
    #         user, password, group=group, project=project
    #     )
    #     gallery_cli.download_latest(save_path=old_dump_zip)

    #     # compare newest & old dump data
    #     compare_dump_zip(
    #         dump_zip, old_dump_zip, save_result_path=self.artifact_dir
    #     )
    #     # upload newest dump
    #     gallery_name = f"pilot_dump_{timestrip}"
    #     gallery_cli.upload_file(dump_zip, name=gallery_name)

    @pytest.mark.skipif(test_level == "daily", reason="commit-wise only")
    def test_compile(self):
        cmd = (
            "python3 projects/pilot/dev/publish/trace_compile_model.py"
            + " --sub-project test"
            + " --publish-version v0.0.1"
            + " --compile-mode local"
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    # @pytest.mark.skipif(test_level == "commit", reason="daily build only")
    @pytest.mark.skipif(True, reason="Not ready NOW!")
    def test_publish(self):
        ts = time.strftime("%y.%m.%d", time.localtime())
        ts = ".".join([str(int(t)) for t in ts.split(".")])
        os.environ["gitlabTargetBranch"] = "refs/tags/pilot-test-v" + ts
        cmd = """
        set -e
        sh projects/pilot/dev/publish/publish.sh
        """
        subprocess.check_call(cmd, shell=True)

    @pytest.mark.skipif(True, reason="Not stable NOW!")
    def test_fillback_eval(self):
        cmd = """
        set -e
        python3 projects/pilot/tools/fillback_eval/pipeline.py \
            --sub-project test \
            --fillback-type evs \
            --enable-report-diff \
        """
        subprocess.check_call(cmd, shell=True)


@pytest.mark.skipif(test_level != "bev_dev", reason="Only for bev develop")
class TestBEVDevPipeline:
    def setup(self):
        self.setting = "test_lmdb"
        os.environ["HAT_PILOT_MODEL_SETTING"] = self.setting
        self.artifact_dir = "artifacts"
        if not os.path.exists(self.artifact_dir):
            os.mkdir(self.artifact_dir)

    def test_bev_dev(self):
        _model_thresh = json.dumps(
            dict(
                bev_3d_vehicle=dict(
                    score_threshold=0.15,
                    roi_score_threshold=[0.15] * 7,
                ),
                bev_3d_vrumerge=dict(
                    score_threshold=[0.17, 0.19],
                    roi_score_threshold=[0.17, 0.19],
                ),
                online_mapping=dict(
                    lane=0.57,
                    roadedge=0.45,
                ),
                bev_arrow=dict(
                    score_threshold=0.2,
                    iou_threshold=0.2,
                ),
                bev_junction=dict(
                    score_threshold=0.2,
                    iou_threshold=0.2,
                ),
                bev_roadmarking=dict(
                    score_threshold=0.2,
                    iou_threshold=0.2,
                ),
                bev_static_obstacle=dict(
                    score_threshold=0.3,
                    iou_threshold=0.2,
                ),
                bev_parkingrod=dict(
                    threshold_parkingrod_feature=0.2,
                ),
                bev_sod3d=dict(
                    score_threshold=0.2,
                ),
            )
        )
        _multivew_pack = "/horizon-bucket/matrix2/cicd_test/pack_data/MSD-27717/ADAS_20230915-104020_252_$Index.pack"  # noqa

        for model_type in [
            BEVModelType.bev_7v_temporal,
            BEVModelType.bev_5v,
        ]:
            for model_setting in [
                BEVModelSetting.pilot51_master,
            ]:
                cmd = f"""  # noqa
                set -e

                MODEL_TYPE={model_type}
                SETTING={model_setting}
                NAME_POSTFIX="test"
                VERSION=v0.0.1
                NUM_MACHINES=1
                NUM_GPUS_PER_MACHINE=2
                PROJECT_ID=PDT20220001
                BUCKET="matrix,matrix2,SD_Algorithm"
                TASK_SCENE=test

                # training
                python3 projects/pilot/tools/train/pilot_train_pipeline.py \
                    --model-type $MODEL_TYPE \
                    --model-setting $SETTING \
                    --model-name-postfix $NAME_POSTFIX  \
                    --model-version $VERSION \
                    --num-machines $NUM_MACHINES \
                    --num-gpus-per-machine $NUM_GPUS_PER_MACHINE \
                    --project-id $PROJECT_ID \
                    --mount-bucket  $BUCKET \
                    --local \
                    --task-scene $TASK_SCENE \
                    --pipeline-test \

                # packinfer
                export HAT_PILOT_MODEL_SETTING={model_setting}
                export HAT_PILOT_MODEL_CHECKPOINT=tmp_output/pilot_multitask_{model_type}/qat-checkpoint-last.pth.tar
                python3 tools/predict.py \
                    --config projects/pilot/configs/{model_type}/pack_infer_multitask.py \
                    --stage pack_infer \
                    --hat-pilot-model-thresh '{_model_thresh}' \
                    --hat-pilot-multiview-pack '{_multivew_pack}' \
                    --hat-pilot-pack-infer-vis '1' \
                    --pipeline-test

                # validation
                # use train ckpt
                export HAT_PILOT_MODEL_SETTING={model_setting}
                export HAT_PILOT_MODEL_CHECKPOINT=tmp_output/pilot_multitask_{model_type}/qat-checkpoint-last.pth.tar
                python3 tools/predict.py \
                    --config projects/pilot/configs/{model_type}/val_multitask.py \
                    --stage qat \
                    --device-ids 0,1 \
                    --pipeline-test \

                """

                subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    pytest.main(["-s", "-x"])

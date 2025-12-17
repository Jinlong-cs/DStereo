import json
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

import yaml
from aidisdk.client import AIDIClient
from hatbc.aidi.env import HOSTFILE

from hat.utils.apply_func import _as_list
from hat.utils.bucket import url_to_local_path
from hat.utils.config import Config
from hat.utils.dag_node import EnvironContainer, HATOp, ModuleContainer

DIR_PATH = os.path.dirname(__file__)


class PilotHATOp(HATOp):
    def __init__(
        self,
        entrance: Optional[str],
        model_setting: str,
        model_version: Optional[str] = None,
        model_name_postfix: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        task_dependency: Optional[Dict[str, Union[str, List[str]]]] = None,
        enable_tracking: Optional[str] = False,
        tracking_version: Optional[str] = "v3",
        hostfile: Optional[str] = HOSTFILE,
        port: Optional[int] = 8000,
        launcher: Optional[str] = "mpi",
    ):
        super().__init__(entrance, enable_tracking, hostfile, port, launcher)
        self.model_setting = model_setting
        self.working_env["HAT_PILOT_MODEL_SETTING"] = str(model_setting)
        self.model_version = model_version
        if enable_tracking:
            assert tracking_version.lower() in ["v2", "v3"]
            self.tracking_version = tracking_version.lower()
        if model_version:
            self.working_env["HAT_PILOT_MODEL_VERSION"] = model_version
        self.model_name_postfix = model_name_postfix
        if model_name_postfix:
            self.working_env[
                "HAT_PILOT_MODEL_NAME_POSTFIX"
            ] = model_name_postfix  # noqa
        self.model_checkpoint = model_checkpoint
        if model_checkpoint:
            self.working_env["HAT_PILOT_MODEL_CHECKPOINT"] = model_checkpoint

        # TODO: Get rid of this weird setting
        if task_dependency is None:
            task_meta_path = os.path.join(DIR_PATH, "../../task_meta.yaml")
            with open(task_meta_path, "r") as rf:
                _default_pilot_task_dependency = yaml.safe_load(rf)
            self.task_dependency = _default_pilot_task_dependency
        else:
            self.task_dependency = task_dependency
        assert isinstance(self.task_dependency, dict)

    def forward(
        self,
        stage: str,
        config_path: str,
        device_ids: Optional[Union[int, List[int]]] = 0,
        num_machines: Optional[int] = 1,
        pipeline_test: Optional[bool] = False,
        working_path: Optional[str] = None,
        working_env: Optional[dict] = None,
        target_tasks: Optional[Union[str, List[str]]] = None,
        dry_run: Optional[bool] = False,
    ):
        """Pilot hat train single-stage op.

        Args:
            stage: Stage.
            config_path: Config.
            target_tasks: Target tasks.
            device_ids: GPU id.
            num_machines: Num of machines.
            pipeline_test: Pipeline test.
            working_path: Add to PYTHONPATH.
            working_env: Env variable.
            dry_run:
                Skip actual cmd run.

        Returns:
            working_meta: working meta.
        """
        # add working path to pythonpath
        self.update_working_path(working_path=working_path)
        # update working_env
        self.update_working_env(working_env=working_env)
        # build training tasks for trainig stage
        task_names, tasks = self.get_training_tasks(
            stage=stage,
            config_path=config_path,
            target_tasks=target_tasks,
        )
        self.working_env["HAT_PILOT_TASKS"] = json.dumps(tasks)
        self.working_env["HAT_NUM_MACHINES"] = str(num_machines)

        super().forward(
            stage=stage,
            config_path=config_path,
            device_ids=device_ids,
            num_machines=num_machines,
            pipeline_test=pipeline_test,
            working_path=working_path,
            dry_run=dry_run,
        )

        model_name = self.get_model_name(
            config_path=config_path,
            stage=stage,
        )
        # resume working_env
        self.resume_working_env(working_env)
        self.working_env.pop("HAT_PILOT_TASKS", None)
        self.working_env.pop("HAT_NUM_MACHINES", None)
        # update working meta
        stage_meta = dict(
            model_name=model_name,
            stage=stage,
            tasks=tasks,
            task_names=task_names,
        )
        self.working_meta[stage] = stage_meta
        self.working_meta["newest_stage"] = deepcopy(stage_meta)

        return self.working_meta

    def get_training_tasks(
        self,
        stage: str,
        config_path: str,
        target_tasks: Optional[Union[str, List[str]]] = None,
    ):
        # update training stage
        tmp_env = deepcopy(self.working_env)
        tmp_env["HAT_TRAINING_STEP"] = stage

        with ModuleContainer(), EnvironContainer(tmp_env):
            tasks = Config.fromfile(config_path)["tasks"]
            valid_task_names = [task["name"] for task in tasks]
            valid_training_tasks = {task["name"]: task for task in tasks}

        if target_tasks is not None:
            result_task_names = set()
            for task in _as_list(target_tasks):
                depneds = _as_list(self.task_dependency.get(task, []))
                depneds = [t for t in depneds if t in valid_task_names]
                result_task_names.update(depneds)
                if task not in valid_task_names:
                    continue
                result_task_names.add(task)

            result_task_names = list(result_task_names)
        else:
            result_task_names = valid_task_names
        result_tasks = [
            valid_training_tasks[name] for name in result_task_names
        ]

        return result_task_names, result_tasks

    def get_model_name(self, config_path: str, stage: Optional[str] = None):
        tmp_env = deepcopy(self.working_env)
        if stage is not None:
            tmp_env["HAT_TRAINING_STEP"] = stage
        with ModuleContainer(), EnvironContainer(update_environ=tmp_env):
            config = Config.fromfile(config_path)
            model_name = config.get("model_name", None)

        return model_name


class PilotHATTrainPipeline(PilotHATOp):
    def __init__(
        self,
        model_setting: str,
        model_version: Optional[str] = None,
        model_name_postfix: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        load_ckpt_resume_mode: Optional[bool] = False,
        task_dependency: Optional[Dict[str, Union[str, List[str]]]] = None,
        enable_tracking: Optional[str] = False,
        tracking_version: Optional[str] = "v3",
        entrance: Optional[str] = "tools/train.py",
        hostfile: Optional[str] = HOSTFILE,
        port: Optional[int] = 8000,
        launcher: Optional[str] = "mpi",
    ):
        super().__init__(
            entrance=entrance,
            model_setting=model_setting,
            model_version=model_version,
            model_name_postfix=model_name_postfix,
            model_checkpoint=model_checkpoint,
            task_dependency=task_dependency,
            enable_tracking=enable_tracking,
            tracking_version=tracking_version,
            hostfile=hostfile,
            port=port,
            launcher=launcher,
        )
        self.load_ckpt_resume_mode = load_ckpt_resume_mode

    def forward(
        self,
        stages: Union[str, List[str]],
        config_path: str,
        target_tasks: Optional[Union[str, List[str]]] = None,
        device_ids: Optional[Union[int, List[int]]] = 0,
        num_machines: Optional[int] = 1,
        pipeline_test: Optional[bool] = False,
        working_path: Optional[str] = None,
        working_env: Optional[dict] = None,
        rerun_with_resume: Optional[bool] = False,
        dry_run: Optional[bool] = False,
    ):
        """Pilot hat train multi-stage op.

        Args:
            stages: List of stage.
            config_path: Config.
            target_tasks: Target tasks.
            device_ids: GPU id.
            num_machines: Num of machines.
            pipeline_test: Pipeline test.
            working_path: Add to PYTHONPATH.
            working_env: Env variable.
            rerun_with_resume:
                If true, pull newest artifact for ckpt resume.
            dry_run:
                Skip actual cmd run.

        Returns:
            working_meta: working meta.
        """
        client = AIDIClient()
        self.update_working_env(working_env=working_env)
        config_path = os.path.abspath(url_to_local_path(config_path))
        model_name = self.get_model_name(config_path=config_path)
        resume_stage_idx = self.get_resume_stage(model_name, stages)
        # When load_ckpt_resume_mode = True and rerun_with_resume = True,
        # Only support artifact not in same run.
        # If same, load_ckpt_resume_mode NOT effect.
        if (
            self.load_ckpt_resume_mode
            and rerun_with_resume
            and resume_stage_idx >= 0
        ):
            resume_stage = stages[resume_stage_idx]
            resume_artifact = client.experiment.artifact(
                f"{model_name}-{resume_stage}:{self.model_version}"
            )
            cur_run = client.experiment.run().run_name
            resume_artifact_run = resume_artifact.logged_by().run_name
            if cur_run == resume_artifact_run:
                self.load_ckpt_resume_mode = False
            else:
                self.load_ckpt_resume_mode = True

        for i, stage in enumerate(_as_list(stages)):
            if i > 0:
                if "HAT_PILOT_MODEL_CHECKPOINT" in self.working_env:
                    self.working_env.pop("HAT_PILOT_MODEL_CHECKPOINT")
                if "HAT_PILOT_RESUME_TRAINING" in self.working_env:
                    self.working_env.pop("HAT_PILOT_RESUME_TRAINING")
            if self.load_ckpt_resume_mode:
                if i == 0:
                    self.working_env["HAT_PILOT_RESUME_TRAINING"] = "1"
                    assert (
                        "HAT_PILOT_MODEL_CHECKPOINT" in self.working_env
                    ), "Please set model_checkpoint!"
            else:
                if rerun_with_resume:
                    if i < resume_stage_idx:
                        continue
                    if i == resume_stage_idx:
                        self.working_env["HAT_PILOT_RESUME_TRAINING"] = "1"
                        if self.tracking_version == "v2":
                            self.working_env[
                                "HAT_PILOT_MODEL_CHECKPOINT"
                            ] = f"aidi://{model_name}/{self.model_version}/{stage}"  # noqa
                        else:
                            self.working_env["HAT_PILOT_MODEL_CHECKPOINT"] = (
                                f"aidi_artifact://{model_name}/{stage}/"
                                f"{self.model_version}/{stage}-checkpoint-last.pth.tar"  # noqa
                            )

            super().forward(
                stage=stage,
                config_path=config_path,
                device_ids=device_ids,
                num_machines=num_machines,
                pipeline_test=pipeline_test,
                working_path=working_path,
                target_tasks=target_tasks,
                dry_run=dry_run,
            )

        self.resume_working_env(working_env=working_env)
        return self.working_meta

    def get_resume_stage(
        self,
        model_name: str,
        stages: Union[str, List[str]],
    ):
        client = AIDIClient()
        resume_stage_idx = -1
        for i, stage in enumerate(_as_list(stages)):
            is_exist = False
            if self.tracking_version.lower() == "v3":
                if client.experiment.artifact_exist(
                    f"{model_name}-{stage}:{self.model_version}"
                ):
                    is_exist = True
            else:
                if client.model.exist(model_name, self.model_version, stage):
                    is_exist = True
            if is_exist:
                resume_stage_idx = i
        return resume_stage_idx


class PilotHATPredictor(PilotHATOp):
    def __init__(
        self,
        project_id: str,
        model_setting: str,
        model_version: Optional[str] = None,
        model_name_postfix: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        task_dependency: Optional[Dict[str, Union[str, List[str]]]] = None,
        enable_tracking: Optional[str] = False,
        entrance: Optional[str] = "tools/predict.py",
        hostfile: Optional[str] = HOSTFILE,
        port: Optional[int] = 8000,
        launcher: Optional[str] = "mpi",
    ):
        super().__init__(
            entrance=entrance,
            model_setting=model_setting,
            model_version=model_version,
            model_name_postfix=model_name_postfix,
            model_checkpoint=model_checkpoint,
            task_dependency=task_dependency,
            enable_tracking=enable_tracking,
            hostfile=hostfile,
            port=port,
            launcher=launcher,
        )
        self.project_id = project_id
        self.working_env["PROJECT_ID"] = self.project_id

    def forward(
        self,
        config_path: str,
        stage: str,
        device_ids: Optional[Union[int, List[int]]] = 0,
        num_machines: Optional[int] = 1,
        pipeline_test: Optional[bool] = False,
        working_path: Optional[str] = None,
        working_env: Optional[dict] = None,
        target_tasks: Optional[Union[str, List[str]]] = None,
        model_thresh: Optional[Dict] = None,
        prediction_name_suffix: Optional[str] = None,
        eval_data_setting: Optional[str] = None,
        dry_run: Optional[bool] = False,
    ):
        """Pilot hat pred op.

        Args:
            stage: Stage.
            config_path: Config.
            target_tasks: Target tasks.
            device_ids: GPU id.
            num_machines: Num of machines.
            pipeline_test: Pipeline test.
            working_path: Add to PYTHONPATH.
            working_env: Env variable.
            model_thresh: Task thresh.
            prediction_name_suffix: Prediction suffix.
            eval_data_setting: Eval data setting.
            dry_run:
                Skip actual cmd run.

        Returns:
            working_meta: working meta.
        """
        # add working path to pythonpath
        self.update_working_path(working_path=working_path)
        self.update_working_env(working_env=working_env)
        config_path = os.path.abspath(url_to_local_path(config_path))
        # setting environment for pilot training
        if model_thresh is not None:
            self.working_env["HAT_PILOT_MODEL_THRESH"] = json.dumps(
                model_thresh
            )
        if prediction_name_suffix is not None:
            self.working_env[
                "HAT_PILOT_PREDICTION_NAME_SUFFIX"
            ] = prediction_name_suffix

        if eval_data_setting is not None:
            self.working_env["HAT_PILOT_EVAL_DATA_SETTING"] = eval_data_setting
        else:
            self.working_env[
                "HAT_PILOT_EVAL_DATA_SETTING"
            ] = self.model_setting
        super().forward(
            stage=stage,
            config_path=config_path,
            device_ids=device_ids,
            num_machines=num_machines,
            pipeline_test=pipeline_test,
            working_path=working_path,
            target_tasks=target_tasks,
            dry_run=dry_run,
        )

        # update meta
        dataset_ids = self.get_dataset_ids(
            config_path=config_path, stage=stage
        )
        prediction_name = self.get_prediction_name(
            config_path=config_path, stage=stage, suffix=prediction_name_suffix
        )
        self.resume_working_env(working_env=working_env)
        update_meta = dict(
            prediction_name=prediction_name,
            eval_datasets=dataset_ids,
        )
        self.working_meta["newest_stage"].update(update_meta)
        self.working_meta[stage].update(update_meta)

        return self.working_meta

    def get_dataset_ids(self, config_path: str, stage: Optional[str] = None):
        tmp_env = deepcopy(self.working_env)
        if stage is not None:
            tmp_env["HAT_TRAINING_STEP"] = stage
        with ModuleContainer(), EnvironContainer(update_environ=tmp_env):
            config = Config.fromfile(config_path)
            dataset_ids = config.get("dataset_ids", dict())

        return dataset_ids

    def get_prediction_name(
        self,
        config_path: str,
        stage: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        tmp_env = deepcopy(self.working_env)
        if stage is not None:
            tmp_env["HAT_TRAINING_STEP"] = stage
        if suffix is not None:
            tmp_env["HAT_PILOT_PREDICTION_NAME_SUFFIX"] = suffix
        with ModuleContainer(), EnvironContainer(update_environ=tmp_env):
            config = Config.fromfile(config_path)
            model_name = config.get("model_name", None)
            prediction_name = config.get("prediction_name", model_name)

        return prediction_name

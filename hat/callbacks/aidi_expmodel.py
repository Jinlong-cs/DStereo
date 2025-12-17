# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import List, Optional

from hat.core.event import EventStorage, StorageHandler
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import Artifact
from hat.utils.apply_func import _as_list
from hat.utils.deprecate import deprecated_warning
from hat.utils.distributed import rank_zero_only
from hat.utils.logger import ExperimentLogger
from hat.utils.package_helper import require_packages
from .callbacks import CallbackMixin
from .checkpoint import TRAIN_CHECKPOINT_FORMAT, Checkpoint
from .save_traced import DEPLOY_FILE_FORMAT, SaveTraced

__all__ = ["AIDIExpModel", "AIDIExperimentManager"]

logger = logging.getLogger(__name__)

ARTIFACT_NAME_FORMAT = "%s-%s:%s"
ARTIFACT_ALIAS_FORMAT = "%s-%s"


@OBJECT_REGISTRY.register
class AIDIExperimentManager(CallbackMixin):
    """AIDIExperimentManager Callback.

    It is used for uploading and tracking model to the experimental
    model in AIDI platform.

    Args:
        model_name: Name of experimental model directory.
        save_model: Save the type of the model."last" or "best".
        model_version: Model version.
        group_name: Group name of experimental run.
        upload_progressive_checkpoint: If ``True`` upload step or epoch
            checkpoint to aidi exp model saved by "Checkpoint" callback,
            interval controlled by "Checkpoint" callback. Loop end checkpoint
            is by default uploaded to aidi exp model and not controlled by
            this argument. Default: ``False``.
        overwrite_file: If ``True`` the previous uploaded file
            (like checkpoint file) will be overwritten by new ones.
            Only work when `upload_progressive_checkpoint=True`.
        tags: Catalog label.
    """

    @require_packages("aidisdk")
    def __init__(
        self,
        model_name: str,
        save_model: str,
        model_version: Optional[str] = None,
        group_name: Optional[str] = None,
        upload_progressive_checkpoint: bool = False,
        overwrite_file: bool = False,
        tags: Optional[List[str]] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.save_model = save_model
        self.model_version = model_version
        self.group_name = group_name
        self.tags = tags
        self.upload_progressive_checkpoint = upload_progressive_checkpoint
        self.overwrite_file = overwrite_file
        self._client = None
        self._enable_tracking = None

        tracking_setting = bool(
            int(os.environ.get("HAT_ENABLE_MODEL_TRACKING", "0"))
        )
        self.experiment_logger = ExperimentLogger(
            enable_tracking=tracking_setting,
            logger_type="aidi",
        )

    @property
    def client(self):
        self.stage = os.getenv("HAT_TRAINING_STEP", None)
        if self._client is None:
            self._client = self.experiment_logger.logger.client

        return self._client

    @property
    def enable_tracking(self):
        if self._enable_tracking is None:
            self._enable_tracking = (
                self.experiment_logger.logger.enabled_tracking(self.client)
            )
        return self._enable_tracking

    @rank_zero_only
    def on_loop_begin(self, storage: EventStorage, **kwargs):
        self.stage = os.environ.get("HAT_TRAINING_STEP", None)
        # init group
        group_name = (
            self.group_name
            if self.group_name
            else f"training-group-{self.stage}"
        )  # noqa E501
        self.experiment_logger.init_group(group_name)

        # set storage producer is ready
        if self.enable_tracking:
            for key in [
                StorageHandler.metrics_keys_epoch,
                StorageHandler.metrics_keys_step,
            ]:
                StorageHandler.producer_ack(storage, key)

    @rank_zero_only
    def on_step_end(
        self,
        storage,
        callbacks,
        epoch_id=None,
        global_step_id=None,
        num_steps=None,
        **kwargs,
    ):
        # upload ckpt
        if self.upload_progressive_checkpoint:
            self._log_model_out_artifact(
                callbacks,
                epoch_id=epoch_id,
                global_step_id=global_step_id,
                loop_prefix="step",
            )

        # log metric
        self._log_metrics(
            storage=storage,
            metric_interval="step",
            step_id=global_step_id,
            epoch_id=epoch_id,
        )

    @rank_zero_only
    def on_epoch_end(
        self,
        storage,
        epoch_id,
        global_step_id,
        callbacks,
        **kwargs,
    ):
        # upload ckpt
        if self.upload_progressive_checkpoint:
            self._log_model_out_artifact(
                callbacks,
                epoch_id=epoch_id,
                global_step_id=global_step_id,
                loop_prefix="epoch",
            )

        # log metric
        self._log_metrics(
            storage=storage,
            metric_interval="epoch",
            step_id=global_step_id,
            epoch_id=epoch_id,
        )

    @rank_zero_only
    def on_loop_end(self, callbacks, epoch_id, global_step_id, **kwargs):

        # log model output artifact
        self._log_model_out_artifact(
            callbacks,
            epoch_id=epoch_id,
            global_step_id=global_step_id,
            is_loop_last=True,
        )

    def _log_metrics(
        self,
        storage: EventStorage,
        metric_interval: str,
        step_id: Optional[int] = None,
        epoch_id: Optional[int] = None,
    ):
        if self.enable_tracking:
            # handle metrics from event storage
            if metric_interval == "step":
                key = StorageHandler.metrics_keys_step
            else:
                key = StorageHandler.metrics_keys_epoch
            metrics_list = StorageHandler.consume(storage, key)
            if metrics_list:
                logger.debug("consume metrics list: %s" % metrics_list)
                self.experiment_logger.logger.log_metrics(
                    metrics_list=metrics_list,
                    epoch_id=epoch_id,
                    step_id=step_id,
                )

    def _get_checkpoint_callback(self, callbacks: List[CallbackMixin]):
        for cb in callbacks:
            if isinstance(cb, Checkpoint):
                return cb
        return None

    def _log_model_out_artifact(
        self,
        callbacks: List[CallbackMixin],
        epoch_id: int,
        global_step_id: int,
        loop_prefix: Optional[str] = None,
        is_loop_last: bool = False,
    ):

        cb = self._get_checkpoint_callback(callbacks)

        if cb is None:
            return

        if loop_prefix and not (cb.interval_by == loop_prefix):
            return

        current_loop_id = (
            epoch_id if cb.interval_by == "epoch" else global_step_id
        )  # noqa E501

        if is_loop_last or ((current_loop_id + 1) % cb.save_interval == 0):
            ckpt_name = TRAIN_CHECKPOINT_FORMAT % (
                cb.name_prefix,
                self.save_model,
            )
            ckpt_file = os.path.join(
                cb.save_dir,
                ckpt_name,
            )

            artifact_name = ARTIFACT_NAME_FORMAT % (
                self.model_name,
                self.stage,
                self.save_model,
            )

            if self.overwrite_file:
                alias_suffix = "latest"
            else:
                alias_suffix = (
                    "latest"
                    if is_loop_last and not loop_prefix
                    else f"{cb.interval_by}_{current_loop_id}"
                )

            artifact_alias = ARTIFACT_ALIAS_FORMAT % (
                self.stage,
                alias_suffix,
            )
            if self.model_version:
                artifact_alias = [artifact_alias] + [self.model_version]

            artifact_files = [ckpt_file]
            if is_loop_last:
                for callback in callbacks:
                    if isinstance(callback, SaveTraced):
                        pt_file = os.path.join(
                            callback.save_dir,
                            DEPLOY_FILE_FORMAT
                            % (callback.name_prefix, "last"),
                        )
                        artifact_files.append(pt_file)

            self.experiment_logger.logger.log_artifact(
                artifact_name=artifact_name,
                artifact_type="model",
                artifact_aliases=_as_list(artifact_alias),
                artifact_tags=[self.stage],
                files=artifact_files,
                overwrite=self.overwrite_file,
            )


@OBJECT_REGISTRY.register
class AIDIExpModel(AIDIExperimentManager):
    """AIDIExpModel Callback.

    It is used for uploading and tracking model to the experimental
    model in AIDI platform.

    Args:
        model_name: Name of experimental model directory.
        task_type: Algorithm task type to which the model belongs. Currently
            supported "detection", "classification", "segmentation",
            "landmarks","skeleton", "face recognition",
            "person re-identification","multi-object tracking",
            "face anti-spoofing", "traffic lane","multitask", "head pose",
            "OCR", "action recognition", "regression".
        save_model: Save the type of the model."last" or "best".
        upload_progressive_checkpoint: If ``True`` upload step or epoch
            checkpoint to aidi exp model saved by "Checkpoint" callback,
            interval controlled by "Checkpoint" callback. Caution, the
            previous checkpoint is overwritten by new checkpoints. Loop
            end checkpoint is by default uploaded to aidi exp model and
            not controlled by this argument. Default: ``False``
        desc: Directory description.
        tags: Catalog label.
        model_version: Model version.
        platforms: The type of hardware platforms supported by this model
            version.Currently supported "J5","X3","J3","X2","J2","GPU",
            "Hisilicon", "NVIDIA","Matrix", "FPGA", "X1", "CPU", "Other".
        attachments: Stage model attachments.
    """

    @require_packages("aidisdk")
    def __init__(
        self,
        model_name: str,
        task_type: str,
        save_model: str,
        upload_progressive_checkpoint: bool = False,
        desc: Optional[str] = "",
        tags: Optional[List[str]] = None,
        model_version: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
    ):
        super().__init__(
            model_name=model_name,
            save_model=save_model,
            model_version=model_version,
            upload_progressive_checkpoint=upload_progressive_checkpoint,
            tags=tags,
        )

        self.task_type = task_type
        self.desc = desc
        self.platforms = platforms
        self.attachments = attachments
        if self.upload_progressive_checkpoint:
            assert (
                self.model_version
            ), "upload_progressive_checkpoint needs specific model_version"
        deprecated_warning(
            author="mengyang.duan",
            deprecation_version="1.2.2",
            removal_version="1.4.0",
            old_name="AIDIExpModel",
            new_name="AIDIExperimentManager",
            rank_zero_only=True,
        )

    @rank_zero_only
    def on_step_end(
        self,
        storage,
        callbacks,
        global_step_id=None,
        num_steps=None,
        **kwargs,
    ):
        # upload ckpt
        if self.upload_progressive_checkpoint:
            checkpoint_callback = self._get_checkpoint_callback(callbacks)
            if checkpoint_callback is not None:
                if checkpoint_callback.interval_by == "step" and (
                    (global_step_id + 1) % checkpoint_callback.save_interval
                    == 0
                ):
                    ckpt_file = os.path.join(
                        checkpoint_callback.save_dir,
                        TRAIN_CHECKPOINT_FORMAT
                        % (
                            checkpoint_callback.name_prefix,
                            self.save_model,
                        ),
                    )
                    self._rm_exist_model_stage(
                        checkpoint_callback.name_prefix[:-1]
                    )
                    self._create_exp_model_version_stage(
                        ckpt_file, checkpoint_callback.name_prefix[:-1]
                    )
        # log metric
        self._log_metrics(
            storage=storage,
            metric_interval="step",
            step_id=global_step_id,
        )

    @rank_zero_only
    def on_epoch_end(
        self, storage, epoch_id, callbacks, num_epochs=None, **kwargs
    ):

        # upload ckpt
        if self.upload_progressive_checkpoint:
            checkpoint_callback = self._get_checkpoint_callback(callbacks)
            if checkpoint_callback is not None:
                if checkpoint_callback.interval_by == "epoch" and (
                    (epoch_id + 1) % checkpoint_callback.save_interval == 0
                ):
                    ckpt_file = os.path.join(
                        checkpoint_callback.save_dir,
                        TRAIN_CHECKPOINT_FORMAT
                        % (
                            checkpoint_callback.name_prefix,
                            self.save_model,
                        ),
                    )
                    self._rm_exist_model_stage(
                        checkpoint_callback.name_prefix[:-1]
                    )
                    self._create_exp_model_version_stage(
                        ckpt_file, checkpoint_callback.name_prefix[:-1]
                    )

        # log metric
        self._log_metrics(
            storage=storage,
            metric_interval="epoch",
            step_id=epoch_id,
            prefix="Epoch: ",
        )

    @rank_zero_only
    def on_loop_end(self, callbacks, storage: EventStorage, **kwargs):

        checkpoint_callback = self._get_checkpoint_callback(callbacks)
        if checkpoint_callback is not None:
            model_file = os.path.join(
                checkpoint_callback.save_dir,
                TRAIN_CHECKPOINT_FORMAT
                % (checkpoint_callback.name_prefix, self.save_model),
            )
            self._rm_exist_model_stage(checkpoint_callback.name_prefix[:-1])
            model_dict = self._create_exp_model_version_stage(
                model_file, checkpoint_callback.name_prefix[:-1]
            )
            if self.enable_tracking:
                self.client.experiment.log_output_artifact(
                    Artifact.from_model_stage(
                        self.client.model.finditem(**model_dict)
                    )
                )

        if self.enable_tracking:
            # clear experiment
            self.client.experiment.clear()

    def _rm_exist_model_stage(self, stage: str):
        if self.client.model.exist(
            model_name=self.model_name,
            model_version=self.model_version,
            stage=stage,
        ):
            self.client.model.delete(
                model_name=self.model_name,
                model_version=self.model_version,
                stage=stage,
            )

    def _create_exp_model(self):
        if self.client.model.exist(model_name=self.model_name) is True:
            logger.info("Model name already exists")
        else:
            self.client.model.set_public_authority()
            self.client.model.create(
                model_name=self.model_name,
                task_type=self.task_type,
                desc=self.desc,
                tags=self.tags,
            )

    def _create_exp_model_version(self, stage: str):
        self._create_exp_model()
        if (
            self.model_version is not None
            and self.client.model.exist(
                model_name=self.model_name,
                model_version=self.model_version,
            )
            is True
        ):
            logger.info("Model version already exists")
        else:
            result = self.client.model.submit_version_to_model(
                model_name=self.model_name,
                model_version=self.model_version,
                framework="PyTorch",
                platforms=self.platforms,
            )
            self.model_version = result.model_version

    def _create_exp_model_version_stage(self, model_file: str, stage: str):
        self._create_exp_model_version(stage)
        model_dict = {}
        if self.model_version is None:
            stage_version = "@newest"
        else:
            stage_version = self.model_version
        if not self.client.model.exist(
            model_name=self.model_name,
            model_version=stage_version,
            stage=stage,
        ):
            result = self.client.model.submit_stage_to_version(
                model_version=stage_version,
                stage=stage,
                model_name=self.model_name,
                model_file=model_file,
                attachments=self.attachments,
            )
            model_dict[
                "model_name"
            ] = result.model_version_item.model_item.model_name  # noqa E501
            model_dict[
                "model_version"
            ] = result.model_version_item.model_version  # noqa E501
            model_dict["stage"] = result.stage

        else:
            logger.info("Model stage already exists")
            model_dict.update(
                dict(  # noqa C408
                    model_name=self.model_name,
                    model_version=self.model_version,
                    stage=self.stage,
                )
            )

        return model_dict

    def _log_metrics(
        self,
        storage: EventStorage,
        metric_interval: str,
        step_id: int,
        prefix: str = "",
    ):

        # handle metrics from event storage
        if metric_interval == "step":
            key = StorageHandler.metrics_keys_step
        else:
            key = StorageHandler.metrics_keys_epoch
        metrics_list = StorageHandler.consume(storage, key)
        logger.debug("consume metrics list: %s" % metrics_list)
        for (k, v) in metrics_list:
            self.client.experiment.log_metrics(
                # TODO (mengyang.duan): fix this later.
                key=self.stage + ": " + prefix + k,
                value=v,
                step=step_id,
            )

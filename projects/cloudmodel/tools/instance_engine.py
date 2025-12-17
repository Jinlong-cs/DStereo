import logging
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Union

from aidisdk import AIDIClient
from aidisdk.model import DeviceMeta, Model, ModelFileSpec
from aidisdk.model.inference import LocalInferenceApi, RemoteInferenceApi
from hatbc.message import CameraFrame, DenseMap, Instance, LidarFrame
from perception_proto.frame_pb2 import CameraFrame as CameraFrameProto

from hat.utils.apply_func import _as_list
from hat.utils.config import Config

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


class CloudModelInference:
    class InferenceMode(Enum):
        AidiInferServiceRemote = "AidiInferServiceRemote"
        InferModelRemote = "InferModelRemote"
        AidiInferServiceLocal = "AidiInferServiceLocal"
        InferModelLocal = "InferModelLocal"

    def __init__(
        self,
        infer_mode: InferenceMode,
        init_config: dict,
        aidi_client: AIDIClient = None,
    ):
        self.infer_mode = infer_mode
        self.init_config = init_config
        # init inference instance before running
        self.instance: Optional[
            Union[Model, RemoteInferenceApi, LocalInferenceApi]
        ] = None
        self.check_init_config()
        self.aidi_client = AIDIClient() if aidi_client is None else aidi_client

    def check_init_config(self):
        if self.infer_mode in [
            CloudModelInference.InferenceMode.AidiInferServiceRemote,
        ]:
            assert "model_name" in self.init_config
            assert "model_version" in self.init_config
        elif (
            self.infer_mode
            == CloudModelInference.InferenceMode.InferModelRemote
        ):
            assert "model_name" in self.init_config
            assert "model_version" in self.init_config
            assert "run_device" in self.init_config
        elif (
            self.infer_mode
            == CloudModelInference.InferenceMode.InferModelLocal
        ):
            assert "aidi_config" in self.init_config
            assert "run_device" in self.init_config

    def _get_model_name(self):
        if self.infer_mode in [
            CloudModelInference.InferenceMode.AidiInferServiceRemote,
            CloudModelInference.InferenceMode.AidiInferServiceLocal,
            CloudModelInference.InferenceMode.InferModelRemote,
        ]:
            return self.init_config["model_name"]
        elif self.infer_mode in [
            CloudModelInference.InferenceMode.InferModelLocal
        ]:
            return Config.fromfile(self.init_config["aidi_config"])[
                "model_name"
            ]
        else:
            raise ValueError("Invalid inference mode")

    def _init_infer_instance(self):
        if (
            self.infer_mode
            == CloudModelInference.InferenceMode.AidiInferServiceRemote
        ):
            instance = self.load_aidi_inference_service_remote(
                self.init_config["model_serving_config"],
            )
        elif (
            self.infer_mode
            == CloudModelInference.InferenceMode.AidiInferServiceLocal
        ):
            instance = self.load_aidi_inference_service_local(
                self.init_config.get("ip", "localhost"),
                self.init_config.get("local_service_port", 8070),
            )
        elif (
            self.infer_mode
            == CloudModelInference.InferenceMode.InferModelRemote
        ):
            instance = self.load_model_remote(
                self.init_config["model_name"],
                self.init_config["model_version"],
                self.init_config["run_device"],
                aidi_client=self.aidi_client,
            )
        elif (
            self.infer_mode
            == CloudModelInference.InferenceMode.InferModelLocal
        ):
            instance = self.load_model_local(
                self.init_config["aidi_config"],
                self.init_config["run_device"],
            )
        else:
            raise ValueError("Invalid inference mode")
        self.instance = instance

    def load_aidi_inference_service_remote(
        self, model_serving_config: Dict, wait_for_launch=True
    ):
        # aquire a inference
        self.create_aidi_infer_service(
            service_path=model_serving_config["service_path"],
            project_id=model_serving_config["project_id"],
            resource_pool=model_serving_config["resource_pool"],
            instance_min=model_serving_config.get("instance_min", 1),
            instance_max=model_serving_config.get("instance_max", 4),
        )

        inference = self.aidi_client.model_registry.acquire_inference_api(
            service_path=model_serving_config["service_path"],
            instances=model_serving_config.get("instance_max", 1),
        )
        if wait_for_launch:
            while True:
                status = (
                    self.aidi_client.model_registry.inference_service_status(
                        model_serving_config["service_path"]
                    )
                )
                if status == "launch_success":
                    break
        print(f"Launch Success!!!!:{model_serving_config['service_path']}")
        return inference

    @staticmethod
    def load_aidi_inference_service_local(ip, port):
        forward_api = f"http://{ip}:{port}"
        inference = LocalInferenceApi(forward_api)
        return inference

    @staticmethod
    def load_model_remote(
        model_name,
        model_version,
        device: DeviceMeta,
        aidi_client: AIDIClient,
    ):
        aidi_client.model_registry.load(name=f"{model_name}:{model_version}")
        instance = aidi_client.model_registry.load_infer_model(
            name=f"{model_name}:{model_version}",
        )
        instance.init(device)
        return instance

    @staticmethod
    def load_model_local(
        aidi_config,
        device: DeviceMeta,
    ):
        config = Config.fromfile(aidi_config)
        instance = config["model_cls"](
            model_file_spec=ModelFileSpec(
                model_file=config["model_config"],
                attachments=dict(checkpoint_file=config["param_file"]),
            ),
        )
        instance.init(device)
        return instance

    def forward(self, source_data: List[Union[CameraFrame, LidarFrame]]):
        return self.__call__(source_data)

    def __call__(self, source_data: List[Union[CameraFrame, LidarFrame]]):
        if self.instance is None:
            self._init_infer_instance()
        source_data = _as_list(source_data)
        assert all(
            [
                isinstance(i, (CameraFrame, CameraFrameProto, LidarFrame))
                for i in source_data
            ]
        ), "Input should be a list of hatbc.message::CameraFrame or its serialized version"
        if self.infer_mode in [
            CloudModelInference.InferenceMode.AidiInferServiceRemote,
            CloudModelInference.InferenceMode.AidiInferServiceLocal,
        ]:
            return self.instance.forward(source_data)
        elif self.infer_mode in [
            CloudModelInference.InferenceMode.InferModelLocal,
            CloudModelInference.InferenceMode.InferModelRemote,
        ]:
            data = self.instance.preprocess(source_data)
            data = self.instance.predict(data)
            data = self.instance.postprocess(data)
            return data
        else:
            raise ValueError("Invalid inference mode")

    def __del__(self):
        if hasattr(self.instance, "release"):
            self.instance.release()
            print(
                f"release inference service done.. Model:{self.instance.model_name}  Service: {self.instance.service_name} "
            )  # noqa

    def create_aidi_infer_service(
        self,
        service_path,
        project_id,
        resource_pool,
        instance_min=1,
        instance_max=4,
    ):
        if self.aidi_client.model_registry.inference_service_exist(
            service_path
        ):
            return
        name = f"{self.init_config['model_name']}:{self.init_config['model_version']}"
        self.aidi_client.model_registry.create_inference_service(
            service_path=service_path,
            name=name,
            project_id=project_id,
            resource_pool=resource_pool,
            instance_min=instance_min,
            instance_max=instance_max,
        )

    def result2cameraframe(
        self,
        pred: List[List[Union[Instance, DenseMap]]],
        src: List[Union[CameraFrameProto, CameraFrame]],
        keep_src_perceptions=False,
    ):
        model_name = self._get_model_name()
        results = list()

        def _get_perceptions(perception_list):
            id2perceptions = defaultdict(list)
            for perception in perception_list:
                if isinstance(perception, Instance):
                    id2perceptions[
                        (perception.track_id, perception.topic)
                    ].append(perception)
                elif isinstance(perception, DenseMap):
                    id2perceptions[(None, perception.topic)].append(perception)
                else:
                    raise ValueError()
            return id2perceptions

        def _merge_results(src, target):
            ret_perception = list()
            for p_id, p_val in target.items():
                ret_perception.extend(p_val)
                if p_id in src:
                    src.pop(p_id)
            for p_val in src.values():
                ret_perception.extend(p_val)
            return ret_perception

        for pred_i, src_i in zip(pred, src):
            if isinstance(src_i, CameraFrameProto):
                src_i = CameraFrame.from_proto(src_i)
            if keep_src_perceptions:
                src_perceptions = _get_perceptions(src_i.perceptions)
                pred_perceptions = _get_perceptions(pred_i)
                res_perceptions = _merge_results(
                    src_perceptions, pred_perceptions
                )
            else:
                res_perceptions = pred_i
            results.append(
                CameraFrame(
                    topic=model_name,
                    image=src_i.image,
                    camera_param=src_i.camera_param,
                    camera_type=src_i.camera_type,
                    perceptions=res_perceptions,
                )
            )
        return results

import logging
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Union

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.config import Config
from hat.utils.package_helper import require_packages

try:
    from aidisdk import AIDIClient
    from aidisdk.model import DeviceMeta, Model, ModelFileSpec
    from aidisdk.model.inference import LocalInferenceApi, RemoteInferenceApi
    from hatbc.message import CameraFrame
    from perception_proto.frame_pb2 import CameraFrame as CameraFrameProto
except ImportError:
    AIDIClient = None
    DeviceMeta = None
    Model = None
    ModelFileSpec = None
    LocalInferenceApi = None
    RemoteInferenceApi = None
    CameraFrame = None
    CameraFrameProto = None


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


@OBJECT_REGISTRY.register_module
class AIDIPredictor:
    """AIDIPredictor class for performing inference using different modes.

    Args:
        infer_mode (InferenceMode): The inference mode for AIDIPredictor.
        There are three modes for inference:
         1. AidiInferServiceRemote: In this mode, the AIDIPredictor utilizes
            cloudmodel inference services for remote inference.
         2. InferModelRemote: This mode involves initializing the AIDIPredictor
            using the model configuration from the aidi model repository for
            local inference.
         3. InferModelLocal: In this mode, the model is initialized locally
            using the model configuration from the local model repository for
            local inference.
        model_name (str): The name of the model, required for
          AidiInferServiceRemote and InferModelRemote modes.
        model_version (str): The version of the model, required for
          AidiInferServiceRemote and InferModelRemote modes.
        model_serving_config (dict): Configuration for model serving,
          used in AidiInferServiceRemote mode.
        run_device: The device for running inference, used
          in InferModelRemote and InferModelLocal modes.
        aidi_config: Configuration for AIDI, used in InferModelLocal mode.
        aidi_client (AIDIClient): AIDIClient instance.

    """

    class InferenceMode(Enum):
        """Enumeration of inference modes for AIDIPredictor."""

        AidiInferServiceRemote = "AidiInferServiceRemote"
        InferModelRemote = "InferModelRemote"
        InferModelLocal = "InferModelLocal"

    @require_packages("aidisdk", raise_msg="Please pip3 install aidisdk")
    def __init__(
        self,
        infer_mode: InferenceMode,
        model_name: str = None,
        model_version: str = None,
        model_serving_config: dict = None,
        run_device: DeviceMeta = None,
        aidi_config: str = None,
        aidi_client: AIDIClient = None,
    ):
        self.infer_mode = infer_mode
        self.model_name = model_name
        self.model_version = model_version
        self.model_serving_config = model_serving_config
        self.run_device = run_device
        self.aidi_config = aidi_config
        # init inference instance before running
        self.instance: Optional[Union[Model, RemoteInferenceApi, LocalInferenceApi]] = (
            None
        )
        self.check_init_config()
        self.aidi_client = AIDIClient() if aidi_client is None else aidi_client

    def check_init_config(self):
        if self.infer_mode in [
            AIDIPredictor.InferenceMode.AidiInferServiceRemote,
        ]:
            assert self.model_name is not None
            assert self.model_version is not None
        elif self.infer_mode == AIDIPredictor.InferenceMode.InferModelRemote:
            assert self.model_name is not None
            assert self.model_version is not None
            assert self.run_device is not None
        elif self.infer_mode == AIDIPredictor.InferenceMode.InferModelLocal:
            assert self.aidi_config is not None
            assert self.run_device is not None

    def _get_model_name(self):
        if self.infer_mode in [
            AIDIPredictor.InferenceMode.AidiInferServiceRemote,
            AIDIPredictor.InferenceMode.InferModelRemote,
        ]:
            return self.model_name
        elif self.infer_mode in [AIDIPredictor.InferenceMode.InferModelLocal]:
            return Config.fromfile(self.init_config["aidi_config"])["model_name"]
        else:
            raise ValueError("Invalid inference mode")

    def _init_infer_instance(self):
        if self.infer_mode == AIDIPredictor.InferenceMode.AidiInferServiceRemote:
            instance = self.load_aidi_inference_service_remote(
                self.model_serving_config,
            )
        elif self.infer_mode == AIDIPredictor.InferenceMode.InferModelRemote:
            instance = self.load_model_remote(
                self.model_name,
                self.model_version,
                self.run_device,
                aidi_client=self.aidi_client,
            )
        elif self.infer_mode == AIDIPredictor.InferenceMode.InferModelLocal:
            instance = self.load_model_local(
                self.aidi_config,
                self.run_device,
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
                status = self.aidi_client.model_registry.inference_service_status(
                    model_serving_config["service_path"]
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
                attachments={"checkpoint_file": config["param_file"]},
            ),
        )
        instance.init(device)
        return instance

    def forward(self, source_data: List[CameraFrame], output_name: str = None):
        return self.__call__(source_data, output_name=output_name)

    def __call__(self, source_data: List[CameraFrame], output_name: str = None):
        if self.instance is None:
            self._init_infer_instance()
        source_data = _as_list(source_data)
        assert all(
            [isinstance(i, (CameraFrame, CameraFrameProto)) for i in source_data]
        ), "Input should be a list of hatbc.message::CameraFrame or its serialized version"  # noqa
        if self.infer_mode in [
            AIDIPredictor.InferenceMode.AidiInferServiceRemote,
        ]:
            data = self.instance.forward(source_data)
            if output_name is not None:
                data = OrderedDict({output_name: data})
            return data
        elif self.infer_mode in [
            AIDIPredictor.InferenceMode.InferModelLocal,
            AIDIPredictor.InferenceMode.InferModelRemote,
        ]:
            data = self.instance.preprocess(source_data)
            data = self.instance.predict(data)
            data = self.instance.postprocess(data)
            if output_name is not None:
                data = OrderedDict({output_name: data})
            return data
        else:
            raise ValueError("Invalid inference mode")

    def __del__(self):
        if hasattr(self.instance, "release"):
            self.instance.release()
            print(
                f"release inference service done.. Model:{self.instance.model_name}  Service: {self.instance.service_name} "  # noqa
            )  # noqa

    def create_aidi_infer_service(
        self,
        service_path,
        project_id,
        resource_pool,
        instance_min=1,
        instance_max=4,
    ):
        if self.aidi_client.model_registry.inference_service_exist(service_path):
            return
        name = f"{self.init_config['model_name']}:{self.init_config['model_version']}"  # noqa
        self.aidi_client.model_registry.create_inference_service(
            service_path=service_path,
            name=name,
            project_id=project_id,
            resource_pool=resource_pool,
            instance_min=instance_min,
            instance_max=instance_max,
        )

    @property
    def training(self):
        return False

    def eval(self):
        pass

    def cuda(self, device=None):
        pass

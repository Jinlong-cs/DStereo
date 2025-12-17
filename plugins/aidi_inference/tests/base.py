import os
import time

from aidisdk import AIDIClient
from aidisdk.model import DeviceMeta, ModelTags

from hat.utils.config import Config

client = AIDIClient(endpoint="http://aidi.hobot.cc")


class BaseTestModel(object):
    def setup_class(self):
        config_path = os.environ.get("HAT_AIDI_INFERENCE_CONFIG", None)
        assert (
            config_path is not None
        ), "HAT_AIDI_INFERENCE_CONFIG should provided in env"
        self.config = Config.fromfile(config_path)
        self.record = {}
        assert (
            "example_input" in self.config
        ), f"config: {config_path} should provided example_input as test input"
        self.record["num_batch"] = self.config.get("num_batch", 1)
        self.example_input = self.config["example_input"]

    def teardown_class(self):
        print(self.record)

    def deserialize_single(self, result):
        return result

    def test_run_local(self, num_infer=1):
        model_name = self.config["model_name"]
        num_batch = self.record["num_batch"]
        device_meta = DeviceMeta("gpu", "1")
        print(
            "\n**************************** test local running ****************************"  # noqa
        )
        # test newest model
        test_model = client.model_registry.load_model(
            model_name,
            tags=ModelTags.from_dict({}),
        )
        test_model.init(device_meta)
        ts = time.time()
        for _ in range(num_infer):
            # local inference should include pre/post process
            model_input = test_model.preprocess(self.example_input)
            model_output = test_model.predict(model_input)
            result = test_model.postprocess(model_output)
            # deserialize
            result = [self.deserialize_single(r) for r in result]
        print(
            "\n************************* local running successful **************************"  # noqa
        )
        te = time.time() - ts
        self.record["local_speed"] = num_batch / (te / num_infer)
        print(f"Speed(FPS): {self.record['local_speed']}")
        assert result is not None
        return result

    def test_run_aidi(self, num_infer=1):
        model_name = self.config["model_name"]
        num_batch = self.record["num_batch"]
        test_input = self.example_input
        print(
            "\n****************************** test aidi running *****************************"  # noqa
        )
        # test newest model
        inference = client.model_registry.acquire_inference(
            name=model_name,
            tags=ModelTags.from_dict({}),
        )
        inference.wait_for_launch(interval=1)
        ts = time.time()
        for _ in range(num_infer):
            result = inference.request(test_input)
            result = [self.deserialize_single(r) for r in result]
        print(
            "\n*************************** aidi running successful ***************************"  # noqa
        )
        te = time.time() - ts
        self.record["cloud_speed"] = num_batch / (te / num_infer)
        print(f"Speed(FPS): {self.record['cloud_speed']}")
        inference.release()
        assert result is not None
        return result

    def results_compare(self):
        raise NotImplementedError(
            f"Test class {self.__class__.__name__} NOT implement"
            " local & aidi compare func"
        )

    def test_local_aidi_compare_result(self):
        model_name = self.config["model_name"]
        test_input = self.example_input
        # local
        test_model = client.model_registry.load_model(
            model_name,
            tags=ModelTags.from_dict({}),
        )
        device_meta = DeviceMeta("gpu", "0")
        test_model.init(device_meta)
        model_input = test_model.preprocess(test_input)
        model_output = test_model.predict(model_input)
        local_result = test_model.postprocess(model_output)
        # cloud
        inference = client.model_registry.acquire_inference(
            name=model_name,
            tags=ModelTags.from_dict(self.config["model_tags"]),
        )
        inference.wait_for_launch(interval=1)
        aidi_result = inference.request(test_input)
        inference.release()

        compare_rets, is_pass = self.results_compare(local_result, aidi_result)
        self.record["local_cloud_diff"] = compare_rets
        assert is_pass, "local & aidi results are inconsistent"

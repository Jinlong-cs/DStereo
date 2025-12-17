from hdflow.fillback import BaseIssueConfig, BasePredictConfig


def _update_fillback_runtime_config(meta_data):
    input_template = {
        "front_main": {
            "filter": {"name": {"prefix": "ADAS", "pattern": "_5.pack"}},
            "deploy": [{"port": 6690, "device": "j5a"}],
        },
        "front_narrow": {
            "filter": {"name": {"prefix": "ADAS", "pattern": "_0.pack"}},
            "deploy": [{"port": 6680, "device": "j5a"}],
        },
        "surround_0": {
            "filter": {"name": {"prefix": "ADAS", "pattern": "_1.pack"}},
            "deploy": [{"port": 6682, "device": "j5a"}],
        },
        "surround_1": {
            "filter": {"name": {"prefix": "ADAS", "pattern": "_2.pack"}},
            "deploy": [{"port": 6684, "device": "j5a"}],
        },
        "surround_2": {
            "filter": {"name": {"prefix": "ADAS", "pattern": "_3.pack"}},
            "deploy": [{"port": 6686, "device": "j5a"}],
        },
        "surround_3": {
            "filter": {"name": {"prefix": "ADAS", "pattern": "_4.pack"}},
            "deploy": [{"port": 6688, "device": "j5a"}],
        },
    }
    output_template = [
        {
            "device": "j5a",
            "port": 5570,
            "prefix": "ADAS",
            "channel": 0,
            "check": {"frame": {"loss_rate": 0.1}},
        },
        {"device": "j5a", "port": 5571, "prefix": "ADAS", "channel": 1},
        {"device": "j5a", "port": 5572, "prefix": "ADAS", "channel": 2},
        {"device": "j5a", "port": 5573, "prefix": "ADAS", "channel": 3},
        {"device": "j5a", "port": 5574, "prefix": "ADAS", "channel": 4},
        {"device": "j5a", "port": 5576, "prefix": "ADAS", "channel": 5},
        {
            "device": "j5a",
            "port": 5575,
            "prefix": "FUSION6V",
            "channel": 5,
            "check": {"frame": {"loss_rate": 0.1}},
        },
    ]

    sdkplus = True
    template = {
        "input_template": input_template,
        "output_template": output_template,
    }
    runtime_lib = "/usr/local/gcc-5.4.0/lib64"

    meta_data["sdkplus"] = sdkplus
    meta_data["template"] = template
    meta_data["runtime_lib"] = runtime_lib

    return meta_data


class PredictConfig(BasePredictConfig):
    def __init__(
        self,
        meta,
        name=None,
        app=None,
        dataset=None,
        queue=None,
        num_worker=None,
        fillback_device_num_per_worker=1,
    ):

        # 新增新版回灌参数
        sdk_plus_option = meta["runtime_config"].get("sdkplus", True)
        if sdk_plus_option:
            _update_fillback_runtime_config(meta["runtime_config"])

        super().__init__(
            meta=meta,
            name=name,
            app=app,
            dataset=dataset,
            queue=queue,
            num_worker=num_worker,
            fillback_device_num_per_worker=fillback_device_num_per_worker,
        )

    @property
    def dataset(self):
        return self._get_property("_dataset")

    @dataset.setter
    def dataset(self, data):
        self._dataset = data
        if data is not None:
            self.params.data = data.id
            if data.flag:
                self.params["common_params"]["car_type"] = data.flag
            if data.meta is not None:
                if "app_update" in data.meta:
                    self.params.app_config.update(data.meta["app_update"])
                if "runtime_update" in data.meta:
                    self.params.runtime_config.update(
                        data.meta["runtime_update"]
                    )


class IssueConfig(BaseIssueConfig):
    def __init__(
        self,
        meta,
        name=None,
        app=None,
        dataset=None,
        queue=None,
        num_worker=None,
        fillback_device_num_per_worker=1,
    ):
        # 新增新版回灌参数
        sdk_plus_option = meta.get("sdkplus", True)
        if sdk_plus_option:
            _update_fillback_runtime_config(
                meta["spec_conf"]["runtime_config"]
            )
        super().__init__(
            meta=meta,
            name=name,
            app=app,
            dataset=dataset,
            queue=queue,
            num_worker=num_worker,
            fillback_device_num_per_worker=fillback_device_num_per_worker,
        )

    @property
    def dataset(self):
        return self._get_property("_dataset")

    @dataset.setter
    def dataset(self, data):
        self._dataset = data
        if data is not None:
            dataset_ids = data.ids
            if isinstance(dataset_ids, int):
                dataset_ids = str(dataset_ids)
            if isinstance(dataset_ids, (list, tuple)):
                dataset_ids = ",".join(list(map(str, dataset_ids)))
            self.params["artifact_ids"] = dataset_ids
            if data.flag:
                self.params["common_params"]["car_type"] = data.flag
            if data.meta is not None:
                if "merge_key" in data.meta:
                    self.params["spec_conf"]["merge_key"] = data.meta[
                        "merge_key"
                    ]
                if "common_update" in data.meta:
                    self.params["common_params"].update(
                        data.meta["common_update"]
                    )
                if "spec_update" in data.meta:
                    self.params["spec_conf"].update(data.meta["spec_update"])
                if "runtime_update" in data.meta:
                    self.params["spec_conf"]["runtime_config"].update(
                        data.meta["runtime_update"]
                    )

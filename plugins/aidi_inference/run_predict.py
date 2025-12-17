# Copyright (c) Horizon Robotics. All rights reserved.
try:
    # TODO find rootcause
    # we have to import mxnet first if exists,
    # otherwise will raise an error....
    import mxnet
except Exception:
    mxnet = None
import argparse
import logging

from aidisdk import AIDIClient
from aidisdk.model import DeviceMeta

from hat.utils.config import Config

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--local",
        action="store_true",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    client = AIDIClient()

    if args.local:
        model_name = f"{cfg.model_name}:{args.version}"
        client.model_registry.load(name=model_name)
        test_model = client.model_registry.load_infer_model(model_name)
        device_meta = DeviceMeta("gpu", "0")
        test_model.init(device_meta)
        # local inference should include pre/post process
        model_input = test_model.preprocess(cfg.example_input)
        model_output = test_model.predict(model_input)
        result = test_model.postprocess(model_output)
    else:
        inference = client.model_registry.acquire_inference_api(
            service_path=cfg.serve_path,
            instances=1,
        )
        inference.wait(timeout=100)
        result = inference.forward(cfg.example_input)
        print("******** online predict success! **********")
        inference.release()

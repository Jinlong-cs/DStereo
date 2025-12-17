import argparse
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Union

from aidisdk import AIDIClient
from aidisdk.experiment import Bar, Line, Table
from aidisdk.experiment.artifact import Artifact

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
IS_LOCAL = not os.path.exists("/running_package")


def find_args(
    args: str,
    key: str,
    default_value: Any = None,
    with_idx: bool = False,
):
    args_list = list(filter(lambda x: len(x) > 0, args.split(" ")))
    if key in args_list:
        key_idx = args_list.index(key)
        value_idx = key_idx + 1
        return (
            (args_list[value_idx], value_idx)
            if with_idx
            else args_list[value_idx]
        )
    return (default_value, None) if with_idx else default_value


class AIDIExperimentLogger:

    client = AIDIClient()

    @classmethod
    def log_artifact(
        cls,
        artifact_name: str,
        artifact_type: str,
        artifact_aliases: List[str],
        artifact_tags: List[str],
        files: Optional[List[str]] = None,
        overwrite: bool = False,
    ):
        """Log Artifact to AIDIExperiment.

        Args:
            artifact_name: Artifact name.
            artifact_type: Artifact type.
            artifact_aliases: Artifact aliases.
            artifact_tags: Artifact tags.
            files: Files to add to artifact. Defaults to None.
            overwrite: Whether to overwrite artifact. Defaults to False.
        """

        if cls.enabled_tracking():
            output_artifact = Artifact(
                artifact_name,
                type=artifact_type,
                aliases=artifact_aliases,
                tags=artifact_tags,
            )

            if files:
                for file in files:
                    if os.path.exists(file):
                        output_artifact.add_file(file, overwrite=overwrite)
                    else:
                        logger.warning(f"File {file} does not exits.")

            cls.client.experiment.log_artifact(
                artifact=output_artifact,
                overwrite=overwrite,
            )

    @classmethod
    def enabled_tracking(cls):
        """Check whether to enabled aidi tracking."""

        enable_tracking = bool(
            int(os.environ.get("HAT_ENABLE_MODEL_TRACKING", "0"))
        )

        try:
            enabled = cls.client.experiment.enabled
        except Exception as e:
            logger.warning(
                f"AIDI Experiment: Get aidi experiment.enabled failed: {str(e)}",  # noqa E501
            )
            enabled = False
        return enabled and enable_tracking

    @classmethod
    def log_config(cls, config: Dict):
        """Log config to AIDIExperiment.

        Args:
            config: Config info
        """
        try:
            if cls.enabled_tracking():
                cls.client.experiment.log_config(config)
        except Exception as e:
            logger.warning(f"failed to log_config: {str(e)}")

    def load_json(self, json_file: str):
        assert os.path.exists(json_file), f"File {json_file} does not exits."

        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    @classmethod
    def init_group(cls, group: str):
        """Init AIDIExperiment group.

        Args:
            group: Group name.
        """
        try:
            if cls.enabled_tracking():
                cls.client.experiment.init_group(group)
        except Exception as e:
            logger.warning(f"failed to init_group: {str(e)}")


class BasicScriptHandler(object):
    def parse_input_args(self, script_args, **kwargs):
        """Parse script args from command line.

        Args:
            script_args: Input script args.

        Returns:
            script_args: Parsed args.
        """
        return script_args

    def log_to_aidi_experiment(
        self,
        output_dir: str,
        artifact_name: str,
        artifact_type: str = None,
        artifact_aliases: List[str] = None,
        artifact_tags: List[str] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """Log data and artifact to AIDIExperiment.

        Args:
            output_dir: Directory path of json file.
            artifact_name: Artifact name.
            artifact_type: Artifact type.
            artifact_aliases: Artifact aliases.
            artifact_tags: Artifact tags.
            overwrite:  Whether to overwrite artifact. Defaults to False.
        """
        # log table, plot, config
        self.log_info_from_json(output_dir)

        # log artifact
        self.log_artifact_from_json(
            output_dir,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            artifact_aliases=artifact_aliases,
            artifact_tags=artifact_tags,
            overwrite=overwrite,
        )

    def log_info_from_json(self, output_dir):
        """Read json data and log data to AIDIExperiment.

        Args:
            output_dir: Directory path of json file.
        """
        pass

    def log_artifact_from_json(
        self,
        output_dir: str,
        artifact_name: str,
        artifact_type: str,
        artifact_aliases: List[str],
        artifact_tags: List[str],
        files: Optional[List[str]] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """Read json data and log artifact to AIDIExperiment.

        Args:
            output_dir: Directory path of json file.
            artifact_name: Artifact name.
            artifact_type: Artifact type.
            artifact_aliases: Artifact aliases.
            artifact_tags: Artifact tags.
            files: Files to add to artifact. Defaults to None.
            overwrite:  Whether to overwrite artifact. Defaults to False.
        """
        pass

    def init_group(self, group_name: Optional[str] = None):
        if group_name:
            self.exp_logger.init_group(group_name)

    @property
    def exp_logger(self) -> AIDIExperimentLogger:
        return AIDIExperimentLogger()


class CompileStandaloneScriptHandler(BasicScriptHandler):
    def parse_input_args(
        self,
        script_args,
        enable_tracking,
        input_artifact_key: str = None,
        **kwargs,
    ):

        script_args_list = list(
            filter(lambda x: len(x) > 0, script_args.split(" "))
        )
        in_file = script_args_list[0]

        if in_file.startswith("aidi_artifact://"):
            (
                _,
                _,
                model_name,
                training_stage,
                entry_alias,
                ckpt_name,
            ) = in_file.split(os.sep)
            input_artifact_name = (
                f"{model_name}-{training_stage}:{entry_alias}"
            )

            input_artifact = client.experiment.use_artifact(
                input_artifact_name, trace=enable_tracking
            )
            key = (
                ckpt_name if input_artifact_key is None else input_artifact_key
            )  # noqa E501
            in_file_path = input_artifact.get_file(key)
            script_args_list[0] = in_file_path

        if enable_tracking and "--dump-cmp-info" not in script_args_list:
            script_args_list.append("--dump-cmp-info")

        return " ".join(script_args_list)

    def log_info_from_json(self, output_dir):
        if output_dir is None:
            output_dir = "tmp_compile"

        compile_info = self.exp_logger.load_json(
            json_file=os.path.join(
                output_dir,
                "_compile_standalone_info.json",
            ),
        )

        self.exp_logger.log_config(
            {
                "env": compile_info["env"],
                "compile_params": compile_info["params"],
            }
        )

        out_files = compile_info["out_files"]

        if "perf_json" in out_files:
            perf_json = out_files["perf_json"]
            assert os.path.exists(perf_json)
            perf_result = json.load(open(perf_json, "r"))
            self._log_perf_result(perf_result)

            if bool(compile_info["params"]["debug"]):
                self._log_perf_details(perf_result)

            self.exp_logger.client.experiment.log_summary(
                perf_result["summary"]
            )

    def log_artifact_from_json(
        self,
        output_dir: str,
        artifact_name: str,
        artifact_type: str = None,
        artifact_aliases: List[str] = None,
        artifact_tags: List[str] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        if artifact_aliases is None:
            artifact_aliases = ["latest"]
        if artifact_tags is None:
            artifact_tags = ["latest"]
        if artifact_type is None:
            artifact_type = "compile"
        if output_dir is None:
            output_dir = "tmp_compile"

        compile_info = self.exp_logger.load_json(
            json_file=os.path.join(
                output_dir, "_compile_standalone_info.json"
            ),
        )

        out_files = compile_info["out_files"]

        self.exp_logger.log_artifact(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            artifact_aliases=artifact_aliases,
            artifact_tags=artifact_tags,
            files=list(out_files.values()),
            overwrite=overwrite,
        )

    def _log_perf_result(self, perf_result):
        compute_utilization = perf_result["summary"].pop(
            "interval computing unit utilization"
        )
        perf_result["summary"].pop("interval computing units utilization")
        load_bandwidth = perf_result["summary"].pop(
            "interval loading bandwidth (megabytes/s)"
        )
        store_bandwidth = perf_result["summary"].pop(
            "interval storing bandwidth (megabytes/s)"
        )

        iterval_number = int(perf_result["summary"]["interval number"])

        bar_table0 = Table(
            name="loading_bandwidth",
            columns=[
                "class",
                "loading_bandwidth",
            ],
            data=[
                {
                    "class": index,
                    "loading_bandwidth": int(load_bandwidth[index]),
                }
                for index in range(iterval_number)
            ],
        )
        self.exp_logger.client.experiment.log_plot(
            "loading_bandwidth",
            bar_table0,
            Bar(x="class", y="loading_bandwidth"),
        )

        bar_table1 = Table(
            name="storing_bandwidth",
            columns=["class", "storing_bandwidth"],
            data=[
                {
                    "class": index,
                    "storing_bandwidth": int(store_bandwidth[index]),
                }
                for index in range(iterval_number)
            ],
        )
        self.exp_logger.client.experiment.log_plot(
            "storing_bandwidth",
            bar_table1,
            Bar(x="class", y="storing_bandwidth"),
        )

        line_table = Table(
            name="computing_utilization",
            columns=["class", "computing_utilization"],
            data=[
                {
                    "index": index,
                    "computing_utilization": float(compute_utilization[index]),
                }
                for index in range(iterval_number)
            ],
        )
        self.exp_logger.client.experiment.log_plot(
            "computing_utilization",
            line_table,
            Line(
                x="class",
                y="computing_utilization",
                stroke="computing_utilization",
            ),
        )

    def _log_perf_details(self, perf_result):
        layer_details = perf_result["summary"].pop("layer details")
        layer_details[0] = [
            ld.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")  # noqa E501
            for ld in layer_details[0]
        ]
        data = [
            {k: v for k, v in zip(layer_details[0], layer_details[index])}
            for index in range(1, len(layer_details))
        ]
        layer_detail_table = Table(
            name="layer_details",
            columns=layer_details[0],
            data=[
                {k: v for k, v in zip(layer_details[0], layer_details[index])}
                for index in range(1, len(layer_details))
            ],
        )
        self.exp_logger.client.experiment.log_table(layer_detail_table)

        iterval_number = int(perf_result["summary"]["interval number"])
        name_lists, util_lists = [], []
        detail_keys = [
            "dsu",
            "convolution",
            "transform",
        ]
        for key in detail_keys:
            full_key = f"interval {key} unit utilization"
            if full_key in perf_result["summary"]:
                util_lists.append(perf_result["summary"].pop(full_key))
                name_lists.append(f"{key}_utilization")

        for name, data in zip(name_lists, util_lists):
            line_table = Table(
                name=name,
                columns=["class", name],
                data=[
                    {
                        "index": index,
                        name: float(data[index]),
                    }
                    for index in range(iterval_number)
                ],
            )
            self.exp_logger.client.experiment.log_plot(
                name,
                line_table,
                Line(
                    x="class",
                    y=name,
                    stroke=name,
                ),
            )

    def init_group(self, group_name: Optional[str] = None):
        if group_name is None:
            group_name = "compile_task"
        return super().init_group(group_name)


class PackHbmScriptHandler(BasicScriptHandler):
    def parse_input_args(
        self,
        script_args,
        enable_tracking,
        input_artifact_key: str = None,
        **kwargs,
    ):

        args_list = script_args.split("--")
        for arg in args_list:
            if arg.startswith("input-hbm-list"):
                input_hbm_str = arg.replace("input-hbm-list ", "")
                input_hbm_list = list(
                    filter(lambda x: len(x) > 0, input_hbm_str.split(" "))
                )
                break

        input_lists = []
        for input_name in input_hbm_list:
            if input_name.startswith("aidi_artifact://"):
                (
                    _,
                    _,
                    model_name,
                    training_stage,
                    entry_alias,
                    ckpt_name,
                ) = input_name.split(os.sep)
                input_artifact_name = (
                    f"{model_name}-{training_stage}:{entry_alias}"  # noqa E501
                )

                input_artifact = client.experiment.use_artifact(
                    input_artifact_name, trace=enable_tracking
                )
                key = (
                    ckpt_name
                    if input_artifact_key is None
                    else input_artifact_key
                )  # noqa E501
                input_lists.append(input_artifact.get_file(key))
            else:
                input_lists.append(input_name)
        script_args = script_args.replace(
            input_hbm_str, " ".join(input_lists) + " "
        )

        if enable_tracking and "--dump-cmp-info" not in script_args:
            script_args += " --dump-cmp-info"

        return script_args

    def log_info_from_json(self, output_dir):
        if output_dir is None:
            output_dir = "tmp_compile"
        pack_info = self.exp_logger.load_json(
            json_file=os.path.join(output_dir, "_pack_hbm_info.json"),
        )
        self.exp_logger.log_config(pack_info["env"])

    def log_artifact_from_json(
        self,
        output_dir: str,
        artifact_name: str,
        artifact_type: str = None,
        artifact_aliases: List[str] = None,
        artifact_tags: List[str] = None,
        overwrite: bool = False,
    ):

        if artifact_aliases is None:
            artifact_aliases = ["latest"]
        if artifact_tags is None:
            artifact_tags = ["latest"]
        if artifact_type is None:
            artifact_type = "compile"
        if output_dir is None:
            output_dir = "tmp_compile"

        pack_info = self.exp_logger.load_json(
            json_file=os.path.join(output_dir, "_pack_hbm_info.json"),
        )
        out_files = pack_info["out_files"]

        output_files = []
        for _, v in out_files.items():
            if isinstance(v, str):
                output_files.append(v)
            elif isinstance(v, List):
                output_files += v

        self.exp_logger.log_artifact(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            artifact_aliases=artifact_aliases,
            artifact_tags=artifact_tags,
            files=output_files,
            overwrite=overwrite,
        )

    def init_group(self, group_name: Optional[str] = None):
        if group_name is None:
            group_name = "compile_task"
        return super().init_group(group_name)


SCRIPT_TO_HANDLER = {
    "train.py": BasicScriptHandler,
    "predict.py": BasicScriptHandler,
    "compile_standalone.py": CompileStandaloneScriptHandler,
    "pack_hbm.py": PackHbmScriptHandler,
}


class ScriptRunner:
    def __init__(self, script) -> None:

        self.script = script
        self.handler = self._get_handler()

    def _get_handler(self):
        """Get script handler according script name."""
        script_name = self.script.split("/")[-1]
        obj = SCRIPT_TO_HANDLER.get(script_name, None)

        if obj is None:
            raise ValueError(f"Failed to find handler for {script_name}")
        return obj()

    @classmethod
    def run(
        cls,
        script: str,
        script_args: Union[str, List[str]],
        enable_tracking: bool = False,
        group_name: str = None,
        artifact_name: str = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """Execute script and log data to AIDIExperiment.

        Args:
            script: Input script.
            script_args: Parsed args.
            enable_tracking: Whether to enable log to AIDIExperiment.
            group_name: Group name for AIDIExperiment. Defaults to None.
            artifact_name: Artifact name. Defaults to None.
            overwrite: Whether to overwrite artifact. Defaults to False.
        """

        handler = cls(script).handler

        if enable_tracking:
            handler.init_group(group_name)

        if isinstance(script_args, List):
            script_args = " ".join(script_args)

        cmd = f"python3 -W ignore {script} {script_args}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)

        if enable_tracking:
            handler.log_to_aidi_experiment(
                output_dir=find_args(script_args, "--output"),
                artifact_name=artifact_name,
                overwrite=overwrite,
                **kwargs,
            )

    @classmethod
    def parse_script_args(
        cls,
        script: str,
        script_args: str,
        **kwargs,
    ):
        """Parse script args from command line.

        Args:
            script: Input script.
            script_args: Input script args.

        Returns:
            script_args: Parsed args.
        """
        return cls(script).handler.parse_input_args(
            script_args=script_args,
            **kwargs,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=False,
        default=None,
        help="experiment name used in AIDI MLOPs Experiment system",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        type=str,
        required=False,
        help="Run name in AIDI MLOPs Exmeriment",
    )
    parser.add_argument(
        "--group-name",
        default=None,
        type=str,
        required=False,
        help="Group name in Run",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=False,
        default=None,
        help="project_id.",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        default="",
        help="bucket for saving in AIDI MLOPs",
    )

    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=False,
        help="export HAT_ENABLE_MODEL_TRACKING=1, which enable aidi tracking",
    )

    parser.add_argument(
        "--input-artifact-name",
        type=str,
        default=None,
        help="experiment name used in AIDI Experiment system",
    )
    parser.add_argument(
        "--output-artifact-name",
        type=str,
        default=None,
        help="experiment name used in AIDI Experiment system",
    )
    parser.add_argument(
        "--overwrite-file",
        action="store_true",
        help="whether to overwrite previous file, used in `compile_standalone` and `pack_hbm` script",  # noqa E501
    )

    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="HAT tools script to run",
    )
    parser.add_argument(
        "--script-args",
        type=str,
        required=True,
        help="arguments to the `script`",
    )

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    client = AIDIClient()

    try:
        args, unknow_args = parse_args()

        script_args = args.script_args
        if "--config" in script_args:
            cfg_file = find_args(script_args, key="--config")
        elif "-c" in script_args:
            cfg_file = find_args(script_args, key="-c")
        else:
            cfg_file = None

        # set env
        os.environ["HAT_ENABLE_MODEL_TRACKING"] = str(
            int(args.enable_tracking)
        )

        if IS_LOCAL and args.experiment_name:
            if client.experiment.get_experiment(args.experiment_name) is None:
                client.experiment.create_experiment(
                    name=args.experiment_name,
                    project_id=args.project_id,
                    experiment_path=args.experiment_path,
                )
            with client.experiment.init(
                experiment_name=args.experiment_name,
                run_name=args.run_name,
                enabled=args.enable_tracking,
            ) as run:
                run.log_runtime(runtime="local", config_file=cfg_file)

        script_runner = ScriptRunner(args.script)
        script_args = script_runner.parse_script_args(
            script=args.script,
            script_args=script_args,
            enable_tracking=args.enable_tracking,
            input_artifact_key=args.input_artifact_name,
        )

        script_runner.run(
            script=args.script,
            script_args=script_args,
            enable_tracking=args.enable_tracking,
            group_name=args.group_name,
            artifact_name=args.output_artifact_name,
            overwrite=args.overwrite_file,
        )

    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        raise e

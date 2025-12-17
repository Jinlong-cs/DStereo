import argparse
import logging
import os
import shutil
import subprocess
import zipfile

from file_helper import upload_to_gallery
from get_result import get_infer_res
from trace_compile_model import trace_and_compile

from hat.utils.config import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

HAT_BUCKET_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "/horizon-bucket/HDLTAlgorithm"
)

UNZIP_ROOT = "tmp_compile_model"
ZIP_FILE_PATH = "compile_model.zip"

multitask_yuv_data = {
    "1x3x640x1024": f"{HAT_BUCKET_PATH}/users/tian.li/consistency/as33_day/resize_2/16715.yuv",  # noqa
    "1x3x320x512": f"{HAT_BUCKET_PATH}/users/tian.li/consistency/as33_day/resize_4/10257.yuv",  # noqa
    "1x3x640x960": f"{HAT_BUCKET_PATH}/users/tian.li/consistency/cc02_x3c_day/resize_2/1890.yuv",  # noqa
    "1x3x320x480": f"{HAT_BUCKET_PATH}/users/tian.li/consistency/cc02_x3c_day/resize_4/1965.yuv",  # noqa
    "1x3x192x512": f"{HAT_BUCKET_PATH}/users/tian.li/consistency/cc02_x3c_day/crop/1682.yuv",  # noqa
}

iqa_yuv_data = {
    "1x3x320x512": f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency/512x320/1490521.yuv",  # noqa
    "1x3x320x480": f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency/480x320/1270505.yuv",  # noqa
    "1x3x256x480": f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency/480x256/1270604.yuv",  # noqa
}

PILOT_LTC_ENTRY_TYPE_MAP = {
    "as33": "dev-pilot3-model-verifier",
    "c385_x3c": "dev-pilot3-model-verifier",
    "c385_x3c_parking": "dev-pilot3-model-verifier",
    "cc02_x3c": "dev-pilot3-model-verifier",
    "ev52_x3c": "dev-pilot3-model-verifier",
    "galaxy_x3c_pe": "dev-pilot5-model-verifier",
    "niofy_CN_x3c_pe": "dev-pilot5-6v-model-verifier",
}

PILOT_LTC_PROJECT_TYPE_MAP = {
    "as33": "as33",
    "c385_x3c": "c385_x3c",
    "c385_x3c_parking": "c385_x3c_parking",
    "cc02_x3c": "cc02_x3c",
    "ev52_x3c": "ev52_x3c",
    "galaxy_x3c_pe": "pilot5",
    "niofy_CN_x3c_pe": "pilot5_6v",
}

PILOT_LTC_MODEL_TYPE_MAP = {
    "as33": "default",
    "c385_x3c": "default",
    "c385_x3c_parking": "default",
    "cc02_x3c": "default",
    "ev52_x3c": "default",
    "galaxy_x3c_pe": "adas_iqa",
    "niofy_CN_x3c_pe": "s5v",
}


def hbdk_verify(
    sub_project,
    compile_model_url,
):
    if os.path.exists(UNZIP_ROOT):
        shutil.rmtree(UNZIP_ROOT)
    os.mkdir(UNZIP_ROOT)

    pub_cfg = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__),
            f"cfg/pub_cfg_{sub_project}.py",
        ),
    )

    # get yuv data from model cfg
    model_data_dict = {}
    for name, cfg_i in pub_cfg["models"].items():
        shape = cfg_i["input_shape"]
        yuv_data = (
            multitask_yuv_data if "iqa_parsing" not in name else iqa_yuv_data
        )
        assert (
            shape in yuv_data
        ), f"The yuv_data with shape:{shape} is missing!"
        args = cfg_i.get("extra_args", "").split(" ")
        try:
            pyramid_stride = args[args.index("--pyramid-stride") + 1]
        except ValueError:
            pyramid_stride = None

        info = dict(  # noqa
            yuv_data=yuv_data[shape],
            yuv_shape=shape,
            pyramid_stride=pyramid_stride,
        )
        model_data_dict[name + ".hbm"] = info

    opt_level = pub_cfg["optimization_level"]

    # download model
    download_cmd = f"wget -c -O {ZIP_FILE_PATH} {compile_model_url}"
    subprocess.check_call(download_cmd, shell=True)

    # unzip & save model hbm file
    z = zipfile.ZipFile(ZIP_FILE_PATH, "r")
    for i in z.namelist():
        dir = os.path.dirname(i)
        file = os.path.basename(i)
        if file in model_data_dict:
            if os.path.basename(dir) == "so":
                continue
            task_name = file.strip(".hbm")
            hbm_path = os.path.join(UNZIP_ROOT, file)
            pt_path = hbm_path.replace(".hbm", ".pt")
            pt_zip_path = os.path.join(dir, f"model_opt_{opt_level}.pt")
            with open(hbm_path, "wb") as w:
                w.write(z.read(i))
            try:
                with open(pt_path, "wb") as w:
                    w.write(z.read(pt_zip_path))
            except Exception as e:
                logger.error(e)
                raise ValueError(
                    "Please check zip file, .hbm and .pt should be in the same dir"  # noqa
                )
            info = model_data_dict.pop(file)
            model_data_dict[task_name] = dict(  # noqa
                hbm_file=hbm_path,
                yuv_data=info["yuv_data"],
                yuv_shape=info["yuv_shape"],
                pyramid_stride=info["pyramid_stride"],
                pt_file=pt_path,
            )
    for task, info in model_data_dict.items():
        logger.info(
            f"----------------- hbdk model verify task: {task} ---------------"
        )
        ps = info["pyramid_stride"]
        hbdk_verify_cmd = f"""
        hbdk-model-verifier --hbm {info['hbm_file']} \
            --model-pt {info['pt_file']} \
            --yuv-shape {'x'.join(info['yuv_shape'].split('x')[2:])} \
            --model-input {info['yuv_data']} --image-stride {ps} --skip-bpu
        """
        logger.info(hbdk_verify_cmd)
        subprocess.check_call(hbdk_verify_cmd, shell=True)
        logger.info(
            "------------------- hbdk model verify task done ----------------"
        )


def check_consistency(
    sub_project,
    pub_version,
    pub_model_url,
    infer_result,
    entry_type="dev-pilot3-model-verifier",
    num_retry=1,
):
    assert isinstance(num_retry, int)
    assert isinstance(pub_model_url, str) and isinstance(infer_result, str)

    project = PILOT_LTC_PROJECT_TYPE_MAP[sub_project]
    model_type = PILOT_LTC_MODEL_TYPE_MAP[sub_project]
    trigger_script = os.path.join(
        os.path.dirname(__file__), "check_consistency.sh"
    )

    _cmd = f"""
    set -e
    sh {trigger_script} {entry_type} {project} {model_type} \
    {pub_version} {pub_model_url} {infer_result}

    """

    logger.info(_cmd)
    result = subprocess.check_output(_cmd, shell=True)

    job_id, success = str(result, encoding="utf-8").strip().split(":")
    if success == "false":
        logger.info(" Consistency check NOT pass ")
        raise ValueError
    elif success == "invalid":
        logger.info(" Jenkins job unkown error ")
        if num_retry > 0:
            logger.info(" Try to resubmit Jenkins job ")
            check_consistency(
                sub_project,
                pub_version,
                pub_model_url,
                infer_result,
                entry_type,
                num_retry=num_retry - 1,
            )
        else:
            raise ValueError
    else:
        logger.info(" Consistency check pass ")

    jenkins_console_url = "https://haihui.chen:1111a9c180388f1f1d18a976424497a5d0@ci.hobot.cc/view/QA/job/QA-Auto/job/QA02_DEV/job/dev-{}-model-verifier/{}/console"  # noqa

    logger.info(
        f"Please check url for more information: {jenkins_console_url.format(entry_type, job_id)}"  # noqa
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub-project",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--publish-version",
        type=str,
        required=False,
        default=None,
        help="if specify version, will overwrite model_version in cfg",
    )
    parser.add_argument(
        "--pub-model-url",
        required=False,
        default=None,
        help="if specify compile model url, DONT do compile again",
    )
    parser.add_argument(
        "--infer-result-path",
        required=False,
        default=None,
        help="if specify the path of infer result file, DONT do infer again",
    )
    parser.add_argument(
        "--hbdk-verify-perf",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    sub_project = args.sub_project
    publish_version = args.publish_version

    # Step 1: model compile and get pub_url
    pub_model_url = args.pub_model_url
    if pub_model_url is None:
        pub_model_url = trace_and_compile(
            sub_project=sub_project,
            publish_version=publish_version,
            compile_mode="aidi",
            overwrite=True,
            return_compile_url=True,
            release=True,
        )

    pub_model_url = (
        pub_model_url
        if isinstance(pub_model_url, str)
        else pub_model_url.get()
    )

    # Step 2: get infer results for consistency check
    infer_result = args.infer_result_path
    if infer_result is None:
        infer_result_local = get_infer_res(sub_project, publish_version)
        assert ".tar.gz" in infer_result_local, "Unknown format"
        base_name = os.path.basename(infer_result_local).replace(".tar.gz", "")
        infer_result = upload_to_gallery(
            infer_result_local,
            base_name,
            "auto.pilot_algo.publish",
            "consistency_results",
            username="pilot.runner",
            password="!Kj70Ic7feMh%ER$",
        )

    # Step 3: hbdk model verify
    if args.hbdk_verify_perf:
        hbdk_verify(sub_project, pub_model_url)

    # Step 4: check algo & backfill consistency
    check_consistency(
        sub_project="as33" if sub_project == "test" else sub_project,
        pub_version=publish_version,
        pub_model_url=pub_model_url,
        infer_result=infer_result,
        entry_type=PILOT_LTC_ENTRY_TYPE_MAP[sub_project],
        num_retry=2,
    )

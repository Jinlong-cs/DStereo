import getpass
import logging
import multiprocessing
import os
import re
import shutil
import traceback
import urllib
import zipfile
from typing import List

from aidisdk import AIDIClient
from aidisdk.eval.api_define import EvalReportTypeEnum
from hatbc.filestream.bucket import BucketClient
from hdflow.eval import AIDIEVSEval
from hdflow.eval.report import (
    generate_diff_excel,
    generate_issue_diff_report,
    is_highlight_func,
    obs_filter_func,
)
from hdflow.fillback import AIDIFillbackPredict, AIDIIssueRegressionEval
from hdflow.fillback.utils import unzip_function
from hdflow.utils.io import CacheFile

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = AIDIClient(endpoint="http://aidi.hobot.cc")


class Dispatcher:
    def __init__(self):
        self.__map = {}

    def __getitem__(self, key):
        return self.__map[key]

    def register(self, key, cls):
        self.__map[key] = cls


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        def error(msg, *args):
            return multiprocessing.get_logger().error(msg, *args)

        try:
            result = self.__callable(*args, **kwargs)

        except Exception:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up

            raise

        # It was fine, give a normal answer
        return result

    pass


def load_zip_publish_model(
    update_hbm: str, hbm_types: List[str], cachefile_save_dir: str
):
    cache_file = CacheFile(url=update_hbm)
    cachefile_path = cache_file.name
    unzip_function(cachefile_path, cachefile_save_dir)
    cachefile_save_files = os.listdir(cachefile_save_dir)
    update_hbm_file = []
    for hbm_type in hbm_types:
        update_hbm_file.append(
            os.path.join(
                cachefile_save_dir,
                cachefile_save_files[0],
                hbm_type,
            )
        )
    return update_hbm_file


def load_aidi_publish_model(path: str, save_path: str = "./"):
    publish_name, version = path.split(":")
    compile_task = list(
        client.modelpublish.get_compile_task_list(publish_name, version)
    )
    if len(compile_task) > 1:
        logger.warning(
            f"Expect len(compile_task) == 1, "
            f"but len(compile_task)={len(compile_task)}!"
            f"Using first compile task result!"
        )
    hbm_pack_zip = compile_task[0].download_result()
    zip_rel_path = "so/model.hbm"
    with zipfile.ZipFile(hbm_pack_zip, "r") as z:
        z.extract(zip_rel_path, path=save_path)
    return os.path.join(save_path, zip_rel_path)


def download_eval_report_by_job_id(id: int, save_path: str = "./"):
    # TODO: aidisdk support such func
    # get report url
    eval_ret = client.eval_entry.query_eval_job(id)
    report_url = eval_ret.result
    parsed_url = urllib.parse.urlparse(report_url)
    match = re.search(r"dutkey=([^&]+)", parsed_url.fragment)
    if match:
        dutkey = match.group(1)
        print(dutkey)
    else:
        raise AssertionError(
            f"Job id report url: {report_url}, dutkey NOT found!"
        )
    # download report
    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    client.eval.download_excel_report(
        dutkeys=[dutkey],
        output_dir=tmp_dir,
        report_type=EvalReportTypeEnum.MERGE,
    )
    xlsx_file = os.listdir(tmp_dir)[0]
    assert xlsx_file.endswith(".xlsx"), "Download eval report xlsx ERROR!"
    ret = save_path
    if not save_path.endswith(".xlsx"):
        ret = os.path.join(save_path, xlsx_file)
    shutil.copy(os.path.join(tmp_dir, xlsx_file), ret)
    shutil.rmtree(tmp_dir)
    return ret


logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = AIDIClient(endpoint="http://aidi.hobot.cc")
bucket_client = BucketClient()

EVS_XLSX_DMP_ROOT = (
    f"dmpv2://matrix2/users/{getpass.getuser()}/fillback_eval_xlsx"
)
ISSUE_XLSX_DMP_ROOT = f"dmpv2://matrix2/users/{getpass.getuser()}/issue_xlsx"


def evs_report_diff(
    name,
    result,
    dataset,
    save_dir,
):
    # TODO: Use the uniform version identifier to
    # get the comparison report
    eval_job_compared = dataset.meta["eval_job_compared"]
    report = download_eval_report_by_job_id(result.job_id)
    report_compared = download_eval_report_by_job_id(eval_job_compared)
    ret = generate_diff_excel(
        report,
        report_compared,
        filter_func=obs_filter_func,
        is_highlight_func=is_highlight_func,
        save_path=save_dir,
    )
    # upload bucket and send to web
    dmp_dir = os.path.join(EVS_XLSX_DMP_ROOT, name)
    if not bucket_client.exists(dmp_dir):
        bucket_client.mkdir(dmp_dir, recursive=True)
    dmp_path = os.path.join(dmp_dir, os.path.basename(ret))
    bucket_client.upload(
        local_file=ret,
        url=dmp_path,
    )
    logger.info(
        f"Fillback eval report cluster output: "
        f"{os.path.join(save_dir, os.path.basename(ret))}, "
        f"bucket url: {dmp_path}"
    )


def issue_report_diff(
    name,
    result,
    dataset,
    save_dir,
):
    # TODO: Use the uniform version identifier to
    # get the comparison report
    issue_job = result["id"]
    issue_job_compared = dataset.meta["issue_job_compared"]
    ret = generate_issue_diff_report(
        issue_job,
        issue_job_compared,
        save_path=save_dir,
    )
    # upload bucket and send to web
    dmp_dir = os.path.join(ISSUE_XLSX_DMP_ROOT, name)
    if not bucket_client.exists(dmp_dir):
        bucket_client.mkdir(dmp_dir, recursive=True)
    dmp_path = os.path.join(dmp_dir, os.path.basename(ret))
    bucket_client.upload(
        local_file=ret,
        url=dmp_path,
    )
    logger.info(
        f"Issue regression report cluster output: "
        f"{os.path.join(save_dir, os.path.basename(ret))}, "
        f"bucket url: {dmp_path}"
    )


predictor_dispatch = Dispatcher()
evaluator_dispatch = Dispatcher()
reporter_dispatch = Dispatcher()
predictor_dispatch.register("evs", AIDIFillbackPredict)
evaluator_dispatch.register("evs", AIDIEVSEval)
reporter_dispatch.register("evs", evs_report_diff)
predictor_dispatch.register("issue", AIDIIssueRegressionEval)
evaluator_dispatch.register("issue", None)
reporter_dispatch.register("issue", issue_report_diff)

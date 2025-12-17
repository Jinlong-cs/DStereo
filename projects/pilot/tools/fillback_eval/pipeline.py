import argparse
import logging
import multiprocessing
import os
import uuid
from multiprocessing.pool import ThreadPool
from tempfile import TemporaryDirectory

from aidisdk import AIDIClient
from aidisdk.pkgmanager.api_define import (
    LevelEnum,
    LevelUsrEnum,
    Members,
    SceneEnum,
    StorageLocationTypeEnum,
    VersionStatusEnum,
    VersionTypeEnum,
)
from hatbc.filestream.bucket import BucketClient
from hdflow.fillback.utils import update_zip_package
from hdflow.utils import url_to_local_path
from pipeline_utils import (
    LogExceptions,
    evaluator_dispatch,
    load_aidi_publish_model,
    load_zip_publish_model,
    predictor_dispatch,
    reporter_dispatch,
)

from hat.utils import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = AIDIClient(endpoint="http://aidi.hobot.cc")
bucket_client = BucketClient()

FILLBACK_EVAL_CONFIGS_ROOT = (
    f"{os.path.dirname(os.path.abspath(__file__))}/configs"
)
FILLBACK_EVAL_DATASETS_ROOT = (
    f"{os.path.dirname(os.path.abspath(__file__))}/datasets"
)


def dataset_processing(
    predictor,
    evaluator,
    reporter,
    name,
    task,
    dataset,
    app,
    queue,
    num_worker,
    disable_eval=False,
    enable_report_diff=False,
):
    # TODO: Support skip predict
    predict_job_name = "pred_" + name
    assert (
        len(predict_job_name) < 50
    ), f"The fillback job name {predict_job_name} is too long!"
    ret = predictor(
        dataset=dataset,
        name=predict_job_name,
        app=app,
        queue=queue,
        num_worker=num_worker,
    )
    if not disable_eval:
        eval_job_name = "eval_" + name
        assert (
            len(eval_job_name) < 50
        ), f"The eval job name {eval_job_name} is too long!"
        if evaluator:
            ret = evaluator(
                task=task,
                gt=dataset.gt,
                predict=ret.job_id,
                setting=task,
                name=eval_job_name,
            )
        if enable_report_diff:
            if not os.path.exists("/running_package"):
                save_dir = f"tmp_output/{name}"
            else:
                save_dir = f"/job_data/{name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ret = reporter(
                name,
                ret,
                dataset,
                save_dir,
            )

    return ret


def main(
    config,
    datasets,
    version,
    fillback_type,
    app=None,
    update_hbm=None,
    fillback_cluster=None,
    disable_eval=False,
    enable_report_diff=False,
):
    cfg = Config.fromfile(config)
    datasets = Config.fromfile(datasets).dataset_ids
    version = version if version else cfg.get("version")
    predictor = predictor_dispatch[fillback_type](
        config=getattr(cfg, fillback_type, None),
        max_num_fail=5,
        dry_run_return_param=False,
    )
    evaluator = None
    if evaluator_dispatch[fillback_type]:
        evaluator = evaluator_dispatch[fillback_type](
            config=getattr(cfg, fillback_type + "_eval", None),
            dry_run_return_param=False,
        )
    reporter = reporter_dispatch[fillback_type]
    # get hbm file
    update_hbm = (
        cfg.get("update_hbm", None) if update_hbm is None else update_hbm
    )
    if update_hbm is not None:

        if update_hbm.endswith(".hbm"):
            update_hbm_file = url_to_local_path(update_hbm)
        elif update_hbm.endswith(".zip"):
            assert cfg.get("hbm_types"), "Please offer the hbm types!!!"
            temp_dir = TemporaryDirectory(
                suffix=os.path.basename(os.path.splitext(update_hbm)[0])
            )
            update_hbm_file = load_zip_publish_model(
                update_hbm, cfg.get("hbm_types"), temp_dir.name
            )
        else:
            # hbm on aidi publish model
            update_hbm_file = load_aidi_publish_model(update_hbm)

    app = cfg.app if app is None else app
    app_iter = client.pkgmanager.search_packages(pkg_name=app)
    for item in app_iter:
        if item.pkg_name == app:
            app_address = item.gpfs_address
            break
    if update_hbm is not None:
        # replace hbm in pkg
        logger.info(f"Start replace hbm, app: {app}, hbm: {update_hbm}")
        update_pkg = update_zip_package(
            src=url_to_local_path(app_address),
            replace_arc_paths=cfg.get("app_update_arc_path"),
            update_files=update_hbm_file,
            save_path="./",
        )
        base_name = app.replace(".zip", "")
        new_app_name = (
            base_name + f"_replace_hbm_{cfg.timestamp}_{str(uuid.uuid4())[:7]}"
        )
        upload_pkg = client.pkgmanager.upload_packages(
            file=update_pkg,
            version="v1.0.0",
            ipd_id=cfg.ipd_number,
            level=LevelEnum.APPLICATION,
            level_usr=LevelUsrEnum.PUBLIC,
            members=Members(department=["auto"]),
            scene=SceneEnum.ADAS,
            version_status=VersionStatusEnum.NOTSTARTED,
            version_type=VersionTypeEnum.DEVELOP,
            storage_location=StorageLocationTypeEnum.OSS,
            pkg_name=new_app_name,
            file_name=new_app_name + ".zip",
        )
        logger.info(
            f"Upload pkg successful! pkg name: "
            f"{new_app_name}, pkg id: {upload_pkg.package_id}"
        )
        cur_app = new_app_name
    else:
        cur_app = app
    if fillback_type == "evs":
        # covert to app bucket url
        cur_app = list(client.pkgmanager.search_packages(pkg_name=cur_app))[
            0
        ].gpfs_address
    if cfg.num_parallel_job > 1:
        pool = ThreadPool(cfg.num_parallel_job)
        multiprocessing.log_to_stderr()
    ret_list = []
    for task, item in datasets.items():
        for dataset in item:
            if cfg.num_parallel_job > 1:
                ret = pool.apply_async(
                    LogExceptions(dataset_processing),
                    (
                        predictor,
                        evaluator,
                        reporter,
                        "_".join(
                            [
                                task,
                                str(dataset.desc),
                                version.replace(".", "_"),
                                str(uuid.uuid4())[:4],
                            ]
                        ),
                        task,
                        dataset,
                        cur_app,
                        cfg.task_queue
                        if fillback_cluster is None
                        else fillback_cluster,
                        cfg.num_worker,
                    ),
                    {
                        "disable_eval": disable_eval,
                        "enable_report_diff": enable_report_diff,
                    },
                )
            else:
                ret = dataset_processing(
                    predictor,
                    evaluator,
                    reporter,
                    "_".join(
                        [
                            task,
                            str(dataset.desc),
                            version.replace(".", "_"),
                            str(uuid.uuid4())[:4],
                        ]
                    ),
                    task,
                    dataset,
                    cur_app,
                    cfg.task_queue
                    if fillback_cluster is None
                    else fillback_cluster,
                    cfg.num_worker,
                    disable_eval=disable_eval,
                    enable_report_diff=enable_report_diff,
                )
            ret_list.append(ret)
    if cfg.num_parallel_job > 1:
        pool.close()
        return [ret.get() for ret in ret_list]
    else:
        return ret_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-project", type=str, required=True)
    parser.add_argument("--app", type=str, default=None)
    parser.add_argument(
        "--update-hbm",
        type=str,
        default=None,
    )
    parser.add_argument("--version", type=str, required=False, default=None)
    parser.add_argument(
        "--fillback-type", type=str, required=True, choices=["evs", "issue"]
    )
    parser.add_argument("--disable-eval", action="store_true", default=False)
    parser.add_argument(
        "--enable-report-diff", action="store_true", default=False
    )
    parser.add_argument(
        "--fillback-cluster", type=str, required=False, default=None
    )
    args = parser.parse_args()

    config = os.path.join(FILLBACK_EVAL_CONFIGS_ROOT, f"{args.sub_project}.py")
    datasets = os.path.join(
        FILLBACK_EVAL_DATASETS_ROOT,
        f"{args.sub_project}_{args.fillback_type}_datasets.py",
    )
    main(
        config=config,
        datasets=datasets,
        version=args.version,
        fillback_type=args.fillback_type,
        app=args.app,
        update_hbm=args.update_hbm,
        fillback_cluster=args.fillback_cluster,
        disable_eval=args.disable_eval,
        enable_report_diff=args.enable_report_diff,
    )

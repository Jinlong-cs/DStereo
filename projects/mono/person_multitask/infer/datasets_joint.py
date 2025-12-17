import os

from hatbc.filestream.bucket.client import get_gpfs_bucket_mount_root
from hpflow.eval_platform_adaptor import EvalPlatformJsonWritter
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import parse_datapaths

gpfs_root = get_gpfs_bucket_mount_root().get("auto_tmp")
dataset_root = f"{gpfs_root}/shu.zhang/cache_eval_dataset/cache_testsets_01"
os.makedirs(dataset_root, exist_ok=True)
cache_dir = os.path.join(os.path.expanduser("~"))


# results will save to:
# executor's savedir + workflow's env save_dir + dataset's save_dir


def download_4pe_predict_id(dataset_id, predict_id=None):
    from hatbc.adas_eval import EvaluationClient
    from tqdm import tqdm

    if predict_id is None:
        raise ValueError
    save_path = os.path.join(
        cache_dir, "joint_4pe", dataset_id, f"{predict_id}.json"
    )
    if os.path.exists(save_path):
        os.remove(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    client = EvaluationClient()
    client.auth.login_by_ldap()

    B_PER_MB = 1000 * 1000
    chunk_size = 20 * 20 * B_PER_MB
    temp_fn = f"{dataset_id}-{predict_id}"
    pbar = tqdm(desc=temp_fn, unit="MB")

    response = client.prediction.download(int(predict_id))

    with open(save_path, "wb") as fwrite:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fwrite.write(chunk)
            chunk_mb = len(chunk) / float(B_PER_MB)
            pbar.update(chunk_mb)
    pbar.close()

    return save_path


HPFLOW_JOINT4PE_PREDICT_ID = os.environ.get("HPFLOW_JOINT4PE_PREDICT_ID", None)
assert HPFLOW_JOINT4PE_PREDICT_ID is not None
pairs = HPFLOW_JOINT4PE_PREDICT_ID.split("|")
pairs = [
    {"dataset": i.split("-")[0], "projectid4pe": i.split("-")[1]}
    for i in pairs
]

datapaths = {
    # person posneg
    PTypeEnum.kPedPosNeg: [
        # 10652 joint eval
        dict(
            img_root=os.path.join(dataset_root, pair["dataset"]),
            anno_path=os.path.join(
                dataset_root,
                "joint_4pe",
                download_4pe_predict_id(pair["dataset"], pair["projectid4pe"]),
            ),
            dataset_id=pair["dataset"],
            save_dir=pair["dataset"],
            eval_platform_result_writter=EvalPlatformJsonWritter,
        )
        for pair in pairs
    ],
}

datasets = parse_datapaths(
    datapaths,
    download_from_evalplatform=True,
    use_bucket_gt_json=False,
    bucket_first=True,
    bucket_name="auto_eval",
)

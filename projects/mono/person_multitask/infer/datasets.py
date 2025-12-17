import argparse
import logging
import os

from hatbc.filestream.bucket.client import get_bucket_mount_root
from hpflow.eval_platform_adaptor import EvalPlatformJsonWritter
from hpflow.structure.ptype import PTypeEnum
from hpflow.utils import parse_datapaths

logger = logging.getLogger(__name__)


mount_root = get_bucket_mount_root().get("adas")
dataset_root = os.path.join(mount_root, "hui.xue/eval")  # noqa
os.makedirs(dataset_root, exist_ok=True)
# results will save to:
# executor's savedir + workflow's env save_dir + dataset's save_dir


def get_parser():
    parser = argparse.ArgumentParser(
        description="argument for dataset builder"
    )
    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument("--eval-dataset-task", default=None)
    args, argv = parser.parse_known_args()
    if argv:
        logger.warning(
            f"unrecognized arguments in dataset" f" parser : {' '.join(argv)}"
        )

    return args


args = get_parser()

datapaths_default = {
    # # person age
    PTypeEnum.kPedAge: [
        # 10652
        dict(
            img_root=os.path.join(dataset_root, "6027225"),
            anno_path=os.path.join(dataset_root, "6027225.json"),
            dataset_id="6027225",
            eval_platform_result_writter=EvalPlatformJsonWritter,
            save_dir="6027225",
        )
    ],
}

str2ptype = {
    "kPedHeadBBox2D": PTypeEnum.kPedHeadBBox2D,
    "kPedAge": PTypeEnum.kPedAge,
    "kPedPose": PTypeEnum.kPedPose,
    "kPedPosNeg": PTypeEnum.kPedPosNeg,
    "kPedOcclusion": PTypeEnum.kPedOcclusion,
    "kPedOrientation": PTypeEnum.kPedOrientation,
}

if args.eval_dataset_name is not None and args.eval_dataset_task is not None:
    datapaths = {
        str2ptype[args.eval_dataset_task]: [
            dict(
                img_root=os.path.join(
                    dataset_root, args.eval_dataset_name
                ),  # noqa
                anno_path=os.path.join(
                    dataset_root, f"{args.eval_dataset_name}.json"
                ),  # noqa
                dataset_id=args.eval_dataset_name,
                save_dir=args.eval_dataset_name,
                eval_platform_result_writter=EvalPlatformJsonWritter,
            ),
        ]
    }
else:
    datapaths = datapaths_default

datasets = parse_datapaths(
    datapaths,
    download_from_evalplatform=False,
    bucket_first=True,
    bucket_name="auto_eval",
)

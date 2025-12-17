from dataset_utils import get_default_obs_gt_info
from hdflow.fillback import FillbackEvalDataset

dataset_ids = dict(
    obs=[
        # 测试数据集
        FillbackEvalDataset(
            id=493744,
            flag="x03",
            desc="just_test",
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 8846},
        ),
    ]
)

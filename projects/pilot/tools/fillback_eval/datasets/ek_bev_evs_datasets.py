from dataset_utils import get_default_obs_gt_info
from hdflow.fillback import FillbackEvalDataset

dataset_ids = dict(
    obs=[
        # 样例
        FillbackEvalDataset(
            id=415564,
            desc="ek_byd",
            gt=get_default_obs_gt_info(),
        ),
    ]
)

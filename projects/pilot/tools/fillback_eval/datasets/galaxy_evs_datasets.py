from dataset_utils import get_default_obs_gt_info
from hdflow.fillback import FillbackEvalDataset

dataset_ids = dict(
    obs=[
        # # 异形车
        FillbackEvalDataset(
            id=512559,
            flag="x03",
            desc="special_car",
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 25568},
        ),
        # # 环境误检
        FillbackEvalDataset(
            id=512570,
            flag="x03",
            desc="env_fp",
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 25569},
        ),
        # 夜晚炫光
        FillbackEvalDataset(
            id=512569,
            flag="x02",
            desc="night_light",
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 25694},
        ),
        # # 3D弯道白天 done
        FillbackEvalDataset(
            id=536280,
            flag="x03",
            desc="3d_curve",
            # gt=get_default_obs_gt_info(odom_version="2.0.0", gt_version="3.0.0"),  # noqa
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 25565},
        ),
        # # 3D截断白天 done
        # FillbackEvalDataset(
        #     id=536313,
        #     flag="x03",
        #     gt=get_default_obs_gt_info(odom_version="2.0.0", gt_version="3.0.0"),  # noqa
        #     # meta={"eval_job_compared": 8848},
        # ),
        # # 3D截断夜晚 done
        # FillbackEvalDataset(
        #     id=555962,
        #     flag="x03",
        #     gt=get_default_obs_gt_info(odom_version="2.0.0", gt_version="3.0.0"),  # noqa
        #     # meta={"eval_job_compared": 8848},
        # ),
        # # 行人柱状物白天 done
        FillbackEvalDataset(
            id=556498,
            flag="x03",
            desc="ped_pillar_fp_day",
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 25581},
        ),
        # # 行人柱状物夜晚
        # FillbackEvalDataset(
        #     id=556494,
        #     flag="x03",
        #     gt=get_default_obs_gt_info(),
        #     # meta={"eval_job_compared": 8848},
        # ),
        # 行人车身误检
        FillbackEvalDataset(
            id=557428,
            flag="x03",
            desc="ped_veh_fp",
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 25579},
        ),
        # # 行人鬼影 done
        # FillbackEvalDataset(
        #     id=557314,
        #     flag="x02",
        #     gt=get_default_obs_gt_info(),
        #     # meta={"eval_job_compared": 8848},
        # ),
        # 行人车身误检 x02 漏刷
        FillbackEvalDataset(
            id=521705,
            flag="x02",
            desc="ped_veh_fp_x02",
            gt=get_default_obs_gt_info(),
            meta={"eval_job_compared": 25580},
        ),
        # # 近处大车
        # FillbackEvalDataset(
        #     id=557433,
        #     flag="x03",
        #     gt=get_default_obs_gt_info(),
        #     # meta={"eval_job_compared": 8848},
        # ),
    ]
)

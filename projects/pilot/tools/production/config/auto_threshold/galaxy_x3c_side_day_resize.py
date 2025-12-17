task_threshold = dict(
    # 4PE
    vehicle_detection=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6036259,
                setting_names=[
                    "all_region-det-veh-full_4cam_forward0m_70",
                ],
            ),
        ],
    ),
    rear_detection=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6036266,
                setting_names=[
                    "all_region-det-veh-rear_4cam_forward0m_70",
                ],
            ),
        ],
    ),
    person_detection=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6036190,
                setting_names=[
                    "all_region-det-ped_4cam_forward0m_60",
                ],
            ),
        ],
    ),
    cyclist_detection=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6036261,
                setting_names=[
                    "all_region-det-cyclist_4cam_forward0m_60",
                ],
            ),
        ],
    ),
    # 2PE
    # vehicle_wheel_detection=dict(
    #     threshold=dict(
    #         report_type="DetEvalReport",
    #         metric_type="ThresholdAtMaxDetRate",
    #         metric_kwargs=dict(),
    #     ),
    #     datasets=[
    #         dict(
    #             dataset_id=6029463,
    #             setting_names=[
    #                 "all_region-det-veh-wheel_5cam_forward0m_40",
    #             ],
    #         ),
    #     ],
    # ),
    rear_plate_detection=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6037603,
                setting_names=[
                    "plate_minH20_iou0.5_ig0.3",
                ],
            ),
        ],
    ),
    person_face_detection=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6037601,
                setting_names=[
                    "face_minH32_iou0.5_ig0.15",
                ],
            ),
        ],
    ),
    # 3D
    vehicle_roi_3d=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6036169,
                setting_names=[
                    "4cam_0-50m_IoU0.5.yaml",
                ],
            ),
        ],
    ),
    person_roi_3d=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6042391,
                setting_names=[
                    "4cam_0-50m_IoU0.1.yaml",
                ],
            ),
        ],
    ),
    cyclist_roi_3d=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6042392,
                setting_names=[
                    "4cam_0-50m_IoU0.1.yaml",
                ],
            ),
        ],
    ),
)

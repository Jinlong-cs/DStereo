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
                dataset_id=6036255,
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
                dataset_id=6036245,
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
                dataset_id=6036197,
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
                dataset_id=6036240,
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
    # rear_plate_detection=dict(
    #     threshold=dict(
    #         report_type="DetEvalReport",
    #         metric_type="ThresholdAtMaxDetRate",
    #         metric_kwargs=dict(),
    #     ),
    #     datasets=[
    #         dict(
    #             dataset_id=6028207,
    #             setting_names=[
    #                 "det-plate-5cam_minH50_IOU0.5",
    #             ],
    #         ),
    #     ],
    # ),
    # person_face_detection=dict(
    #     threshold=dict(
    #         report_type="DetEvalReport",
    #         metric_type="ThresholdAtMaxDetRate",
    #         metric_kwargs=dict(),
    #     ),
    #     datasets=[
    #         dict(
    #             dataset_id=6028696,
    #             setting_names=[
    #                 "det-face-5cam_minH32_IOU0.3",
    #             ],
    #         ),
    #     ],
    # ),
    # 3D
    vehicle_roi_3d=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            dict(
                dataset_id=6042394,
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
                dataset_id=6042390,
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
                dataset_id=6042388,
                setting_names=[
                    "4cam_0-50m_IoU0.1.yaml",
                ],
            ),
        ],
    ),
)

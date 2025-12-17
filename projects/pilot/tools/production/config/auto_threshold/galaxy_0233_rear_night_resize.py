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
                dataset_id=6040926,
                setting_names=[
                    "all_region-det-veh-full_pinhole_rear_forward0m_70",
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
                dataset_id=6036244,
                setting_names=[
                    "all_region-det-veh-rear_pinhole_rear_forward0m_70",
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
                dataset_id=6042156,
                setting_names=[
                    "all_region-det-ped_pinhole_rear_forward0m_50",
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
                dataset_id=6042157,
                setting_names=[
                    "all_region-det-cyclist_pinhole_rear_forward0m_50",
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
                dataset_id=6042396,
                setting_names=[
                    "1cam_0-50m_IoU0.5.yaml",
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
                dataset_id=6042358,
                setting_names=[
                    "1cam_0-50m_IoU0.1.yaml",
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
                dataset_id=6042362,
                setting_names=[
                    "1cam_0-50m_IoU0.1.yaml",
                ],
            ),
        ],
    ),
)

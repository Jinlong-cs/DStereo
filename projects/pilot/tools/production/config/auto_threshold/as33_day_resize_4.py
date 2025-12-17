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
                dataset_id=6028996,
                setting_names=[
                    "all_region-det-veh-full_5cam_forward0m_40",
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
                dataset_id=6028916,
                setting_names=[
                    "all_region-det-veh-rear_5cam_forward0m_40",
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
                dataset_id=6028981,
                setting_names=[
                    "all_region-det-ped_5cam_forward0m_20",
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
                dataset_id=6028984,
                setting_names=[
                    "all_region-det-cyclist_5cam_forward0m_20",
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
                dataset_id=6028207,
                setting_names=[
                    "det-plate-5cam_minH20_IOU0.5",
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
                dataset_id=6028696,
                setting_names=[
                    "det-face-5cam_minH32_IOU0.3",
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
                dataset_id=6036059,
                setting_names=[
                    "4cam_0-30m_IoU0.5.yaml",
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
                dataset_id=6029018,
                setting_names=[
                    "person_5cam_0-20m_IoU0.1_thresh0",
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
                dataset_id=6029021,
                setting_names=[
                    "cyclist_5cam_0-20m_IoU0.1_thresh0",
                ],
            ),
        ],
    ),
)

task_threshold = dict(
    # 4PE
    vehicle_detection=dict(
        threshold=dict(
            report_type="DetEvalReport",
            metric_type="ThresholdAtMaxDetRate",
            metric_kwargs=dict(),
        ),
        datasets=[
            # 常规
            dict(
                dataset_id=6029004,
                setting_names=[
                    "all_region-det-veh-full_5cam_forward0m_40",
                ],
            ),
            # 眩光
            dict(
                dataset_id=6036555,
                setting_names=[
                    "all_region-det-veh-full_5cam_car_night_light_forward0m_40",  # noqa
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
                dataset_id=6028918,
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
                dataset_id=6028983,
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
                dataset_id=6028985,
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
                    "det-plate-5cam_minH50_IOU0.5",
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
                dataset_id=6036055,
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
                dataset_id=6029020,
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
                dataset_id=6029019,
                setting_names=[
                    "cyclist_5cam_0-20m_IoU0.1_thresh0",
                ],
            ),
        ],
    ),
)

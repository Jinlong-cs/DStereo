def get_default_obs_gt_info(odom_version=None, gt_version=None):

    if odom_version:
        odom_version = f"{odom_version}-adas"

    return {
        "force_batch": 2,
        "odom_data_source": "vehicle_can",
        "odom_version": odom_version,
        "gt_data_source": "pandar_gt",
        "gt_version": gt_version,
        "gt_label_version": gt_version,
        "csv_prefix": "FUSION6V",
        "csv_suffix": "_5.pack",
    }

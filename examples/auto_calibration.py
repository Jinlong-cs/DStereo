from examples.classification.mobilenetv1 import *

calibration_trainer.update(
    dict(
        num_steps=10,
        auto_calibration=True,
        auto_calibration_config=dict(
            observer_list=["mse", "kl", "percentile"],
            percentile_list=[99.9, 99.99, 99.999],
            preload_data=True,
        ),
    )
)

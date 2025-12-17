from examples.classification.mobilenetv1 import *

val_callback["val_on_train_end"] = True

calibration_trainer.update(
    dict(
        weight_reconstruction=True,
        weight_reconstruction_config=dict(preload_data=True),
    )
)

# just for saving int_infer pth and pt
int_infer_trainer["model_convert_pipeline"]["converters"] = [
    dict(type="Float2QAT", convert_mode=convert_mode),
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "calibration-checkpoint-last.pth.tar"
        ),
    ),
    dict(type="QAT2Quantize", convert_mode=convert_mode),
]

calibration_predictor["model_convert_pipeline"]["converters"] = [
    dict(type="Float2QAT", convert_mode=convert_mode),
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "calibration-checkpoint-last.pth.tar"
        ),
        verbose=True,
    ),
]

int_infer_predictor["model_convert_pipeline"]["converters"] = [
    dict(type="Float2QAT", convert_mode=convert_mode),
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "calibration-checkpoint-last.pth.tar"
        ),
    ),
    dict(type="QAT2Quantize", convert_mode=convert_mode),
]

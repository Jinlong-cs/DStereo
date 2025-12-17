import os

pipeline_test = bool(int(os.environ.get("HAT_PIPELINE_TEST", False)))
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

global_desc = dict(
    roi_input=dict(fp_x=480, fp_y=256, width=960, height=512),
    vanishing_point=[480, 256],
)

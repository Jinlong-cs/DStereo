import os

from horizon_plugin_pytorch.quantization import March

# env variables
num_machines = int(os.getenv("HAT_NUM_MACHINES", "1"))
pipeline_test = os.getenv("HAT_PIPELINE_TEST", "0") == "1"
training_step = os.getenv("HAT_TRAINING_STEP", "int_infer")
enable_model_tracking = os.getenv("HAT_ENABLE_MODEL_TRACKING") == "1"

model_checkpoint = os.getenv("HAT_WK_MODEL_CHECKPOINT")
model_setting = os.getenv("HAT_WK_MODEL_SETTING", "cn_wk_v0.0.1")
model_name_postfix = os.getenv("HAT_WK_MODEL_NAME_POSTFIX", "fan.lv")
model_version = os.getenv("HAT_WK_MODEL_VERSION", "v0.0.1")
model_thresh = os.getenv("HAT_WK_MODEL_THRESH")
tasks = os.getenv("HAT_WK_TASKS")
resume_training = os.getenv("HAT_WK_RESUME_TRAINING", None)
march = os.getenv("HAT_TRAINING_MARCH", March.BAYES)

# personal setting
num_worker = os.getenv("HAT_WK_NUM_WORKER", 6)
batch_size_factor = os.getenv("HAT_WK_BATCHSIZE_FACTOR", 16)
input_size = os.getenv("INPUT_SIZE", (234, 456))
pred_batch_size = os.getenv("HAT_WK_PRED_BATCHSIZE", 96)

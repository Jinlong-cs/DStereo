import os

# env variables
num_machines = int(os.getenv("HAT_NUM_MACHINES", "1"))
pipeline_test = os.getenv("HAT_PIPELINE_TEST", "0") == "1"
training_step = os.getenv("HAT_TRAINING_STEP", "int_infer")
enable_model_tracking = os.getenv("HAT_ENABLE_MODEL_TRACKING") == "1"

model_checkpoint = os.getenv("HAT_TL_MODEL_CHECKPOINT")
model_setting = os.getenv("HAT_TL_MODEL_SETTING", "cn_2pe_v0.0.1")
model_name_postfix = os.getenv("HAT_TL_MODEL_NAME_POSTFIX", "fan.lv")
model_version = os.getenv("HAT_TL_MODEL_VERSION", "v0.0.1")
model_thresh = os.getenv("HAT_TL_MODEL_THRESH")
tasks = os.getenv("HAT_TL_TASKS")
resume_training = os.getenv("HAT_TL_RESUME_TRAINING", None)

# personal setting
num_worker = os.getenv("HAT_TL_NUM_WORKER", 2)
batch_size_factor = os.getenv("HAT_TL_BATCHSIZE_FACTOR", 4)
input_size = os.getenv("INPUT_SIZE", (96, 96))
pred_batch_size = os.getenv("HAT_TL_PRED_BATCHSIZE", 64)

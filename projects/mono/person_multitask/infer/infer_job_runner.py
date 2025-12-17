import os
import subprocess

MAX_MEMORY_REQUIRED_IN_GB = 7.0


def is_training_on_cluster():
    """
    Whether is training on cluster.
    """
    return os.environ.get("PBS_JOBNAME", None) is not None


current_dir = os.path.dirname(os.path.abspath(__file__))
HPFLOW_PERSON_6_CLASS = "0"
WORKER_PER_GPU = 10 if is_training_on_cluster() else 0
DEVICE = "0,1,2,3,4,5,6,7" if is_training_on_cluster() else "0"
NUM_WORKER = len(DEVICE.split(",")) * WORKER_PER_GPU

infer_type = "2pe_only"  # ["2pe_only", "joint4pe_and_2pe", "joint4pe_only"]

SCORE_4PE = 0.35


datasets_2pes = [
    dict(task="kPedOrientation", id="6037189", sensor="x8b"),
    # dict(task="kPedOcclusion", id="6027664", sensor=10652),
    # dict(task="kPedAge", id="6027225", sensor=10652),
    # dict(task="kPedPose", id="6028197", sensor=10652),
    # dict(task="kPedPosNeg", id="6027322", sensor=820),
    # dict(task="kPedHeadBBox2D", id="6027154", sensor=820),
    # dict(task="kPedPosNeg", id="6029231", sensor=820),
    # dict(task="kPedPosNeg", id="6038932", sensor='x8b'),
    # #
    # dict(task="kPedPose", id="6026670", sensor=820),
    # dict(task="kPedPosNeg", id="6027352", sensor=10652),
    # dict(task="kPedPosNeg", id="6027070", sensor=820),
    # dict(task="kPedPosNeg", id="6026850", sensor=10652),
]

EXP_PATHS = ["hobot-dag-3222327_hat-v2-6-v8data-20230629-193617"]
EVAL_QAT = ["qat", "float"][0]

HPFLOW_UPLOAD_FLAGS = [
    # "PersonMultitask2PE-HAT-5TaskV2.2-With_calib-20220303-014349"
    "test",
] * len(EXP_PATHS)

HPFLOW_JOINT4PE_PREDICT_IDS = [
    # "6026833-7093138|6027228-7100207",
    # "6026833-7088034",
    # "6026833-7088034",
    # "6026833-7088034",
    # "6026833-7093138|6027228-7100207",
    # "6026833-7088034",
    # "6026833-7093138|6027228-7100207",
    # "6026833-7093138|6027228-7100207",
    # "6026833-7093138|6027228-7100207",
    # "6026833-7093138|6027228-7100207",
    "6026833-7093138|6027228-7100207",
    # "6026860-7085128",
]

assert len(EXP_PATHS) == len(HPFLOW_UPLOAD_FLAGS)
if infer_type in ["joint4pe_only", "joint4pe_and_2pe"]:
    assert (
        len(EXP_PATHS)
        == len(HPFLOW_JOINT4PE_PREDICT_IDS)
        == len(HPFLOW_UPLOAD_FLAGS)
    )


template_infer_2pe = """
set -e
export HPFLOW_UPLOAD_FLAGS={HPFLOW_UPLOAD_FLAGS}
export HPFLOW_PERSON_6_CLASS={HPFLOW_PERSON_6_CLASS}
hpflow_run evaluate_upload \
--data {current_dir}/datasets.py \
--config {current_dir}/hat_person_model.py \
--ctx {DEVICE} \
--num-workers {NUM_WORKER} \
--upload-num-workers 16 \
--eval-cfg-model-name {EXP_NAME} \
--eval-dataset-name {EVAL_DATASET_ID} \
--eval-dataset-task {EVAL_DATASET_TASK} \
--eval-qat {EVAL_QAT} \
--upload \
--overwrite \
--project-id PDT20220001
"""  # noqa
#

template_infer_joint4pe = """
set -e
export HPFLOW_JOINT4PE_PREDICT_ID='{HPFLOW_JOINT4PE_PREDICT_ID}'
export HPFLOW_UPLOAD_FLAGS={HPFLOW_UPLOAD_FLAGS}
export HPFLOW_PERSON_6_CLASS={HPFLOW_PERSON_6_CLASS}
hpflow_run evaluate_upload \
--data hat_eval/datasets_joint.py \
--config hat_eval/hat_person_model.py \
--overwrite \
--ctx {DEVICE} \
--num-workers {NUM_WORKER} \
--upload-num-workers 16 \
--eval-cfg-model-name {EXP_NAME} \
--eval-dataset-name {EVAL_DATASET_ID} \
--eval-dataset-task {EVAL_DATASET_TASK} \
--upload \
--project-id PDT2020005
"""  # noqa

for idx in range(len(EXP_PATHS)):
    if infer_type in ["joint4pe_and_2pe", "joint4pe_only"]:
        cmd = template_infer_joint4pe.format(
            DEVICE=DEVICE,
            NUM_WORKER=NUM_WORKER,
            EXP_NAME=EXP_PATHS[idx],
            HPFLOW_UPLOAD_FLAGS=HPFLOW_UPLOAD_FLAGS[idx],
            current_dir=current_dir,
            HPFLOW_JOINT4PE_PREDICT_ID=HPFLOW_JOINT4PE_PREDICT_IDS[idx],
            SCORE_4PE=SCORE_4PE,
            HPFLOW_PERSON_6_CLASS=HPFLOW_PERSON_6_CLASS,
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)


if infer_type in ["2pe_only", "joint4pe_and_2pe"]:
    for datasets_2pe in datasets_2pes:
        for idx in range(len(EXP_PATHS)):
            cmd = template_infer_2pe.format(
                DEVICE=DEVICE,
                NUM_WORKER=NUM_WORKER,
                EXP_NAME=EXP_PATHS[idx],
                HPFLOW_UPLOAD_FLAGS=HPFLOW_UPLOAD_FLAGS[idx],
                current_dir=current_dir,
                EVAL_DATASET_ID=datasets_2pe["id"],
                EVAL_DATASET_TASK=datasets_2pe["task"],
                HPFLOW_PERSON_6_CLASS=HPFLOW_PERSON_6_CLASS,
                EVAL_QAT=EVAL_QAT,
            )
            print(cmd)
            subprocess.check_call(cmd, shell=True)

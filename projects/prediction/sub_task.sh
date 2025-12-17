export HAT_DIR="/home/users/shengzhe.dai/cnn1/HAT"
export CONFIGFILE=${HAT_DIR}"/projects/prediction/configs/densetnt_traj.py"
export PYTHONPATH=${HAT_DIR}:${PYTHONPATH}
export TASKTYPE="densetnt_traj"

## run local
# python3 -W ignore ${HAT_DIR}/tools/train.py --stage float --config ${CONFIGFILE}
# python3 -W ignore ${HAT_DIR}/tools/train.py --stage calibration --config ${CONFIGFILE}
# python3 -W ignore ${HAT_DIR}/tools/train.py --stage qat --config ${CONFIGFILE}
# python3 -W ignore ${HAT_DIR}/tools/train.py --stage int_infer --config ${CONFIGFILE}

# # If you want to run validation
# python3 -W ignore ${HAT_DIR}/tools/predict.py --stage float --config ${CONFIGFILE}
# python3 -W ignore ${HAT_DIR}/tools/predict.py --stage calibration --config ${CONFIGFILE}
# python3 -W ignore ${HAT_DIR}/tools/predict.py --stage qat --config ${CONFIGFILE}
# python3 -W ignore ${HAT_DIR}/tools/predict.py --stage int_infer --config ${CONFIGFILE}

# # If you want to compile the model and upload
# cd ${TRAJ_PROCESS_DIR}
# python3 ${HAT_DIR}/projects/prediction/tools/compile.py --task-type ${TASKTYPE} --hat-path ${HAT_DIR} --compile 1 --upload 0

# # If you want to recover the config file
# python3 ${SCRIPT_DIR}/local_scripts/train_recover.py
# python3 -W ignore ${HAT_DIR}/tools/analyze/model_profiler.py -c ${CONFIGFILE}


# # run cluster
python3 -W ignore ${HAT_DIR}/projects/prediction/tools/train/train_pipeline.py \
--model-type densetnt_traj_stage1 \
--model-version 'v1.0.0' \
--model-setting 'task_densetnt_stage1_master' \
--project-id 'PDT20220001' \
--current-cluster 'project-4090-pilot5-pnc-idc-newage' \
# --sleep
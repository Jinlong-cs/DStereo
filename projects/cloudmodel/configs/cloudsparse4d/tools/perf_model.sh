# 该脚本用于评测cloud sparse4d infer性能
# 包括时序、非时序推理，以及在titanxp、3090等gpu上推理性能（占用显存、fps等）
export PYTHONPATH=`pwd`

export aidi_eval=True
export val_batch_size=1
export num_gpus_per_machine=1
export temporal_eval=False
export cluster="share-debug-bcloud"
export model_version="v1.1"
export eval_dataset_id=6042946

# 非时序
python3 $(dirname $0)/../utils/perf_model_pipeline.py \
    --val-batch-size $val_batch_size \
    --model-version $model_version \
    --num-gpus-per-machine $num_gpus_per_machine \
    --aidi-eval $aidi_eval \
    --temporal-eval $temporal_eval \
    --current-cluster $cluster \
    --eval-dataset-id $eval_dataset_id \
    --local \
    # --sleep \

# 时序
export temporal_eval=True
python3 $(dirname $0)/../utils/perf_model_pipeline.py \
    --val-batch-size $val_batch_size \
    --model-version $model_version \
    --num-gpus-per-machine $num_gpus_per_machine \
    --aidi-eval $aidi_eval \
    --temporal-eval $temporal_eval \
    --current-cluster $cluster \
    --eval-dataset-id $eval_dataset_id \
    --local \

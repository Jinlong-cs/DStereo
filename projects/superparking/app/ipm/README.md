# IPM Mulitask Running Instructions

This is a guide for running IPM on SuperParking project.

## Installation

Please refer to [HAT Installation Guide](../../../../README.md)

## Training

The training process is the same as other HAT projects. Users can refer to `HAT/tools/train.py` for more details.

The used configuration file is `HAT/projects/superparking/app/ipm/ipm_multitask.py`.

### Example script

```fish
#!/usr/bin/fish

set HAT_HOME /home/users/(whoami)/projects/HAT
set -x PYTHONPATH $HAT_HOME $PYTHONPATH
echo "HAT_HOME: $HAT_HOME"

if contains pipeline-test $argv
    set pipeline_test true
else
    set pipeline_test false
end
echo "pipeline_test: $pipeline_test"


function local_train
    set -l cmd "python3 $HAT_HOME/tools/train.py \
        --config $HAT_HOME/projects/superparking/app/ipm/ipm_multitask.py \
        --stage $argv[1] \
        --device-ids 2,3 "

    if test $pipeline_test
        set cmd "$cmd --pipeline-test"
    end

    # strip multiple spaces
    set cmd (echo $cmd | tr -s ' ')

    echo "Train in local: $cmd"
    eval $cmd
end

function aidi_train
    set pre_dir (pwd)

    cd "$HAT_HOME/plugins/k8s_submit"
    set -l cmd "python3 submit.py \
        --config $HAT_HOME/projects/superparking/app/ipm/ipm_multitask.py \
        --cluster $argv[1] \
        --num-machines 2 \
        --num-gpus-per-machine 8"

    set cmd (echo $cmd | tr -s ' ')

    echo "Train in AIDI: $cmd"
    eval $cmd

    cd $pre_dir
end


if contains local $argv
    for stage in float qat int_infer
        local_train $stage
    end
else
    aidi_train share-2080ti-aliyun
end

```

## Inference

## Evaluation

### Local evaluation

HAT has a unified evaluation process. Users can refer to `HAT/tools/predict.py` for more details.

#### Example script

```fish
python3 tools/predict.py \
    --config projects/superparking/app/ipm/ipm_multitask.py \
    --stage qat \
    --ckpt <model-checkpoint-path> \
    --device-ids 2,3
```

### AIDI evaluation

### Special evaluation process

The special evaluation process runs with `HAT/projects/superparking/inference/ipm_multitask_inference.py`. （ Although the file name is `inference`, it is actually can be used in evaluation.）

This process runs locally.

**This method is going to be deprecated. Use this process as a backup plan.**

#### Example script

```fish
python3 projects/superparking/inference/ipm_multitask_inference.py \
    --stage int_infer \
    --task psd \
    --status val \
    --ckpt tmp_output/ipm_multitask/int_infer-deploy-checkpoint-last-843363e8.pt \
    --gpu-ids 2,3 \
    --config projects/superparking/app/ipm/ipm_multitask.py
```

## Compilation

#!/bin/bash
#Copyright: Horizon Robotic
#Function: Deploy quantized model(QAT) and related files to jfs.

usage="Usage: bash deploy_qat.sh configs/tcn_qat_demo.py"

if [[ $# -lt 1 ]]; then
    echo "${usage}"
    exit 1
fi

cfg_file=${1}

export SETUPTOOLS_USE_DISTUTILS=stdlib
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
HAT_PROJECT_PATH=${SHELL_FOLDER}/../../../
export PYTHONPATH=${HAT_PROJECT_PATH}:$PYTHONPATH
export NLU_NORM_FILE_PATH=${HAT_PROJECT_PATH}projects/halo/nlu/utils/nlu/nlu/data/standard/dict

# 抽取config文件指定配置内容
function get_config_content() {
    target=$(grep "${1} = .*" "${cfg_file}")
    target=$(echo "${target}" | grep -e "\".*\"" -o)
    target=${target:1:0-1}
    echo "${target}"
}

task_name=$(get_config_content task_name)
acc_threshold=$(get_config_content quantinized_acc_loss)
project_path=projects/halo/nlu

cd ../../../
# step1. float 及 int_infer 模型指标比对
echo "NOTICE: step1. compare float and int_infer model."
if ! python tools/predict.py --stage float --config ${project_path}/"${cfg_file}"; then
    echo "ERROR: tools/predict.py --stage float failed running" >&2
    exit 1
fi

if ! python tools/predict.py --stage int_infer --config ${project_path}/"${cfg_file}"; then
    echo "ERROR: tools/predict.py --stage int_infer failed running" >&2
    exit 1
fi

#获取float报告中的acc损失数值
is_acc_loss=0
while read -r line; do
    if [[ ${line} = "**match_ratio(accuracy)**" ]]; then
        is_acc_loss=1
        continue
    fi
    if [[ ${is_acc_loss} = 1 ]]; then
        acc_float=${line}
        break
    fi
done <${project_path}/output/${task_name}/predict_result/float_predictor.output

#获取int_infer报告中的acc损失数值
is_acc_loss=0
while read -r line; do
    if [[ ${line} = "**match_ratio(accuracy)**" ]]; then
        is_acc_loss=1
        continue
    fi
    if [[ ${is_acc_loss} = 1 ]]; then
        acc_int_infer=${line}
        break
    fi
done <${project_path}/output/${task_name}/predict_result/int_infer_predictor.output

#损失校验
if [[ ${acc_float} == "0.0" ]]; then
    echo "ERROR: acc_float is zero, please check your model report."
    exit 1
fi
acc_loss=$(echo "( ${acc_float} - ${acc_int_infer} ) / ${acc_float}" | bc)
if [[ "$(echo "${acc_loss} < ${acc_threshold}" | bc)" -eq 1 ]]; then
    echo "NOTICE: acc_loss ${acc_loss} is qualified."
else
    echo "NOTICE: acc_loss ${acc_loss} is not within acc_threshold ${acc_threshold}, please try to quantized model again."
fi

# step2. hbm模型编译
echo "NOTICE: step2. compile hbm model."
if ! python tools/deploy/compile_perf.py --config ${project_path}/"${cfg_file}"; then
    echo "ERROR: tools/deploy/compile_perf.py failed running" >&2
    exit 1
fi

# step3. 导出embedding层、crf层参数
echo "NOTICE: step3. extract special layers' parameters"
if ! python projects/halo/nlu/deploy/extract_special_layers.py --config ${project_path}/"${cfg_file}"; then
    echo "ERROR: projects/halo/nlu/deploy/extract_special_layers.py failed running" >&2
    exit 1
fi

# step4. embedding层参数量化为int8
echo "NOTICE: step4. transfer embedding layer into int8"
if ! python projects/halo/nlu/deploy/float2int8.py --config ${project_path}/"${cfg_file}"; then
    echo "ERROR: projects/halo/nlu/deploy/float2int8.py failed running" >&2
    exit 1
fi

# step5. 导出onnx模型
echo "NOTICE: step5. export onnx model file"
if ! python tools/deploy/export_onnx.py --config ${project_path}/"${cfg_file}"; then
    echo "ERROR: tools/deploy/export_onnx.py failed running" >&2
    exit 1
fi
cd ${project_path}
if ! python deploy/remove_initializer_from_onnx.py --input output/${task_name}/float.onnx --output output/${task_name}/float.onnx; then
    echo "ERROR: deploy/remove_initializer_from_onnx.py failed running" >&2
    exit 1
fi

# step6. 将模型发版至配置的jfs路径
echo "NOTICE: step6. push model to jfs path"

if ! ./deploy/push_to_jfs_qat.sh "${cfg_file}"; then
    echo "ERROR: ./deploy/push_to_jfs_qat.sh failed running" >&2
    exit 1
fi

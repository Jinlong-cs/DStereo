#!/bin/bash
#Copyright: Horizon Robotic
#Function: Deploy quantized model (PTQ)and related files to jfs.

usage="Usage: bash deploy_ptq.sh configs/tcn_ptq_demo.py"

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

# 检查日志中是否包含报错信息
function check_log_error() {
    log_file=${1}
    res=$(grep -c error "${log_file}")
    if [[ "${res}" -ne 0 ]]; then
        echo "ERROR: error occur in ${log_file}" >&2
        exit 1
    fi
}

platform=$(get_config_content platform)
task_name=$(get_config_content task_name)
# step1. 采样生成校准集数据
echo "NOTICE: step1. extract calibration dataset"
if ! python deploy/extract_calibration_dataset.py "${cfg_file}"; then
    echo "ERROR: deploy/extract_calibration_dataset.py failed running" >&2
    exit 1
fi

# step2. 导出embedding层、crf层参数
echo "NOTICE: step2. extract special layers' parameters"
if ! python deploy/extract_special_layers.py --config "${cfg_file}"; then
    echo "ERROR: deploy/extract_special_layers.py failed running" >&2
    exit 1
fi

# step3. 导出onnx模型
echo "NOTICE: step3. export onnx model file"
if ! python ../../../tools/deploy/export_onnx.py --config "${cfg_file}"; then
    echo "ERROR: ../../../tools/deploy/export_onnx.py failed running" >&2
    exit 1
fi
if ! python deploy/remove_initializer_from_onnx.py --input output/${task_name}/float.onnx --output output/${task_name}/float.onnx; then
    echo "ERROR: deploy/remove_initializer_from_onnx.py failed running" >&2
    exit 1
fi

# step4. 基于 hb_mapper checker 工具检验模型算子支持情况
echo "NOTICE: step4. hb_mapper checker"
if [[ "${platform}" = j3 ]]; then
    march_option=bernoulli2
elif [[ "${platform}" = j5 ]]; then
    march_option=bayes
else
    echo "ERROR: only support platform = [j3|j5]" >&2
    exit 1
fi
echo "NOTICE: march option: ${march_option}"
hb_mapper checker --model-type onnx --march ${march_option} --model output/${task_name}/float.onnx
deploy_save_dir=output/${task_name}/deploy/${platform}
if [[ ! -d "${deploy_save_dir}" ]]; then
    mkdir "${deploy_save_dir}"
fi
mv hb_mapper_checker.log ${deploy_save_dir}
check_log_error ${deploy_save_dir}/hb_mapper_checker.log
echo "NOTICE: check hb_mapper checker log in : ${deploy_save_dir}/hb_mapper_checker.log"

# step5. 准备生成bin模型所需的二进制校准文件
echo "NOTICE: step5. prepare bin calibration files for makertbin"
if ! python deploy/get_binary_inputs.py "${cfg_file}"; then
    echo "ERROR: deploy/get_binary_inputs.py failed running" >&2
    exit 1
fi

# step6. 基于 hb_mapper makertbin 生成bin模型
echo "NOTICE: step6. hb_mapper makertbin"
cp deploy/makertbin.yaml ${deploy_save_dir}
makertbin_yaml=${deploy_save_dir}/makertbin.yaml
sed -i "s|^\s*march.*|    march: \"${march_option}\"|" "${makertbin_yaml}"
hb_mapper makertbin --config ${makertbin_yaml} --model-type onnx
mv hb_mapper_makertbin.log ${deploy_save_dir}
check_log_error ${deploy_save_dir}/hb_mapper_makertbin.log

# step7. 基于量化前后的onnx模型，进行性能比对
echo "NOTICE: step7. compute performance loss after quantification"
if ! python deploy/check_model_accuracy.py "${cfg_file}"; then
    echo "ERROR: deploy/check_model_accuracy.py failed running" >&2
    exit 1
fi

# 获取报告中的acc损失数值
is_acc_loss=0
while read -r line; do
    if [[ ${line} = "**acc_loss**" ]]; then
        is_acc_loss=1
        continue
    fi
    if [[ ${is_acc_loss} = 1 ]]; then
        acc_loss=${line}
        break
    fi
done <${deploy_save_dir}/quantinized_model_eval_report.log

# step8. 如果量化损失在阈值范围内，则将模型发版至配置的jfs路径
echo "NOTICE: step8. push model to jfs path"
# 判断精度损失是否在阈值范围内（acc_loss < acc_threshold）
if [[ ${acc_loss} == "NaN" ]]; then
    echo "ERROR: acc_loss is NaN, please check your model report."
    exit 1
fi
acc_loss=${acc_loss:0:-1}
acc_threshold=$(get_config_content quantinized_acc_loss)
upload_path=$(get_config_content upload_path)
upload_version=$(get_config_content upload_version)
upload_jfs_path=${upload_path}${upload_version}

if [[ "$(echo "${acc_loss} < ${acc_threshold}" | bc)" -eq 1 ]]; then
    echo "NOTICE: acc_loss ${acc_loss} is qualified, push to jfs"
    echo "NOTICE: you are trying to push models to ${upload_jfs_path}, are you sure? [y/n]"
    read -r answer
    if [[ ${answer} = "y" ]] || [[ ${answer} = "Y" ]]; then
        if ! ./deploy/push_to_jfs.sh "${cfg_file}"; then
            echo "ERROR: ./deploy/push_to_jfs.sh failed running" >&2
            exit 1
        fi
    else
        echo "NOTICE: stop pushing"
        exit 1
    fi
else
    echo "NOTICE: acc_loss ${acc_loss} is not within acc_threshold ${acc_threshold}, please try to quantized model again."
fi

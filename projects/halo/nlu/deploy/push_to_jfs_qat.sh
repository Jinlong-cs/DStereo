#!/bin/bash
#Copyright: Horizon Robotic
#Function: Push related model files(QAT) and data to jfs.

usage="Usage: ./deploy/push_to_jfs_qat.sh configs/tcn_qat_offcial.cfg"

if [[ $# -lt 1 ]]; then
    echo "${usage}"
    exit 1
fi

cfg_file=${1}

# 抽取config文件指定配置内容
function get_config_content() {
    target=$(grep "${1} = .*" "${cfg_file}")
    target=$(echo "${target}" | grep -e "\".*\"" -o)
    target=${target:1:0-1}
    echo "${target}"
}

# 训练时的数据存储jfs路径
base_jfs_path=$(get_config_content base_jfs_path)
upload_path=$(get_config_content upload_path)
upload_version=$(get_config_content upload_version)
upload_jfs_path=${upload_path}${upload_version}
platform=$(get_config_content platform)
hbdk_version=$(get_config_content hbdk_version)
task_name=$(get_config_content task_name)
deploy_save_dir=output/${task_name}/deploy

if [[ ! -d "${deploy_save_dir}" ]]; then
    echo "ERROR: ${deploy_save_dir} not exist,please check!"
    exit 1
fi
to_upload_path=${deploy_save_dir}/to_upload
if [[ ! -d "${to_upload_path}" ]]; then
    mkdir ${to_upload_path}
else
    rm -f ${to_upload_path}/*
fi

if [[ "${platform}" = j5 ]]; then
    release_path=${upload_jfs_path}/qat/release_${platform}_hbdk${hbdk_version}
else
    echo "ERROR: only support platform = j5 in QAT mode" >&2
    exit 1
fi

if ! hdfs dfs -test -e "${base_jfs_path}"; then
    echo "ERROR: ${base_jfs_path} not exist,please check!"
    exit 1
fi

#准备待上传的特殊层参数，标签配置，词典配置
hdfs dfs -get ${base_jfs_path}/label/dict.txt ${to_upload_path}
hdfs dfs -get ${base_jfs_path}/label/domain.txt ${to_upload_path}
hdfs dfs -get ${base_jfs_path}/label/intent.txt ${to_upload_path}
hdfs dfs -get ${base_jfs_path}/label/slots.txt ${to_upload_path}
cp ${deploy_save_dir}/special_layers/crf.start_transitions.bin ${to_upload_path}
cp ${deploy_save_dir}/special_layers/crf.end_transitions.bin ${to_upload_path}
cp ${deploy_save_dir}/special_layers/crf.transitions.bin ${to_upload_path}
cp ${deploy_save_dir}/special_layers/embedding.float.weight.bin ${to_upload_path}/embedding.float.weight.bin
cp ${deploy_save_dir}/special_layers/embedding.weight.bin ${to_upload_path}/embedding.weight.bin

if hdfs dfs -test -e "${upload_jfs_path}"; then
    echo "NOTICE: ${upload_jfs_path} directory exists! Are you trying to cover all the files? [y/n]"
    read -r answer
    if [[ ${answer} = "y" ]] || [[ ${answer} = "Y" ]]; then
        hdfs dfs -rm -r "${upload_jfs_path}"/qat/ckpt
        if hdfs dfs -test -e "${release_path}"; then
            hdfs dfs -rm -r "${release_path}"
        fi
    else
        echo "NOTICE: stop pushing"
        exit 1
    fi
else
    echo "NOTICE: ${upload_jfs_path} directory does not exists! Create directory and push files."
    hdfs dfs -mkdir "${upload_jfs_path}"
fi

if hdfs dfs -test -e "${upload_jfs_path}"/data; then
    echo "NOTICE: ${upload_jfs_path}/data exists."
    hdfs dfs -rm -r "${upload_jfs_path}"/data/dataset
    hdfs dfs -rm -r "${upload_jfs_path}"/data/label
else
    hdfs dfs -mkdir "${upload_jfs_path}"/data
    hdfs dfs -chmod 777 "${upload_jfs_path}"/data
fi

hdfs dfs -mkdir "${upload_jfs_path}"/qat
hdfs dfs -mkdir "${upload_jfs_path}"/qat/ckpt
hdfs dfs -mkdir "${upload_jfs_path}"/onnx
hdfs dfs -mkdir "${release_path}"

echo "NOTICE: pushing files"
# ckpt推送
ckpt_file=output/${task_name}/*-checkpoint-best.pth.tar
hdfs dfs -put ${ckpt_file} ${upload_jfs_path}/qat/ckpt
hdfs dfs -put output/${task_name}/int_infer-checkpoint-last.pth.tar ${upload_jfs_path}/qat/ckpt

# onnx模型相关配置推送
hdfs dfs -put "${to_upload_path}"/* "${upload_jfs_path}"/onnx
hdfs dfs -put "${deploy_save_dir}"/../float.onnx "${upload_jfs_path}"/onnx/base_best.onnx

# hbm模型相关配置推送
hdfs dfs -put "${to_upload_path}"/* "${release_path}"
hdfs dfs -put "${deploy_save_dir}"/base_best*hbm "${release_path}"/base_best.hbm

# data推送
hdfs dfs -cp "${base_jfs_path}"/* "${upload_jfs_path}"/data
echo "Successful! Please check: ${upload_jfs_path}"

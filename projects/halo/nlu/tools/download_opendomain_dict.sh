#!/bin/bash
#Copyright: Horizon Robotic
#Function: download open_domain dict from hdfs

#open domain词表的jfs路径
remote_dir=jfs://jfs-hdfs/user/nlp/open_domain_dict

if [[ ! "$(pwd)" == *HAT/projects/halo/nlu/tools ]]; then
    echo "ERROR: running dir for this script is wrong! Please check it!" >&2
    exit 1
fi

local_dir=./files/open_domain_dict
if [[ ! -d "${local_dir}" ]]; then
    mkdir -p "${local_dir}"
    echo "INFO: mkdir ${local_dir}."
fi

rm -rf "${local_dir:?}"/*
echo "INFO: remove files in ${local_dir}."

if ! hdfs dfs -test -e "${remote_dir}"; then
    echo "ERROR: ${remote_dir} does not exist! Check path again." >&2
    exit 1
fi

if ! hdfs dfs -get "${remote_dir}"/* "${local_dir}"/; then
    echo "ERROR: fail to get open domain dict from ${remote_dir}" >&2
    exit 1
else
    echo "INFO: open domain dict files got."
fi

#! /usr/bin/bash
# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
# sh upload_bev_model.sh py36 1.6.16
set -e

PYTHON_VERSION=$1
HAT_VERSION=$2
SCRIPTS_DIR=$(readlink -f "$(dirname "$0")")
cd "$SCRIPTS_DIR" || exit 1

# check python version
if [[ "$PYTHON_VERSION" != "py36" ]] && [[ "$PYTHON_VERSION" != "py38" ]];
then
    echo "Invalid python verison $PYTHON_VERSION"
    exit 1
fi

# check hat version
if [ -z "$HAT_VERSION" ]; then
    echo "Please input hat version"
    exit 1
fi

# check file exist
if [[ ! -d "./bev_release_models" ]];
then
    echo "Local folder bev_release_models doesn't exist"
    echo "please download it first if you want to continue uploading"
    exit 1
fi

function setenv() {
    wget -q http://file.ddk.hobot.cc/oe_file/lftp_compiled.tar.gz
    tar zxf lftp_compiled.tar.gz
    rm -rf lftp_compiled.tar.gz
}

function uploadBEV() {
    ./lftp/bin/lftp -e "mirror -R ./bev_release_models /bev_j5/$HAT_VERSION/$PYTHON_VERSION/bev_release_models;exit" \
        -u 'openexplorer,c5R,2!pG' vrftp.horizon.ai
}

setenv
uploadBEV

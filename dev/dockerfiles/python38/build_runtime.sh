#!/bin/bash

DIR="$( cd "$( dirname "$0"  )" && pwd )"
HAT_DIR="${DIR}/../../.."

cd ${HAT_DIR}

COMMIT_ID="$( git rev-parse HEAD )"
TAG_NAME="$( grep -ri "__version__" hat/version.py | grep -w -Eo '[0-9\.]*' )"
DOCKER_FILES_DIR=dev/dockerfiles/python38/

echo "Build runtime docker for commit ${COMMIT_ID:0:7} with version ${TAG_NAME}"

echo "------------------- Build torch2.0.1 cu118 ... -------------------"
    
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch201_cu118.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-${TAG_NAME} \
    --no-cache

echo "------------------- Build torch2.0.1 cpu ... -------------------"
    
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch201_cpu.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cpu-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cpu-${TAG_NAME} \
    --no-cache

echo "------------------- Build torch1.13.0 cu116 ... -------------------"
    
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch1130_cu116.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-${TAG_NAME} \
    --no-cache


echo "------------------- Build torch1.13.0 cu116 cudnn890... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch1130_cu116_cudnn890.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-cudnn890-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cu116-cudnn890-${TAG_NAME} \
    --no-cache

echo "------------------- Build torch1.13.0 cpu ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch1130_cpu.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cpu-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.13.0-cpu-${TAG_NAME} \
    --no-cache


echo "------------------- Build torch1.10.2 cu102 ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch1102_cu102.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu102-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu102-${TAG_NAME} \
    --no-cache

echo "------------------- Build torch1.10.2 cu111 ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch1102_cu111.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-${TAG_NAME} \
    --no-cache

echo "------------------- Build torch1.10.2 cpu ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_runtime_torch1102_cpu.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cpu-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cpu-${TAG_NAME} \
    --no-cache


echo "------------------- Build torch1.10.2 cu111 for isp ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file aidi_isp_runtime_torch1102_cu111.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat-isp:runtime-py3.8-torch1.10.2-cu111-${COMMIT_ID:0:7} docker.hobot.cc/dlp/hat-isp:runtime-py3.8-torch1.10.2-cu111-${TAG_NAME} \
    --build-args "TAG_NAME=${TAG_NAME}" \
    --no-cache

echo "Build runtime docker successfully"

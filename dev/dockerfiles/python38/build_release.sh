#!/bin/bash

DIR="$( cd "$( dirname "$0"  )" && pwd )"
HAT_DIR="${DIR}/../../.."

cd ${HAT_DIR}
TAG_NAME="$( grep -ri "__version__" hat/version.py | grep -w -Eo '[0-9\.]*' )"
DOCKER_FILES_DIR=dev/dockerfiles/python38/

echo "Build release docker for tag ${TAG_NAME}"

echo "------------------- Build torch2.0.1 cu118 ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file release_torch201_cu118.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:release-py3.8-torch2.0.1-cu118-${TAG_NAME} \
    --no-cache

echo "------------------- Build torch1.13.0 cu116 ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file release_torch1130_cu116.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:release-py3.8-torch1.13.0-cu116-${TAG_NAME} \
    --no-cache


echo "------------------- Build torch1.10.2 cu102 ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file release_torch1102_cu102.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:release-py3.8-torch1.10.2-cu102-${TAG_NAME} \
    --no-cache

echo "------------------- Build torch1.10.2 cu111 ... -------------------"
python3 dev/dockerfiles/build_docker.py \
    --docker-file-dir ${DOCKER_FILES_DIR} \
    --docker-file release_torch1102_cu111.Dockerfile \
    --docker-names docker.hobot.cc/dlp/hat:release-py3.8-torch1.10.2-cu111-${TAG_NAME} \
    --no-cache

echo "Build release docker successfully"

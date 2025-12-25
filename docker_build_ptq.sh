#!/bin/bash

# 设置变量
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OS_VERSION=$(uname -s)
IMAGE_NAME="stereo_depth_ptq:${TIMESTAMP}_${OS_VERSION}"
INFER_IMAGE_NAME="stereo_depth_infer:${TIMESTAMP}_${OS_VERSION}"
DOCKER_HUB_REPO="ccr-29eug8s3-pub.cnc.bj.baidubce.com/dcloud/${IMAGE_NAME}"
INFER_DOCKER_HUB_REPO="ccr-29eug8s3-pub.cnc.bj.baidubce.com/dcloud/${INFER_IMAGE_NAME}"

# 构建Docker镜像
echo "Building Docker image..."
docker build --build-arg BUILD_DATE=$(date +%s) -t $IMAGE_NAME -f Dockerfile.ptq .

# 登录Docker Hub
echo "Logging in to Docker Hub..."
docker login --username=pqcong --password=Digua1234 ccr-29eug8s3-pub.cnc.bj.baidubce.com

# 标记镜像
echo "Tagging image..."
docker tag $IMAGE_NAME $DOCKER_HUB_REPO
docker tag $IMAGE_NAME $INFER_DOCKER_HUB_REPO

# 推送镜像到Docker Hub
echo "Pushing image to Docker Hub..."
docker push $DOCKER_HUB_REPO
docker push $INFER_DOCKER_HUB_REPO

echo "Process completed successfully!"
echo "Local image name: $IMAGE_NAME"
echo "Remote image name: $DOCKER_HUB_REPO"
echo "Infer remote image name: $INFER_DOCKER_HUB_REPO"
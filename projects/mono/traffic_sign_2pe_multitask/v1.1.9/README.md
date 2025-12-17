1. 调整遮挡分支的aug 配置
2. filter invalid roi
3. loss ignore label -1
4. smooth alpha 0.01
5. 遮挡分支移除base augmentation(高斯噪声，运动模糊等)
6. 分类分支移除运动模糊，高斯模糊等augmentation，保留高斯噪声
6. RandomCropNearBBoxV2

# Dockerfile

本目录下包含了 `HAT` 的 runtime 镜像和 release 镜像的构建信息。

## aidi_runtime_

包含公司内部环境如 `hdfs`、`gcc7` 等的镜像文件，可用于：
- 在 aidi 平台上作为训练任务的执行环境
- 在测试集群上用作 `HAT` 的测试环境
> 注意：此镜像仅为执行环境，不含 `HAT` 本身

## release_

包含 `HAT` 本身的最小执行环境，用于对外发布

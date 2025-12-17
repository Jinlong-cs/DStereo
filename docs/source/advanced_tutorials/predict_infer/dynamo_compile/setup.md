# 环境配置

HAT中提供了TensorRT一键安装脚本，运行：
```shell
sh dev/install_torch1130_tensorrt.sh
```
目前仅支持torch1.13.0 + cuda11.6。

需要注意的是，当安装完成后，tensorRT会被放置在`tmp_env`下(可以将其复制到其他期望的地址)，需要在默认环境变量中增加该依赖。

```shell
# in file ~.bashrc
...
export LD_LIBRARY_PATH=tmp_env/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/lib/:$LD_LIBRARY_PATH
export PATH=tmp_env/TensorRT-8.5.1.7/targets/x86_64-linux-gnu/bin/:$PATH
...
```

此外，我们还提供了基础docker，方便大家的使用
```
docker.hobot.cc/imagesys/hat:runtime-py3.8-torch1.13.0-cu116-1.3.3-tesorrt8.5.1.7
```

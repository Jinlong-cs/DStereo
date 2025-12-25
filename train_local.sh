# 代码输出的checkpoint路径
ln -s /horizon-bucket/d-robotics-bucket/zengpeng.sun/work_dirs/ work_dirs
# 训练数据的软链接
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/ public
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/kws_outdoor_dataset/ kws_data_output
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/models/ tmp_pretrained_models
ln -s /horizon-bucket/d-robotics-bucket/AIOT_algorithm_data/Depth_data Depth_data
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/SyntheticDataGeneration SyntheticDataGeneration
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/SyntheticDataGeneration/NVIDIA NVIDIA
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/SyntheticDataGeneration/TartanAir/TartanAir/ TartanAir

export PYTHONPATH=/docker-mount/zengpeng.sun/DStereo/:$PYTHONPATH
# 训练的主要入口文件
# python3 tools/train.py -s float -c DStereo/DStereoPlus.py
python3 tools/predict.py -s float -c DStereo/DStereoPlus.py
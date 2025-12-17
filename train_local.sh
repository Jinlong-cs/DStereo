find . -maxdepth 1 -type l -exec rm -f {} +
# 代码输出的checkpoint路径
ln -s /docker-mount/zengpeng.sun/work_dirs/ work_dirs

# 训练数据的软链接

#TaranAir
ln -s /docker-mount/zengpeng.sun/DStereoDepthDatasets/TartanAir/ TartanAir

# 存放了sceneflow和IRS的list文件
# public/stereo_data/sceneflow_train.list
# 存放了IRS的list文件
# public/Public_Datasets/IRS/IRSDataset_TRAIN.list
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/ public

# 存放了SIDOD和FallingThings
# NVIDIA/SIDOD/mixed_distractor/
# NVIDIA/FallingThings/fat/mixed/
ln -s /horizon-bucket/d-robotics-bucket/bohao.zhang/SyntheticDataGeneration/NVIDIA NVIDIA


export PYTHONPATH=/docker-mount/zengpeng.sun/DStereo_ori_V2.1/:$PYTHONPATH
# 训练的主要入口文件
python3 tools/train.py -s float -c DStereo/DStereoPlus.py

# 预测流程
# python3 tools/predict.py -s float -c DStereo/DStereoPlus.py

# 导出onnx模型
# python3 tools/deploy/export_onnx.py --config DStereo/DStereoPlus.py
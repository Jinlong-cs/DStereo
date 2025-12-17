"""Pilot数据打包流程简介
===========================================================

pilot提供了一键式运行数据打包的工具。
"""

############################################
# 数据准备
# ------------------------------------------
#
# 1. 打包需要准备好img和对应的annotation，其中img需要放到同一文件夹下，annotation需要存储在一个json文件中。
#
# .. code-block:: none
#
#    root
#    |--data
#    |    |--xxx.jpg
#    |--data.json
#
#
# 2. 以cyclist_detecion的打包任务为例annotation中需要包括以下信息，其中image_key需要和img文件#夹中的img文件名一一对应
#
# .. code-block:: python
#   :emphasize-lines: 0
#
#   {
#       "height": 1280,
#       "image_key": "6cfe2a9d6720ae817b9657ff373e4e5d.jpg",
#       "image_source": " ", "image_uuid": "6127e29ac5b551017febf34a",
#       "person": [{
#           "attrs":
#           {
#               "OverlappedBox": "no",
#               "age": "Adult",
#               "confidence": "High",
#               "ignore": "no",
#               "occlusion": "occluded",
#               "pred_category": "person",
#               "score": 1.019287109375,
#               "type": "Pedestrian"
#           },
#           "data": [1527, 540, 1603.441, 766],
#           "id": 1,
#           "label_type": "boxes",
#           "luid": "145cbfb7-5f09-42cb-8c17-a166cc8093c3",
#           "struct_type": "rect",
#           "track_id": -1
#       }],
#       "video_index": "1",
#       "video_name": "1",
#       "width": 2048
#   }
#

############################################
# 运行打包代码
# ------------------------------------------
#
# 1. 进入脚本所在目录
#
# .. code-block:: shell
#
#
#       cd projects/pilot/pack_tools
#
# 2. 创建data文件夹
#
# .. code-block:: shell
#
#       mkdir data
#
# 3. 运行打包脚本，其中各个任务的config存在configs目录下, {path}是存储img data和annotation的根目录
#
# .. code-block:: shell
#
#       python3 pack.py --config configs/{task}/train.py --num-worker 8 --root_path {path} --visualize
#

############################################
# 对打包anno transform config做修改
# ------------------------------------------
#
# 打包的时候会对框做anno transform 会做一些过滤，如果需要更改匹配原则可以更改各个任务config下的config.yaml 文件

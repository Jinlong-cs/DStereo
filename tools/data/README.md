该目录下为数据工具， 包括数据打包、可视化等，其中：

`imagenet_packer.py`:  Imagenet数据集打包工具，使用方式如下：
    ```bash
    python3 tools/datasets/imagenet_packer.py --src-data-dir ${src-data-dir} --target-data-dir ${target-data-dir} --split-name train --num-workers 10 --pack-type lmdb
    ```

`imagenet_viz.py`: Imagenet数据集可视化工具，使用方式如下：
    ```bash
    python3 tools/data/imagenet_viz.py --data-path ${data-path} --viz-num 10 --plot
    ```

`mscoco_packer.py`: MSCOCO数据集打包工具，使用方式如下：
    ```bash
    python3 tools/datasets/mscoco_packer.py --src-data-dir ./tmp_data/mscoco/ --target-data-dir ./tmp_data/mscoco --split-name train --pack-type lmdb
    ```

`mscoco_viz.py`: MSCOCO数据集可视化工具，使用方式如下：
    ```bash
    python3 tools/data/mscoco_viz.py --data-path ${data-path} --viz-num 10 --plot
    ```

`video.py`: 编码视频工具，使用方式如下：
    ```bash
    python3 tools/data/video.py  --images-dir {imgs_path} --video-path ${video-path} --fps 5 --order-type descending
    ```

`voc2coco.py`: 数据集格式转换工具，提供从voc格式到coco格式转换，仅支持打包的voc数据，使用方式如下：
    ```bash
    python3 tools/data/voc2coco.py -p ${pack-data-path} -o ${out-dir}
    ```

更多的数据打包工具见[TOOLCHAIN](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/tree/master/projects/toolchain/tools/datasets)

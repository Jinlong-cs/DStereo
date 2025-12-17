# Car keypoint detection
|   model |  dataset |   backbone     |   Input shape      |   config  |  ckpt download | pretrian backbone |
| :----------:          | :-------:|  :--------:    |  :------------:    | :------: |     :--------:  |  :--------:  |
| HeatmapKeypointModel | CarFusion | EfficientNet-b0 | 128x128 | configs/keypoint/keypoint_efficientnetb0_carfusion.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/keypoint_efficientnetb0_carfusion/* --ftp-password='c5R,2!pG' | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/efficientnet_imagenet/float-checkpoint-best.pth.tar --ftp-password='c5R,2!pG' |

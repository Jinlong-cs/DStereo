# Binocular depth estimation
|   model              |  dataset |   backbone     |   Input shape      |   config  |  ckpt download        |
| :----------:          | :-------:|  :--------:    |  :------------:    | :------: |        :--------:           |
| StereoNet | SceneFlow | StereoNeck | 540x960 | configs/disparity_pred/stereonet/stereonet_stereonetneck_sceneflow.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/stereonet_stereonetneck_sceneflow/* --ftp-password='c5R,2!pG' |
| StereoNetPlus | SceneFlow | StereoNeck | 544x960 | configs/disparity_pred/stereonet/stereonetplus_mixvargenet_sceneflow.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/stereonetplus_mixvargenet_sceneflow/* --ftp-password='c5R,2!pG' |

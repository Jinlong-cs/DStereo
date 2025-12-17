# BEV
|   model               |  dataset |   backbone     |   Input shape      |   config                  |  ckpt download        |
| :----------:          | :-------:|  :--------:    |  :------------:    | :----------------------:  |  :--------:           |
| bev_gkt_multitask            | nuscenes |  MixVarGENet   |    (512, 960)      |configs/bev/bev_gkt_mixvargenet_multitask_nuscenes.py  |wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/bev_gkt_mixvargenet_multitask_nuscenes/* --ftp-password='c5R,2!pG' |
| bev_ipm_4d_multitask  | nuscenes | efficientnetb0 |    (512, 960)       |configs/bev/bev_ipm_4d_efficientnetb0_multitask_nuscenes.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/bev_ipm_4d_efficientnetb0_multitask_nuscenes/* --ftp-password='c5R,2!pG' |
| bev_ipm_multitask       | nuscenes | efficientnetb0 |    (512, 960)      | configs/bev/bev_ipm_efficientnetb0_multitask_nuscenes.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/bev_ipm_efficientnetb0_multitask_nuscenes/* --ftp-password='c5R,2!pG' |
| bev_lss_multitask     | nuscenes | efficientnetb0 |    (256, 704)      |configs/bev/bev_lss_efficientnetb0_multitask_nuscenes.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/bev_lss_efficientnetb0_multitask_nuscenes/* --ftp-password='c5R,2!pG' |
| detr3d                | nuscenes | fcose3d_efficientnetb3  | (512, 1408)         |configs/bev/detr3d_efficientnetb3_nuscenes.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/detr3d_efficientnetb3_nuscenes/* --ftp-password='c5R,2!pG' |
| petr                  | nuscenes | fcose3d_efficientnetb3 | (512, 1408) | configs/bev/petr_efficientnetb3_nuscenes.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/petr_efficientnetb3_nuscenes/* --ftp-password='c5R,2!pG' |

# Backbone Pretrained ckpt
|   backbone         |  ckpt download |
| :----------:       |  :-----------: |
| efficientnetb0     |  wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/efficientnet_imagnet/* --ftp-password='c5R,2!pG' |
| MixVarGENet        |  wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/mixvargenet_imagenet/* --ftp-password='c5R,2!pG' |
|fcose3d_efficientnetb3 | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/RELEASE_VERSION/py38/modelzoo/qat_origin_modelzoo/fcos3d_efficientnetb3_nuscenes/* --ftp-password='c5R,2!pG' |

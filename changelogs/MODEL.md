# Changelog of model

## [unReleased] - 2024-MM-DD

### Added

- [models] Add white list of torch compile. ([#5585](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5585))

### Changed
- 

### Deprecated
-

### Removed
- [models] Remove Calibration2QAT converter. ([#5573](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5573))

### Fixed
-

## [2.2.0] - 2024-01-28

### Added
- [models] Add MeanVFE voxel encoder & 3D sparse conv backbone & ASPP neck for lidar cloud model. ([#5519](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5519))

- [models] Add keypoints project with distortion support for sparse4d.([#5453](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5453))

- [models] Update e2e modules and unit-tests for pilot-bev. ([#5439](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5439))

- [models] Add qconfig template. ([#5515](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5515))

- [models] Update task modules and unit-tests for pilot-bev. ([#5435](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5435))

- [models] Update losses modules and unit-tests for pilot-bev. ([#5430](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5430))

- [models] Add vision transformer. ([#5468](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5468))

- [models] Add detection model DINO. ([#5431](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5431))

### Changed
- [models] Modify parkingrod model rod in slot logic. ([#5551](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5551))

- [models] Modify some attribute names of cloudmodel outputs, change "category" to "type". ([#5369](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5369))

- [models] Fix motr amp qat train nan. ([#5400](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5400))

- [models] Add bevformer models.([#5464](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5464))

- [models] Fix toolchain ci.([#5479](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5479))

- [models] `QAT2Quantize` support speedup int_infer by gpu and cutlass. ([#5358](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5358))

### Deprecated
-

### Removed
-

### Fixed
- [models] Fix view error in swin-transformer. ([#5446](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5446))

- [models] `QAT2Quantize` set `fast_mode` default False. ([#5539](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5539))

- [models] Fix `SaveTensor` in eval mode. ([#5568](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5568))

## [2.1.0] - 2023-12-17

### Added
- [models] Add models for avspeech task. ([#4747](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4747))

- [models] Add qconfig refresher in `ModelConvertPipeline` to allow modification of qconfig. ([#5067](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5067))

- [models] Add FastViT model. ([#4800](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4800))

- [models] Support `torch.compile` in torch2.0.1. ([#5092](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5092))

- [models] Support trt for sd cone detection model. ([#5241](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5241))

- [models] Support trt for lane parsing7cls model. ([#5297](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5297))

- [models] Add j6 toolchain models and intinfer models. ([#5335](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5335))

- [models] Add Deformable Detr model. ([#5248](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5248))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
- [models] Fix dynamic shape error in transformer when using torchdynamo and tensorrt ([#5223](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5223)).

## [2.0.0] - 2023-10-30

### Added
- [models] Replace the cat operation in the decoder structure with the add operation of DenseTNT stage one. ([#5061](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5061))

- [models] Cut out unnecessary structures of DenseTNT to reduce latenct. ([#5023](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5023))

- [models] Reduce the endpoint number of DenseTNT from 2048 to 1024. ([#5039](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5039))

- [models] Add MobileOne model. ([#4728](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4728))

- [models] Add CSPDarknet/CSPPAFPN model. ([#5008](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5008))

- [models] Add custom implementation of `SyncBatchNorm`, which will be faster in training pipeline when torch>=1.13.0. ([#5041](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5041))

- [models] Add decoder and loss for gaze pccr. ([#5028](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5028))

### Changed
- [models] Change score output shape to topk of DenseTNT and fix the visualization. ([#5103](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5103))

- [models] Change DenseTNT structure to enable qat training, finish float evaluation. ([#4934](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4934))

- [models] DenseTNT stage2 qat optimization. ([#5097](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5097))

- [models] Support trt and warmup for laneseg model. ([#5091](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5091))

- [models] Update gaze pccr fitting code. ([#5115](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5115))

### Deprecated
-

### Removed
-

### Fixed
-

## [1.4.1] - 2023-09-20

### Added
- [models] Add behav model quant. ([#4845](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4845))

- [models] Add LutNet model. ([#4893](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4893))

- [models] Add split transform for isp_me model. ([#4869](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4869))

- [models] Add dn and auxhead for sparse4d. ([#4642](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4642))

- [models] Add toolchain trajectory prediction model. ([#4281](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4484))

- [models] Add static gesture model. ([#4442](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4442))

- [models] Add attribute decoder with the DetObjects format. ([#4549](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4549))

- [models] Support multi-backbone for sparse4d. ([#4430](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4430))

- [models] Add toolchain disparity pred stereonetplus model. ([#4547](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4547))

- [models] Add RepVGG/QARepVGG model. ([#4249](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4249))

- [models] Support tensorrt fx extension. ([#4546](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4546)) 

- [models] Support tensort for cloud seg model. ([#4585](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4585))

- [models] Support dynamic batch size for torchdynamo and tensorrt. ([#4586](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4586))

- [models] Update stereonetplus head. ([#4775](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4775))

- [models] Add isp me detector, two types of me include GenISP and NISP. ([#4814](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4814))

 
### Changed
- [models] Mid multimodal fusion add duo-vcsrange feature. ([#4779](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4779))

- [models] Add modules for Lidar&Camera multimodel and temporal fusion modules. ([#4516](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4516))

- [models] Add `top_layer`, `quant_input`, `dequant_output` args for ResNet. ([#4687](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4687))

- [models] Modify DenseTNT head and add stage two training. ([#4627](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4627))

### Deprecated
-

### Removed
-

### Fixed

- [models] Fix proxy error of classifier. ([#4573](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4573))

## [1.4.0] - 2023-07-05

### Added
- [models] Add odd of valid obstacles for trajectory prediction model. ([#4281](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4281))

- [models] Update sparse4d init_weights. ([#4390](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4390))

- [models] Add Sparse4D structure. ([#4348](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4348))

- [models] Add Sparse4D head. ([#4305](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4305))

- [models] Add Sparse4D blocks. ([#4338](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4338))

- [models] Add Sparse4D instance_bank. ([#4330](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4330))

- [models] Add decoder and target of 3D detection for Sparse4D. ([#4306](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4306))

- [models] Add BatchVoxelization for pointpillar. ([#4220](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4220))

- [models] Add centerpoint model. ([#4258](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4258))

- [models] Add Detr3d & Petr. ([#4248](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4248))

- [models] Add motr models.([#4251](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4251))
  
- [models] Add `SECONDNeck`.([#4262](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4262))

- [models] Add `Stereonet`.([#4267](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4276))

- [models] Add lidar multitask models.([#4289](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4289))

- [models] Add qat onnx config of transformer and lidar.([#4458](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4458))

- [models] Add norm layers in PAFPN.([#4393](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4393))

- [models] Add ResizePatcher and DownscaleNeck.([#4301](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4301))

- [models] Add HeatmapKeypointModel for carfusion keypoint detection.([#4392](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4392))


### Changed
- [models] Refactor face3d structure. (#[4387](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4387))

- [models] Update network of attribute task.([#4268](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4268))

- [models] Migrate qat policy for motr detr3d and petr.([#4282](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4282))
  
- [models] Support PointPillars model use fx.([#4275](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4275))

- [models] MaxPostProcess support return only max value and DequantModule support sequence inputs.([#4377](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4377))

### Deprecated
-

### Removed
-

### Fixed
- [models] Fix efficientnet dropconnect rate. ([#4255](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4255))

- [models] Fix pafpn norm_cfg bug. ([#4401](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4401))

## [1.3.2] - 2023-05-13

### Added
- [models] Modify SNDRMBv2 model. ([#4236](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4236))

- [modles] Add SPIN and HMR for human3d. ([#3831](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3831))

- [modles] Add HRNet for landmark detection. ([#4191](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4191))

- [models] Add SegEdgeloss for segmentaion. ([#4160](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4160))

- [models] Remove valid head for trajectory prediction. ([#4005](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4005))

- [models] Add SNDRMixDWNet for SNDR. ([#4108](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4108))

- [models] Add quantmodule. ([#4071](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4071))

- [models] Add fasternet for snapdragon SoC. ([#4089](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4089))

- [models] Add 3d target modules. ([#3963](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3963))

- [models] Add mobilenext for snapdragon SoC. ([#4066](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4066))

- [models] Add efficientnet for snapdragon Soc. ([#4060](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4060))


- [models] Add mobilenetv2 for snapdragon SoC. ([#4044](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4044))

- [models] Add fcos3d configs. ([#3934](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3934))

- [models] Add loss to attribute model. ([#4253](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4253))

- [models] Add lane instance segmentation training modules ([#3958](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3958))
- [models] Add `Anchor3DGeneratorStride` and `LidarTargetAssigner` modules for Lidar3d task.([#3967](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3967))
- [models] Add `PointPillars`.([#3969](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3969))

- [models] Add `TimeSformer` model.([#4006](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4006))

- [models] Add giou/diou/ciou loss of SGNetLoss. ([#4049](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4049))

### Changed
- [model] Migrate calibration for segmentation models. ([#3982](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3982))

- [model] Migrate calibration for detection models. ([#4070](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4070))

- [model] Migrate calibration for bev models. ([#4117](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4117))
  
- [model] Revert bev spatial transformer. ([#4207](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4207))

- [models] Update `lidar_encoder`:  `PillarFeatureNet` and `PointPillarScatter`.([#3968](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3968))

- [models] Fix DETR model resume training by adding freeze_bn callback.([#4027](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4027))

- [models] Fix bool type error in DynamicFcosTarget.([#4086](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4086))

### Deprecated
-

### Removed
-

### Fixed
-

## [1.3.1] - 2023-03-19

### Added
- [models] Update person position code. ([#3954](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3954))

- [models] Add views_fuson structer. ([#3953](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3953))

- [models] Add group fpn. ([#3974](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3974))

- [models] Update densetnt's config, transform, collate, model head, structure and viz. ([#3896](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3896)

- [models] Update retinanet models according to dev-tool. ([#3936](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3936))

- [models] Add lidar input bev grid related functions. ([#3740](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3740))

- [model] Add BMSegTarget and update BMSegmentor to support MultitaskGraphModel. ([#3867](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3867))

- [models] Add fcos3d postprocess. ([#3829](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3829))

- [model] Add fcos xj3 model configs. ([#3845](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3845))

- [model] Add global loss_weight to FocalLossV2. ([#3860](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3860))

- [models] Add fatigue eye-ldmk neck.([#3883](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3883))

- [model] Add fcos swin transformer model. ([#3826](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3826))

- [model] Support multi-warps for a feature in spatial transfomer. ([#3868](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3868))

- [models] Add fcos3d models. ([#3754](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3754))

- [model] Add Mixup transforms and fix mono dataset load bug. ([#3766](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3766))

- [model] Add rpp op and three stage structure. ([#3503](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3503))

- [model] Add loss_custom_weight function. ([#3392](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3392/))

- [model] Add configs of fcos efficientnet b2 b3 model. ([#3699](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3699))

- [model] Add configs of deeplabv3plus model. ([#3667](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3667))

- [model] Add face3d eye3d error. ([#3701](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3701))

- [model] Add behavior transforms for lateral move and distance threshold restriction. ([#3684])(https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3684)

- [model] Update heatmap focal loss to support class weight. ([#3710])(https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3710)

- [model] Add DETR model and configs. ([#3732](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3732))

- [model] Add pre-trained weights convert code in swin transformer. ([#3797](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3797))

- [models] Adjust the features of VectorNet trajectory model. ([#3856](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3856))

- [model] Add run length encoding postprocess for dense classification. ([#3823](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3823))

- [project] Update VehicleSideFCOSTarget and VehicleSideFCOSDecoder. ([#4831](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4831))

### Changed

- [model] Fix groupfpn groupbase. ([#4030](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4030))

- [project] Fix pilot roi decoder of point task. ([#3929](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3929))

- [model] Migrate calibration for classification models. ([#3900](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3900))

- [projects] Add pilot model production dag ([#3697](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3697))


### Deprecated

-

### Removed

-

### Fixed

- [model] Fix precision problem in roi3DDecoder. ([#3722](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3722))

- [model] Fix precision problem of fcos efficientnet models. ([#3749](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3749))

- [model] Fix loading pre-trained weights in swin transformer. ([#3841](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3841))


## [1.3.0] - 2023-02-16

### Added

- [model] Fix resize quantization parameters not saved to ckpt.([#3692](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3692))

- [model] Add mmvad model and unittest.([#3502](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3502))

- [model] Add Centerpoint head. ([#3681](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3681))

- [model] Add Lidar semi-supervised 3d-detection model. ([#3645](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3645))

- [model] Update densetnt transform, head and config.([#3625](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3625))

- [model] Add lidar model module code. ([#3641](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3641)

- [model] Add model structure for face3d_pp. ([#3572](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3572))

- [model] Adapt the sampling policy of road features in vectornet to software. ([#3516](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3516))

- [model] Add densetnt head, structure, loss, metrics and config. ([#3517](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3517))

- [model] Support the 3d task evaluation of side and rear withbn model. ([#3413](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3413))

- [model] Add OnlineReParamBlock. ([#3306](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3306))

- [model] Add CenterNet loss target_generator and 2d pbrec dataset.  ([#3402](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3402))

- [model] Add EfficientNet.  ([#3574](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3574))

- [model] Add Structure & HungarianBBox3DLoss & HungarianBBoxAssigner3D for MVT4D. ([#3540](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3540))

- [model] add BevFeatEncoder & transformer bricks used in BevFeatEncoder for MVT4D. ([#3559](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3559))

- [model] Add Max postprocess. ([#3555](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3555))

- [model] Add postprocess for MVT4D big model including bev3d_multiclass_nms func, NMSFreeBBoxCoder, MVTPostProcess. ([#3566](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3566))

- [model] Add deformable Detection3D Head for MVT4D big model including stage2nd img/bev head. ([#3571](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3571))

- [model] Add efficientnasnets and efficientnasnetm model. ([#3634](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3634))

- [model] Add vargconvnet model. ([#3643](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3643))

- [model] Add ganet model. ([#3648](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3648))

- [model] Add horizon swin transformer model. ([#3631](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3631))

### Changed

- [model] Optimize model module codes and bug fixes in sp/mono. ([#3715](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3715))

### Deprecated

-

### Removed

-

### Fixed

- [model] Fix the wrong output of lidar-det compile model. ([#3688](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3688))

- [model] Fix tests in arcloss due to torch1.13. ([#3664](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3664))

- [model] Update multitask_graph_model and module_patch. ([#3522](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3522))

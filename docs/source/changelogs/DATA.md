# Chagelog of data

## [unReleased] - 2024-MM-DD

### Added
- 

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
-

## [2.2.0] - 2024-01-28

### Added
- [data] Dataset auto3dv supports reading lidar point cloud data. ([#5514](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5514))

- [data] Move the rest changes for pilot-bev v3. ([#5491](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5491))

- [data] Update targets transform and unit-tests for pilot-bev. ([#5474](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5474))

- [data] Update auto3dv transforms and unit-tests for pilot-bev. ([#5451](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5451))

- [data] Update data collates and samples for pilot-bev. ([#5370](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5370))

- [data] Support read odo status code in PackDataset and pack infer when odo is invaild.  ([#5526](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5526))

### Changed
- [data] Modify hard code for collate_3d. ([#5470](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5470))

### Deprecated
-

### Removed
-

### Fixed
- [data] Fix map_size in lmdb. ([#5459](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5459))

## [2.1.0] - 2023-12-17

### Added
- 

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
-

## [2.0.0] - 2023-10-30

### Added
- [data] Add a simplified goal sampling transform for DenseTNT. ([#4945](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4945))
- [data] Update transforms for cloudsparse4d. ([#5047](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5047))
- [data] Add ETH-XGaze dataset. ([#5068](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5068))

### Changed
-

### Deprecated
-

### Removed
-

### Fixed

- [data] Fix `StatefulDistributedSampler` index in elastic resume.([#4916](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4916))

## [1.4.1] - 2023-09-20

- [data] Fix bug of temporalLMDBDataset of seq frame sampling. ([#4679](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4679))

- [data] Add dataset, dataloader, sampler and utiles for avspeech task.([#4712](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4712))

### Removed
- [data] Remove carp directory. ([#4662](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4662))

### Changed
- [data] Add raw FE transform. ([#4893](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4893))

- [data] Add split transform for isp_me model. ([#4869](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4869))

- [data] Mid multimodal fusion add duo-vcsrange feature. ([#4779](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4779))

- [data] Update processed_dataset. ([#4846](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4846))

- [data] Add modules for Lidar&Camera multimodel and temporal fusion modules. ([#4516](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4516))

- [data] Add WeNet dataset and processor. ([#4687](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4687))

- [data] Update basetrajdatasetv2. ([#4661](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4661))

- [data] Update dataset to adapt fus data. ([#4658](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4658))

- [data] Add mixed occ_type in RandomOcclusion. ([#4616](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4616))

- [data] Add sampler of each dataset in concat dataset and chunk shuffle. ([#4508](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4508))

- [data] Update resume dataloader index by global variable. ([#4490](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4490))

- [data] Update attribute collate function. ([#4470](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4470))

- [data] Revise face3d lmdb packer for saving cropped img and fix it. ([#4473](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4473))

- [data] Add jaw occlusion for face3d. ([#4597](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4597))

- [data] Add perf transform time cost tools ([#4571](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4571))

- [data] Add fixed dataset and dataloader output function ([#4520](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4520))

- [data] Add dataset memory and time cost perf tools ([#4605](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4605))

### Deprecated
-

### Removed
-

### Fixed

- [data] Fix NoneType Error when lidar param is None. ([#4632](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4632))

- [data] Fix PackDateset for SD. ([#4541](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4541))

- [data] Fix introduced BPUPyramidResizer init error. ([#4560](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4560))

- [data] Fix twice jpeg compression of LMDB packing in lmdb auto2d. ([#4814](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4814))


## [1.4.0] - 2023-07-05

### Added

- [data] Add lidargt transfrom to traj prediction pipeline.([#4595](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4595))

- [data] Update gen obstacles goals v2 and head and metrics.([#4575](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4575))

- [data] Add Add pred traj refactor dateset.([#4319](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4319))

- [transform] Add raw pad transform method.([#4440](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4440))

- [data] Add gesture lmdb dataset. ([#4444](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4444))

- [data] Add avspeech dataset. ([#4404](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4404))

- [data] Fix transform and unit test bug for gesture task. ([#4252](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4252))

- [transform] Add fixlength pad transform. ([#4386](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4386))

- [transform] Update sparse4d adaptor transform. ([#4383](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4383))

- [transform] Add transform to read raw data. ([#4361](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4361))

- [data] Add temporal lmdb dataset. ([#4332](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4332))

- [sampler] Add dist_group_in_batch_sampler. ([#4308](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4308))

- [data] Add dataset to read real3d data generated by 4dgt. ([#4183](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4183))

- [data] Support reading the pilot lmdb dataset and transforming the data to densebox format. ([#4325](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4325))

- [data] Add avspeech augment transforms. ([#4125](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4125))

- [data] Add MaskImageEdgeTransform transform. ([#4226](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4226)])

- [tools] Add mot17 dataset. ([#4242](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4242))

- [data] Add nuscenes lidar dataset. ([#4258](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4258))

- [data] Add ignore_key in colloate_3d. ([#4314](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4314))

- [data] Update data tools for attribute model. ([#4441](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4441))

- [data] Add error_logging in lmdb. ([#4402](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4402))

- [data] Read roi from evalset/pack and Add DynamicCropImgPatch. ([#4409](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4409))

- [transform] Add AddGaussianNoise, GenerateHeatmapTarget, RandomPadLdmkData for keypoint data. ([#4437](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4437))

### Changed
- [transform] Update mask transform to support different layouts. ([#4423](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4423))

- [data] Update DenseboxDataset and transform for attribute classification task. ([#4274](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4274))

- [data] Add pack_flag to ResampleDataset，ConcatDataset and pack_dataset. ([#4835](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4835))

-

### Deprecated
-

### Removed
-

### Fixed
- [data] Fix rec to lmdb packer for parsing. ([#4174](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4174))

- [data] Fix key miss bug in lmdbauto2d. ([#4342](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4342#note_554218))

- [data] Fix orig_img bugs in image_auto2d. ([#4831](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4831))

## [1.3.2] - 2023-05-13

### Added
- [transform] Add transform for face3d-grid-norm. ([#4259](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4259))

- [data] Fix problem in landmark dataset and transform. ([#4222](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4222))

- [data] Add landmark lmdb dataset. ([#4149](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4149))

- [data] Add diff trajs and metric updater for DenseTNT. ([#4102](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4102))

- [data] Support camera standardization for 3d transforms. ([#4101](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4101))

- [data] Add new 3d transforms. ([#3956](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3956))

- [data] Add collate function to concatenate data. ([#4231](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4231))

- [transform] Fix transforms bug for BPUPyramidResizer when march is Bayes. ([#3983](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3983))

- [data] Add SegResizeAffine transform. ([#3869](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3869))

- [data] Add TimeSformer dataset and transforms. ([#3926](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3926))

- [data] Add New Features in SGNet Dataset Pipeline. ([#4040](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4040))

- [transform] Add IterableDetRoIListTransform for crop. ([#4162](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4162))

### Changed
- [transform] Optimize training speed of camera_standardization. ([#3972](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3972))

- [data] Update parsing anno transform and support opencv4. ([#3974](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3947))

- [data] Update src_cam to meta_info. ([#3992](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3992))
- [data] Support load LMDB to HorizonLidar3D dataset. ([#3942](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3942))

### Deprecated
-

### Removed
-

### Fixed
- [data] Fix transform 3d index error for empty image. ([#4119](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4119))
- [data] Fix distributed random dataset shuffle. ([#3951](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3951))

## [1.3.1] - 2023-03-19

### Added
- [transform] Add gridsample infos transform for face3d perspective projection. ([#3940](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3940))

- [data] Add human3d packer. ([#3853](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3853))

- [transform] Add multi-views transfomer. ([#3851](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3851))

- [data] Fix randomcrop bug when using crop_around_gt. ([#3791](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3791))

- [data] Output calib_all in YUVDataset. ([#3802](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3802))

- [data] Add LiDAR lmdb dataset.([#3892](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3892))

- [data] Add Nuscenes datasets. ([#3743](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3743))

- [data] Add kitti lidar3D dataset. ([#3731](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3731))

- [data] Add Lidar3D transforms. ([#3801](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3801))

- [transform] Decouple face3d transform and add cliff. ([#3765](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3765))

- [data] Update data transform to support fatigue. ([#3858](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3858))

- [data] Add NuscenesMonoDataset and NuscenesMonoFromImage. ([#3832](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3832))

- [data] Add splitted dataflow for behavior prediction. ([#4059])(https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4059)
### Changed

- [data] Merge pyramid of J3 and J5. ([#3621](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3621))

### Deprecated

- [data] Deprecate old pyramid resizer. ([#3912](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3912))

### Removed

-

### Fixed

- [data] Fix unit test in test_multipath_structure. ([#3705](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3705))

- [data] Fix ModelEvalRawDataset::get_image_info arguments bug. ([#3767](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3767))

## [1.3.0] - 2023-02-16

### Added

- [data] Update camera_standardization.([#3541](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3541))

- [data] Add avspeech datasets.([#3537](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3537))

- [data] Add ConvertDataType to transforms common.([#3579](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3579))

- [data] Add flank task anno transformer.([#3582](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3582))

- [data] Fix anon transform in data.([#3545](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3545))

- [data] Add features to MultitaskInfLoader.([#3484](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3484))

- [data] Fix bugs in SP added modules. ([#3486](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3486))

- [data] Add det2d mosaic data augmentation in data.([#3497](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3497))

- [data] Add densetnt transform and collate. ([#3421](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3421))

- [data] Add text and audio visual align transforms.([#3496](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3496))

- [data] Fix world_step error in RankSplitDataloader. ([#3542](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3542))

- [data] Add rank split dataset. ([#3451](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3451))

- [data] Add eye vis dataset.([#3477](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3477))

- [data] Add audio denoise transforms.([#3454](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3454))

- [data] Add lipmove data iterator. ([#2403](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/2403))

- [data] Support training multiview rec with version v0.1. ([#3493](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3493))

- [data] Add multiview rec data transforms for MVT4D. ([#3498](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3498))

- [data] Adapt valid head in vectornet behavior prediction. ([#3536](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3536))

- [data] Add culane dataset and metric. ([#3646](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3646))

- [data] Add DETR model data preparation.([#3670](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3670))

### Changed

- [data] Optimize data pipeline and bugfixes for sp/mono project. ([#3712](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3712))
- [data] Optimize BgrToYuv444V2 by reducing device switching and removing some unneed operations for better gpu running performance. ([#4418](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4418))

### Deprecated

-

### Removed

-

### Fixed

- [data] Fix to-buf in pack dataset. ([#3682](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3682))

# Changelog of util/core and test.

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
- [utils] Fx2trt backend support custom max batch_size. ([#5456](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5456))

- [utils] Save tensor support torch201. ([#5545](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5545))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
-

## [2.1.0] - 2023-12-17

### Added

- [utils] Support environments of python3.10. ([#5230](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5230))

- [utils] Add deterministic utils.([#5164](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5164/))

### Changed

- [utils] Cancel strict check on torch version in HAT. ([#5259](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5259))

### Deprecated
-

### Removed
-

### Fixed
- [utils] Raise error when passing unexpected keys to `set_default_qconfig` in qconfig_manager. ([#5067](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5067))

- [utils] Fix missing `_metadata` in static_dict when load checkpoint. ([#5197](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5197))

## [2.0.0] - 2023-10-30

### Added

- [utils] Support torch2.0.1. ([#4969](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4969))

### Changed

- [utils] Add ignore_key in join_path when recursive execute func in list. ([#4917](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4917))

- [e2e] Change argsort to stable-argsort, for consistance compare. ([#4924](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4924))

### Deprecated
-

### Removed
-

### Fixed
- 

## [1.4.1] - 2023-09-20

### Added

- [util] Add gpu affinity. ([#4621](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4621))

- [registry] Add error log of not registry. ([#4521](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4521))

- [util] Update version of protobuf. ([#4445](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4445))

### Changed
- [core] Fillback get_vcs2bev_img_mat to only matrix. ([#4537](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4537))

- [configs] Move configs to examples. ([#4512](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4512))

- [utils] Update requirement check utils. ([#4743](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4743/))

- [utils] Make `hatbc` optional. ([#4673](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4673))


### Deprecated
-

### Removed
-

### Fixed

- [registry] Fix error log of registry. ([#4564](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4564))

- [utils] Fix `ExperimentLogger` bug. ([#4519](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4519))

- [utils] Fix `aidi experiment` bug that keeps creating thread. ([#4538](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4538))

## [1.4.0] - 2023-07-05

### Added
- [core] Add ct nms to replace the distance matching filtering method in let nms. ([#4205](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4205))

- [core] Add cylindrical differentiable camera. ([#4223](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4223))

- [core] Add differentiable camera module and fix uint8 reduce bug of unit-test. ([#4165](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4165))

### Changed
- [visualize] Update bev3d vis to default use virtual camera. ([#4612](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4612))

- [configs] Tune unet hyper-params to regenerate recorded miou. ([#4169](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4169))
- [misc] update __init__.py. ([#4446](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4446))

### Deprecated
-

### Removed
-

### Fixed
- [utils] Fix tensor duplication when moving to cuda. ([#4291](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4291))

- [core] Fix yaw index error in let_nms and bev3d_nms. ([#4214](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4214))

- [qconfig] Add default weight qconfig kwargs in set_default_qconfig. ([#4341](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4341))

## [1.3.2] - 2023-05-13

### Added
- [engine] Add share_callbacks in validation. ([#4016](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4016))

- [core] Add zoom/crop/pad for virtual camera and so3 domain project api. ([#3945](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3945))

- [core] Add `rotate_iou_v2` implemented by numba. ([#4020](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4020))

### Changed
- [configs] Minor modifications about toolchain. ([#3965](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3965))

- [engine] Increase Lmdb read's timeout. ([#3933](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3933))

- [core] Fix virtualcam fov bug. ([#3865](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3865))

- [core] Split bev3d_nms and let_nms and add agnostic nms option. ([#3836](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3836))

- [requirement] Update hatbc and aidisdk version. ([#4017](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4017))

### Deprecated
-

### Removed
-

### Fixed
- [engine] Make fx_wrap compatible with all plugin version. ([#4004](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4004))

## [1.3.1] - 2023-03-19

### Added

- [configs] Update configs releated to COCO metric. ([#3949](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3949))

- [configs] Update traj pred dataset config and discard precutin flag in behav pred. ([#3889](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3889))

### Changed

-

### Deprecated

-

### Removed

-

### Fixed

-

## [1.3.0] - 2023-02-16

### Added

- [utils] Add function img_array2tensor and img_tensor2array in utils.apply_func. ([#4296](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4296))

- [core] Fix bev_size usage error.([#3635](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3635))

- [core] Update get_vcs2bev_img_mat by IPMCamera and add ImagePointsInterpolation for anno. ([#3474](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3474))

- [utils] Add frame to video tools.  ([#3511](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3511))

- [test] Add numerical consistent test for pilot model train. ([#3485](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3485))

### Changed

- [QAC] Add support of torch1.13. ([#3590](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3590))

- [utils] refactor saved-tensor interface and fix memory leak bug. ([#2913](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/2913))

- [utils] Change default mapsize of lmdb while reading. ([#3476](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3476))

- [QAC] Fixed the version of some dependent packages. ([#3550](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3550))

- [QAC] Fixed the version of matplotlib. ([#3676](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3676))

### Deprecated

-

### Removed

-

### Fixed

- [utils] Fix loss and postprocess of traj pred due to torch1.13. ([#3674](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3674))

- [utils] Fix bug of model_patch. ([#3666](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3666))

- [test] Fix ci error. ([#3626](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3626))

- [test] Fix daily build ci error in traj_pred. ([#3628](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3628))

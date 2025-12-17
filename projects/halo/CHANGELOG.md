# Changelog of halo

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
- 

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
- [halo] Add metric and config of pupil segmentation. ([#5112](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5112))

### Changed
- [config] Change face ldmk68 config. ([#5138](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5138))

### Deprecated
-

### Removed
-

### Fixed
-

## [2.0.0] - 2023-10-30

### Added
- [data] Add gaze eve dataset. ([#5050](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5050))

- [config] Add glint points detection config and packer. ([#4891](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4891))

- [tool] Add pupil segmentation packer. ([#5036](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5036))

- [model] Add pupil segmentation network. ([#5043](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5043))

- [loss] Add loss module of pupil segmentation. ([#5059](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5059))

- [data] Add gazemap dataset. ([#5069](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5069))

- [data] Add dataset and transform of pupil segmentation. ([#5083](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5083))

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

## [1.4.1] - 2023-09-20

### Added
- [config] Add config files for avspeech tasks. ([#4749](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4749))

- [config] Add human3d config. ([#4791](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4791))

- [metric] Add human3d metric and loss. ([#4741](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4741))

- [model] Update human3d model code. ([#4660](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4660))

- [data] Add human3d dataset. ([#4708](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4708))

- [metric] Support faceid multi gar eval and remove projects dir reliance. ([#4485](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4485))

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
- [metric] Fix bugs in eye3d_pose metrics. ([#4850]https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4850)

## [1.4.0] - 2023-07-05

### Added

- [config] Add faceid baseline training config. ([#4462](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4462))

- [model] Add flat_output args in faceid vargnet backbone. ([#4450](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4450))

- [metric] Add faceid GAR metric. ([#4408](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4408))

- [model] Add hand3d loss module. ([#4322](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4322))

- [tool] Add faceid eval and benchmark. ([#4328](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4328))

- [tool] Add faceid metric. ([#4269](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4269))

### Changed
-

### Deprecated
-

### Removed
-

### Fixed

- [metric] Fix faceid metric import projects bug. ([#4448](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4448))

- [model] Fix hand3d manolayer module. ([#4438](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4438))

- [model] Fix hand3d mano decoder module. ([#4447](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4447))

## [1.3.2] - 2023-05-13

### Added

- [model] Add hand3d model and metric module. ([#4257](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4257))

- [model] Add hand3d config module. ([#4263](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4263))

- [tool] Add emotion data packer. ([#4239](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4239))

- [tool] Add face landmark packer. ([#4185](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4185))

- [config] Add eye_visibility config. ([#4138](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4138))

- [tool] Add landmark packer. ([#4140](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4140))

- [config] Add eye status binary classification config. ([#4146](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4146))

- [tool] Add eye_visibility flip transform. ([#4131](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4131))

- [tool] Add pack2raw tool for SNDR. ([#4111](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4111))

- [model] Add gaze grid sample model and corresponding dataset. ([#4091](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4091))

- [data] Add fas lmdb dataset. ([#4056](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4056))

- [config] Add mixvargenet config for SNDR. ([#4044](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4044))

- [tools] Add face anti spoofing packer([#4053](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4053))

- [config] Add eye status config. ([#4034](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4034))

- [metric] Add eye status metrics. ([#4029](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4029))

- [config] Add halo image quality config. ([#3995](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3995))

- [config] Add facemtl config. ([#3989](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3989))

- [transforms] Add facemtl transforms. ([#3990](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3990))

- [data] Add sampler for eye status. ([#3976](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3976))

- [structure] Add facemtl structure (face3d). ([#3977](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3977))

- [model] Add facemtl head. ([#3973](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3973))

- [structure] Add eye status classifier. ([#3921](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3921))

- [structure] Add eyeldmk groupconv module. ([#4025](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4025))

### Changed

- [data] Update eye_visibility datasets. ([#4095](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4095))

- [model] Update faceattr-related code to meet master version. ([#4050](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4050))

- [config] Fix face3d config bug. ([#4042](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4042))

- [config] Fix facemtl config bug. ([#4037](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4037))

- [transforms] Fix GenerateGaussianVector bug when x!=y. ([3938](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3938))

### Deprecated
-

### Removed
-

### Fixed
- [tool] Fix circular import problem in faceldmk packer. ([#4190](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4190))

- [model] Fix fatigue eye status ldmk head qat bug. ([#4129](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4129))

- [metric] Fix face3d metric. ([#4110](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4110))

- [metric] Fix eye_status metric and config. ([#4090](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4090))

- [data] Fix dataset and transforms in gaze online rotate pipeline. ([3987](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3987))

## [1.3.1] - 2023-03-19

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

## [1.3.0] - 2023-02-16

### Added

- [config] Add face3d pp config and datahub. ([#3671](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3671))

- [config] Add phone config.([#3471](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3471))

- [projects/halo/nlu] Refactor projects/halo/nlu pipeline. ([#3439](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3439))

### Changed

-

### Deprecated

-

### Removed

-

### Fixed

-

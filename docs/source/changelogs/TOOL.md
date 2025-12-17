# Changelog of tools and plugin tools.

## [unReleased] - 2024-MM-DD

### Added
- [tools] Support hbdk4>=4.0.14 ([#5593](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5593))

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
- [docs] Add float training config example and docs. ([#5270](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5270))

- [tools] Add quant analysis module. ([#5538](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5538))

- [plugins] Add `nccl_test`. ([#5441](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5441))

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
- [tools] Add warnings filter function. ([#5215](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5215))

### Changed
- [plugins] Add `shecdule_type` parameter. ([#5193](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5193))

### Deprecated
-

### Removed
-

### Fixed
- [tools] Follow hbdk4 pipeline changes and modify hbir graph on QAT module. ([#5240](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5240))
- [plugins] Fix `torchrun` launcher stack bug in multi-node task. ([#5276](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5276))

## [2.0.0] - 2023-10-30

### Added
- [tools] Add export, infer, compile tools for hbir. ([#4641](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4641))

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

- [tools] Add torchdynamo support and tensorrt inference acc. ([#4464](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4464))

### Changed
- [tools] Update api generator. ([#4491](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4491))

- [tools] Reorganize the tools directory. ([#4512](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4512))

- [docs] Reorganize the docs. ([#4608](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4608))

- [plugins] Set `torchrun` as the multi-machine DDP launcher on AIDI instead of mpirun. ([#4578](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4578))

- [plugins] Update aidi inference. ([#4626](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4626))

- [plugins] Add `job_desc` to elastic job. ([#4648](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4648))

- [util] Support DCU. ([#4751](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4751))

### Deprecated
-

### Removed
-

### Fixed
- [tools] Fix num-machines and num-gpus-per-machine args of k8s submit script. ([#4918](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4918))

- [tools] Fix api generator. ([#4500](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4500))

- [tools] Fix key error when upload compile perf paras. ([#4545](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4545))
  
- [plugins] Fix torchrun env bug in multi-machines task. ([#4611](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4611))

- [plugins] Fix torchrun bug in single-machine task. ([#4617](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4617))

- [tools] Fix config error of repeated fromfile. ([#4620](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4620))

- [tools] Fix the bug of using gpu_affinity in a DCU environment. ([#4827](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4827))

- [plugins] Fix experiment_path bug. ([#4874](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4874))

## [1.4.0] - 2023-07-05

### Added

- [tools] Add modify config options in predict.py. ([#4287](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4287))

- [tools] Add disable-inference of aidi deploy. ([#4256](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4256))

- [tools] Add parsing40cls and multitask detection aidi inference model. ([#3981](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3981))

- [tools] Add split mot17 dataset. ([#4242](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4242))

- [tools] Support fault tolerant training. ([#4298](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4298))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
-

## [1.3.2] - 2023-05-13

### Added

- [tool] Add seed in predict. ([#4203](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4203))

- [tool] Add tracking for compile. ([#3878](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3878))

### Changed

- [tools] Update `submit` script, support description of job and dag. ([#4094](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4094))

- [tools] Update `compile_standalone` script, support compiling hbm with hbir as input. ([#3901](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3901))

- [tools] Update bucket name in aidi experiment. ([#3943](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3943))

- [tools] Update `submit` script, support `single_job`. ([#3794](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3794))

- [tools] Update update aidi deploy readme. ([#3960](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3960))

### Deprecated
-

### Removed
-

### Fixed
-

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

- [plugins] Fix docker image in plugins. ([#3696](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3696))

- [tools] Refactor aidi inference deploy tools. ([#3809](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3809))

- [tools] Fix aidi deploy intergret test bug. ([#3913](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3913))

## [1.3.0] - 2023-02-16

### Added

- [plugins] Add output bucket mount in ``submit.py``. ([#3587](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3587))

### Changed

- [plugins] Update aidi inference. ([#3601](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3601))

- [plugins] Fix aidi inference. ([#3680](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3680))

### Deprecated

-

### Removed

- [tools] Remove some deprecated in submit. ([#3439](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3337))

### Fixed

-

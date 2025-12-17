# Changelog of callback and profile

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
- [callbacks] Update callbacks modules and unit-tests for pilot-bev. ([#5350](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5350))
- [callbacks] Add step_id type for OnlineModelTrick. ([#5477](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5477))
- [callbacks] Add discobj output origin image size config. ([#5535](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5535))

### Changed
- [callbacks] Format output of ANCSaveFusionFeature same with software. ([#5316](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5316))

### Deprecated
-

### Removed
-

### Fixed
-

## [2.1.0] - 2023-12-17

### Added
- [profilers] Improve SimpleProfiler and open read data speed monitor by default. ([#5247](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5247))
- [profilers] Add `DynamoProfiler` for debugging torch compile([#5196](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5196))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
- [profilers] Fix data monitor log time bug. ([#5339](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5339))
- [profilers] Fix monitor logger repeat init bug. ([#5343](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5343))

## [2.0.0] - 2023-10-30

### Added
- [callback] Add remain step percent logging in training([#4915](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4915))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
- [callback] Fix estimate_remain_time_and_step base on num_steps of StatsMonitor([#5049](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5049))

## [1.4.1] - 2023-09-20

### Added
- [callback] Add optimizer param updater. ([#4249](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4249))
- [callback] Add pruner callback. ([#4756](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4756))

### Changed
- [callback] Support freeze hierarchical sub-modules in FreezeModule. ([#4687](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4687))

- [profiler] Fix `pytorch_profiler` bugs. ([#4557](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4557))

- [profiler] Add memory snapshot to `GPUMemoryProfiler`. ([#4619](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4619))

- [callback] `OnlineModelTrick` support elastic resume. ([#4648](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4648))

### Deprecated
-

### Removed
-

### Fixed

- [profile] Separate the Python profiler into a separate file. ([#4488](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4488))
- [callbacks] Fix ema_model resume bug. ([#4548](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4548))
- [callbacks] Fix `aidi_expmodel` log artifact. ([#4591](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4591))
- [profile] Fix profiler filename bug. ([#4650](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4650))
- [profile] Fix GPUMemoryProfiler bug in torch1.10.2. ([#4654](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4654))


## [1.4.0] - 2023-07-05

### Added

- [callback] Add only save ddp flag in checkpoint callback. ([#4451](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4451))
- [profiler] Optimize simple profiler. ([#4329](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4329))

- [callback] Add aidi eval function to attribute model. ([#4265](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4265))

- [callback] Add eval tracking in aidiexpmodel. ([#4247](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4247))
- [callback] Add rotbox detection handler. ([#4359](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4359))
- [callback] Optimize logic of enable tracking. ([#4370](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4370))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed

- [profile] Fix batch_processor in profile. ([#4245](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4245))
- [callback] Fix `StepDecayLrUpdater`. ([#4226](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4426))

## [1.3.2] - 2023-05-13

### Added
- [callback] Add Visualzier callbacks for multitask cloudmodel. ([#3889](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3906))


### Changed
- [profile] Update `pytorch_profiler`. ([4099](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4099/))
- [callback] Use horizon.jit.save/load to save and load pt file. ([#4143](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4143))


### Deprecated
-

### Removed
-

### Fixed
-

## [1.3.1] - 2023-03-19

### Added

- [callbacks] Add cpu&mem setting in aidieval handler.([#3884](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3884))

- [callbacks] Dump model params for dump data callback.([#3461](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3461))


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

- [callbacks] Add resume training and upload progressive checkpoint for aidi exp model callback.([#3473](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3473))

- [callback] Add `AIDIExperimentManager` callback. ([#3576](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3576))

- [profiler] Add CPUMemoryProfiler used to profile the CPU memory bottleneck. ([#3472](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3472))

- [profiler] Add StageCPUMemoryProfiler used to profile the CPU memory bottleneck in one stage. ([#3611](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3611))

### Changed

- [callback] Add epoch and step info to loop_end checkpoint. ([#3505](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3505))

### Deprecated

-

### Removed

-

### Fixed

-

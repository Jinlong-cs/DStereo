# Changelog of engine

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
- [engine] Integrate auto calibration in Calibrator. ([#5067](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5067))

- [engine] Support automatically setting `update_interval` in KL observer for best performance. ([#5067](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5067))

- [engine] Add DeepSpeed Trainer to support ZeRO(Zero Redundancy Optimizer). ([#5183](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5183))

- [engine] Add AidiPredictor for inferring models from the Aidi model repository. ([#5271](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5271))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
- [optimizer] Fix case when module has parameters and sub module. ([#5248](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5248))


## [2.0.0] - 2023-10-30

### Added
- [engine] Integrate weight reconstruction in Calibrator. ([#5007](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5007))

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

- [engine] Fix grad accumulation. ([#4803](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4803))

- [engine] Support upload of training tricks. ([#4799](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4799))

- [engine] Fix rank of calibration. ([#4755](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4755))

- [engine] Support cuda-graph example. ([#4416](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4416))

- [optimizers] Add custom_param_optimizer to allow finer control over the 
params passed to the optimizer. ([#4492](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4492))

### Changed
- 

### Deprecated
-

### Removed
-

### Fixed
- [optimizers] Fix custom_param_optimizer bug. ([#4801](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4801))

- [engine] Fix device_ids bug in Inference. ([#4879](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4879))

## [1.4.0] - 2023-07-05

### Added

- [engine] Support grad accumulation between different batch. ([#4204](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4204))

### Changed
- 

### Deprecated
- [engine] Support new graph split method in multitask graph model and multistage processor. ([#4182](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4182))
- [engine] Deprecate old model converter. ([#4224](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4224))

### Removed
-

### Fixed

- [engine] Add batch args to basic processor backend callback. ([#4435](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4435))

- [engine] Fix channels_last in multistage batchprocessor. ([#4219](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4219))

## [1.3.2] - 2023-05-13

### Added

- [engine] Support channels last in training. ([#4142](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4142))

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

## [1.3.1] - 2023-03-19

### Added

- [engine] Support bfloat16 in AMP training. ([#3773](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3773))

- [engine] Add calibrator v2. ([#3560](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3560))

### Changed

- [engine] NCCL is set to WARN by default. ([#3728](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3728))

### Deprecated

-

### Removed

-

### Fixed

- [engine] Fix calibration v2 compatible issue. ([#3840](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3840)) 

## [1.3.0] - 2023-02-16

### Added

-

### Changed

-

### Deprecated

-

### Removed

-

### Fixed

- [engine] Fix multistage split backward processor where backbone cannot update gradient. ([#3435](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3435))

- [engine] Fix integration test of loop resume by adding back checkpoint attribute to trainer. ([#3581](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3581))

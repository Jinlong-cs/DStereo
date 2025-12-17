# Changelog of metric and evaluation

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
- [metrics] Add add discobj ap_score_threshold and fppi metric. ([#5555](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5555))
- [metric] Add bev3d corner error and percentile error. ([#5516](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5516))
- [metrics] Add ay and yaw_rate in NuscenesTrackingMetric error computation. ([#5454](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5454))
- [metrics] Update e2e metrics and unit-tests for pilot-bev. ([#5428](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5428))
- [metrics] Update online-mapping modules and unit-tests for pilot-bev. ([#5378](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5378))
- [metrics] Update discobj metric modules and unit-tests for pilot-bev. ([#5392](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5392))
- [metrics] Update bev elevation metrics and unit-tests for pilot-bev. ([#5374](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5374))
- [metrics] Fix parkingrod recall decline. ([#5537](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5537))
- [metrics] Fix parkingrod eval metric. ([#5591](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5591))
- [evaluation] Add utils for offline bev3d tag val. ([#5323](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5323))
- [evaluation] Support batch flat and max detection rate for bev3d offline eval. ([#5333](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5333))
- [evaluation] Add bev3d tag eval init and metric calulate. ([#5336](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5336))
- [evaluation] Add instance process function for tag bev3d eval. ([#5367](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5367))
- [evaluation] Support tag eval update, compute and build table. ([#5375](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5375))
- [evaluation] Support tag combination for bev3d. ([#5405](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5405))

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
-
- [evaluation] Support tracklet-wise temporal evaluation for bev3d aidieval. ([#4823](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4823))

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
- [metric] Fix division error when num_inst is zero in compute function of EvalMetric class. ([#5154](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/5154))

## [1.4.1] - 2023-09-20

### Added
- [evalution] Add LocalEval. ([#4436](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4436))

- [evalution] Add classification and det2d fp metrics for aidieval. ([#4471](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4471))

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
- [metric] Fix let-iou valid box match repeatedly error. ([#4707](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4707))
- [evaluation] Fix y_thresh none type error when multiprocessing. ([#4693](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4693))
- [evaluation] Fix seg local eval error. ([#4570](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4570))
- [evaluation] Fix unit test for bev3d aidieval. ([#4556](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4556))

## [1.4.0] - 2023-07-05

### Added
- [evalution] Add drot@90% metrics for bev3d aidieval. ([#4411](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4411))

- [metric] Add range mode para in bev3d confusion matrix. ([#4380](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4380))

- [metric] BEVDetEval support custom result key. ([#4331](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4331))

- [metric] Add mot17 tracking metric. ([#4242](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4242))

- [metric] Remove false warning about coco metric distributed sampler. ([#4422](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4422))

- [metric] Add attribute multi-label metric. ([#4254](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4254))

- [visuakize] Add bev rotate box detection draw samples and visualize. ([#4327](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4327))

- [metric] Add bev rotate box detextion aidi eval metric. ([#4270](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4270))

- [metric] Add init file and change file name. ([#4321](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4321))

- [metric] Add target_precisions, target recalls, target thresholds in summarize_ap function. ([#4391](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4391))

### Changed
- [metric] Optim bev3d pred file save. ([#4243](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4243))

### Deprecated
-

### Removed
-

### Fixed
- [metric] Fix aidi eval sod3d task bug. ([#4349](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4349))

- [metric] Fix aidi eval plot visable and seg confusion_matrix color bug. ([#4364](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4364))

## [1.3.2] - 2023-05-13

### Added
- [evaluation] Add APH metric for Bev_3D aidieval. ([#4115](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4115))

- [evaluation] Add tracking metric. ([#3882](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3882))

- [evaluation] Add confusion matrix metric for BEV3D. ([#4073](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4073))

- [metric] Add nuscenes tracking metric. ([#4104](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4104))

- [evaluation] Support scence_tag for Bev_3D aidieval. ([#3999](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3999))

- [evaluation] Support aidieval job submission for bev3d. ([#3961](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3961))

- [visualize] Add ipm visualize.([#3923](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3923))

- [evaluation] Add let-ap and let-apl metrics for bev3d evalution on aidi all-region datasets. ([#3897](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3897))

- [metric] Add kitti3d detection metric. ([#4024](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4024))

### Changed
- [evaluation] Split bev_3d aidieval result.json and update eval_vcs_range. ([#4187](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4187))

- [metric] Add IPN in landmark metric. ([#4067](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4067))

- [metric] Fix BEV det metric file over-size. ([#4096](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/4096))
### Deprecated
-

### Removed
-

### Fixed
-

## [1.3.1] - 2023-03-19

### Added

- [metric] Return detailed result in BEVDetEval. ([#3777](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3777))

- [metric] Support image_tag in detection_3d evaluation. ([#3776](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3776))

- [evaluation] Add aidieval submit to two datasets for bev3d task. ([#3758](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3758))

- [evaluation] Add multiprocess for seg eval. ([#3716](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3716))

- [evaluation] Expand detection sep mode presentation form. ([#3762](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3762))

- [evaluation] Update eval image match logic and add remove ignore data flag to speedup eval and aidi list sample. ([#3991](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3991))

- [metric] Support distributed coco evaluation. ([#3843](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3843))

- [evaluation] Update detection tabel name to compatible with eror compare tabel. ([#3871](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3871))

- [metric] Support multiprocessing in detection_3d evaluation. ([#3838](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3838))

### Changed

-

### Deprecated

-

### Removed

-

### Fixed

- [metric] Fix a metric type error in sgnet whiling updating metrics. ([#3806](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3806))

## [1.3.0] - 2023-02-16

### Added

- [metric] Add new behavior prediction metrics and decouple traj_head and behav_head log generation. ([#3456](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3456))

- [metric] Add dataset and transformer for face3d based on perspective projection. ([#3557](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3557))

- [metric] Add pose and eye location metric for face3d_pp. ([#3573](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3573))

- [metric] Add person position metric. ([#3515](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3515))

- [evaluation] Update seg evaluation according to adas_eval. ([#3531](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3531))

### Changed

- [metric] Bev3d pr curves support more line color and bev3d draw color bbox support draw class id on image. ([#3578](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3578))

### Deprecated

-

### Removed

-

### Fixed

- [metric] Fix block of voc metric. ([#3458](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3458))

- [metric] Fix integration test of voc metric by rolling back. ([#3499](https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/-/merge_requests/3499))

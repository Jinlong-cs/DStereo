# Horizon Algorithm Toolkit (海图)

Documentation: [http://model.aidi.hobot.cc/api/gateway/v1/docs/latest?name=HAT](http://model.aidi.hobot.cc/api/gateway/v1/docs/latest?name=HAT)


## Introduction
HAT aims at developing efficient and user-friendly AI(Deep Learning) algorithm toolkit(based on Pytorch APIs) for Horizon BPU.

It also provides implementations of the state-of-the-art (SOTA) deep learning models including Image Classification, Objdect Detection, Semantic Segmentation tasks.


## Features
1. Based on the public pytorch 2.0.1 and [horizon pytorch plugin](http://gitlab.hobot.cc/ptd/ap/dlp/horizon_plugin_pytorch/).
2. SOTA results reported in research papers. And all examples are compatible with Horizon BPU.


## Installation
See [installation instructions](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/docs/source/quick_start/installation.md).



## Getting Started
See [quick_start with HAT](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/docs/source/quick_start/quick_start.md).


## Model Zoo
See [model zoo](http://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/HAT/blob/master/docs/source/model_zoo/model_zoo.md).
## How to contribute code
1. Clone this repo and checkout own branch
2. Modify code and commit

Before the first git commit command, please install the develop environment by `./dev/prepare_develop_env.sh`.

After installed, `pre-commit` checks the style of code in every commit. We will see something like the following when you run `git commit`:
```
Check added large files..................................................Passed
Fix end-of-file..........................................................Passed
Trailing whitespace......................................................Passed
Check merge conflict.....................................................Passed
Check python imports.....................................................Passed
Auto format python code..................................................Passed
Check pep8...............................................................Passed
```

3. Merge request:
This repo has automatic integration testing. cicd pipeline can be triggered automatically by merge request.

Developers can push code safely after got "Build Result: SUCCESS".


4. Merge by reviewer

**Note: We welcome contributions in any form, not just code! Please check [HAT Contribution Guide](https://oshcjymfmk.feishu.cn/docs/doccnRg02I4wBuERufaOBjXs1hc) for details about contributions.**

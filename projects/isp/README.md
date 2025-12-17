# Welcome to ISP project!

### How to start

Before start, please make sure you have read [readme](../../README.md) and got
everything ready.

```bash
git clone git@gitlab.hobot.cc:ptd/algorithm/ai-platform-algorithm/HAT.git
cd HAT

# Mount bucket. If you don't have permission, ask manager to help you.
# If you don't known how to mount, refer to http://wiki.hobot.cc/pages/viewpage.action?pageId=180630622
export HAT_BUCKET=/horizon-bucket

# Train locally
export PYTHONPATH=`pwd`
# Replace with other config if necessary.
# Remember to replace `ckpt_dir` and `json_save_prefix` in config accordding to your own environment.
python3 tools/train.py --config projects/isp/app/det/entry.py --stage float

# AIDI evaluation
# If it is your first time to do AIDI evaluation, please initialize your local configure. Please refer to http://user-manual.aidi.hobot.cc/docs/aidi-model/aidi-model-1d00rpmb21mor for more detail.
python3 tools/predict.py --config projects/isp/app/det/entry.py --stage float

```

Or you can use Makefile to simplify cmd.

```bash
cp projects/isp/GNUmakefile.template GNUmakefile

# Change `config` in GNUmakefile if you want.

# Train model in local.
make train

# Prediction in local.
make prediction

# Submit job to AIDI.
# 1. Change aidi `cluster` in GNUmakefile if you want.
# 2. Remember to set `local_train` to False
# 3. If you want to do qat, remember to set step to `qat` in config
make submit
```

### Raw Perception Experiment

#### J3 dataset

Please refer to https://horizonrobotics.feishu.cn/wiki/IwvPwHzSIi9Kwrk93mBcCO30nnd.

### About GNUmakefile

> By default, when make looks for the makefile, it tries the following names, in order: GNUmakefile, makefile and Makefile.

[Click me for more details.](https://www.gnu.org/software/make/manual/html_node/Makefile-Names.html)

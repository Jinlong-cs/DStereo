#!/usr/bin/env bash
# for torch1.13.0+cu116 ci env

set -e

export PATH=~/.local/bin:$PATH

# install deps
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
pip3 install --user torch==1.13.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${pip_ext}
pip3 install --user torchvision==0.14.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${pip_ext}
pip3 install --user torchaudio==0.13.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${pip_ext}
pip3 install --use-deprecated=legacy-resolver --user -U -r requirements.txt ${pip_ext}

# internal requirements
pip3 install --user -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu116/torch1130 --trusted-host art-internal.hobot.cc ${pip_ext}
pip3 install --user -U horizon-plugin-profiler ${pip_ext}
pip3 install --user pyyaml==5.3.1 ${pip_ext}  # temp for habtc
pip3 install --user hatbc==0.10.0b202309070700+b39a57d ${pip_ext}
pip3 install --user aidisdk==0.15.0 ${pip_ext}
pip3 install --user hat_sim ${pip_ext}
pip3 install -U --user pyramid_resizer ${pip_ext}

pip3 install --user aiohttp ${pip_ext}
# add optional apex
#hadoop fs -get hdfs://hobot-bigdata/user/zhigang.yang/horizon_algorithm_toolkit/dependence/apex/*

# install optional deps for test
# horizon_driving_dataset for 2.5d and bev task
# TODO waymo-open-dataset required by lidar task
pip3 install --use-deprecated=legacy-resolver --user timm hbdk>=3.26.5 horizon_driving_dataset==0.0.28 ${pip_ext}

# code style
echo "----------------lint with pre-commit----------------"
echo -e "\nfail_fast: true" >>.pre-commit-config.yaml
# TODO: should only run with diff files
pre-commit run --all-files
echo "----------------lint success!-----------------------"

# add optional openlab
echo "----------------install openlab----------------"
pip3 install --user yapf ${pip_ext}
pip3 install --user -U pycocotools ${pip_ext}
pip3 install --user mmcv-full==1.7.1 -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/mmcv/cu116/torch1130 ${pip_ext}
pip3 install --user mmdet -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/mmdet --trusted-host art-internal.hobot.cc ${pip_ext}
echo "-------------install openlab success!-------------"

# add optional nuscenes-devkit
echo "----------------install nuscenes-devkit----------------"
pip3 install --user nuscenes-devkit==1.1.11 ${pip_ext}
echo "-------------install nuscenes-devkit success!-------------"

pip3 install --user memray==1.6.0 ${pip_ext}
pip3 install --user deepspeed ${pip_ext}
aidi_host="http://aidi.hobot.cc"
ci_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNTgwMDE5NDMsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InpoaWdhbmcueWFuZyIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.LIISKqNmERemrWg-Q4i6Amwy7rU8zyThvrfD3EhE6MQhj772kp2R_JGHnnt6FVavrLzfiReYI9UiLlUOuN08abFfI_lqyNP1lB6-9NhnzHf-2gTmV2VVUwiSP6SPIyR1R8Wbwb_mMI8Yd1_zyoAE5r597fX8g06MchvLpRlUJheZUsgzKyMJyOTxnPtUVOHpkpel1i6gEx7S4DFBtDqNLw3oJMAdJ75DuIkTusbaW9Ko6NEWm1Jyrjz79zPRCY5w14JwiL13iEX6kpJcM7Q9b_Rumm7-QWh5hOKaluntgRlKLdlgI2d27dEE3QYQxyWvGwKnQ515LIBXyKtrP1xB1A"
aidisdk config --token ${ci_token} --endpoint ${aidi_host}

# face3d
pip3 install --user kornia lpips==0.1.4 ${pip_ext}

# hand3d
pip3 install --user smplx[all] chumpy ${pip_ext}

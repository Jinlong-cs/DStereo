#!/usr/bin/env bash
# for torch1.13.0+cu116 ci basic env

set -e

export PATH=~/.local/bin:$PATH

# install deps
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
pip3 install --user torch==1.13.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${pip_ext}
pip3 install --user torchvision==0.14.0+cu116 -i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple ${pip_ext}
pip3 install --use-deprecated=legacy-resolver --user -U -r requirements.txt ${pip_ext}

# internal requirements
pip3 install --user hbdk>=3.26.5 ${pip_ext}
pip3 install --user -U horizon-plugin-pytorch -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu116/torch1130 --trusted-host art-internal.hobot.cc ${pip_ext}
pip3 install --user -U horizon-plugin-profiler ${pip_ext}

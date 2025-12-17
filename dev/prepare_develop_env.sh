#!/usr/bin/env bash
set -x
# run ./dev/prepare_develop_dev.sh in <HAT>/

# add --user if install to local, or outside virturalenv.
pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
pip3 install $1 -r requirements/develop.txt ${pip_ext}
pre-commit install

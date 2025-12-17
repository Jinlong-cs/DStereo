#!/usr/bin/env bash

set -e

mkdir -p tmp_models

sh ./projects/toolchain/dev/bernoulli2/align_bpu_pipline.sh

sh ./projects/toolchain/dev/bayes/align_bpu_pipline.sh
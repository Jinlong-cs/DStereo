#!/usr/bin/env bash

set -e

export PYTHONPATH=`pwd`:${PYTHONPATH}
export SCRIPT_DIR=./projects/toolchain/dev/model_metric/
export OUTDIR=./model_zoo

# remove cache
rm -rf ${OUTDIR}
rm -rf ${OUTDIR}.tgz
mkdir -p ${OUTDIR}

bash ${SCRIPT_DIR}/bayes_perf.sh
bash ${SCRIPT_DIR}/bernoulli2_perf.sh

tar -zcvf ${OUTDIR}.tgz model_zoo

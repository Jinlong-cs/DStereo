#!/usr/bin/env bash

set -e

cfg=tests/data/toy_multitask/multitask.py


python3 tools/train.py --stage float -c ${cfg} --pipeline-test $@
python3 tools/train.py --stage qat -c ${cfg} --pipeline-test $@
python3 tools/train.py --stage int_infer -c ${cfg} --pipeline-test $@

if [ "${USE_DCU}" != "1" ]; then

sed -i "s/delay_sync=False/delay_sync=True/g" ${cfg}
python3 tools/train.py --stage float -c ${cfg} --pipeline-test $@
sed -i "s/delay_sync=True/delay_sync=False/g" ${cfg}


sed -i "s/grad_accumulation_step=1/grad_accumulation_step=2/g" ${cfg}
python3 tools/train.py --stage float -c ${cfg} --pipeline-test $@
sed -i "s/grad_accumulation_step=2/grad_accumulation_step=1/g" ${cfg}

fi


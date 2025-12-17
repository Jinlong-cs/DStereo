#!/bin/sh

set -e

prepare_env() {
    export PATH=/opt/env/horizon_env/bin/:~/.local/bin:/root/.local/bin:/home/cicd/.local/bin/:$PATH
    
    # install deps
    pip_ext="-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"
    plugin_whl_dcu=https://art-internal.hobot.cc:443/artifactory/custom-algo-pypi/horizon-plugin-pytorch/dcu
    pip3 install --user numpy==1.23.5 ${pip_ext}
    pip3 install --use-deprecated=legacy-resolver --user -r requirements.txt ${pip_ext}
    pip3 install --user -U --pre horizon-plugin-pytorch -f ${plugin_whl_dcu} --trusted-host art-internal.hobot.cc ${pip_ext}
    pip3 install --user -U horizon-plugin-profiler ${pip_ext}
    pip3 install -U --user pyramid_resizer ${pip_ext}
    pip3 install --user hat_sim ${pip_ext}
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
    pip3 install --user mmcv-full -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/mmcv/cu111/torch1100 -i http://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc --trusted-host art-internal.hobot.cc
    pip3 install --user mmdet -f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/mmdet --trusted-host art-internal.hobot.cc ${pip_ext}
    echo "-------------install openlab success!-------------"
    
    # add optional nuscenes-devkit
    echo "----------------install nuscenes-devkit----------------"
    pip3 install --user nuscenes-devkit==1.1.9 ${pip_ext}
    echo "-------------install nuscenes-devkit success!-------------"
    
    # auto-matrix changed the version of scikit-learn
    pip3 install --user scikit-learn==0.22 ${pip_ext}
    pip3 install --user memray==1.6.0 ${pip_ext}
    aidi_host="http://aidi.hobot.cc"
    ci_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNTgwMDE5NDMsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InpoaWdhbmcueWFuZyIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.LIISKqNmERemrWg-Q4i6Amwy7rU8zyThvrfD3EhE6MQhj772kp2R_JGHnnt6FVavrLzfiReYI9UiLlUOuN08abFfI_lqyNP1lB6-9NhnzHf-2gTmV2VVUwiSP6SPIyR1R8Wbwb_mMI8Yd1_zyoAE5r597fX8g06MchvLpRlUJheZUsgzKyMJyOTxnPtUVOHpkpel1i6gEx7S4DFBtDqNLw3oJMAdJ75DuIkTusbaW9Ko6NEWm1Jyrjz79zPRCY5w14JwiL13iEX6kpJcM7Q9b_Rumm7-QWh5hOKaluntgRlKLdlgI2d27dEE3QYQxyWvGwKnQ515LIBXyKtrP1xB1A"
    aidisdk config --token ${ci_token} --endpoint ${aidi_host}
    
    # face3d
    pip3 install --user kornia lpips==0.1.4 ${pip_ext}
    
    # hand3d
    pip3 install --user smplx[all] chumpy ${pip_ext}
}

run_integeration_and_uint_test() {
    export PATH=/opt/env/horizon_env/bin:~/.local/bin:/root/.local/bin:/home/cicd/.local/bin/:$PATH
    export PYTHONPATH=$(pwd):${PYTHONPATH}
    export USE_DCU=1
    # run tests
    echo "----------------run stable test with pytest----------------"
    make serial-unit-test
    make intergration-tests
    echo "----------------test success!-----------------------"
}

run_dcu_ci() {
    prepare_env
    run_integeration_and_uint_test
}

run_as_root() {
    expect dev/run_dcu_ci_as_root.sh "dev/dcu_ci.sh run_dcu_ci"
}

"$@"

stage('torch compat test') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('torch-compat') {
        mound_test_data()
        print_info()
        exec_torch_compat_test()
    }
}

stage('build torch compat docker') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('torch-compat') {
        build_torch_compat_docker()
    }
}

def exec_torch_compat_test() {
    sh '''#!/bin/bash +x
        nohup bash -ex dev/gpu_monitors.sh &
    '''
    sh '''#!/bin/bash -exl
        set -e
        bash -ex dev/gpfs_mount_test.sh
        bash -ex dev/prepare_ci_torch1130_cu116_env.sh
        bash -ex dev/stable_ci_test.sh
        bash -ex dev/ci_test.sh
    '''
    cobertura coberturaReportFile: 'cov_compat.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

def mound_test_data() {
    sh '''#!/bin/bash -exl
        ln -snf /horizon-bucket/HDLTAlgorithm/data/pack_data/ tmp_data
        ln -snf /horizon-bucket/HDLTAlgorithm/data/orig_data/ tmp_orig_data
        ls tmp_data
        ls tmp_orig_data
    '''
}

def print_info() {
    sh 'pwd'
    sh 'env'
    sh 'ls'
}

def build_torch_compat_docker(){
    sh '''#!/bin/bash -exl
        # bash -ex dev/dockerfiles/python38/build_runtime.sh
    '''
}

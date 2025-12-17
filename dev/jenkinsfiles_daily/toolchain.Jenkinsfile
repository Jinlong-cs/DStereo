stage('daily build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        // add_user()
        mound_test_data()
        print_info()
        exec_daily_test()
    }
}

def print_info() {
    sh 'pwd'
    sh 'env'
    sh 'ls'
}

def add_user() {
    sh '''
        useradd cicd -u 10271
        useradd jenkins -u 1000
        usermod -a -G root,jenkins cicd
        chmod g+w -R ${WORKSPACE}
    '''
}

def mound_test_data() {
    sh '''#!/bin/bash -exl
        ln -snf /horizon-bucket/HDLTAlgorithm/data/pack_data/ tmp_data
        ln -snf /horizon-bucket/HDLTAlgorithm/data/orig_data/ tmp_orig_data
        ln -snf /horizon-bucket/HDLTAlgorithm/models/bayes_release_models ./tmp_pretrained_models
        ls tmp_data
        ls tmp_orig_data
        ls tmp_pretrained_models
    '''
}

def exec_daily_test() {
    sh '''#!/bin/bash +x
        nohup bash -ex dev/gpu_monitors.sh &
    '''
    sh '''#!/bin/bash -exl
        set -e
        bash -ex dev/gpfs_mount_test.sh
        bash -ex dev/prepare_ci_torch1130_cu116_basic_env.sh
        bash projects/toolchain/dev/daily_build.sh ${BUILD_URL}
    '''
    cobertura coberturaReportFile: 'cov_daily.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

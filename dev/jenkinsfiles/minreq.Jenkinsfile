stage('serial build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }

    if (env.gitlabActionType != 'PUSH') {
        horizonContainer('build') {
            // add_user()
            prepare_ci_env()

            sh 'pwd'
            sh 'env'
            sh 'ls'
            sh '''#!/bin/bash +x
                nohup bash -ex dev/gpu_monitors.sh &
            '''

            stage('unittest'){
                withEnv(["CHECK_NEED_REBASE_WITH_SELF=1"]) {
                    exec_build_ut()
                }
            }
        }
    }
}

stage('publish') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
    }
}


def prepare_ci_env() {
    sh '''#!/bin/bash -exl
        bash -ex dev/prepare_ci_torch201_cu118_basic_env.sh
    '''
}

def exec_build_ut() {
    sh '''#!/bin/bash -exl
        if [ $TAG_NAME ]; then
            # cd pipeline
            ls
        else
            # mount cicd data from HAT gpfs
            ln -snf /horizon-bucket/HDLTAlgorithm/data/pack_data/ tmp_data
            ln -snf /horizon-bucket/HDLTAlgorithm/data/orig_data/ tmp_orig_data
            ls tmp_data
            ls tmp_orig_data
            # ci pipeline
            set -e
            bash -ex dev/gpfs_mount_test.sh
            pip3 list
            bash -ex dev/ci_test.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

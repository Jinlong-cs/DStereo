stage('daily build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        mound_test_data()
        exec_build()
    }
}

def mound_test_data() {
    sh '''#!/bin/bash -exl
        ln -snf /horizon-bucket/HDLTAlgorithm/data/pack_data/ tmp_data
        ln -snf /horizon-bucket/HDLTAlgorithm/data/orig_data/ tmp_orig_data
        ls tmp_data
        ls tmp_orig_data
    '''
}

def exec_build() {
    sh 'pwd'
    sh 'env'
    sh 'ls'
    sh '''#!/bin/bash +x
        nohup bash -ex dev/gpu_monitors.sh &
    '''
    sh '''#!/bin/bash -exl
        if [ $TAG_NAME ]; then
            # cd pipeline
            ls
        else
            set -e
            bash -ex dev/gpfs_mount_test.sh
            # maybe remove to halo daily_build.sh
            bash -ex dev/prepare_ci_torch1102_cu111_env.sh
            bash -ex projects/halo/cv/dev/daily_build.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

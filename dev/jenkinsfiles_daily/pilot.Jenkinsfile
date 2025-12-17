stage('daily build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        // add_user()
        mount_test_data()
        exec_build()
    }
}

def add_user() {
    sh '''
        useradd cicd -u 10271
        useradd jenkins -u 1000
        usermod -a -G root,jenkins cicd
        chmod g+w -R ${WORKSPACE}
    '''
}

def mount_test_data() {
    sh '''#!/bin/bash -exl
        ln -snf /horizon-bucket/HDLTAlgorithm/data/pack_data/ tmp_data
        ln -snf /horizon-bucket/HDLTAlgorithm/data/orig_data/ tmp_orig_data
        ln -snf /horizon-bucket/HDLTAlgorithm/models/ tmp_models
        ls tmp_data
        ls tmp_orig_data
        ls tmp_models
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
            # all of pilot daily tests
            export WORKING_PATH=`pwd`
            export PBS_JOBNAME=${JOB_ID}
            bash projects/pilot/dev/daily_build.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
    archiveArtifacts allowEmptyArchive: true, artifacts: 'artifacts/*.csv', fingerprint: false, onlyIfSuccessful: false
}

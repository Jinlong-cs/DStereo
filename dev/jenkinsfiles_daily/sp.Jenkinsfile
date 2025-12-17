stage('daily build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        // add_user()
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
            # daily ci no longer maintained.
            # export WORKING_PATH=`pwd`
            # export PBS_JOBNAME=${JOB_ID}
            # bash projects/superparking/dev/daily_build.sh
            set -e
            bash -ex projects/superparking/dev/ci_test.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

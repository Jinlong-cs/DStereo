stage('serial build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        exec_build()
    }
    stage("doc") {
        publishHTML target:[
            allowMissing: false,
            alwaysLinkToLastBuild: false,
            keepAll: true,
            reportDir: 'projects/superparking/release_package/docs/build/html/',
            reportFiles: 'index.html',
            reportName: "Sphinx Docs"
        ]
    } // stage("doc")
}


def exec_build() {
    sh 'pwd'
    sh 'env'
    sh 'ls'
    sh '''#!/bin/bash +x
        nohup bash -ex dev/gpu_monitors.sh &
    '''
    sh '''#!/bin/bash -exl
        if [ ${gitlabActionType} = "TAG_PUSH" ]; then
            # cd pipeline
            ls
        else
            # superparking ci pipeline
            set -e
            bash -ex projects/superparking/dev/ci_test.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

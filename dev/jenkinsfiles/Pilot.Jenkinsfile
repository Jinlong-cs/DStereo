stage('serial build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }

    horizonContainer('build') {
        if (env.gitlabActionType != 'PUSH' && env.gitlabActionType != 'TAG_PUSH')  {
        
            sh 'pwd'
            sh 'env'
            sh 'ls'
            sh '''#!/bin/bash +x
                nohup bash -ex dev/gpu_monitors.sh &
            '''
            
            stage('pilot_integration'){
                withEnv(["CHECK_NEED_REBASE_WITH_SELF=1"]) {
                    exec_build_pilot()
                }
            }
        }
    }
}

stage('publish') {
    horizonContainer('build') {
        if (env.gitlabActionType == 'TAG_PUSH') {
            run_pilot_release()
        }
    }
}

def exec_build_pilot() {
    sh '''#!/bin/bash -exl
        if [ ${gitlabActionType} = "TAG_PUSH" ]; then
            # cd pipeline
            ls
        else
            # pilot ci pipeline
            set -e
            bash -ex projects/pilot/dev/ci_test.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
    archiveArtifacts allowEmptyArchive: true, artifacts: 'projects/pilot/release_package/artifacts/*.csv', fingerprint: false, onlyIfSuccessful: false
}


def run_pilot_release() {
    sh """#!/bin/bash -exl
        bash -ex projects/pilot/dev/publish/publish.sh
    """
}

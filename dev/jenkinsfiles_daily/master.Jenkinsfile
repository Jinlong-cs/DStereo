stage('daily build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        // add_user()
        mound_test_data()
        print_info()
        exec_daily_test()
        build_python38_runtime_docker()
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
        ls tmp_data
        ls tmp_orig_data
    '''
}

def exec_daily_test() {
    sh '''#!/bin/bash +x
        nohup bash -ex dev/gpu_monitors.sh &
    '''
    sh '''#!/bin/bash -exl
        set -e
        bash -ex dev/gpfs_mount_test.sh
        bash -ex dev/prepare_ci_torch201_cu118_env.sh
        bash -ex dev/stable_ci_test.sh
        bash -ex dev/ci_test.sh
    '''
    cobertura coberturaReportFile: 'cov_daily.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

def build_python38_runtime_docker(){
    // set_docker_repo_auth
    withCredentials([usernamePassword(credentialsId: '666831ca-4c0f-411d-a707-6436a1c4a33f', usernameVariable: 'username', passwordVariable: 'password')]) {
        sh ''' 
            set -e
            df -h
            nvidia-smi
            export NCCL_DEBUG=WARN            
            export DOCKER_CONFIG=/tmp/buildkit/.docker
            mkdir -p $DOCKER_CONFIG 
            auth=`echo -n "${username}:${password}" | base64`
            echo '{"auths":{"docker.hobot.cc":{"auth":"'${auth}'"}}}' > $DOCKER_CONFIG/config.json
            cat $DOCKER_CONFIG/config.json
        '''
    }
    
    // build docker
    sh '''#!/bin/bash -exl
        export DOCKER_CONFIG=/tmp/buildkit/.docker
        bash -ex dev/dockerfiles/python38/build_runtime.sh
    '''
}

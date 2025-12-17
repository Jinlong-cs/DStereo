stage('python compat test') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('python-compat') {
        mound_test_data()
        print_info()
        exec_python_compat_test()
        upload_whl()
    }
}

stage('build python compat docker') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('python-compat') {
        build_python_compat_docker()
    }
}

def exec_python_compat_test() {
    sh '''#!/bin/bash +x
        nohup bash -ex dev/gpu_monitors.sh &
    '''
    sh '''#!/bin/bash -exl
        set -e
        bash -ex dev/gpfs_mount_test.sh
        bash -ex dev/prepare_ci_torch201_cu118_env.sh
        bash -ex dev/ci_test.sh
        bash -ex dev/stable_ci_test.sh
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

def upload_whl() {
    withCredentials([usernamePassword(
                                credentialsId: 'bc8aeaa1-6e19-4727-afd5-beb62e7b5884',
                                usernameVariable: 'username',
                                passwordVariable: 'password')]) {

        sh """
            echo '
            [distutils]
            index-servers = hobot-local

            [hobot-local]
            repository: https://pypi.hobot.cc/hobot-local/
            username: ${username}
            password: ${password}
            ' > ~/.pypirc
        """

    }
    sh """#!/bin/bash -exl
        # python3 setup.py bdist_wheel --python-tag py310 upload -r hobot-local
    """
}

def build_python_compat_docker(){
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
        bash -ex dev/dockerfiles/python310/build_runtime.sh
    '''
}

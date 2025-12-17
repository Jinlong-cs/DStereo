stage('serial build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        // add_user()
        mound_test_data()
        print_info()
        exec_build()
    }
}
stage('publish') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('build') {
        if (env.gitlabActionType == 'TAG_PUSH') {
            mound_test_data()
            print_info()
            exec_stable_build()
            retry(3) {
                upload_whl()
            }
            retry(3) {
                upload_doc()
            }
        }
    }
}

def print_info() {
    sh 'pwd'
    sh 'env'
    sh 'ls'
}

def mound_test_data() {
    sh '''#!/bin/bash -exl
        ln -snf /horizon-bucket/HDLTAlgorithm/data/pack_data/ tmp_data
        ln -snf /horizon-bucket/HDLTAlgorithm/data/orig_data/ tmp_orig_data
        ls tmp_data
        ls tmp_orig_data
    '''
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
    sh '''#!/bin/bash +x
        nohup bash -ex dev/gpu_monitors.sh &
    '''
    sh '''#!/bin/bash -exl
        if [ $TAG_NAME ]; then
            # cd pipeline
            ls
        else
            # ci pipeline
            set -e
            bash -ex dev/gpfs_mount_test.sh
            bash -ex dev/prepare_ci_torch201_cu118_env.sh
            export RELEASE_VERSION=True && bash -ex dev/ci_test.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov_release.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
}

def exec_stable_build() {
    sh '''#!/bin/bash -exl
        if [ $TAG_NAME ]; then
            # cd pipeline
            ls
        else
            # ci pipeline
            set -e
            bash -ex dev/gpfs_mount_test.sh
            bash -ex dev/prepare_ci_torch201_cu118_env.sh
            export RELEASE_VERSION=True && bash -ex dev/stable_ci_test.sh
        fi
    '''
    cobertura coberturaReportFile: 'cov.xml', enableNewApi: true, onlyStable: false, maxNumberOfBuilds: 0, failNoReports: false
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
        export RELEASE_VERSION=True && python3 setup.py bdist_wheel upload -r hobot-local
    """
}


def upload_doc() {
    hat_version = sh (script: "ls dist", returnStdout: true).trim().split('-')[1].replaceAll('\\+', '\\-')
    sh """#!/bin/bash -exl
        wget -c https://gallery.hobot.cc/download/algorithmplatform/aitc/aiplatform_hitc/project/release/linux/x86_64/general/basic/latest -O `pwd`/hitc.tar
        tar -xvf `pwd`/hitc.tar
        chmod +x `pwd`/output_linux/hitc
        # cp `pwd`/output_linux/hitc /usr/local/bin/

        `pwd`/output_linux/hitc init -t eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjIyNTgwMDE5NDMsIlRva2VuVHlwZSI6ImxkYXAiLCJVc2VyTmFtZSI6InpoaWdhbmcueWFuZyIsIk9yZ2FuaXphdGlvbiI6InJlZ3VsYXItZW5naW5lZXIiLCJPcmdhbml6YXRpb25JRCI6MX0.LIISKqNmERemrWg-Q4i6Amwy7rU8zyThvrfD3EhE6MQhj772kp2R_JGHnnt6FVavrLzfiReYI9UiLlUOuN08abFfI_lqyNP1lB6-9NhnzHf-2gTmV2VVUwiSP6SPIyR1R8Wbwb_mMI8Yd1_zyoAE5r597fX8g06MchvLpRlUJheZUsgzKyMJyOTxnPtUVOHpkpel1i6gEx7S4DFBtDqNLw3oJMAdJ75DuIkTusbaW9Ko6NEWm1Jyrjz79zPRCY5w14JwiL13iEX6kpJcM7Q9b_Rumm7-QWh5hOKaluntgRlKLdlgI2d27dEE3QYQxyWvGwKnQ515LIBXyKtrP1xB1A
        `pwd`/output_linux/hitc doc upload --name HAT --version ${hat_version} --desc "hat" --rootpath ./docs/build/html --rootfile index.html -y
        # for meiyan, they can't open aidi
        # yum -y install sshpass
        sshpass -p sxs.Yox-13s scp -r ./docs/build/html hatdoc@10.40.11.14:/var/www/
    """
}


def build_python38_release_docker(){
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
        bash -ex dev/dockerfiles/python38/build_release.sh
    '''
}

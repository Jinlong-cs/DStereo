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
        if (env.gitlabActionType == 'PUSH') {
            // add_user()
            prepare_ci_env()
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


def add_user() {
    sh '''
        useradd cicd -u 10271
        useradd jenkins -u 1000
        usermod -a -G root,jenkins cicd
        chmod g+w -R ${WORKSPACE}
    '''
}

def merge_master_check() {
    sh '''
        # ssh key init
        git config --global user.name hat-public
        git config --global user.email hat-public@horizon.ai
        git config --global color.ui false
        rm -rf ~/.ssh
        mkdir ~/.ssh
        cp ${SSH_KEY_FILE} ~/.ssh/id_ed25519
        chmod 600 ~/.ssh/id_ed25519
        ls -lsha ~/.ssh/
        eval `ssh-agent`
        ssh-add  ~/.ssh/id_ed25519
        ssh-add -l

        # new clone for merge base id
        git clone "${gitlabSourceRepoSshUrl}" -b "${gitlabSourceBranch}" test_branch
        cd test_branch
        merge_base=$(git merge-base origin/master HEAD)
        cd ../
        rm -rf test_branch

        master_id=$(git rev-parse origin/master)

        if [ "$merge_base" != "$master_id" ]; then
            echo "Please merge or rebase current branch from the latest master!"
            exit 1
        fi
    '''
}

def prepare_ci_env() {
    sh '''#!/bin/bash -exl
        bash -ex dev/prepare_ci_torch201_cu118_env.sh
    '''
}

def exec_build_ut() {
    sh '''#!/bin/bash -exl
        if [ $TAG_NAME ]; then
            # cd pipeline
            ls
        else
            # check mr format
            python3 dev/QAC/check_mr.py
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


def exec_stable_build() {
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
            bash -ex dev/prepare_ci_torch201_cu118_env.sh
            bash -ex dev/stable_ci_test.sh
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
        python3 setup.py bdist_wheel upload -r hobot-local
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

stage('serial build') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }

    if (env.gitlabActionType != 'PUSH') {
        container('build') {
            add_user()

            sh 'pwd'
            sh 'env'
            sh 'ls'
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
        echo "StrictHostKeyChecking no" >>  /etc/ssh/ssh_config
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

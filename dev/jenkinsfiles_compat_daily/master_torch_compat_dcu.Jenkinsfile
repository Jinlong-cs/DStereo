stage('torch compat test') {
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    horizonContainer('torch-compat') {
        mound_test_data()
        print_info()
        exec_torch_compat_test()
    }
}

def exec_torch_compat_test() {
    sh '''
        sh dev/gpfs_mount_test.sh
        sh dev/dcu_ci.sh run_as_root
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

from aidisdk import AIDIClient, get_docker_image
import os
from aidisdk.compute.dag import DagJobCustomConfig, DagStatus, JobHook
from aidisdk.compute.job_abstract import (
    GitCodeItem,
    RunningResourceConfig,
    StartUpConfig,
    MountItem, JobMountType, MountMode,
)
from aidisdk.compute.package_abstract import (
    CodePackageConfig,
    LocalPackageItem,
)
import time
import numpy as np
client = AIDIClient()

def vis_yuanxiang(parent_jobs=[]):
    data_root = "/horizon-bucket/d-robotics-bucket/bohao.zhang/yuanxiang/yuanxiang/"
    end_dag_list = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        for scene in filenames:
            if not scene.endswith('.bag'):
                continue
            scene_root = os.path.join(dirpath, scene.replace(".bag", ""))
            bag_root = os.path.join(dirpath, scene)
            
            tmp_name = "%s_vis_%s_%s" % (taskname, scene.replace(".bag", "").replace("-", ""), str(time.time()).replace(".", ""))
            vis = dag.new_job(
                name=tmp_name,
                job_type="train",
                code_package=CodePackageConfig(
                    raw_package=LocalPackageItem(
                        lpath="/home/users/bohao.zhang/Projects/PTQ/HAT_PTQ/dataflow/", encrypt_passwd="123765", follow_softlink=False
                    ),
                    # git_packages=[],
                ),
                startup=StartUpConfig(
                    command="source /opt/ros/noetic/setup.bash && source /root/slam_ws/devel/setup.bash && python3 infer_disp_to_mcap_yuanxiang_720p.py %s %s" % (scene_root, taskname),
                    startup_dir="/running_package/code_package"
                ),
                queue_name=gpu_queue,
                # msg_file_path="prelabel.log",
                running_resource=RunningResourceConfig(
                    docker_image="docker.hobot.cc/aiot/stereo_dataprepare:vis",  # noqa
                    instance=1,
                    cpu=12,
                    gpu=1,
                    cpu_mem_ratio=8,
                    walltime=20000,
                ),
                mount=[MountItem(JobMountType.BUCKET, "d-robotics-bucket", MountMode.READ_AND_WRITE),],
                priority=5,
                schedule_type=None,
                max_retries=0,
                support_async_run=False,
                desc=tmp_name,
                parent_jobs=parent_jobs,
            )
            print("vis job done. ", tmp_name) 

def vis_zed_ros2_db3(data_root, parent_jobs=[]):   
    end_dag_list = []
    for scene in os.listdir(data_root):
        if not scene.endswith(".db3"):
            continue
        scene_root = os.path.join(data_root, scene.replace(".db3", ""))
        bag_root = os.path.join(data_root, scene)
        # parent_jobs = []
        if not os.path.exists(scene_root):
            continue
        tmp_name = "%s_vis_%s" % (taskname, scene.replace(".db3", "").replace("-", ""))
        vis = dag.new_job(
            name=tmp_name,
            job_type="train",
            code_package=CodePackageConfig(
                raw_package=LocalPackageItem(
                    lpath="/home/users/bohao.zhang/Projects/PTQ/HAT_PTQ/dataflow/", encrypt_passwd="123765", follow_softlink=False
                ),
                # git_packages=[],
            ),
            startup=StartUpConfig(
                command="source /opt/ros/noetic/setup.bash && source /root/slam_ws/devel/setup.bash && python3 infer_disp_to_mcap_zed_720p.py %s %s" % (scene_root, taskname),
                startup_dir="/running_package/code_package"
            ),
            queue_name=gpu_queue,
            # msg_file_path="prelabel.log",
            running_resource=RunningResourceConfig(
                docker_image="docker.hobot.cc/aiot/stereo_dataprepare:vis",  # noqa
                instance=1,
                cpu=12,
                gpu=1,
                cpu_mem_ratio=8,
                walltime=20000,
            ),
            mount=[MountItem(JobMountType.BUCKET, "d-robotics-bucket", MountMode.READ_AND_WRITE),],
            priority=5,
            schedule_type=None,
            max_retries=0,
            support_async_run=False,
            desc=tmp_name,
            parent_jobs=parent_jobs,
        )
        print("vis job done. ", tmp_name)

def do_eval(parent_jobs=[]):
    # eval
    for dataset in ["Middlebury", "KITTI", "InStereo2K"]:
        tmp_name = "%s_infer_%s" % (taskname, dataset)
        infer = dag.new_job(
            name=tmp_name,
            job_type="train",
            startup=StartUpConfig(
                command="bash eval_infer.sh %s %s" % (dataset, "/tmp"),
                startup_dir="/running_package/code_package"
            ),
            queue_name=gpu_queue,
            # msg_file_path="prelabel.log",
            running_resource=RunningResourceConfig(
                docker_image="docker.hobot.cc/aiot/hat:runtime-py3.8-torch2.0.1-cu118-2.2.1-dag",  # noqa
                instance=1,
                cpu=12,
                gpu=1,
                cpu_mem_ratio=8,
                walltime=20000,
            ),
            mount=[MountItem(JobMountType.BUCKET, "d-robotics-bucket", MountMode.READ_AND_WRITE),],
            priority=5,
            schedule_type=None,
            max_retries=0,
            support_async_run=False,
            desc=tmp_name,
            parent_jobs=parent_jobs,
        )
        print("eval job done. ", tmp_name)

def vis_zed(data_root, fx, fy, parent_jobs=[]):
    for scene in os.listdir(data_root):
        scene_root = os.path.join(data_root, scene)
        if not os.path.isdir(scene_root):
            continue
        tmp_name = ("%s_vis_%s_%s" % (taskname, scene, str(time.time()).replace(".", ""))).lower().replace("-", "_")[-80:]
        vis = dag.new_job(
            name=tmp_name,
            job_type="train",
            code_package=CodePackageConfig(
                raw_package=LocalPackageItem(
                    lpath="/home/users/bohao.zhang/Projects/PTQ/HAT_PTQ/dataflow/", encrypt_passwd="123765", follow_softlink=False
                ),
                # git_packages=[],
            ),
            startup=StartUpConfig(
                command="source /opt/ros/noetic/setup.bash && source /root/slam_ws/devel/setup.bash && python3 infer_disp_to_mcap_zed_720p.py %s %s" % (scene_root, taskname),
                startup_dir="/running_package/code_package"
            ),
            queue_name=gpu_queue,
            # msg_file_path="prelabel.log",
            running_resource=RunningResourceConfig(
                docker_image="docker.hobot.cc/aiot/stereo_dataprepare:vis",  # noqa
                instance=1,
                cpu=12,
                gpu=1,
                cpu_mem_ratio=8,
                walltime=20000,
            ),
            mount=[MountItem(JobMountType.BUCKET, "d-robotics-bucket", MountMode.READ_AND_WRITE),],
            priority=5,
            schedule_type=None,
            max_retries=0,
            support_async_run=False,
            desc=tmp_name,
            parent_jobs=parent_jobs,
        )
        print("vis job done. ", tmp_name)

def vis_zed_folder(scene_root, fx, fy, parent_jobs=[]):
    scene = os.path.basename(scene_root)
    tmp_name = ("%s_vis_%s_%s" % (taskname, scene, str(time.time()).replace(".", ""))).lower().replace("-", "_")[-80:]
    vis = dag.new_job(
        name=tmp_name,
        job_type="train",
        code_package=CodePackageConfig(
            raw_package=LocalPackageItem(
                lpath="/home/users/bohao.zhang/Projects/PTQ/HAT_PTQ/dataflow/", encrypt_passwd="123765", follow_softlink=False
            ),
            # git_packages=[],
        ),
        startup=StartUpConfig(
            command="source /opt/ros/noetic/setup.bash && source /root/slam_ws/devel/setup.bash && python3 infer_disp_to_mcap_zed_720p.py %s %s" % (scene_root, taskname),
            startup_dir="/running_package/code_package"
        ),
        queue_name=gpu_queue,
        # msg_file_path="prelabel.log",
        running_resource=RunningResourceConfig(
            docker_image="docker.hobot.cc/aiot/stereo_dataprepare:vis",  # noqa
            instance=1,
            cpu=12,
            gpu=1,
            cpu_mem_ratio=8,
            walltime=20000,
        ),
        mount=[MountItem(JobMountType.BUCKET, "d-robotics-bucket", MountMode.READ_AND_WRITE),],
        priority=5,
        schedule_type=None,
        max_retries=0,
        support_async_run=False,
        desc=tmp_name,
        parent_jobs=parent_jobs,
    )
    print("vis job done. ", tmp_name)

def add_job(taskname, scene_name, scene_root, parent_jobs, filename):
    tmp_name = ("%s_vis_%s_%s" % (taskname, scene_name, str(time.time()).replace(".", ""))).lower().replace("-", "_")[-80:]
    vis = dag.new_job(
        name=tmp_name,
        job_type="train",
        code_package=CodePackageConfig(
            raw_package=LocalPackageItem(
                lpath="/home/users/bohao.zhang/Projects/PTQ/HAT_PTQ/dataflow/", encrypt_passwd="123765", follow_softlink=False
            ),
            # git_packages=[],
        ),
        startup=StartUpConfig(
            command="source /opt/ros/noetic/setup.bash && source /root/slam_ws/devel/setup.bash && python3 viz_%s.py %s %s" % (filename, scene_root, taskname),
            startup_dir="/running_package/code_package"
        ),
        queue_name=gpu_queue,
        # msg_file_path="prelabel.log",
        running_resource=RunningResourceConfig(
            docker_image="docker.hobot.cc/aiot/stereo_dataprepare:vis",  # noqa
            instance=1,
            cpu=12,
            gpu=1,
            cpu_mem_ratio=8,
            walltime=20000,
        ),
        mount=[MountItem(JobMountType.BUCKET, "d-robotics-bucket", MountMode.READ_AND_WRITE),],
        priority=5,
        schedule_type=None,
        max_retries=0,
        support_async_run=False,
        desc=tmp_name,
        parent_jobs=parent_jobs,
    )
    print("vis job done. ", tmp_name)

if __name__ == "__main__":
    taskname = ("ai_stereo_orinal_%s" % str(time.time())).replace(".", "").replace("-", "_")[-80:]
    docker_image = ""
    cpu_queue = "share-idc-newage-cpu"
    # gpu_queue = "project-4090-robot-algorithm-idc-newage"
    gpu_queue = "share-4090-small-idc-newage"
    # gpu_queue = "preempt-4090-gpu-idc-newage"
    # gpu_queue = "svc-aip-cpu"

    tmp_name = taskname
    dag = client.dag.new_dag(
        name=tmp_name,
        project_id="GA2020005",
        queue_name=gpu_queue,
        code_package=CodePackageConfig(
                raw_package=LocalPackageItem(
                    lpath="/home/users/zengpeng.sun/DStereoV2.1/", encrypt_passwd="123765", follow_softlink=False
                ),
                # git_packages=[],
            ),
        desc="dataflow %s dag" % taskname,
        priority=5,
    )
    print("new_dag done.", tmp_name)
    parent_jobs = []

    train = dag.new_job(
        name="train_%s" % taskname,
        job_type="train",
        startup=StartUpConfig(
            command="bash train.sh",
            startup_dir="/running_package/code_package"
        ),
        queue_name=gpu_queue,
        running_resource=RunningResourceConfig(
            # docker_image="docker.hobot.cc/aiot/hat:runtime-py3.8-torch2.0.1-cu118-2.2.1-dag",  # noqa
            docker_image="docker.hobot.cc/aiot/hat231:runtime-py3.10-torch1.13.0-cu116-cudnn890-2.3.1-dag",  # noqa
            instance=1,
            cpu=12,
            gpu=8,
            # gpu=1,
            cpu_mem_ratio=8,
            walltime=20000,
        ),
        mount=[MountItem(JobMountType.BUCKET, "d-robotics-bucket", MountMode.READ_AND_WRITE),],
        priority=5,
        schedule_type=None,
        max_retries=0,
        support_async_run=False,
        desc="train %s job" % taskname,
    )
    parent_jobs.append(train)
    print("train job done.")


    # 提交dag
    client.dag.submit_dag(dag, timeout=60)
    # 提交dag 全局上下文参数

    # 核对Dag是否正常进入队列，正常开始排队。可以及时发现问题。
    client.dag.check_into_queue(dag=dag, timeout=60)

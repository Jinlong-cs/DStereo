"""Temporary script for submitting AIDI elastic training tasks."""

import json
from datetime import datetime
from typing import List, Optional

import requests
from aidisdk.compute.job_abstract import (
    MountItem,
    MountMode,
    RunningResourceConfig,
    StartUpConfig,
)
from aidisdk.compute.package_abstract import CodePackageConfig
from aidisdk.compute.single_job import SingleJobModule
from aidisdk.infra.session import Session

PRIORITY = {
    "5": "high",
    "4": "medium",
    "3": "low",
}

# TODO(mengyang.duan): remove
ENDPOINT = "http://aidi.hobot.cc"
ENDPOINT_API_URL = "http://api.aidi.hobot.cc/infra/api/v1alpha/job_manager/job/create"  # noqa E501


class SingleJob(SingleJobModule):
    def create(
        self,
        job_name: str,
        job_type: str,
        project_id: str,
        queue_name: str,
        running_resource: RunningResourceConfig,
        startup: StartUpConfig,
        code_package: CodePackageConfig,
        job_desc: Optional[str] = None,
        mount: List[MountItem] = None,
        priority: int = 3,
        max_reschedulings: int = 0,
        reschedule_policy: str = "default",
    ):

        job_name = job_name.replace(
            "@datetime", datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if mount:
            mount_config = {"buckets": []}
            for mount_item in mount:

                mount_config["buckets"].append(
                    {
                        "name": mount_item.name,
                        "read_only": mount_item.mode == MountMode.READ_ONLY,
                    }
                )

        else:
            mount_config = {}

        self.handle_package(queue_name, code_package, startup)
        working_path = (
            code_package.fs_package.package_dir_in_job
            + "/"
            + code_package.fs_package.package_name
        )
        single_job_config = {
            "job_name": job_name,
            "job_type": "torch",
            "job_max_running_time_minutes": running_resource.walltime,
            "priority": PRIORITY.get(str(priority), "high"),
            "queue": queue_name,
            "project_id": project_id,
            "command": f"sh {working_path}/job.sh",
            "mount_config": mount_config,
            "envs": [
                {"name": "USER_TOKEN", "value": f"{self.session.token}"},
                {"name": "AIDI_ENDPOINT", "value": f"{self.session.endpoint}"},
            ],
            "run_policy": {
                "max_reschedulings": max_reschedulings,
                "reschedule_policy": reschedule_policy,
            },
            "code_source": {
                "package": {
                    "save_paths": [code_package.fs_package.rpath],
                    "password": code_package.fs_package.encrypt_passwd,
                    "dir_name": code_package.fs_package.package_name,
                }
            },
            "job_spec": {
                "image": running_resource.docker_image,
                "pod_count": running_resource.instance,
                "resource": {
                    "gpu": running_resource.gpu,
                    "cpu": running_resource.cpu,
                },
            },
        }

        if job_desc:
            single_job_config["job_desc"] = job_desc

        print(single_job_config)
        headers = {
            "X-Forwarded-User": self.session.token,
            "Content-Type": "application/json",
        }

        json_data = json.dumps(single_job_config)
        response = requests.post(
            ENDPOINT_API_URL,
            data=json_data,
            headers=headers,
        )

        if response.status_code == 200:
            response_text = json.loads(response.text)
            job_id = response_text["data"].get("job_id")
            aidi_web_url = f"http://model.aidi.hobot.cc/job/training-jobs/ongoing-jobs/{job_id}?taskType=torch"  # noqa E501
            job_manager_url = "http://model.aidi.hobot.cc/job/training-jobs"
            print("任务提交成功, 请到以下链接查看任务列别和任务详情: ")
            print(f"任务列表: {job_manager_url}")
            print(f"任务详情: {aidi_web_url}")
        else:
            print("任务提交失败: ")
            print(response.status_code)
            print(response.text)


def create_job(
    job_name: str,
    job_type: str,
    project_id: str,
    queue_name: str,
    running_resource: RunningResourceConfig,
    startup: StartUpConfig,
    code_package: CodePackageConfig,
    job_desc: Optional[str] = None,
    mount: List[MountItem] = None,
    priority: int = 3,
    max_reschedulings: int = 0,
    reschedule_policy: str = "",
):
    session = Session(endpoint=ENDPOINT)
    if reschedule_policy:
        assert reschedule_policy in [
            "Always",
        ], f"`job_restart_mode` should be one of ['Always', ''], but get {reschedule_policy}"  # noqa E501

    SingleJob(session).create(
        job_name=job_name,
        job_type=job_type,
        job_desc=job_desc,
        project_id=project_id,
        queue_name=queue_name,
        running_resource=running_resource,
        startup=startup,
        code_package=code_package,
        mount=mount,
        priority=priority,
        max_reschedulings=max_reschedulings,
        reschedule_policy=reschedule_policy,
    )

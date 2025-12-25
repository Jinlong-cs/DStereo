import argparse
import contextlib
import json
import logging
import os
import random
import subprocess
import time

from hat.utils.elastic import ElasticState

IS_LOCAL = not os.path.exists("/running_package")


class DistributedLock:
    def __init__(self, lock_file) -> None:
        self.lock_file = lock_file

        with contextlib.suppress(FileExistsError):
            os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)

    def release_lock(self):
        if os.path.exists(self.lock_file):
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                pass

    def create_lock(self):
        try:
            os.mknod(self.lock_file)
        except FileExistsError:
            pass


if IS_LOCAL:
    lock_file = ".elastic/lockfile"
else:
    lock_file = "/job_data/elastic/lockfile"
locker = DistributedLock(lock_file)


def get_env(pass_envs):
    envs = []
    for k, v in list(pass_envs.items()):
        envs.append("export " + str(k) + "=" + str(v) + ";")
    return " ".join(envs)


def update_elastic_state(state: str):
    rand_num = random.randint(1, 10)
    time.sleep(rand_num)
    if not os.path.exists(lock_file):
        locker.create_lock()
        try:
            ElasticState.set_cur_state(state)
        except json.decoder.JSONDecodeError:
            pass


def submit(job_index, nnodes, nproc_per_node, command):

    locker.release_lock()

    def run(prog, use_elastic):
        print(f"launch prog: {prog}")
        try:
            result = subprocess.run(prog, shell=True)
            if result.returncode != 0:
                print(f"subprocess returncode is: {result.returncode}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                # failed: set RUNNING
                if use_elastic:
                    update_elastic_state(state=ElasticState.RUNNING)
                os._exit(result.returncode)
            else:
                # success: set FINISHED
                if use_elastic:
                    update_elastic_state(state=ElasticState.FINISHED)

        except subprocess.CalledProcessError as e:
            print(f"subprocess({e.cmd}) failed({e.returncode})! {e.output}")
            os._exit(-1)

    rdzv_id = os.getenv("JOB_ID") + f"_{job_index}"
    is_restart_job = "JOB_RESTART" in os.environ
    if is_restart_job:
        rdzv_id += f"_JOB_RESTART_{os.getenv('JOB_RESTART')}"

    rdzv_conf = "'protocol=https,ssl_cert={},ssl_cert_key={},ca_cert={},join_timeout=1800'".format(  # noqa E501
        os.getenv("ETCD_SSL_CERT"),
        os.getenv("ETCD_SSL_CERT_KEY"),
        os.getenv("ETCD_SSL_CERT_CA"),
    )

    cmd = (
        "torchrun --nnodes={} --nproc_per_node={}"
        + " --rdzv_id={} --rdzv_backend=etcd-v2 --rdzv_conf={}"
        + " --rdzv_endpoint={} {} "
    ).format(
        nnodes,
        nproc_per_node,
        rdzv_id,
        rdzv_conf,
        os.getenv("ETCD_ENDPOINT"),
        command.replace("python3", "") + " --MASTER_PORT=59999",
    )

    pass_envs = {}
    if "PYTHONPATH" in os.environ:
        pass_envs["PYTHONPATH"] = os.environ["PYTHONPATH"]
    if "WORKING_PATH" in os.environ:
        pass_envs["WORKING_PATH"] = os.environ["WORKING_PATH"]

    command_mark = f'"{command}"'
    os.environ["CURRENT_COMMAND"] = command_mark

    local_dir = os.getcwd() + "/"
    working_dir = local_dir
    prog = get_env(pass_envs) + " cd " + working_dir + "; " + cmd

    use_elastic = "USE_ELASTIC" in os.environ
    if use_elastic:
        elastic_state = ElasticState()
    else:
        elastic_state = None

    if is_restart_job:
        cmds_infos = elastic_state.all_state_info
        if command_mark in list(cmds_infos.keys()):
            if elastic_state.cur_command_state() == "FINISHED":
                logging.info(
                    f"`{command}` has been executed, skip in restart job."
                )  # noqa E501
            else:
                if elastic_state.get_cur_checkpoint() is not None:
                    os.environ["ELASTIC_NEED_RESUME"] = "1"

                # start from last broken command
                run(prog=prog, use_elastic=use_elastic)
        else:
            run(prog=prog, use_elastic=use_elastic)

    else:
        run(prog=prog, use_elastic=use_elastic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-idx",
        type=int,
        required=True,
        help="Job(script) index",  # noqa E501
    )
    parser.add_argument(
        "--nnodes",
        type=str,
        required=True,
        default="1:1",
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",  # noqa E501
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        required=True,
        help="Number of workers per node.",
    )
    parser.add_argument("command", nargs="+", help="command for plugin program")
    args = parser.parse_args()
    cmd = " ".join(args.command)
    submit(args.job_idx, args.nnodes, args.nproc_per_node, cmd)

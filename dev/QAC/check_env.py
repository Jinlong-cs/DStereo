import argparse
import os
import subprocess
import time

from checker import (
    check_cpu_info,
    check_gpu_info,
    check_need_rebase,
    check_time_info,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-need-rebase-with-self", action="store_true", default=False
    )
    parser.add_argument(
        "--check-need-rebase-with-master", action="store_true", default=False
    )
    parser.add_argument(
        "--interval", type=int, default=10, help="Query interval"
    )
    parser.add_argument("cmd_args", nargs="*", help="cmd")
    args = parser.parse_args()
    return args


def main(cmd, check_rebase_with_self, check_rebase_with_master, interval):

    begin_time = time.time()
    check_need_rebase(None, check_rebase_with_self, check_rebase_with_master)

    p = subprocess.Popen(cmd, shell=True, env=os.environ)
    while p.poll() is None:
        check_need_rebase(
            p.pid, check_rebase_with_self, check_rebase_with_master
        )
        check_time_info(p.pid, begin_time)
        check_cpu_info(p.pid)
        check_gpu_info(p.pid)
        time.sleep(interval)
    if p.poll() != 0:
        raise RuntimeError("Error with exit code: {}".format(p.poll()))


if __name__ == "__main__":
    args = parse_args()

    cmd = " ".join(args.cmd_args)

    check_rebase_with_self = args.check_need_rebase_with_self
    if not check_rebase_with_self:
        check_rebase_with_self = (
            os.environ.get("CHECK_NEED_REBASE_WITH_SELF", "0") == "1"
        )

    check_rebase_with_master = args.check_need_rebase_with_master
    if not check_rebase_with_master:
        check_rebase_with_master = (
            os.environ.get("CHECK_NEED_REBASE_WITH_MASTER", "0") == "1"
        )

    main(
        cmd,
        check_rebase_with_self=check_rebase_with_self,
        check_rebase_with_master=check_rebase_with_master,
        interval=args.interval,
    )

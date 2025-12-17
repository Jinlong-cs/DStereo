import os
import subprocess
import warnings

from .utils import psutil_kills

__all__ = ["GitChecker", "check_need_rebase"]


class GitChecker:
    @staticmethod
    def fetch():
        try:
            subprocess.check_output(
                ["git", "fetch", "origin"], stderr=subprocess.STDOUT
            )
            return True
        except subprocess.CalledProcessError as e:
            warnings.warn(
                "Meet Exception: {}, ignored...".format(e.stdout.decode())
            )
            return False

    @staticmethod
    def get_branch_name():
        branch_name = os.environ.get("gitlabSourceBranch", None)
        if branch_name is None:
            branch_name = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"]
                )
                .decode("ascii")
                .strip()
            )
        branch_name = f"origin/{branch_name}"
        return branch_name

    @staticmethod
    def should_rebase_with_self():
        try:
            branch_name = GitChecker.get_branch_name()

            current_branch_commit_id = os.environ.get(
                "gitlabMergeRequestLastCommit", None
            )
            if current_branch_commit_id is None:
                current_branch_commit_id = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .decode("ascii")
                    .strip()
                )
            remote_branch_commit_id = (
                subprocess.check_output(["git", "rev-parse", branch_name])
                .decode("ascii")
                .strip()
            )
            if remote_branch_commit_id != current_branch_commit_id:
                msg = "Detected branch {} is changed. Remote = {}, current = {}".format(  # noqa
                    branch_name,
                    remote_branch_commit_id,
                    current_branch_commit_id,
                )
                return (True, msg)

            return (False, "")

        except subprocess.CalledProcessError as e:
            warnings.warn(
                "Meet Exception: {}, ignored...".format(e.stdout.decode())
            )
            return (False, "")

    @staticmethod
    def should_rebase_with_master():
        try:
            branch_name = GitChecker.get_branch_name()

            merge_base_commit_id = (
                subprocess.check_output(
                    ["git", "merge-base", "origin/master", branch_name]
                )
                .decode("ascii")
                .strip()
            )
            master_commit_id = (
                subprocess.check_output(["git", "rev-parse", "origin/master"])
                .decode("ascii")
                .strip()
            )
            if merge_base_commit_id != master_commit_id:
                msg = "Detected need to merge with master. Master = {}, merge Base = {}".format(  # noqa
                    master_commit_id, merge_base_commit_id
                )
                return (True, msg)

            return (False, "")
        except subprocess.CalledProcessError as e:
            warnings.warn(
                "Meet Exception: {}, ignored...".format(e.stdout.decode())
            )
            return (False, "")


def check_need_rebase(pid, check_rebase_with_self, check_rebase_with_master):
    if not check_rebase_with_self and not check_rebase_with_master:
        return
    if not GitChecker.fetch():
        return
    if check_rebase_with_self:
        flag, msg = GitChecker.should_rebase_with_self()
        if flag:
            if pid is not None:
                psutil_kills(pid)
            raise RuntimeError(f"\n\n\n{msg}\n\n\n")
    if check_rebase_with_master:
        flag, msg = GitChecker.should_rebase_with_master()
        if flag:
            if pid is not None:
                psutil_kills(pid)
            raise RuntimeError(f"\n\n\n{msg}\n\n\n")

# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
import subprocess
from datetime import datetime
from typing import Tuple

logger = logging.getLogger(__name__)

# current version
__version__ = "2.2.1"


def is_git_repo(cwd: str) -> bool:
    try:
        _ = (
            subprocess.check_output(["git", "status", "-s"], cwd=cwd)
            .decode("utf-8")
            .strip()
        )
        is_git_repo = True
    except Exception:
        logger.warning("Not in a git repo, get temporary version number")
        is_git_repo = False
    return is_git_repo


def get_commit_id(cwd: str) -> str:
    commit_id = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("utf-8")
        .strip()
    )

    return commit_id


def get_git_repo_info() -> Tuple:
    cwd = os.path.dirname(os.path.abspath("__file__"))

    repo_url = os.environ.get("HAT_GIT_REPO")
    commit_id = os.environ.get("HAT_GIT_COMMIT_ID")

    if repo_url and commit_id:
        return repo_url, commit_id
    else:
        if is_git_repo(cwd):
            # repo url
            repo_url = (
                subprocess.check_output(["git", "remote", "-v"], cwd=cwd)
                .decode("utf-8")
                .strip()
                .replace("\t", " ")
                .split(" ")[1]
            )  # noqa

            # Note: The result of the `git remote -v` high versions of git
            # does not endswith `.git`.
            if not repo_url.endswith(".git"):
                repo_url += ".git"

            # check diff
            diff = (
                subprocess.check_output(["git", "diff", "HEAD"], cwd=cwd)
                .decode("utf-8")
                .strip()
            )
            if len(diff) > 0:  # has diff
                commit_id = "unknow"
            else:
                # commit_id
                commit_id = get_commit_id(cwd)

        else:
            repo_url = "unknow"
            commit_id = "unknow"

        return repo_url, commit_id


def get_setup_version(version: str = __version__) -> str:
    # for setup in pre release version
    release_version = os.getenv("RELEASE_VERSION")

    if release_version is None:
        cwd = os.path.dirname(os.path.abspath("__file__"))

        # check if in a git repo and if there are uncommitted changes
        # get last commit time
        if is_git_repo(cwd):
            commit_id = os.getenv("gitlabMergeRequestLastCommit")
            if commit_id is not None:
                commit_time_unix = int(
                    subprocess.check_output(
                        ["git", "log", "-1", "--pretty=format:%ct", commit_id],
                        cwd=cwd,
                    )
                    .decode("ascii")
                    .strip()
                )

            else:
                commit_time_unix = int(
                    subprocess.check_output(
                        ["git", "log", "-1", "--pretty=format:%ct"], cwd=cwd
                    )
                    .decode("ascii")
                    .strip()
                )
                commit_id = get_commit_id(cwd)
            commit_time = datetime.fromtimestamp(commit_time_unix)
            commit_id = commit_id[:7]
        else:
            commit_time = datetime.now()

            commit_id = "unknown"
        commit_timestamp = commit_time.strftime("%Y%m%d%H%M")

        version += ".dev{}+{}".format(commit_timestamp, commit_id)
    return version


def get_tmp_version(version: str = __version__) -> str:
    # if HAT has not been installed, get a temporary version number
    # with format __version__.dev{timestamp}+unknown
    version += ".dev{}+{}".format(datetime.now().strftime("%Y%m%d%H%M"), "unknown")
    return version


def write_version_file(version: str) -> None:
    version_path = os.path.join(os.path.dirname(__file__), "_version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))

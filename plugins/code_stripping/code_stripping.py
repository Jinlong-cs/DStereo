# Copyright (c) Horizon Robotics. All rights reserved.

import argparse
import glob
import os
import re
import subprocess
from typing import List

from process_files import clip_file_code, process_files
from termcolor import cprint

from hat.utils.config import Config


def run(cmd, prefix=">> ", color="green", with_ret=False):
    """Run shell with log output."""
    cprint(prefix + cmd, color)
    if with_ret:
        (status, output) = subprocess.getstatusoutput(cmd)
        output = output.split("\n")
    else:
        os.system(cmd)
        output = None
    return output


def copy(src, dst):
    """Copy file or folder to dst."""
    dst_dir = os.path.dirname(dst)
    if os.path.isdir(src):
        if dst.endswith("/"):
            dst_dir = os.path.dirname(dst_dir)
        dst = dst_dir

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    run(f"cp -rf {src} {dst}", prefix="[code] ")


def rm(path):
    """Remove file or folder."""
    if not os.path.exists(path):
        return
    dst_init_dirname = os.path.dirname(path)
    if os.path.isdir(path):
        if path.endswith("/"):
            dst_init_dirname = os.path.dirname(dst_init_dirname)
    dst_init_path = os.path.join(dst_init_dirname, "__init__.py")
    if os.path.exists(dst_init_path):
        run(f"rm -rf {dst_init_path}", prefix="[skip] ", color="red")
    run(f"rm -rf {path}", prefix="[skip] ", color="red")


def process_one_line(src_dir, target_dir, each_line, op="copy"):
    """Process one line in file_list or skip_file_list."""
    skip_src_len = len(src_dir) if src_dir.endswith("/") else len(src_dir) + 1
    assert op in ["copy", "rm"], "The op must be one of copy or rm !"
    src = os.path.join(src_dir, each_line)
    src_match_results = glob.glob(src, recursive=True)

    # remove duplicate elements
    src_match_results = list(set(src_match_results))
    assert len(
        src_match_results
    ), "The {} don't match anything, please check your config!".format(src)

    for src_match_result in src_match_results:
        src_match_result_tmp = src_match_result[skip_src_len:]
        dst = os.path.join(target_dir, src_match_result_tmp)
        if op == "copy":
            copy(src_match_result, dst)
        if op == "rm":
            rm(dst)


def refine_all_in_file(temp_info, keep_in_all):
    """Refine __all__ in __init__.py."""
    if len(temp_info) > 1:
        in_all_modules = temp_info[1:-1]
        for tmp in in_all_modules:
            res = re.findall('["](.*?)["]', tmp)
            if len(res) > 0:
                if res[0] not in keep_in_all:
                    temp_info.remove(tmp)
            if len(temp_info) == 2:
                temp_info = []
    else:
        in_all_modules = re.findall('["](.*?)["]', temp_info[0])
        keep_tmp = []
        for tmp in in_all_modules:
            if tmp.strip() in keep_in_all:
                keep_tmp.append(tmp.strip().strip(","))
        all_keeps = [
            '"{}"'.format(keep_tmp_each) for keep_tmp_each in keep_tmp
        ]
        if len(keep_tmp):
            temp_info[0] = "__all__ = [{}]\n".format(", ".join(all_keeps))
        else:
            temp_info = []
    return temp_info


def refine_import_in_file(
    res, import_modules, temp_info, keep_in_all, ori_imports
):
    """Refine import in __init__.py."""
    keep_tmp = []
    for tmp in import_modules:
        if tmp.strip().strip(",") not in ori_imports:
            if len(temp_info) > 1:
                temp_info.remove(tmp)
        else:
            if tmp.strip().strip(",") in ori_imports:
                keep_in_all.append(tmp.strip().strip(","))
                keep_tmp.append(tmp.strip().strip(","))
    if len(temp_info) > 1:
        if len(temp_info) == 2:
            temp_info = []
    else:
        if len(keep_tmp):
            temp_info[0] = "from {} import {}\n".format(
                res[0][0], ", ".join(keep_tmp)
            )
        else:
            temp_info = []
    return temp_info, keep_in_all


def refine_init_file(dst_dir, dst_file, each_file):
    """Refine __init__.py."""
    keep_in_all = []
    if os.path.exists(dst_dir):
        if not os.path.exists(dst_file):
            # get files and dirs in dst_dir
            all_dir_file = os.listdir(dst_dir)
            dirs = []
            files = []
            for item in all_dir_file:
                if os.path.isfile(os.path.join(dst_dir, item)):
                    files.append(item)
                else:
                    dirs.append(item)
            files = [f.replace(".py", "") for f in files]

            old_lines = open(each_file, "r").readlines()
            keep_in_file = []
            temp_info = []
            all_flag = False
            import_num_flag = 0
            all_num_flag = 0
            for line in old_lines:
                if "__all__" in line:
                    all_flag = True
                if all_flag:
                    # change __all__
                    all_num_flag += line.count("[")
                    all_num_flag -= line.count("]")
                    temp_info.append(line)
                    if all_num_flag != 0:
                        continue
                    temp_info = refine_all_in_file(temp_info, keep_in_all)
                # change from xxx import yyy
                else:
                    import_num_flag += line.count("(")
                    import_num_flag -= line.count(")")
                    temp_info.append(line)
                    if import_num_flag != 0:
                        continue
                    res = re.findall("from (.*?) import (.*)", temp_info[0])
                    if len(res) > 0:
                        if len(temp_info) > 1:
                            import_modules = temp_info[1:-1]
                        else:
                            import_modules = res[0][1].strip().split(",")
                        res_tmp = res[0][0].split(".")

                        if res_tmp[1] in files:
                            for tmp in import_modules:
                                keep_in_all.append(tmp.strip().strip(","))
                        elif res_tmp[1] in dirs:
                            dst_dir_tmp = os.path.join(dst_dir, res_tmp[1])
                            dst_file_tmp = os.path.join(
                                os.path.dirname(dst_file),
                                res_tmp[1],
                                os.path.basename(dst_file),
                            )
                            each_file_tmp = os.path.join(
                                os.path.dirname(each_file),
                                res_tmp[1],
                                os.path.basename(each_file),
                            )
                            keep_in_all_tmp = refine_init_file(
                                dst_dir_tmp, dst_file_tmp, each_file_tmp
                            )
                            temp_info, keep_in_all = refine_import_in_file(
                                res,
                                import_modules,
                                temp_info,
                                keep_in_all,
                                keep_in_all_tmp,
                            )
                        elif not bool(res_tmp[1]):
                            temp_info, keep_in_all = refine_import_in_file(
                                res,
                                import_modules,
                                temp_info,
                                keep_in_all,
                                dirs + files,
                            )
                        else:
                            ignore_init = ["from ._version import __version__"]
                            if not (temp_info[0].strip() in ignore_init):
                                temp_info = []
                keep_in_file += temp_info
                temp_info = []
            with open(dst_file, "w") as fid:
                for write_line in keep_in_file:
                    fid.write(write_line)
            cprint(f"[code] create {dst_file}", "green")
        else:
            init_infos = open(dst_file, "r").readlines()
            in_all_flag = False

            num_flag = 0
            for init_info in init_infos:
                if "__all__" in init_info:
                    in_all_flag = True
                if in_all_flag:
                    num_flag += init_info.count("[")
                    num_flag -= init_info.count("]")

                    init_info_strip = init_info.strip()
                    init_info_strip = re.findall(
                        '["](.*?)["]', init_info_strip
                    )
                    for init_info_tmp in init_info_strip:
                        keep_in_all.append(init_info_tmp)
                if num_flag == 0:
                    in_all_flag = False
    return keep_in_all


def main(args):
    if not os.path.exists(args.src_dir):
        cprint("Please check the src dir for cropping.", "red")
        return
    if os.path.exists(args.target_dir):
        if args.override:
            cprint(f"Override {args.target_dir} without clean", "red")
        else:
            cprint(
                f"{args.target_dir} already exists, cannot be overwriten",
                "red",
            )
            return
    args.target_dir = os.path.abspath(args.target_dir)
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    cfg = Config.fromfile(args.file_list)

    file_list = cfg.get("file_list", None)
    if file_list is not None:
        for item in file_list:
            process_one_line(args.src_dir, args.target_dir, item, op="copy")

    skip_file_cfg = cfg.get("skip_file_list", None)
    if skip_file_cfg is not None:
        for item in skip_file_cfg:
            process_one_line(args.src_dir, args.target_dir, item, op="rm")

    copy_samples = cfg.get("copytodst", None)
    if copy_samples is not None:
        for src_copy_path, dst_copy_path in copy_samples.items():
            src = os.path.join(args.src_dir, src_copy_path)
            dst = os.path.join(args.target_dir, dst_copy_path)
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(dst)
            run(f"cp -rf {src} {dst}", prefix="[code] ")

    # refine __init__.py
    init_dirs = cfg.get(
        "init_dirs",
        {
            "hat": "hat",
            "tests": "tests",
        },
    )
    if isinstance(init_dirs, List):
        init_dirs_tmp = {}
        for item in init_dirs:
            init_dirs_tmp.update({item: item})
        init_dirs = init_dirs_tmp

    for src_module, dst_module in init_dirs.items():
        src_module_path = os.path.join(args.src_dir, src_module)
        dst_module_path = os.path.join(args.target_dir, dst_module)
        src_module_path_len = (
            len(src_module_path)
            if src_module_path.endswith("/")
            else len(src_module_path) + 1
        )
        relative_paths = []
        for path, _, files in os.walk(src_module_path):

            abs_paths = [
                os.path.join(path, x) for x in files if x == "__init__.py"
            ]

            relative_path = [path[src_module_path_len:] for path in abs_paths]
            relative_paths += relative_path
        for each_file in relative_paths:
            src_file = os.path.join(src_module_path, each_file)
            dst_file = os.path.join(dst_module_path, each_file)
            dst_dir = os.path.dirname(dst_file)
            refine_init_file(dst_dir, dst_file, src_file)

    if args.toolchain_crop:
        cprint("Begin change import and delete function", "red")
        delete_func_lists = []
        pluginpackages = ["aidisdk", "hatbc"]
        packagename = "thirdparty"
        hat_dir = os.path.join(args.target_dir, "hat")
        process_files(hat_dir, delete_func_lists, pluginpackages, packagename)

    if args.clip_code:
        clip_flag_begin = "clipping: begin"
        clip_flag_end = "clipping: end"
        for path, _, files in os.walk(args.target_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(path, file)
                clip_file_code(file_path, clip_flag_begin, clip_flag_end)
    # rm pycache
    run(
        'find * | grep -E "__pycache__" | xargs rm -rf',
        prefix="[code] ",
    )
    run(
        'find {} | grep -E "__pycache__" | xargs rm -rf'.format(
            os.path.join(args.target_dir, "*")
        ),
        prefix="[code] ",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", type=str, required=True)
    parser.add_argument(
        "--src-dir", type=str, default="../../", help="The root dir of HAT."
    )
    parser.add_argument("--target-dir", type=str, default="./release")
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--toolchain-crop", action="store_true", default=False)
    parser.add_argument("--clip-code", action="store_true", default=False)
    args = parser.parse_args()
    main(args)

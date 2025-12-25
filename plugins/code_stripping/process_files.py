# Copyright (c) Horizon Robotics. All rights reserved.

import os

from termcolor import cprint


def get_strip_num(line: str) -> int:
    """Get the number of characters indented."""
    num = 0
    for c in line:
        if c != " ":
            return num
        num += 1


def rewrite_file(file, func_lists, pluginpackages, packagedir):
    """Delete the functions of func_lists in file, and change the import."""
    func_def_lists = ["def {}(".format(i) for i in func_lists]
    ori_lines = open(file, "r").readlines()

    has_func = False
    has_pluginpackage = False
    old_lines = []
    for old_line in ori_lines:
        # search func whether in file
        for func_def_list in func_def_lists:
            if func_def_list in old_line:
                has_func = True
        # search pluginpackage whether in file and change it
        for pluginpackage in pluginpackages:
            if "import {}".format(pluginpackage) in old_line.strip():
                if "import {}".format(pluginpackage) == old_line.strip():
                    old_line = "import {} as {}\n".format(
                        packagedir + pluginpackage, pluginpackage
                    )
                    has_pluginpackage = True
                else:
                    old_line = old_line.replace(
                        pluginpackage, packagedir + pluginpackage
                    )
                    has_pluginpackage = True
            elif "from {}".format(pluginpackage) in old_line.strip():
                old_line = old_line.replace(pluginpackage, packagedir + pluginpackage)
                has_pluginpackage = True
        old_lines.append(old_line)
    assert len(old_lines) == len(ori_lines)
    # delete_func in this file
    if has_func:
        num_strip = 0
        num_flag = 0
        delete_indx = []
        start_flag = False
        start_index = None
        for index, old_line in enumerate(old_lines):
            if old_line == "\n":
                continue
            tmp_strip_num = get_strip_num(old_line)
            if start_flag:
                if num_strip >= tmp_strip_num and num_flag == 0:
                    end_index = index
                    delete_indx.append([start_index, end_index])
                    start_flag = False
                    start_index = 0

            for func_def_list in func_def_lists:
                if func_def_list in old_line:
                    start_index = index
                    num_strip = tmp_strip_num
                    start_flag = True

            num_flag += old_line.count("(")
            num_flag -= old_line.count(")")

        if start_index != 0 and start_flag:
            delete_indx.append([start_index])

        # delete func in file
        func_lastline_flag = False
        for delete_idx in delete_indx[::-1]:
            if len(delete_idx) == 1:
                del old_lines[delete_idx[0] :]
                func_lastline_flag = True
            else:
                del old_lines[delete_idx[0] : delete_idx[1]]
        if func_lastline_flag and old_lines[-1] == "\n":
            del old_lines[-1]

    # rewrite file
    if has_func or has_pluginpackage:
        with open(file, "w") as f:
            for line in old_lines:
                f.write(line)
        cprint(f"[code] rewrite {file}", "green")


def process_files(src, func_lists, pluginpackages, packagename):  # noqa
    """Delete the functions in func_lists in the src directory,
    and make the packages in pluginpackages imported from hat.packagename.
    """
    package_replace_str = "hat.{}.".format(packagename)
    all_dir_file = os.listdir(src)
    dirs = []
    files = []
    for item in all_dir_file:
        if os.path.isfile(os.path.join(src, item)):
            files.append(os.path.join(src, item))
        else:
            if item == "__pycache__" or item == packagename:
                continue
            dirs.append(os.path.join(src, item))
    if len(files):
        for file in files:
            rewrite_file(file, func_lists, pluginpackages, package_replace_str)
    for dir_each in dirs:
        process_files(dir_each, func_lists, pluginpackages, packagename)


def clip_file_code(file_path, clip_flag_begin, clip_flag_end):
    old_lines = open(file_path, "r").readlines()
    delete_indexs = []
    start_index = None
    for idx, line in enumerate(old_lines):
        if clip_flag_begin in line:
            start_index = idx
        elif clip_flag_end in line:
            delete_indexs.append((start_index, idx))
            start_index = None
    if delete_indexs:
        for delete_index in delete_indexs[::-1]:
            del old_lines[delete_index[0] : delete_index[1] + 1]
        with open(file_path, "w") as f:
            for line in old_lines:
                f.write(line)
            cprint(f"[code] rewrite {file_path}", "red")

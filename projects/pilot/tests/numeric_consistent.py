import os
import subprocess
import zipfile

import numpy as np
import pandas as pd


def compress_dir(path, save_zip, without_dir=False):
    if os.path.exists(save_zip):
        os.remove(save_zip)
    z = zipfile.ZipFile(save_zip, "w", compression=zipfile.ZIP_BZIP2)
    pre_len = len(path)
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # relative path
            if not without_dir:
                arc_path = file_path[pre_len:].strip(os.path.sep)
            else:
                arc_path = os.path.basename(file_path)
            z.write(file_path, arc_path)
    z.close()


def compare_dump_zip(zip_path, diff_zip_path, save_result_path=None):
    zip = zipfile.ZipFile(zip_path, "r")
    diff_zip = zipfile.ZipFile(diff_zip_path, "r")
    zip_name = set(zip.namelist())
    diff_zip_name = set(diff_zip.namelist())
    unintersect_file = (zip_name | diff_zip_name) - (zip_name & diff_zip_name)
    intersect_file = zip_name & diff_zip_name
    all_file = list(intersect_file) + list(unintersect_file)
    report = [
        {"dump_data": f.replace("/", "-"), "check_result": None}
        for f in all_file
    ]
    report = pd.DataFrame(report)
    # diff intersect file
    for idx, file in enumerate(intersect_file):
        tmp_data = "tmp_data.pkl"
        tmp_diff_data = "tmp_diff_data.pkl"
        data = zip.read(file)
        diff_data = diff_zip.read(file)
        with open(tmp_data, "wb") as w, open(tmp_diff_data, "wb") as wd:
            w.write(data)
            wd.write(diff_data)
        diff_result = "diff_result_" + file.replace("/", "-").replace(
            ".pkl", ""
        )
        compare_cmd = f"""
        python3 tools/analyze/compare_dump_data.py --base-file {tmp_data} \
            --cmp-file {tmp_diff_data} \
            --diff-inputs \
            --diff-outputs \
            --diff-grad \
            --output-dir {save_result_path} \
            --file-name {diff_result} \
        """
        subprocess.check_call(compare_cmd, shell=True)
        diff = pd.read_csv(
            os.path.join(save_result_path, diff_result + ".csv")
        )
        check = [d == "Pass" for d in diff["check_result"].to_list()]
        if np.all(check):
            report.loc[idx]["check_result"] = "Pass"
        else:
            report.loc[idx]["check_result"] = "Failed"
    # unintersect file cannot diff
    for idx, file in enumerate(unintersect_file):
        if file not in zip_name:
            report.loc[idx + len(intersect_file)][
                "check_result"
            ] = "Miss in newest dump "
        else:
            report.loc[idx + len(intersect_file)][
                "check_result"
            ] = "Miss in old dump "
    report.to_csv(os.path.join(save_result_path, "diff_result_all.csv"))

import argparse
import os
import subprocess

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        help="numeric consistent stage",
    )

    return parser.parse_args()


class TestNumericConsistent(object):
    def test_numeric_consistent(self, cfg_path, action):
        os.environ["HAT_CFG_PATH"] = cfg_path
        patch_cfg = f"{os.path.dirname(__file__)}/configs/template.py"
        print("patch_cfg: ", patch_cfg)
        cfg_module = os.path.basename(cfg_path).split(".")[0]
        print("=" * 20, f"Begin {cfg_module} {action} Numeric Consistent Test")
        # float training or prediction
        extra_args = "--backend GLOO" if action == "predict" else ""
        float_cmd = f"python3 tools/{action}.py --config {patch_cfg} --stage float {extra_args}"  # noqa
        print("float training cmd: ", float_cmd)
        subprocess.check_call(float_cmd, shell=True)

        # float compare
        prefix = "_val" if action == "predict" else ""
        float_compare_cmd = (
            "python3 tools/compare_dump_data.py "
            + f"--base-file tmp_dump_data/float{prefix}_{cfg_module}_base.pkl "
            + f"--cmp-file  tmp_dump_data/float{prefix}_{cfg_module}_cmp.pkl "
            + "--diff-inputs --diff-outputs --diff-grad --diff-params "
            + f"--file-name float{prefix}_{cfg_module}_diff_reports"
        )
        print("float compare cmd: ", float_compare_cmd)
        subprocess.check_call(float_compare_cmd, shell=True)

        csv_file = f"float{prefix}_{cfg_module}_diff_reports.csv"
        diff = pd.read_csv(csv_file)
        check = np.array([d == "Pass" for d in diff["check_result"].to_list()])
        assert np.all(
            check
        ), f"Numeric inconsistency happened in {cfg_path} {action}"
        print(f"{cfg_module} {action} numeric consistent test pass!!!!")


if __name__ == "__main__":
    args = parse_args()
    numeric_test = TestNumericConsistent()
    numeric_test.test_numeric_consistent(args.config, args.action)

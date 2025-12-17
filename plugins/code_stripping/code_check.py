# Copyright (c) Horizon Robotics. All rights reserved.

import argparse
import subprocess


def test_main(args):
    cmd = f"""
       cd {args.target_dir}
       make basic-data
       make unit-test
       make intergration-test
       make doc
    """
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, default="./release")
    args = parser.parse_args()
    test_main(args)

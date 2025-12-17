import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

package_name = "horizon_torch_samples"
license_str = open("LICENSE").read()
_ROOT_DIR = os.path.dirname(__file__)
_REQUIRE_DIR = "./requirements"


def load_requirements(file_names):  # type: ignore
    valid_lines = []
    for file_name in file_names:
        with open(file_name, "r") as fi:
            for line in fi.readlines():
                line = line.strip()
                if len(line) > 0 and line[0].isalpha():  # noqa
                    valid_lines.append(line)

    return valid_lines


def load_py_module(fname, pkg="hat"):  # type: ignore
    spec = spec_from_file_location(
        os.path.join(pkg, fname), os.path.join(_ROOT_DIR, pkg, fname)
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


version = load_py_module("version.py")
version_number = version.get_setup_version()
version.write_version_file(version_number)

# Define package extras. These are only installed if you specify them.
# Use like `pip install horizon-hat[dev, test]`.
extras = {
    # split requirements to build.txt, test.txt, dev.txt, doc.txt, and so on
    "dev": load_requirements([os.path.join(_REQUIRE_DIR, "develop.txt")])
}

# ext_index='-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc'  # noqa
# Install hat from local repo: `pip install -e . $ext_index`
# Install hat from remote: `pip install horizon-hat $ext_index`
print("Building wheel {}-{}".format(package_name, version_number))

cli_target = "hat/cli"
if not os.path.exists(cli_target):
    os.makedirs(cli_target)

name_lists = [
    "tools/train.py",
    "tools/predict.py",
    "tools/compile_perf.py",
    "tools/calops.py",
    "tools/compile_standalone.py",
    "tools/model_checker.py",
    "plugins/k8s_submit/submit.py",
    "plugins/k8s_submit/generate_submit_files.py",
]
for name in name_lists:
    os.system("cp -rf {} {}".format(name, cli_target))

setup(
    name=package_name,
    version=version_number,
    author="HAT Contributors",
    description="Horizon Algorithm Toolkit",
    license=license_str,
    packages=find_packages(
        exclude=(
            "tests.*",
            "tests",
        )
    ),
    include_package_data=True,
    install_requires=load_requirements(
        [
            os.path.join(_REQUIRE_DIR, "build.txt"),
            os.path.join(_REQUIRE_DIR, "internal.txt"),
        ]
    ),
    extras_require=extras,
)

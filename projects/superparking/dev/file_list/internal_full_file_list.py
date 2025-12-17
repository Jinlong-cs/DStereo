from base_file_list import file_list, skip_file_list  # noqa

file_list += [
    "plugins/k8s_submit/*",
    "plugins/code_stripping/code_check.py",
    "requirements.txt",
    "requirements",
    "requirements/*",
    ".gitignore",
    "dev/*",
    "tests/__init__.py",
    "setup.py",
    "Makefile",
    "README.md",
    "LICENSE",
    # TODO(REMOVE): for release training
    "hat/data/datasets/legacy_mxnet_rec_dataset.py",
    "hat/callbacks/aidi_expmodel.py",
    "hat/callbacks/online_model_trick.py",
    # TODO(REMOVE): for release eval
    "hat/core/eval_platform_adaptor.py",
    "hat/data/datasets/image_auto2d.py",
    "hat/callbacks/aidi_eval.py",
    "projects/superparking/tools/*",
    "projects/superparking/dev/*",
    "projects/superparking/tests/*",
]

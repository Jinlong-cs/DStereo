from base_file_list import file_list, skip_file_list

file_list += [
    "plugins/k8s_submit/*",
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
    # TODO(REMOVE): pilot aidi tools
    "projects/pilot/tools/*",
    "projects/pilot/dev/publish/*",
    "projects/pilot/tests/*",
]

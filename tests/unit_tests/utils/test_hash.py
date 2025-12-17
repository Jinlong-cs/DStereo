import hashlib
import os
import shutil

import pytest

from hat.utils.hash import (
    generate_sha256_file,
    get_hash_file_if_hashed_and_local,
)


@pytest.mark.parametrize(
    ["remove_old", "file_ext"],
    [
        pytest.param(False, ".pth.tar"),
        pytest.param(True, ".pth.tar"),
        pytest.param(True, ".pt"),
        pytest.param(False, ".pt"),
    ],
)
def test_generate_sha256_file(tmpdir, remove_old, file_ext):
    in_file = os.path.join(tmpdir, f"float-checkpoint-best{file_ext}")
    with open(in_file, "w") as f:
        f.write("state_dict:{}")

    sha256 = hashlib.sha256()
    with open(in_file, "rb") as f:
        data = f.read()
        sha256.update(data)
    sha = sha256.hexdigest()

    if remove_old:
        old_file = os.path.join(
            tmpdir, f"float-checkpoint-best-9d3ef576{file_ext}"
        )
        with open(old_file, "w") as f:
            f.write("state_dict:{1}")

    hashed_file = generate_sha256_file(in_file, remove_old)
    assert os.path.exists(in_file)

    if remove_old:
        assert not os.path.exists(
            old_file
        ), f"{old_file} should have been deleted"

    assert str(hashed_file) == str(
        os.path.join(tmpdir, f"float-checkpoint-best-{sha[:8]}{file_ext}")
    )


@pytest.mark.parametrize(
    ["file_ext", "check_hash"],
    [
        pytest.param(".pth.tar", True),
        pytest.param(".pth.tar", False),
        pytest.param(".pt", True),
        pytest.param(".hbm", False),
    ],
)
def test_get_hash_file_if_hashed_and_local(tmpdir, file_ext, check_hash):

    in_file = os.path.join(tmpdir, f"float-checkpoint-last{file_ext}")
    with open(in_file, "w") as f:
        f.write("state_dict:{1}")

    sha256 = hashlib.sha256()
    with open(in_file, "rb") as f:
        data = f.read()
        sha256.update(data)
    sha = sha256.hexdigest()
    hashed_file = os.path.join(
        tmpdir, f"float-checkpoint-last-{sha[:8]}{file_ext}"
    )

    # 1. input no hash filename, return no hash filename if there is no hashed file  # noqa E501
    ret_file = get_hash_file_if_hashed_and_local(
        in_file, check_hash=check_hash
    )
    assert str(ret_file) == str(in_file)

    # 2. input no hash filename, return hashed filename if these is hashed file
    shutil.copy(in_file, hashed_file)
    os.remove(in_file)
    ret_file = get_hash_file_if_hashed_and_local(
        in_file, check_hash=check_hash
    )
    assert str(ret_file) == str(hashed_file)

    # 3. input hashed filename, return hashed filename if there if hashed file
    ret_file = get_hash_file_if_hashed_and_local(hashed_file)
    assert str(ret_file) == str(hashed_file)

    # 4. input file start with http
    in_file = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    ret_file = get_hash_file_if_hashed_and_local(in_file)
    assert str(ret_file) == str(in_file)

    # skip: may cause OSError: HDFS connection failed
    # 5. input file start with hdfs and the file don't exists,
    # raise assert error
    # in_file = "hdfs://hobot-xxxxx"
    # ret_file = get_hash_file_if_hashed_and_local(in_file)
    # assert str(in_file) == str(ret_file)

from hat.version import get_setup_version, get_tmp_version, write_version_file


def test_get_setup_version():
    get_setup_version("1.0.0")


def test_get_tmp_version():
    get_tmp_version("1.0.0")


def test_write_version_file():
    write_version_file("1.0.0")

import warnings


def test_nvrenderer():
    """Skip NVRenderer because it is not supported in default docker.

    BTW, it is hard to install.
    """
    warnings.warn("Skip NVRenderer test.")

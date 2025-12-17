__all__ = ["_as_list", "_check_type"]


def _as_list(obj):
    """A utility function that converts the argument to a list if it is not
    already.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list or tuple, return it. Otherwise, return `[obj]` as a
    single-element list.

    """
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


def _check_type(obj, expected_type, obj_name):
    assert isinstance(
        obj, expected_type
    ), f"{obj_name} should be instance of {expected_type}, but get {str(type(obj))}"  # noqa

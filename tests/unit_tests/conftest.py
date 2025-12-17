import pytest


def pytest_collection_modifyitems(items):
    """Add timeout marker before unit test"""

    for item in items:
        item.add_marker(pytest.mark.timeout(100))

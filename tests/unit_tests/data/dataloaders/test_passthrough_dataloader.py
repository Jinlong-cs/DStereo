import pytest

from hat.data.dataloaders.passthrough_dataloader import PassThroughDataLoader


@pytest.mark.parametrize(
    ["length", "clone"],
    [
        pytest.param(1, False),
        pytest.param(5, False),
        pytest.param(5, True),
    ],
)
def test_passthrough_dataloader(length, clone):

    example = [1, 2, 3, 4, 5]
    changed_example = [None, 2, 3, 4, 5]

    dataloader = PassThroughDataLoader(example, length=length, clone=clone)
    for idx, data in enumerate(dataloader):
        if idx == 0:
            assert data == example
            data[:] = changed_example
        elif clone:
            assert data == example
        else:
            assert data == changed_example

    assert idx + 1 == length


if __name__ == "__main__":
    pytest.main(["-s", __file__])

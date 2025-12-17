import os  # noqa

from hdflow.cli.execute import main, parse_args  # noqa


def check():
    # 1. check directory
    expected_dirname = os.path.abspath(
        os.path.dirname(os.path.relpath(__file__))
    )
    current_dirname = os.getcwd()
    assert (
        expected_dirname == current_dirname
    ), f"Please running this program under directory {expected_dirname}, but you are in directory {current_dirname}"  # noqa


if __name__ == "__main__":
    check()

    args = parse_args()
    main(args)

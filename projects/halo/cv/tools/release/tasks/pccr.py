from typing import List

from .base import ReleasePath

__all__ = ["pccr"]


class PCCR(ReleasePath):
    """
    Owner: yisu.zhou
    """

    @property
    def models(self) -> List:
        return [
            "hat/models/task_modules/PCCR_gaze/*",
            "hat/models/losses/pccr/*",
        ]

    @property
    def transforms(self) -> List:
        return []

    @property
    def datasets(self) -> List:
        return []

    @property
    def metrics(self) -> List:
        return []

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/models/losses/test_pccr_loss.py",
            "tests/unit_tests/models/task_modules/test_pccr.py",
            "tests/unit_tests/models/task_modules/test_screen.py",
        ]


pccr = PCCR().get_file_list()

import os
from typing import List


class ReleasePath:
    def check_path(self, file_list):
        for name in file_list:
            if name.endswith("*"):
                dirname = os.path.dirname(name)
                # assert os.path.isdir(dirname), f"{dirname} is not found."
                if not os.path.isdir(dirname):
                    print(f"{dirname} is not found.")
            else:
                if not os.path.isfile(name):
                    print(f"{name} is not found.")
                # assert os.path.isfile(name), f"{name} is not found."

    def get_file_list(self):
        file_list = (
            self.models
            + self.transforms
            + self.metrics
            + self.datasets
            + self.dataloader
            + self.samplers
            + self.tests
        )
        self.check_path(file_list)
        return file_list

    @property
    def models(self) -> List:
        return []

    @property
    def transforms(self) -> List:
        return []

    @property
    def metrics(self) -> List:
        return []

    @property
    def datasets(self) -> List:
        return []

    @property
    def dataloader(self) -> List:
        return []

    @property
    def samplers(self) -> List:
        return []

    @property
    def tests(self) -> List:
        return []

import os

from base_2d_galaxy_datasets import datapaths
from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix")


datapaths = EasyDict(datapaths)
buckets = [
    "matrix",
]

from base_2d_night_lmdb_datasets import datapaths as base_2d_datapaths
from easydict import EasyDict
from ev52_x3c_night_lmdb_datasets import datapaths as ev52_x3c_night_datapaths
from maxfa_x3c_night_lmdb_datasets import (
    datapaths as maxfa_x3c_night_datapaths,
)
from project_utils import merge_datapaths

root = "dmpv2://matrix2"

# ----- Required -----

datapaths = {}

datapaths = merge_datapaths(
    datapaths,
    base_2d_datapaths,
    maxfa_x3c_night_datapaths,
    ev52_x3c_night_datapaths,
)

datapaths = EasyDict(datapaths)

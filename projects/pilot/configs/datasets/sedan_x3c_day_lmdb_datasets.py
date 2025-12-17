from base_2d_day_lmdb_datasets import datapaths as base_2d_datapaths
from c385_x3c_day_lmdb_datasets import datapaths as c385_x3c_day_datapaths
from easydict import EasyDict
from project_utils import merge_datapaths

root = "dmpv2://matrix2"

# ----- Required -----

datapaths = {}

datapaths = merge_datapaths(
    datapaths,
    base_2d_datapaths,
    c385_x3c_day_datapaths,
)

datapaths = EasyDict(datapaths)

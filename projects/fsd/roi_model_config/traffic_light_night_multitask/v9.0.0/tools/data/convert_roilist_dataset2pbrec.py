import collections
import os
import pprint
import time
from functools import reduce

import mxnet as mx
import numpy as np
from auto_matrix.data.anno_transformer.anno import get_transformed_anno_list
from auto_matrix.data.dataset import DenseBoxAnnoPacker
from auto_matrix.data.dataset.legacy_densebox import (
    LegacyDenseBoxImageRecordDataset,
)
from gluon_horizon.data.recordio.recordio_packer import RecordIOPacker
from gluon_horizon.data.recordio.recordio_pb_pack import (
    pack,
    pack_head,
    pack_string_data,
)
from mxnet import filestream

from hat.core.anno_ts_utils import ImageRois
from hat.registry import build_from_registry

ROOT_PATHS = "/home/users/fan.lv/hdfs_common/cache_mono_data_pbrec/"


def _default_process_roilist_fn(data):
    assert "roi_list" in data.keys()
    return data["roi_list"]


def _check_exists(path):
    assert filestream.exists(path), "%s does not exists" % path


class DenseBoxRoilistPacker(RecordIOPacker):
    """
    DenseBox roi list packer.

    Parameters
    ----------
    uri : str
        Target path.
    num_workers : int, optional
        The number of parallel workers, by default 1.
    """

    def __init__(self, uri, num_worker=1):
        super().__init__(uri, num_worker)

    def _check_record_data_list(self, record_data_list):
        type_error_msg = (
            "record_data_list should be " "list/tuple of [idx, ImageRois]"
        )
        # assert isinstance(record_data_list, (list, tuple)), type_error_msg
        for record in record_data_list:
            assert len(record) in [2], type_error_msg
            assert isinstance(record[0], int), type_error_msg
            assert isinstance(record[1], ImageRois), type_error_msg

    def set_record_data_list(self, record_data_list):
        """Set the record data list for packing.

        Parameters
        ----------
        record_data_list: iterable
            Can be iterable of (idx, ImageRois)
        """
        self._check_record_data_list(record_data_list)
        self.record_data_list = record_data_list

    def prepare_record(self, idx, data):
        rec_data = None
        data = list(data)
        assert len(data) in [2]
        if len(data) == 2:
            data.append(None)

        header = mx.recordio.IRHeader(0, 0, idx, 0)
        pb_str = pack(
            pack_head(idx, reserve=0, version=1.0),
            data=[
                pack_string_data(idx, str.encode(str(data[0]))),
                pack_string_data(idx, str.encode(data[1].to_json())),
            ],
            label=0,
            extra=[],
        )
        rec_data = mx.recordio.pack(header, pb_str)
        return rec_data


def pack_single_data_desc(rec_path, anno_path, roi_list_path):
    dst_roi_list_path = os.path.join(
        ROOT_PATHS, "/".join(rec_path.split("/")[-4:-1])
    )
    if not os.path.exists(dst_roi_list_path):
        os.makedirs(dst_roi_list_path)
    print(dst_roi_list_path)
    dataset = (
        dict(
            type="DenseboxWithRoilistDataset",
            data_path=rec_path,
            anno_path=anno_path,
            # anno_path=anno_path.replace('.json', '.anno.pb_rec'),
            # rec_idx_file_path=rec_path+'.idx',
            roi_list_path=roi_list_path,
            to_rgb=False,
            task_type="detection",
        ),
    )
    dataset2pe = build_from_registry(dataset)[0]
    if roi_list_path is not None:
        print("Getting roi list")
        roi_list_pairs = get_transformed_anno_list(
            dataset2pe, _default_process_roilist_fn, num_workers=12
        )
        print("Packing roi list")
        roilist_packer = DenseBoxRoilistPacker(
            uri=dst_roi_list_path + "/data.roilist.pb_rec", num_worker=12
        )
        roilist_packer.set_record_data_list(roi_list_pairs)
        roilist_packer.pack()


class DataDesc(object):
    def __init__(
        self,
        rec_path,
        anno_path,
        data_type,
        dst_rec_idx_path,
        dst_anno_path,
        roi_list_path=None,
        dst_roi_list_path=None,
    ):
        self.rec_path = rec_path
        self.anno_path = anno_path
        self.roi_list_path = roi_list_path
        _check_exists(self.rec_path)
        _check_exists(self.anno_path)
        self.data_type = data_type
        self.dst_rec_idx_path = dst_rec_idx_path
        self.dst_anno_path = dst_anno_path
        self.dst_roi_list_path = dst_roi_list_path

    def __repr__(self):
        return pprint.pformat(
            collections.OrderedDict(
                rec_path=self.rec_path,
                anno_path=self.anno_path,
                data_type=self.data_type,
                roi_list_path=self.roi_list_path,
                dst_rec_idx_path=self.dst_rec_idx_path,
                dst_anno_path=self.dst_anno_path,
                dst_roi_list_path=self.dst_roi_list_path,
            )
        )

    @staticmethod
    def assign(
        src_rec_path,
        src_anno_path,
        data_type,
        task_type,
        src_roi_list_path=None,
        output_dir=None,
        pathset=None,
        append_prefix="",
    ):

        src_rec_path = "%s%s" % (append_prefix, src_rec_path)
        src_anno_path = "%s%s" % (append_prefix, src_anno_path)
        if src_roi_list_path is not None:
            src_roi_list_path = "%s%s" % (append_prefix, src_roi_list_path)

        def _get_dst_path(path):

            if path not in pathset:
                pathset.add(path)
            else:
                raise ValueError("duplicate path %s" % path)

            subpath = path.split("/")[-1]

            is_anno = subpath.endswith(".json")
            if is_anno:
                dst_path = subpath.replace(".rec.json", ".json").replace(
                    ".json", ".anno.pb_rec"
                )  # noqa
            else:
                dst_path = subpath + ".idx"

            if output_dir is None:
                path_prefix = os.path.dirname(path)
                dst_path = os.path.join(path_prefix, dst_path)
            else:
                path_prefix = reduce(
                    lambda x, y: x + "_" + y,
                    path[len(append_prefix) + 1 :].split("/")[:-1],
                )  # noqa
                dst_path = os.path.join(
                    output_dir, "%s_%s" % (path_prefix, dst_path)
                )  # noqa

            if dst_path not in pathset:
                pathset.add(dst_path)
            else:
                raise ValueError("duplicate path %s" % dst_path)

            return dst_path

        dst_rec_idx_path = _get_dst_path(src_rec_path)
        dst_anno_path = _get_dst_path(src_anno_path)
        # dst_roi_list_path = dst_anno_path.split(".")[0] + "_roilist.json"
        dst_roi_list_path = dst_anno_path.replace(".anno.", ".roilist.")

        return DataDesc(
            rec_path=src_rec_path,
            anno_path=src_anno_path,
            roi_list_path=src_roi_list_path,
            data_type=data_type,
            dst_rec_idx_path=dst_rec_idx_path,
            dst_anno_path=dst_anno_path,
            dst_roi_list_path=dst_roi_list_path,
        )


def parse_dataset_in_py_format(dataset_map, output_dir=None):

    path_set = set()

    data_descs = []
    for task_type, paths in dataset_map.items():
        if task_type != "traffic_light_lens":
            continue
        if task_type in ["semantic_parsing", "lane_parsing"]:
            data_type = "parsing"
        else:
            data_type = "detection"
        # training paths
        if paths.train_data_paths is not None:
            for path_i in paths.train_data_paths:
                data_descs.append(
                    DataDesc.assign(
                        src_rec_path=path_i.rec_path,
                        src_anno_path=path_i.anno_path,
                        src_roi_list_path=path_i.roi_list_path,
                        output_dir=output_dir,
                        data_type=data_type,
                        task_type=task_type,
                        pathset=path_set,
                    )
                )
        # if paths.val_data_paths is not None:
        #     for path_i in paths.val_data_paths:
        #         data_descs.append(
        #             DataDesc.assign(
        #                 src_rec_path=path_i.rec_path,
        #                 src_anno_path=path_i.anno_path,
        #                 output_dir=output_dir,
        #                 data_type=data_type,
        #                 task_type=task_type,
        #                 pathset=path_set,
        #             )
        #         )

    return data_descs


def _default_process_anno_fn(data):
    assert len(data) == 2
    return data[1]


def convert_single_data_desc(
    img_rec_path,
    ann_rec_path,
    cvted_pbrec_path,
    cvted_pbrec_idx_path,
    num_workers=16,
    with_seg_label=False,
    seg_label_dtype=np.uint8,
    overwrite=False,
):
    print(f"~~~~~ Converting {ann_rec_path} to PbRec ~~~~~")
    tic = time.time()

    dst_anno_path_exists_flag = os.path.exists(cvted_pbrec_path)
    dst_rec_idx_path_exists_flag = os.path.exists(cvted_pbrec_idx_path)
    if (
        dst_rec_idx_path_exists_flag
        and dst_anno_path_exists_flag
        and not overwrite
    ):  # noqa
        msg = f"Ignore {cvted_pbrec_idx_path} and {cvted_pbrec_path}"  # noqa
        print(msg)
    else:
        src_dataset = LegacyDenseBoxImageRecordDataset(
            rec_path=img_rec_path,
            anno_path=ann_rec_path,
            rec_idx_file_path=cvted_pbrec_idx_path,
            read_only=True,
            to_rgb=False,
            with_seg_label=with_seg_label,
            seg_label_dtype=seg_label_dtype,
            as_nd=False,
        )

        if dst_anno_path_exists_flag and not overwrite:  # noqa
            pass
        else:
            print("Getting annotations")
            anno_pairs = get_transformed_anno_list(
                src_dataset, _default_process_anno_fn, num_workers
            )
            print("Packing annotation")
            anno_packer = DenseBoxAnnoPacker(
                uri=cvted_pbrec_path, num_worker=num_workers
            )
            anno_packer.set_record_data_list(anno_pairs)
            anno_packer.pack()

    toc = time.time()
    print("Cost time %.2f" % (toc - tic))


if __name__ == "__main__":
    import argparse

    from auto_matrix.config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="projects/tl_2pe_mtl/cn_sd_day/v0.2.0/config/datasets/train_datasets.py",  # noqa
    )
    parser.add_argument(
        "--output-dir", type=str, default="/home/users/fan.lv/hdfs_common/data"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--copy-src-rec", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=True)

    args = parser.parse_args()

    _check_exists(args.dataset)
    if args.output_dir is not None:
        filestream.makedirs(args.output_dir)

    ext = os.path.splitext(args.dataset)[1]
    assert ext in [".py"]
    data_descs = parse_dataset_in_py_format(
        Config.load_file(args.dataset).datapaths, args.output_dir
    )

    pprint.pprint(data_descs)

    if args.output_dir is None:
        args.copy_src_rec = False

    for data_desc_i in data_descs:
        pack_single_data_desc(
            rec_path=data_desc_i.rec_path,
            anno_path=data_desc_i.anno_path,
            roi_list_path=data_desc_i.roi_list_path,
        )

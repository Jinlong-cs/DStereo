import argparse
import multiprocessing
import os
import pprint
import time
from collections import OrderedDict

import mxnet as mx
import numpy as np
import yaml
from gluon_horizon.data.recordio.recordio_packer import RecordIOPacker
from gluon_horizon.data.recordio.recordio_pb_pack import (
    pack,
    pack_head,
    pack_string_data,
)
from mxnet.base import _as_list
from tqdm import tqdm

from hat.core.anno_ts_utils import ImageRois
from hat.data.datasets.densebox_dataset_with_roilist import (
    LegacyDenseBoxWithRoilistImageRecordDataset,
)


def _check_exists(path):
    assert mx.filestream.exists(path), "%s does not exists" % path


def _default_process_roilist_fn(data):
    return data["roi_list"]


def _worker_fn(anno, anno_transformer):
    return anno_transformer(anno)


class DenseBoxRoilistPacker(RecordIOPacker):
    """DenseBox roi list packer.

    Args:
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
        assert isinstance(record_data_list, (list, tuple)), type_error_msg
        for record in record_data_list:
            assert len(record) in [2], type_error_msg
            assert isinstance(record[0], int), type_error_msg
            assert isinstance(record[1], ImageRois), type_error_msg

    def set_record_data_list(self, record_data_list):
        """Set the record data list for packing.

        Args:
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


class DataDesc(object):
    """Input and output data path."""

    def __init__(
        self,
        rec_path,
        anno_path,
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
        self.dst_rec_idx_path = dst_rec_idx_path
        self.dst_anno_path = dst_anno_path
        self.dst_roi_list_path = dst_roi_list_path

    def __repr__(self):
        return pprint.pformat(
            OrderedDict(
                rec_path=self.rec_path,
                anno_path=self.anno_path,
                roi_list_path=self.roi_list_path,
                dst_rec_idx_path=self.dst_rec_idx_path,
                dst_anno_path=self.dst_anno_path,
                dst_roi_list_path=self.dst_roi_list_path,
            )
        )

    @staticmethod
    def assign(
        src_rec_path, src_anno_path, src_roi_list_path=None, data_root=""
    ):
        src_rec_path = "%s%s" % (data_root, src_rec_path)
        src_anno_path = "%s%s" % (data_root, src_anno_path)
        if src_roi_list_path is not None:
            src_roi_list_path = "%s%s" % (data_root, src_roi_list_path)

        dst_rec_idx_path = f"{src_rec_path}.idx"
        dst_anno_path = os.path.splitext(src_anno_path)[0] + ".anno.pb_rec"
        if src_roi_list_path is not None:
            dst_roi_list_path = (
                os.path.splitext(src_roi_list_path)[0] + ".roilist.pb_rec"
            )
        else:
            dst_roi_list_path = None

        return DataDesc(
            rec_path=src_rec_path,
            anno_path=src_anno_path,
            roi_list_path=src_roi_list_path,
            dst_rec_idx_path=dst_rec_idx_path,
            dst_anno_path=dst_anno_path,
            dst_roi_list_path=dst_roi_list_path,
        )


def parse_dataset_in_yaml_format(
    dataset_map, data_root="/horizon-bucket/mono/"
):
    data_descs = []
    for _, paths in dataset_map.items():
        for path_i in paths:
            if path_i["train_rec"] is not None:
                data_descs.append(
                    DataDesc.assign(
                        src_rec_path=path_i["train_rec"],
                        src_anno_path=path_i["train_json"],
                        src_roi_list_path=path_i["train_roi_list"],
                        data_root=data_root,
                    )
                )
            if path_i.get("val_rec", None) is not None:
                assert path_i["val_json"] is not None
                data_descs.append(
                    DataDesc.assign(
                        src_rec_path=path_i["val_rec"],
                        src_anno_path=path_i["val_json"],
                        src_roi_list_path=path_i["val_roi_list"],
                        data_root=data_root,
                    )
                )
    return data_descs


def get_transformed_list(dataset, anno_transformer, num_workers=0):
    """Get 2 record data list for packing.
    The first one is a list/tuple of image path and annotations, it is used
    for :py:class:`gluon_matrix.data.dataset.HorizonImageDataPacker`.

    Args:
        dataset: :py:class:`mxnet.gluon.data.Dataset`
        anno_transformer: callable
            Transformer to transform annotations.
            This function is called on the following way

            .. code-block:: none

                ret = anno_transformer(dataset[idx])

            If you want to ignore the input sample, just return None.
        num_workers: int, optional
            The number of workers, by default 0.

    Returns:
        transform_annos: list/tuple of obj
            Transformed annotations.
    """
    assert callable(anno_transformer)
    valid_flags = [False] * len(dataset)
    pbar = tqdm(total=len(dataset))

    transformed_anno_pairs = []
    if num_workers > 1 and multiprocessing is not None:
        results = []
        worker_pool = multiprocessing.Pool(num_workers)
        for idx in range(len(dataset)):
            item = dataset[idx]
            ret = worker_pool.apply_async(_worker_fn, (item, anno_transformer))
            results.append(ret)
        for idx in range(len(results)):
            ret = results[idx]
            ret = ret.get()
            valid_flags[idx] = ret is not None
            transformed_anno_pairs.append([idx] + list(_as_list(ret)))
            pbar.update(1)
        worker_pool.close()
        worker_pool.join()
        worker_pool.terminate()
    else:
        for idx in range(len(dataset)):
            item = dataset[idx]
            ret = _worker_fn(item, anno_transformer)
            valid_flags[idx] = ret is not None
            transformed_anno_pairs.append([idx] + list(_as_list(ret)))
            pbar.update(1)
    # filter out invalid pairs
    pbar.close()
    transformed_anno_pairs = [
        p
        for p, flag in zip(transformed_anno_pairs, valid_flags)
        if flag is True
    ]
    return transformed_anno_pairs


def pack_single_data_desc(data_desc, num_workers, overwrite=False):
    print("~~~~~ Packing %s ~~~~~" % data_desc.rec_path)
    tic = time.time()

    src_dataset = LegacyDenseBoxWithRoilistImageRecordDataset(
        rec_path=data_desc.rec_path,
        anno_path=data_desc.anno_path,
        roi_list_path=data_desc.roi_list_path,
        read_only=True,
        to_rgb=False,
        seg_label_dtype=np.uint8,
        rec_idx_file_path=data_desc.dst_rec_idx_path,
    )
    src_dataset._decoder = None
    data_desc.dst_roi_list_path = "tmp.roilist.pb_rec"
    if data_desc.roi_list_path is not None:
        if not mx.filestream.exists(data_desc.dst_roi_list_path) or overwrite:
            print("Getting roi list")
            roi_list_pairs = get_transformed_list(
                src_dataset, _default_process_roilist_fn, num_workers
            )
            print("Packing roi list")
            roilist_packer = DenseBoxRoilistPacker(
                uri=data_desc.dst_roi_list_path, num_worker=num_workers
            )
            roilist_packer.set_record_data_list(roi_list_pairs)
            roilist_packer.pack()

    toc = time.time()
    print("Cost time %.2f" % (toc - tic))


def convert(data_descs, num_workers, overwrite=False):
    for data_desc_i in data_descs:
        pack_single_data_desc(data_desc_i, num_workers, overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datas_mono.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=True)

    args = parser.parse_args()

    _check_exists(args.dataset)

    ext = os.path.splitext(args.dataset)[1]
    if ext in [".yaml", ".yml"]:
        data_descs = parse_dataset_in_yaml_format(
            yaml.safe_load(open(args.dataset, "r"))
        )
    else:
        raise TypeError(
            f"Unsupported dataset {args.dataset} endswith ext {ext}"
        )  # noqa

    pprint.pprint(data_descs)

    convert(data_descs, num_workers=args.num_workers)

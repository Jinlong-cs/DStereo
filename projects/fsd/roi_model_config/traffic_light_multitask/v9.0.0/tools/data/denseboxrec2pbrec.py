import collections
import os
import pprint
import time

import numpy as np
from auto_matrix.data.anno_transformer.anno import get_transformed_anno_list
from auto_matrix.data.dataset import DenseBoxAnnoPacker
from auto_matrix.data.dataset.legacy_densebox import (
    LegacyDenseBoxImageRecordDataset,
)
from mxnet import filestream

ROOT_PATHS = "/home/users/fan.lv/hdfs_common/cache_mono_data_pbrec/"


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
            rec_idx_file_path=None,
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


def _check_exists(path):
    assert filestream.exists(path), "%s does not exists" % path


class DataDesc(object):
    def __init__(self, rec_path, anno_path, task_type, dst_path):
        self.rec_path = rec_path
        self.anno_path = anno_path
        self.task_type = task_type
        _check_exists(self.rec_path)
        _check_exists(self.anno_path)
        self.dst_path = dst_path

    def __repr__(self):
        return pprint.pformat(
            collections.OrderedDict(
                rec_path=self.rec_path,
                anno_path=self.anno_path,
                task_type=self.task_type,
                dst_path=self.dst_path,
            )
        )

    @staticmethod
    def assign(
        src_rec_path,
        src_anno_path,
        task_type,
        output_dir=None,
        pathset=None,
        append_prefix: str = "",
    ):

        src_rec_path = "%s%s" % (append_prefix, src_rec_path)
        src_anno_path = "%s%s" % (append_prefix, src_anno_path)

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
                # path_prefix = reduce(lambda x, y: x + '_' + y, path[len(append_prefix)+1:].split('/')[:-1])  # noqa
                # dst_path = os.path.join(output_dir, '%s_%s' % (path_prefix, dst_path))  # noqa
                dst_path = os.path.join(
                    output_dir, "/".join(path.split("/")[-4:-1])
                )

            if dst_path not in pathset:
                pathset.add(dst_path)
            else:
                raise ValueError("duplicate path %s" % dst_path)

            return dst_path

        dst_path = _get_dst_path(src_rec_path)
        # dst_anno_path = _get_dst_path(src_anno_path)
        # dst_roi_list_path = dst_anno_path.split(".")[0] + "_roilist.json"

        return DataDesc(
            rec_path=src_rec_path,
            anno_path=src_anno_path,
            task_type=task_type,
            dst_path=dst_path,
        )


def parse_dataset_in_py_format(dataset_map, output_dir=None):

    path_set = set()

    data_descs = []
    for task_type, paths in dataset_map.items():
        if paths.train_data_paths is not None:
            for path_i in paths.train_data_paths:
                data_descs.append(
                    DataDesc.assign(
                        src_rec_path=path_i.rec_path,
                        src_anno_path=path_i.anno_path,
                        output_dir=output_dir,
                        pathset=path_set,
                        task_type=task_type,
                    )
                )

    return data_descs


if __name__ == "__main__":
    import argparse

    from auto_matrix.config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="projects/mono/traffic_light_multitask/config/datasets/convert_datasets.py",  # noqa
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
        convert_single_data_desc(
            img_rec_path=data_desc_i.rec_path,
            ann_rec_path=data_desc_i.anno_path,
            cvted_pbrec_path=os.path.join(
                data_desc_i.dst_path, "data.anno.pb_rec"
            ),
            cvted_pbrec_idx_path=os.path.join(
                data_desc_i.dst_path, "data.anno.pb_rec.idx"
            ),
            num_workers=16,
            with_seg_label=False,
            seg_label_dtype=np.uint8,
            overwrite=True,
        )

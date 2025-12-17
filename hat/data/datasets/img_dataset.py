"""Some class are migrate from gluon_person."""
# coding: utf-8
import logging
import os
from typing import Sequence, Union

import cv2
import numpy as np
import torch.utils.data as data

from hat.utils.pack_type.lmdb import Lmdb, LmdbReadList
from hat.utils.pack_type.mxrecord import MXRecord, MXRecordIO, unpack_img

__all__ = ["IndexDataset", "NamedIndexDataset", "ImgLmdbDataset"]


class IndexDataset(data.Dataset):
    """A dataset wrapping over BaseDataset using index to read RecordIO file.

    Each sample is an image.

    Parameters
    ----------
    imgrec_path_list : list[str]
        list of path to rec file.
    rgb : bool, default=True
        True for RGB formatted output (MXNet default).
        False for BGR formatted output (OpenCV default).
    kwargs : ...
        Other arguments.

    """

    def __init__(
        self,
        imgrec_path_list,
        num_images=None,
        read_len_type="lst",
        copy_rec=False,
        rgb=True,
        cv_format=cv2.IMREAD_COLOR,
        **kwargs
    ):
        super(IndexDataset, self).__init__(**kwargs)
        self.imgrec_path_list = imgrec_path_list
        self.num_images = num_images
        self.read_len_type = read_len_type
        self.rgb = rgb
        self.cv_format = cv_format
        if copy_rec:
            for i in range(len(self.imgrec_path_list)):
                if self.imgrec_path_list[i] is None:
                    continue
                src_rec_path = self.imgrec_path_list[i]
                src_idx_path = src_rec_path[:-4] + ".idx"
                src_lst_path = src_rec_path[:-4] + ".lst"
                os.system("hdfs dfs -get %s img_%d.rec" % (src_rec_path, i))
                os.system("hdfs dfs -get %s img_%d.idx" % (src_idx_path, i))
                os.system("hdfs dfs -get %s img_%d.lst" % (src_lst_path, i))
                self.imgrec_path_list[i] = "img_%d.rec" % i

        self.imgrec_reader = None

    def __len__(self):
        if self.num_images is None or self.num_images <= 0:
            self.num_images = 0
            for imgrec_path in self.imgrec_path_list:
                if self.read_len_type == "lst":
                    imglst_path = imgrec_path[:-4] + ".lst"
                    with open(imglst_path, "r") as fn:
                        self.num_images += len(fn.readlines())

                elif self.read_len_type == "idx":
                    imgidx_path = imgrec_path[:-4] + ".idx"
                    with open(imgidx_path, "r") as fn:
                        self.num_images += len(fn.readlines())

                elif self.read_len_type == "rec":
                    # TODO: packtype unsupported
                    record = MXRecordIO(imgrec_path, "r")
                    while True:
                        data = record.read()
                        if data is None:
                            break
                        self.num_images += 1
                    record.close()
                else:
                    raise NotImplementedError
        return self.num_images

    def load_data(self):
        """Load the RecordIO (.rec) file if img rec_reader is None."""
        if self.imgrec_reader is None:
            logging.info(
                "use imgrec to read image, {}".format(self.imgrec_path_list)
            )
            self.imgrec_reader = []
            for imgrec_path in self.imgrec_path_list:
                if imgrec_path is None:
                    self.imgrec_reader.append(None)
                else:
                    imgidx_path = imgrec_path[:-4] + ".idx"
                    self.imgrec_reader.append(
                        MXRecord(
                            imgrec_path, idx_path=imgidx_path, writable=False
                        )
                    )

    def __getitem__(self, idx):
        self.load_data()
        assert self.imgrec_reader[idx[0]] is not None
        _, img = unpack_img(
            self.imgrec_reader[idx[0]].read(idx[1]), self.cv_format
        )
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class NamedIndexDataset(IndexDataset):
    """wrapping over IndexDataset, used by action.

    Parameters
    ----------
    imgrec_path_list : list
        image rec file path list
    num_images : int, optional
        if set to positive and is smaller than len(IndexDataset), the
        len(IndexDataset) will be overwrite by set value, by default None
    read_len_type : str, optional
        method to get dataset length, by default 'lst'
    copy_rec : bool, optional
        whether copy rec from hdfs to local, by default False
    rgb : bool, optional
        whether exchange image 1st channel with 3rd channel, by default True
    kwargs :
        read_by_name: str, optional
            if it is not set or is set to False, NamedIndexDataset is totally
            same with IndexDataset
    """

    def __init__(
        self,
        imgrec_path_list,
        num_images=None,
        read_len_type="lst",
        copy_rec=False,
        rgb=True,
        **kwargs
    ):
        self.read_by_name = kwargs.get("read_by_name", False)
        if self.read_by_name:
            kwargs.pop("read_by_name")
        super(NamedIndexDataset, self).__init__(
            imgrec_path_list,
            num_images=num_images,
            read_len_type=read_len_type,
            copy_rec=copy_rec,
            rgb=rgb,
            **kwargs
        )
        self._img_dict = None

    @property
    def img_dict(self):
        """Generate image name to rec index dictionary."""
        if self._img_dict is None:
            count_rec = 0
            self._img_dict = {}
            for rec_path in self.imgrec_path_list:
                lst_path = rec_path.replace(".rec", ".lst")
                cur_img_list = _read_lst(lst_path)
                for k in cur_img_list:
                    cur_img_list[k] = (count_rec, cur_img_list[k])
                count_rec += 1
                self._img_dict.update(cur_img_list)
        return self._img_dict

    def __getitem__(self, key):
        if self.read_by_name:
            assert isinstance(key, str)
            idxs = self.img_dict[key]
        else:
            idxs = key
        return super(NamedIndexDataset, self).__getitem__(idxs)


def _read_lst(lst_path):
    """Read the lst (.lst) file."""
    image_lst = {}
    with open(lst_path, "r") as fin:
        for line in iter(fin.readline, ""):
            try:
                line = line.decode("utf-8")
            except AttributeError:
                pass
            line = line.strip().split("\t")
            image_index = line[0]
            image_name = line[-1]
            assert image_name not in image_lst
            image_lst[image_name] = int(image_index)
    return image_lst


class NamedIndexDatasetV2(NamedIndexDataset):
    """
    Wrapping over NamedIndexDataset.

    support read index from lmdb to avoid excessive memory consumption.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lst_lmdb = {}
        self.init_lst_lmdb()

    def init_lst_lmdb(self):
        count_rec = 0
        for rec_path in self.imgrec_path_list:
            lst_lmdb_path = rec_path.replace(".rec", ".lst.lmdb")
            if os.path.isdir(lst_lmdb_path):
                # lst_lmdb_path is a directory means loading lmdb.
                self.lst_lmdb[count_rec] = Lmdb(
                    lst_lmdb_path,
                    False,
                    readonly=True,
                    map_size=1024 ** 2 * 10,
                )

            count_rec += 1

    @property
    def img_dict(self):
        """Generate image name to rec index dictionary."""
        if self._img_dict is None:
            count_rec = 0
            self._img_dict = {}
            for rec_path in self.imgrec_path_list:
                lst_lmdb_path = rec_path.replace(".rec", ".lst.lmdb")
                if not os.path.isdir(lst_lmdb_path):
                    # use lst file
                    lst_path = rec_path.replace(".rec", ".lst")
                    cur_img_list = _read_lst(lst_path)
                    for k in cur_img_list:
                        cur_img_list[k] = (count_rec, cur_img_list[k])
                    self._img_dict.update(cur_img_list)
                count_rec += 1

        return self._img_dict

    def __getstate__(self):
        state = self.__dict__
        for _, lst_lmdb in state["lst_lmdb"].items():
            lst_lmdb.close()
        state["lst_lmdb"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_lst_lmdb()

    def __getitem__(self, key):
        if self.read_by_name:
            assert isinstance(key, str)
            if key in self.img_dict:
                idxs = self.img_dict[key]
            else:
                for rec_idx, lst_lmdb in self.lst_lmdb.items():
                    try:
                        idxs = (rec_idx, int(lst_lmdb.read(key).decode()))
                        break
                    except Exception:
                        continue
        else:
            idxs = key
        return super(NamedIndexDataset, self).__getitem__(idxs)


class ImgLmdbDataset(data.Dataset):
    """
    A dataset using key to read image from Lmdb file.

    Args:
        uri: Path to lmdb file, support str and list[str].
        rgb : bool, optional
            whether exchange image 1st channel with 3rd channel.
        cv_format : format for use cv2.decode.
        map_size: map size of default lmdb settings

    """

    def __init__(
        self,
        uris: Union[str, Sequence[str]],
        rgb: bool = True,
        cv_format: int = cv2.IMREAD_COLOR,
        map_size: int = 1024 ** 2 * 100,
    ) -> None:

        self.uris = uris
        self.rgb = rgb
        self.cv_format = cv_format
        self.map_size = map_size
        self.init_lst_lmdb()

    def init_lst_lmdb(
        self,
    ):
        self.lmdb_dataset = LmdbReadList(self.uris, map_size=self.map_size)

    def __getstate__(self):
        state = self.__dict__
        state["lmdb_dataset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_lst_lmdb()

    def __len__(self):
        """Get the length."""
        return len(self.lmdb_dataset)

    def __getitem__(self, key):
        buffer_img = self.lmdb_dataset.read(key)
        encode_img = np.frombuffer(buffer_img, np.uint8)
        img = cv2.imdecode(encode_img, self.cv_format)
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

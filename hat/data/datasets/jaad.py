# -*- coding: utf-8 -*-  # noqa: D100
###################################################################
#   Copyright (C) 2022 Horizon Robotics. All rights reserved.
#   This file defines the datasets for trajectory prediction.
#   Filename    : jaad.py
#   Author      : xiaoming.zhang
#   Date        : Dec 2022
#   Description : Interface for the JAAD dataset and it is a simplified
# version, which contains only the load of cached pickle file, without
# the generation of cached dataset, as it is too redundant.
###################################################################

import pickle
from os import makedirs
from os.path import abspath, exists, join


class JAAD(object):
    def __init__(self, data_path: str = ""):
        """Construct the jaad class.

        Args:
            data_path: path to the folder of the dataset
        """
        self._image_ext = ".png"
        # Paths
        self._jaad_path = data_path
        assert exists(self._jaad_path), "Jaad path does not exist: {}".format(
            self._jaad_path
        )
        self._data_split_ids_path = join(self._jaad_path, "split_ids")

    # Path generators
    @property
    def cache_path(self):
        """Generate a path to save cache files.

        Returns:
            cache_path (str): cached file folder path
        """
        cache_path = abspath(join(self._jaad_path, "data_cache"))
        if not exists(cache_path):
            makedirs(cache_path)
        return cache_path

    def _get_video_ids_split(self, image_set: str, subset: str = "default"):
        """Return a list of video ids for a given data split.

        Args:
            image_set: data split, train, test, val

        Returns:
            vid_ids (list): the list of video ids
        """
        vid_ids = []
        sets = [image_set] if image_set != "all" else ["train", "test", "val"]
        for s in sets:
            vid_id_file = join(self._data_split_ids_path, subset, s + ".txt")
            with open(vid_id_file, "rt") as fid:
                vid_ids.extend([x.strip() for x in fid.readlines()])
        return vid_ids

    def generate_database(self):
        """Generate a database of jaad dataset by integrating all annotations.

        Returns:
            database (dict): a database dictionary
        """
        # Generates a list of behavioral xml file names for  videos
        cache_file = join(self.cache_path, "jaad_database.pkl")
        if exists(cache_file):
            with open(cache_file, "rb") as fid:
                try:
                    database = pickle.load(fid)
                except Exception:
                    database = pickle.load(fid, encoding="bytes")
            return database
        else:
            raise Exception("JAAD cache path doesn't exist.")

    # Trajectory data generation
    def _get_data_ids(self, image_set: str, params):
        """Help generate set id and ped ids (if needed) for processing.

        Args:
            image_set: train, test, val
            params: ref func generate_data_trajectory_sequence

        Returns:
            vid_ids (list): the list of video ids
        """
        _pids = None
        try:
            return (
                self._get_video_ids_split(image_set, params["subset"]),
                _pids,
            )
        except Exception:
            raise Exception("The subset of params is not appropriate.")

    def generate_data_trajectory_sequence(self, image_set: str, **opts):
        """Generate pedestrian tracks.

        Args:
            image_set: train, val, test
            opts: specific options (fixed)
                "fstride": frequency of sampling from the data
                "subset": the name of folder which stores the split ids
                "data_split_type": how to split the data
                "min_track_size": min track length allowable

        Returns:
            sequence (Dict): a dictionary of trajectories
        """
        params = {
            "fstride": 1,
            "subset": "default",
            "data_split_type": "default",
            "min_track_size": 20,
        }
        assert all(
            k in params for k in opts.keys()
        ), "Wrong option(s)." "Choose one of the following: {}".format(
            list(params.keys())
        )
        params.update(opts)

        annot_database = self.generate_database()
        sequence = self._get_trajectories(image_set, annot_database, **params)

        return sequence

    def _get_trajectories(self, image_set: str, annotations: dict, **params):
        """Generate final trajectory data.

        Args:
            image_set: train, val, test
            params: ref func generate_data_trajectory_sequence
            annotations: the annotations database

        Returns:
            ret (Dict): a dictionary of trajectories
        """

        seq_stride = params["fstride"]
        image_seq, pids_seq, box_seq, resolution_seq = [], [], [], []

        video_ids, _pids = self._get_data_ids(image_set, params)

        for vid in sorted(video_ids):
            img_width = annotations[vid]["width"]
            img_height = annotations[vid]["height"]
            pid_annots = annotations[vid]["ped_annotations"]

            for pid in sorted(annotations[vid]["ped_annotations"]):

                if params["data_split_type"] != "default" and pid not in _pids:
                    continue
                if "p" in pid:
                    continue

                images = [
                    join(
                        self._jaad_path, "images", vid, "{:05d}.png".format(f)
                    )
                    for f in pid_annots[pid]["frames"]
                ]

                boxes = pid_annots[pid]["bbox"]

                if len(boxes) / seq_stride < params["min_track_size"]:
                    continue

                ped_ids = [[pid]] * len(boxes)
                image_seq.append(images[::seq_stride])
                box_seq.append(boxes[::seq_stride])
                pids_seq.append(ped_ids[::seq_stride])
                resolutions = [[img_width, img_height]] * len(boxes)
                resolution_seq.append(resolutions[::seq_stride])
        ret = {
            "image": image_seq,
            "resolution": resolution_seq,
            "pid": pids_seq,
            "bbox": box_seq,
        }
        return ret

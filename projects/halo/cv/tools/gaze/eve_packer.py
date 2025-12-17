import argparse
import json
import logging
import multiprocessing
import os
import pickle
import time
from typing import List, Optional

import cv2
import ffmpeg
import h5py
import msgpack
import numpy as np
from torch.multiprocessing import ProcessContext

from hat.data.datasets.data_packer import Packer
from hat.utils.logger import init_logger

logger = logging.getLogger(__name__)


predefined_splits = {
    "train": ["train%02d" % i for i in range(1, 40)],
    "val": ["val%02d" % i for i in range(1, 6)],
    "test": ["test%02d" % i for i in range(1, 11)],
    "etc": ["etc%02d" % i for i in range(1, 3)],
}


class EVEDatasetPacker(Packer):
    """Create EVE Dataset.

    Implement based on paper: https://arxiv.org/abs/2007.13120.
    Args:
        dataset_path (str): the path of eve dataset.
        dataset_type (str): must be train, val or test.
        src_type (str): the source type to pack, must be idx, anno or data.
        save_dir (str): Save path for packed file.
        assumed_frame_rate (int): sampling frequency
        max_sequence_len (int): the maximum sequence length.
        camera_frame_type (str): input image type, must be full, face or eyes.
        num_workers (int): Num workers for reading data using multiprocessing.
        pack_type (str): The file type for packing.
        num_samples (int): the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_type: str,
        src_type: str,
        save_dir: str,
        assumed_frame_rate: int = 10,
        max_sequence_len: int = 30,
        camera_frame_type: str = "eyes",
        num_workers: Optional[int] = 8,
        pack_type: Optional[str] = "lmdb",
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.path = dataset_path
        self.types_of_stimuli = ["image", "video", "wikipedia"]
        self.cameras_to_use = ["basler", "webcam_l", "webcam_c", "webcam_r"]
        self.assumed_frame_rate = assumed_frame_rate
        self.max_sequence_len = max_sequence_len
        self.src_type = src_type
        self.camera_frame_type = camera_frame_type
        self.source_to_fps = {
            "basler": 60,
            "webcam_l": 30,
            "webcam_c": 30,
            "webcam_r": 30,
        }

        assert dataset_type in ["train", "val", "test"]
        self.participants_to_use = predefined_splits[dataset_type]
        assert len(self.participants_to_use) > 0
        assert 30 > assumed_frame_rate
        assert 30 % assumed_frame_rate == 0

        cache_pkl_path = (
            "./tmp_models/segmentation_cache/%dHz_seqlen%d.pkl"
            % (
                assumed_frame_rate,
                max_sequence_len,
            )
        )
        if not os.path.isfile(cache_pkl_path):
            os.makedirs(os.path.dirname(cache_pkl_path), exist_ok=True)
            self.build_segmentation_cache(cache_pkl_path)
            assert os.path.isfile(cache_pkl_path)
        with open(cache_pkl_path, "rb") as f:
            self.sequence_segmentations = pickle.load(f)

        # Register entries
        self.select_sequences()
        if num_samples is None:
            num_samples = len(self.all_subfolders)
        super(EVEDatasetPacker, self).__init__(
            save_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def build_segmentation_cache(self, cache_pkl_path):
        all_folders = sorted(
            [
                d
                for d in os.listdir(self.path)
                if os.path.isdir(self.path + "/" + d)
            ]
        )
        output_to_cache = {}
        for folder_name in all_folders:
            participant_path = "%s/%s" % (self.path, folder_name)
            assert os.path.isdir(participant_path)
            output_to_cache[folder_name] = {}

            subfolders = sorted(
                [
                    p
                    for p in os.listdir(participant_path)
                    if os.path.isdir(os.path.join(participant_path, p))
                    and p.split("/")[-1].startswith("step")
                    and "eye_tracker_calibration" not in p
                ]
            )
            for subfolder in subfolders:
                subfolder_path = "%s/%s" % (participant_path, subfolder)
                output_to_cache[folder_name][subfolder] = {}

                # We assume that the videos are synchronized
                # and have the same length in time.
                for source in ("basler", "webcam_l", "webcam_c", "webcam_r"):
                    current_outputs = []
                    source_path_pre = "%s/%s" % (subfolder_path, source)
                    available_indices = np.loadtxt(
                        "%s.timestamps.txt" % source_path_pre
                    )
                    num_available_indices = len(available_indices)

                    # Determine desired length and skips
                    fps = self.source_to_fps[source]
                    target_len_in_s = (
                        self.max_sequence_len / self.assumed_frame_rate
                    )
                    num_original_indices_in_sequence = fps * target_len_in_s
                    assert num_original_indices_in_sequence.is_integer()
                    num_original_indices_in_sequence = int(
                        num_original_indices_in_sequence
                    )
                    index_interval = int(fps / self.assumed_frame_rate)
                    start_index = 0
                    while start_index < num_available_indices:
                        end_index = min(
                            start_index + num_original_indices_in_sequence,
                            num_available_indices,
                        )
                        picked_indices = list(
                            range(start_index, end_index, index_interval)
                        )
                        current_outputs.append(picked_indices)

                        # Move along sequence
                        start_index += num_original_indices_in_sequence

                    # Store back indices
                    if len(current_outputs) > 0:
                        output_to_cache[folder_name][subfolder][
                            source
                        ] = current_outputs

        # Do the caching
        with open(cache_pkl_path, "wb") as f:
            pickle.dump(output_to_cache, f)

    def select_sequences(self):
        self.all_subfolders = []
        for obj_name, obj_data in self.sequence_segmentations.items():
            if obj_name not in self.participants_to_use:
                continue

            for stimulus_name, stimulus_segments in obj_data.items():
                current_stimulus_type = stimulus_name.split("_")[1]
                if current_stimulus_type not in self.types_of_stimuli:
                    continue

                for camera, all_indices in stimulus_segments.items():
                    if camera not in self.cameras_to_use:
                        continue

                    for indices in all_indices:
                        self.all_subfolders.append(
                            {
                                "camera_name": camera,
                                "participant": obj_name,
                                "subfolder": stimulus_name,
                                "partial_path": "%s/%s"
                                % (obj_name, stimulus_name),
                                "full_path": "%s/%s/%s"
                                % (self.path, obj_name, stimulus_name),
                                "indices": indices,
                                "key": "%s_%s_%s"
                                % (obj_name, stimulus_name, camera),
                            }
                        )

    def load_all_from_source(self, path, source, selected_indices, flag=True):
        assert source in ("basler", "webcam_l", "webcam_c", "webcam_r")

        # Read HDF
        subentry = {}
        with h5py.File("%s/%s.h5" % (path, source), "r") as hdf:
            for k1, v1 in hdf.items():
                if isinstance(v1, h5py.Group):
                    subentry[k1] = np.copy(v1["data"][selected_indices])
                    subentry[k1 + "_validity"] = np.copy(
                        v1["validity"][selected_indices]
                    )
                else:
                    shape = v1.shape
                    subentry[k1] = np.repeat(
                        np.reshape(v1, (1, *shape)),
                        repeats=self.max_sequence_len,
                        axis=0,
                    )

        # Compute rotation matrices from rvec values
        subentry["head_R"] = np.stack(
            [cv2.Rodrigues(rvec)[0] for rvec in subentry["head_rvec"]]
        )

        # Get frames
        video_path = "%s/%s" % (path, source)

        if self.camera_frame_type == "full":
            video_path += ".mp4"
        elif self.camera_frame_type == "face":
            video_path += "_face.mp4"
        elif self.camera_frame_type == "eyes":
            video_path += "_eyes.mp4"
        else:
            raise ValueError(
                "Unknown camera frame type: %s" % self.camera_frame_type
            )

        timestamps, frames = VideoReader(
            video_path, frame_indices=selected_indices, is_async=False
        ).get_frames(flag)

        # Collect and return
        subentry["timestamps"] = np.asarray(timestamps, dtype=np.int64)
        if flag:
            subentry["frames"] = frames

        # Pad as necessary with zero value and zero validity
        for key, value in subentry.items():
            if value.shape[0] < self.max_sequence_len:
                pad_len = self.max_sequence_len - value.shape[0]
                if pad_len > 0:
                    subentry[key] = np.pad(
                        value,
                        pad_width=[
                            (0, pad_len if i == 0 else 0)
                            for i in range(value.ndim)
                        ],
                        mode="constant",
                        constant_values=(
                            False if value.dtype is np.bool_ else 0.0
                        ),
                    )
            if isinstance(value, np.ndarray):
                subentry[key] = subentry[key].tolist()

        return subentry

    def pack_data(self, idx):
        spec = self.all_subfolders[idx]
        path = spec["full_path"]
        source = spec["camera_name"]
        indices = spec["indices"]
        if self.src_type == "data":
            return_frame = True
        else:
            return_frame = False
        entry = self.load_all_from_source(path, source, indices, return_frame)
        pack_datas = []
        data = {}
        for key, value in entry.items():
            for i in range(self.max_sequence_len):
                if i not in data.keys():
                    data[i] = {}
                data[i][key] = value[i]

        for _, value in data.items():
            anno_key = spec["key"] + "_" + str(value["timestamps"])
            if self.src_type == "anno":
                pack_data = value
            elif self.src_type == "data":
                img = np.asarray(value["frames"]).astype(np.uint8)
                pack_data = cv2.imencode(".jpg", img)[1].tobytes()
            elif self.src_type == "idx":
                pack_data = {
                    "scene_token": spec["key"],
                    "timestamp": value["timestamps"],
                    "key": anno_key,
                }
            else:
                assert "src_type must in [idx, anno, data]"
            pack_datas.append(
                (anno_key, msgpack.packb(pack_data, use_bin_type=True))
            )

        return pack_datas

    def _mp_process(self):
        ctx = multiprocessing.get_context(self.start_method)
        queue = ctx.Queue(self.queue_maxsize)

        rank = int(self.num_blocks / self.num_workers)
        if rank * self.num_workers != self.num_blocks:
            rank += 1
        data_idx = list(range(self.num_blocks))

        read_process = [
            ctx.Process(
                target=_read_worker,
                args=(self, data_idx[i * rank : (i + 1) * rank], queue),
            )
            for i in range(self.num_workers)
        ]
        error_queues = [
            multiprocessing.SimpleQueue() for i in range(self.num_workers)
        ]

        write_process = ctx.Process(target=_write_worker, args=(self, queue))

        for p in read_process:
            p.start()

        error_context = ProcessContext(read_process, error_queues)
        write_process.start()

        error_context.join()

        for p in read_process:
            p.join()

        queue.put(None)
        write_process.join()

        for p in read_process + [write_process]:
            if p.is_alive():
                p.terminate()


def _read_worker(packer, data_idx, queue):
    """
    Read data and put into queue by idx of q_in.

    Processed data by packer.pack_data.

    Args:
        packer (Packer): Instantiated packer.
        data_idx (list): Packing index.
        queue (multiprocess.Queue): Queue for packing data.

    """
    try:
        for idx in data_idx:
            blocks = packer.pack_data(idx)
            for key, block in blocks:
                if block is not None:
                    queue.put((key, block))
    except Exception as e:
        queue.put(None)
        raise e


def _write_worker(packer, queue):
    """
    Write processed data into packer.dataset.

    Args:
        packer (Packer): Instantiated packer.
        queue (multiprocess.Queue): Queue for packing data.
    """

    idx = 0
    pre_time = time.time()
    while True:
        data = queue.get()
        if data is None:
            # try to write length for dataset
            packer._write_length(idx)
            break

        q_idx, q_data = data
        packer._write(q_idx, q_data)
        idx += 1
        if idx % 100 == 0:
            cur_time = time.time()
            logger.info(f"time: {cur_time - pre_time}, count: {idx}")
            pre_time = cur_time


class VideoReader(object):
    def __init__(
        self, video_path, frame_indices=None, is_async=True, output_size=None
    ):
        self.is_async = is_async
        self.video_path = video_path
        self.output_size = output_size
        self.frame_indices = frame_indices
        if self.video_path.endswith("_eyes.mp4"):
            self.ts_path = video_path.replace("_eyes.mp4", ".timestamps.txt")
        elif self.video_path.endswith("_face.mp4"):
            self.ts_path = video_path.replace("_face.mp4", ".timestamps.txt")
        elif self.video_path.endswith(".128x72.mp4"):
            self.ts_path = video_path.replace(".128x72.mp4", ".timestamps.txt")
        else:
            self.ts_path = video_path.replace(".mp4", ".timestamps.txt")
        assert os.path.isfile(self.video_path)
        assert os.path.isfile(self.ts_path)
        self.preparations()

    def get_frames(self, return_frame=True):
        assert self.is_async is False

        # Get frames
        input_params, output_params = self.get_params()
        if return_frame:
            buffer, _ = (
                ffmpeg.input(self.video_path, **input_params)
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    loglevel="quiet",
                    **output_params,
                )
                .run(capture_stdout=True, quiet=True)
            )
            frames = np.frombuffer(buffer, np.uint8).reshape(
                -1, self.height, self.width, 3
            )
        else:
            frames = None

        # Get timestamps
        timestamps = self.timestamps
        if self.frame_indices is not None:
            timestamps = self.timestamps[self.frame_indices]

        return timestamps, frames

    def preparations(self):
        # Read video file tags
        probe = ffmpeg.probe(self.video_path)
        video_stream = next(
            (
                stream
                for stream in probe["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )
        self.width = video_stream["width"]
        self.height = video_stream["height"]
        assert self.height != 0
        assert self.width != 0
        if self.output_size is not None:
            self.width, self.height = self.output_size

        # Read timestamps file
        self.timestamps = np.loadtxt(self.ts_path).astype(np.int64)

    def __enter__(self):
        assert self.is_async
        return self

    def get_params(self):
        # Input params (specifically, selection of decoder)
        input_params = {}
        input_params["vsync"] = 0

        # Set output params (resize frame here)
        output_params = {}
        if self.frame_indices is not None:
            # Index picking for range [start_index, end_index)
            assert len(self.frame_indices) > 1
            cmd = "select='%s'" % "+".join(
                [("eq(n,%d)" % index) for index in self.frame_indices]
            )
            output_params["vf"] = (
                output_params["vf"] + "," + cmd
                if "vf" in output_params
                else cmd
            )
        if self.output_size is not None:
            ow, oh = self.output_size
            cmd = "scale=%d:%d" % (ow, oh)
            output_params["vf"] = (
                output_params["vf"] + "," + cmd
                if "vf" in output_params
                else cmd
            )

        return input_params, output_params

    def __iter__(self):
        assert self.is_async
        input_params, output_params = self.get_params()

        # Make the actual call
        self.ffmpeg_call = (
            ffmpeg.input(self.video_path, **input_params)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                loglevel="quiet",
                **output_params,
            )
            .run_async(pipe_stdout=True)
        )
        self.index = self.start_index if self.start_index is not None else 0
        return self

    def __next__(self):
        assert self.is_async
        in_bytes = self.ffmpeg_call.stdout.read(self.height * self.width * 3)
        if not in_bytes:
            raise StopIteration
        if self.index >= len(self.timestamps):
            raise StopIteration
        current_timestamp = self.timestamps[self.index]
        self.index += 1
        return (
            current_timestamp,
            np.frombuffer(in_bytes, dtype=np.uint8).reshape(
                self.height, self.width, 3
            ),
        )

    def __exit__(self):
        if self.is_async:
            self.ffmpeg_call.stdout.close()
            self.ffmpeg_call.wait()


def parse_args():
    parser = argparse.ArgumentParser(description="Pack face3d dataset.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="The path of eve dataset.",
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="dataset_type must be `train`, `val` or `test`.",
    )
    parser.add_argument(
        "--src-type",
        required=True,
        help="src_type must be `idx`, `anno` or `data`.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="The directory for result of packer.",
    )
    parser.add_argument(
        "--assumed-frame-rate",
        default=10,
        help="sampling frequency.",
    )
    parser.add_argument(
        "--max-sequence-len",
        default=30,
        help="maximum sequence length.",
    )
    parser.add_argument(
        "--camera-frame-type",
        default="eyes",
        help="input image type, must be 'full', 'face' or 'eyes'",
    )
    parser.add_argument(
        "--pack-type",
        default="lmdb",
        help="The target pack type for result of packer",
    )
    parser.add_argument(
        "--num-workers",
        default=16,
        help="The number of workers to load image.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    init_logger("work_dirs/hat_logss/eve_dataset_packer")
    pack_path = os.path.join(
        args.save_dir,
        "%s/%s_%s" % (args.dataset_type, args.src_type, args.pack_type),
    )

    packer = EVEDatasetPacker(
        args.dataset_path,
        args.dataset_type,
        args.src_type,
        pack_path,
        args.assumed_frame_rate,
        args.max_sequence_len,
        args.camera_frame_type,
        int(args.num_workers),
        args.pack_type,
    )
    packer()

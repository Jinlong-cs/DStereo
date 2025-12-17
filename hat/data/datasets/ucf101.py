# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import math
import random
from typing import List, Optional

try:
    import av
except Exception:
    av = None
import numpy as np
import torch
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = ["UCF101"]


def get_start_end_idx(
    video_size: int, clip_size: int, clip_idx: int, num_clips: int
):
    """Sample a clip of size clip_size from a video.

    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip.
    If clip_idx is-1, the clip is randomly sampled, otherwise uniformly
    split the video to num_clips clips, and select the start and end index of
    clip_idx-th video clip.

    Args:
        video_size: number of overall frames.
        clip_size: size of the clip to sample from the frames.
        clip_idx: if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips: overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx: the start frame index.
        end_idx: the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


class Sampler(object):
    """Sample frames id.

    Args:
        num_seg: number of segments.
        seg_len: number of sampled frames in each segment.
        valid_mode: True or False.
        select_left: Whether to select the frame to the left in the middle
            when the sampling interval is even in the test mode.
    """

    def __init__(
        self,
        num_seg: int,
        seg_len: int,
        valid_mode: bool = False,
        select_left: bool = False,
    ):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.valid_mode = valid_mode
        self.select_left = select_left

    def _get(self, frames_idx, results):
        data_format = results["format"]
        if data_format == "video":
            assert results["backend"] == "pyav", "only support pyav backend"
            imgs = []
            frames = np.array(results["frames"])
            for idx in frames_idx:
                imgbuf = frames[idx]
                imgs.append(imgbuf)
            imgs = np.stack(imgs)  # thwc

        results["imgs"] = imgs
        return results

    def __call__(self, results):
        frames_len = int(results["frames_len"])
        average_dur = int(frames_len / self.num_seg)
        frames_idx = []
        if not self.select_left:
            for i in range(self.num_seg):
                idx = 0
                if not self.valid_mode:
                    if average_dur >= self.seg_len:
                        idx = random.randint(0, average_dur - self.seg_len)
                        idx += i * average_dur
                    elif average_dur >= 1:
                        idx += i * average_dur
                    else:
                        idx = i
                else:
                    if average_dur >= self.seg_len:
                        idx = (average_dur - 1) // 2
                        idx += i * average_dur
                    elif average_dur >= 1:
                        idx += i * average_dur
                    else:
                        idx = i
                for jj in range(idx, idx + self.seg_len):
                    if results["format"] == "video":
                        frames_idx.append(int(jj % frames_len))
                    elif results["format"] == "frame":
                        frames_idx.append(jj + 1)
                    else:
                        raise NotImplementedError

            return self._get(frames_idx, results)


class VideoDecoder(object):
    """Decode mp4 file to frames.

    Args:
        backend: decoding backend.
        mode: Options includes `train`, `val`, or `test` mode.
        sampling_rate: frame sampling rate (interval between two sampled
            frames).
        num_seg: number of segments.
        num_clips: overall number of clips to uniformly sample from the
            given video.
        target_fps: the input video may has different fps, convert it to
            the target video fps.
    """

    @require_packages("av")
    def __init__(
        self,
        backend: str = "pyav",
        mode: str = "train",
        sampling_rate: int = 32,
        num_seg: int = 8,
        num_clips: int = 1,
        target_fps: int = 30,
    ):
        self.backend = backend
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.num_seg = num_seg
        self.num_clips = num_clips
        self.target_fps = target_fps

    def __call__(self, results):
        file_path = results["filename"]
        results["format"] = "video"
        results["backend"] = self.backend

        assert self.backend == "pyav", "only support pyav backend"
        if self.mode in ["train", "valid"]:
            clip_idx = -1
        elif self.mode in ["test"]:
            clip_idx = 0
        else:
            raise NotImplementedError

        container = av.open(file_path)

        num_clips = 1  # always be 1

        # decode process
        fps = float(container.streams.video[0].average_rate)

        frames_length = container.streams.video[0].frames
        duration = container.streams.video[0].duration

        if duration is None:
            # If failed to fetch the decoding information,
            # decode the entire video.
            decode_all_video = True
            video_start_pts, video_end_pts = 0, math.inf
        else:
            decode_all_video = False
            start_idx, end_idx = get_start_end_idx(
                frames_length,
                self.sampling_rate * self.num_seg / self.target_fps * fps,
                clip_idx,
                num_clips,
            )
            timebase = duration / frames_length
            video_start_pts = int(start_idx * timebase)
            video_end_pts = int(end_idx * timebase)

        frames = None
        # If video stream was found, fetch video frames from the video.
        if container.streams.video:
            margin = 1024
            seek_offset = max(video_start_pts - margin, 0)

            container.seek(
                seek_offset,
                any_frame=False,
                backward=True,
                stream=container.streams.video[0],
            )
            tmp_frames = {}
            buffer_count = 0
            max_pts = 0
            for frame in container.decode(**{"video": 0}):
                max_pts = max(max_pts, frame.pts)
                if frame.pts < video_start_pts:
                    continue
                if frame.pts <= video_end_pts:
                    tmp_frames[frame.pts] = frame
                else:
                    buffer_count += 1
                    tmp_frames[frame.pts] = frame
                    if buffer_count >= 0:
                        break
            video_frames = [tmp_frames[pts] for pts in sorted(tmp_frames)]

            container.close()
            frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
            clip_size = (
                self.sampling_rate * self.num_seg / self.target_fps * fps
            )

            start_idx, end_idx = get_start_end_idx(
                len(frames),
                clip_size,
                clip_idx if decode_all_video else 0,
                1,
            )
            results["frames"] = frames
            results["frames_len"] = len(frames)
            results["start_idx"] = start_idx
            results["end_idx"] = end_idx

        return results


@OBJECT_REGISTRY.register
class UCF101(data.Dataset):  # noqa: D205,D400
    """UCF101 video loader.

    Construct the UCF101 video loader, then sample clips from the videos.

    Args:
        root: The path of packed file.
        file_path: the file path of mp4 file
        mode: Options includes `train`, `val`, or `test` mode.
        num_retries: number of retries.
        num_frames: number of framses to sample.
        transforms: Transforms of video before using.
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        mode: str = "train",
        num_retries: int = 10,
        num_frames: int = 8,
        transforms: Optional[List] = None,
    ):
        self.root = root
        self.file_path = file_path
        self.mode = mode
        self.num_retries = num_retries
        self.num_frames = num_frames
        self.transforms = transforms

        self.decoder = VideoDecoder(mode=self.mode)
        if self.mode == "test":
            self.sampler = Sampler(
                num_seg=self.num_frames, seg_len=1, valid_mode=True
            )
        else:
            self.sampler = Sampler(
                num_seg=self.num_frames, seg_len=1, valid_mode=False
            )
        self.samples = self._load_file()

    def _load_file(self):
        info = []
        with open(self.file_path, "r") as fin:
            for line in fin:
                line_split = line.strip().split()
                filename, labels = line_split
                filename = self.root + "/" + filename
                info.append({"filename": filename, "labels": int(labels)})
        return info

    def __getitem__(self, idx):
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.samples[idx])
                results = self.decoder(results)
                results = self.sampler(results)
            except Exception:
                if ir < self.num_retries - 1:
                    print(
                        "Error when loading {}, have {} trys, \
                        will try again".format(
                            results["filename"], ir
                        )
                    )

                idx = random.randint(0, len(self.samples) - 1)
                continue

            sample, target = (
                results["imgs"],
                torch.as_tensor(np.array([results["labels"]])).squeeze() - 1,
            )
            data = {
                "img": sample,
                "labels": target,
            }
            if self.transforms is not None:
                data = self.transforms(data)

            return data

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return "UCF101"

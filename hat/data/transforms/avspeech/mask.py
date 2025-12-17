# Copyright (c) Horizon Robotics, All rights reserved.
import random
from typing import Callable

import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(
        0, max_len, dtype=torch.int64, device=lengths.device
    )
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_mask(
    size: int,
    device: torch.device = torch.device("cpu"),  # noqa: B008
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),  # noqa: B008
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size.

       this is for streaming encoder.

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


class PadMask:
    """Make mask tensor containing indices of padded part.

    Args:
        length_key: key of length in data
        feat_key: key of feature in data
        subsample_func: function to subsample the mask

    """

    def __init__(
        self,
        length_key: str,
        feat_key: str,
        subsample_func: Callable = None,
    ):
        self.length_key = length_key
        self.feat_key = feat_key
        self.subsample_func = subsample_func

    def __call__(self, data: dict) -> dict:
        lens = data[self.length_key]
        feat = data[self.feat_key]
        max_len = feat.shape[1]

        assert max_len >= lens.max()

        pad_mask = ~make_pad_mask(lens, max_len)  # B T
        pad_mask = pad_mask.unsqueeze(1)  # B 1 T
        if self.subsample_func is not None:
            pad_mask = self.subsample_func(pad_mask)
        pad_mask = pad_mask.unsqueeze(1)  # B 1 1 T
        data["pad_mask"] = pad_mask

        return data


class DynamicChunkMask:
    """Dynamic chunk mask for streaming encoder.

    Args:
        dynamic_left_chunk (bool): use dynamic left chunk
        max_chunksize (int): max chunk size
    """

    def __init__(
        self, dynamic_left_chunk: bool = True, max_chunksize: int = 25
    ):
        self.dynamic_left_chunk = dynamic_left_chunk
        self.max_chunksize = max_chunksize

    def __call__(self, data: dict) -> dict:
        pad_mask = data["pad_mask"]
        max_len = pad_mask.size(-1)
        chunk_size = random.randint(1, max_len)
        num_left_chunks = -1
        if chunk_size > max_len // 2:
            chunk_size = max_len
        else:
            chunk_size = chunk_size % self.max_chunksize + 1
            if self.dynamic_left_chunk:
                max_left_chunks = (max_len - 1) // chunk_size
                num_left_chunks = random.randint(0, max_left_chunks)

        chunk_masks = subsequent_chunk_mask(
            max_len, chunk_size, num_left_chunks, pad_mask.device
        )
        chunk_masks = chunk_masks.unsqueeze(0).unsqueeze(0)  # 1 1 T T
        chunk_masks = chunk_masks & pad_mask
        data["chunk_mask"] = chunk_masks  # B 1 T T

        return data


class StaticChunkMask:
    """Static chunk mask for streaming encoder.

    Args:
        chunk_size: chunk size
        num_left_chunks: number of left chunks
    """

    def __init__(self, chunk_size: int = 8, num_left_chunks: int = 4):
        self.chunk_size = chunk_size
        self.num_left_chunks = num_left_chunks

    def __call__(self, data) -> dict:
        pad_mask = data["pad_mask"]
        max_len = pad_mask.size(-1)
        chunk_masks = subsequent_chunk_mask(
            max_len,
            self.chunk_size,
            self.num_left_chunks,
            pad_mask.device,
        )
        chunk_masks = chunk_masks.unsqueeze(0).unsqueeze(0)  # 1 1 T T
        chunk_masks = chunk_masks & pad_mask
        data["chunk_mask"] = chunk_masks  # B 1 T T
        return data


class ModalityDropout:
    """Drop audio or video modality randomly.

    Args:
        p (float): probability of dropout
        a2v_func (Callable): function to convert audio length to video length
        loss_weight (dict): loss weight
    """

    def __init__(
        self,
        p: float = 0.5,
        a2v_func: Callable = lambda x: (x - 7) // 4,
        loss_weight: dict = None,
        image_feat_size: int = 256,
    ):
        self.p = p  # audio dropout
        self.a2v_func = a2v_func
        if loss_weight is None:
            loss_weight = {"ao": 1.0, "vo": 1.0, "av": 1.0}
        self.loss_weight = loss_weight
        self.image_feat_size = image_feat_size

    def __call__(self, data):
        if "images" not in data:
            afea = data["audio"]
            audio_length = data["audio_lens"]
            B, T_audio, D = afea.shape
            T_images = self.a2v_func(T_audio)
            data["images"] = torch.zeros(
                B, T_images, self.image_feat_size, device=afea.device
            )
            data["images_lens"] = self.a2v_func(audio_length)
            data["loss_weight"] = self.loss_weight["ao"]
            valid = ["audio"]
        elif random.random() < self.p:
            data["audio"] = torch.zeros_like(data["audio"])
            data["loss_weight"] = self.loss_weight["vo"]
            valid = ["images"]
        else:
            data["loss_weight"] = self.loss_weight["av"]
            valid = ["audio", "images"]

        data["valid"] = valid

        return data

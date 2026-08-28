"""Resize-aware geometry for the DStereo s1.0/s0.8 experiment."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

DEFAULT_RESIZE_SCALES = (1.0, 0.8)
DEFAULT_BASE_HEIGHT = 352
DEFAULT_BASE_WIDTH = 640
DEFAULT_SIZE_DIVISOR = 32


@dataclass(frozen=True)
class ResizeAwareSpec:
    """Concrete content and padded tensor geometry for one scale."""

    scale: float
    base_height: int
    base_width: int
    size_divisor: int
    content_height: int
    content_width: int
    tensor_height: int
    tensor_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @property
    def horizontal_scale(self) -> float:
        return self.content_width / float(self.base_width)

    @property
    def vertical_scale(self) -> float:
        return self.content_height / float(self.base_height)

    @property
    def content_shape(self) -> tuple[int, int]:
        return self.content_height, self.content_width

    @property
    def tensor_shape(self) -> tuple[int, int]:
        return self.tensor_height, self.tensor_width

    @property
    def padding(self) -> tuple[int, int, int, int]:
        return self.pad_top, self.pad_bottom, self.pad_left, self.pad_right


def _ceil_to_divisor(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def _split_padding(total: int) -> tuple[int, int]:
    before = total // 2
    return before, total - before


def build_resize_aware_specs(
    scales: Iterable[float] = DEFAULT_RESIZE_SCALES,
    *,
    base_height: int = DEFAULT_BASE_HEIGHT,
    base_width: int = DEFAULT_BASE_WIDTH,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> tuple[ResizeAwareSpec, ...]:
    """Build concrete geometries for all configured resize scales."""

    specs = []
    for raw_scale in scales:
        scale = float(raw_scale)
        content_height = max(1, int(round(base_height * scale)))
        content_width = max(1, int(round(base_width * scale)))
        tensor_height = _ceil_to_divisor(content_height, size_divisor)
        tensor_width = _ceil_to_divisor(content_width, size_divisor)
        pad_top, pad_bottom = _split_padding(tensor_height - content_height)
        pad_left, pad_right = _split_padding(tensor_width - content_width)
        specs.append(
            ResizeAwareSpec(
                scale=scale,
                base_height=base_height,
                base_width=base_width,
                size_divisor=size_divisor,
                content_height=content_height,
                content_width=content_width,
                tensor_height=tensor_height,
                tensor_width=tensor_width,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
            )
        )

    return tuple(specs)


class ResizeAwareStereo:
    """Resize a canonical stereo sample without changing its field of view."""

    def __init__(
        self,
        scales: Sequence[float] = DEFAULT_RESIZE_SCALES,
        *,
        base_height: int = DEFAULT_BASE_HEIGHT,
        base_width: int = DEFAULT_BASE_WIDTH,
        size_divisor: int = DEFAULT_SIZE_DIVISOR,
        max_disp: float = 96.0,
    ) -> None:
        self.base_height = int(base_height)
        self.base_width = int(base_width)
        self.size_divisor = int(size_divisor)
        self.max_disp = float(max_disp)
        self.specs = build_resize_aware_specs(
            scales,
            base_height=self.base_height,
            base_width=self.base_width,
            size_divisor=self.size_divisor,
        )
        self._spec_by_key = {round(spec.scale, 8): spec for spec in self.specs}

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(spec.scale for spec in self.specs)

    def spec_for(self, scale: float) -> ResizeAwareSpec:
        return self._spec_by_key[round(float(scale), 8)]

    def __call__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
        spec = self.spec_for(scale)
        disparity = np.asarray(disparity, dtype=np.float32)
        expected_shape = (self.base_height, self.base_width)

        valid = (
            np.isfinite(disparity)
            & (disparity > 0.0)
            & (disparity < self.max_disp)
        )
        safe_disparity = np.where(valid, disparity, 0.0).astype(
            np.float32, copy=False
        )

        if spec.content_shape == expected_shape:
            left_content = left
            right_content = right
            disparity_content = safe_disparity
            valid_content = valid
        else:
            content_size = (spec.content_width, spec.content_height)
            left_content = cv2.resize(
                left, content_size, interpolation=cv2.INTER_AREA
            )
            right_content = cv2.resize(
                right, content_size, interpolation=cv2.INTER_AREA
            )
            disparity_content = cv2.resize(
                safe_disparity,
                content_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32, copy=False)
            valid_content = cv2.resize(
                valid.astype(np.uint8),
                content_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool, copy=False)

        disparity_content = disparity_content * np.float32(
            spec.horizontal_scale
        )
        disparity_content[~valid_content] = 0.0

        if spec.padding == (0, 0, 0, 0):
            left_tensor = left_content
            right_tensor = right_content
            disparity_tensor = disparity_content
        else:
            border = spec.padding
            left_tensor = cv2.copyMakeBorder(
                left_content,
                *border,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            right_tensor = cv2.copyMakeBorder(
                right_content,
                *border,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            disparity_tensor = cv2.copyMakeBorder(
                disparity_content,
                *border,
                cv2.BORDER_CONSTANT,
                value=0.0,
            )

        metadata = {
            "resize_scale": spec.scale,
            "resize_content_shape": spec.content_shape,
            "resize_tensor_shape": spec.tensor_shape,
            "resize_padding": spec.padding,
            "resize_horizontal_scale": spec.horizontal_scale,
            "resize_vertical_scale": spec.vertical_scale,
        }
        return (
            np.ascontiguousarray(left_tensor),
            np.ascontiguousarray(right_tensor),
            np.ascontiguousarray(disparity_tensor),
            metadata,
        )


def resize_aware_config(
    scales: Sequence[float] = DEFAULT_RESIZE_SCALES,
    *,
    base_height: int = DEFAULT_BASE_HEIGHT,
    base_width: int = DEFAULT_BASE_WIDTH,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
    max_disp: float = 96.0,
) -> dict[str, object]:
    """Return serializable constructor args for the resize transform."""

    specs = build_resize_aware_specs(
        scales,
        base_height=base_height,
        base_width=base_width,
        size_divisor=size_divisor,
    )
    return {
        "scales": [spec.scale for spec in specs],
        "base_height": base_height,
        "base_width": base_width,
        "size_divisor": size_divisor,
        "max_disp": max_disp,
    }


__all__ = [
    "DEFAULT_BASE_HEIGHT",
    "DEFAULT_BASE_WIDTH",
    "DEFAULT_RESIZE_SCALES",
    "DEFAULT_SIZE_DIVISOR",
    "ResizeAwareSpec",
    "ResizeAwareStereo",
    "build_resize_aware_specs",
    "resize_aware_config",
]

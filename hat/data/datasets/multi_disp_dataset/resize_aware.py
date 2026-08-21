"""Geometry helpers for opt-in resize-aware stereo training.

The model is trained in a canonical 640x352 view.  A candidate view keeps
that whole field of view, scales both cameras together, scales disparity only
in the horizontal direction, and then adds black padding until the tensor is
aligned to the model's size divisor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


DEFAULT_BASE_HEIGHT = 352
DEFAULT_BASE_WIDTH = 640
DEFAULT_SIZE_DIVISOR = 32


@dataclass(frozen=True)
class ResizeAwareSpec:
    """Concrete content/tensor geometry for one nominal resize scale."""

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
        """Return the actual horizontal pixel scale used for disparity."""

        return self.content_width / float(self.base_width)

    @property
    def vertical_scale(self) -> float:
        """Return the actual vertical pixel scale after integer rounding."""

        return self.content_height / float(self.base_height)

    @property
    def content_shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` before alignment padding."""

        return self.content_height, self.content_width

    @property
    def tensor_shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` presented to the network."""

        return self.tensor_height, self.tensor_width

    @property
    def padding(self) -> tuple[int, int, int, int]:
        """Return padding as ``(top, bottom, left, right)``."""

        return self.pad_top, self.pad_bottom, self.pad_left, self.pad_right


def _ceil_to_divisor(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def _split_padding(total: int) -> tuple[int, int]:
    """Split padding symmetrically, putting an odd row/column at the end."""

    before = total // 2
    return before, total - before


def build_resize_aware_specs(
    scales: Iterable[float],
    *,
    base_height: int = DEFAULT_BASE_HEIGHT,
    base_width: int = DEFAULT_BASE_WIDTH,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> tuple[ResizeAwareSpec, ...]:
    """Build and validate concrete geometries for ``scales``.

    ``round`` is deliberately used for content dimensions so the effective
    disparity scale is derived from the actual integer width, rather than the
    nominal decimal scale.  This avoids a one-pixel geometry mismatch between
    image and label transforms.
    """

    if base_height <= 0 or base_width <= 0 or size_divisor <= 0:
        raise ValueError("base dimensions and size divisor must be positive")

    specs: list[ResizeAwareSpec] = []
    seen: set[float] = set()
    for raw_scale in scales:
        scale = float(raw_scale)
        key = round(scale, 8)
        if key in seen:
            raise ValueError(f"duplicate resize scale: {raw_scale}")
        seen.add(key)
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"resize scale must be in (0, 1], got {raw_scale}")

        content_width = max(1, int(round(base_width * scale)))
        content_height = max(1, int(round(base_height * scale)))
        tensor_width = _ceil_to_divisor(content_width, size_divisor)
        tensor_height = _ceil_to_divisor(content_height, size_divisor)
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

    if not specs:
        raise ValueError("at least one resize scale is required")
    return tuple(specs)


class ResizeAwareStereo:
    """Apply one validated scale to a canonical stereo sample.

    Inputs and outputs use HWC BGR images and a HW float disparity map.  The
    transform intentionally returns metadata alongside the arrays so logging
    and visualization code can distinguish content dimensions from padded
    tensor dimensions.
    """

    def __init__(
        self,
        scales: Sequence[float],
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
        if self.max_disp <= 0.0:
            raise ValueError("max_disp must be positive")
        self.specs = build_resize_aware_specs(
            scales,
            base_height=self.base_height,
            base_width=self.base_width,
            size_divisor=self.size_divisor,
        )
        self._spec_by_key = {round(spec.scale, 8): spec for spec in self.specs}

    @property
    def scales(self) -> tuple[float, ...]:
        """Return the configured nominal scales in schedule order."""

        return tuple(spec.scale for spec in self.specs)

    def spec_for(self, scale: float) -> ResizeAwareSpec:
        """Resolve a sampler scale and reject unconfigured values."""

        key = round(float(scale), 8)
        try:
            return self._spec_by_key[key]
        except KeyError as error:
            raise ValueError(
                f"scale {scale!r} is not in configured scales {self.scales}"
            ) from error

    @staticmethod
    def _validate_images(left: np.ndarray, right: np.ndarray) -> None:
        for name, image in (("left", left), ("right", right)):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{name} must be HWC with three channels")
            if image.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 BGR, got {image.dtype}")

    def __call__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
        spec = self.spec_for(scale)
        self._validate_images(left, right)
        disparity = np.asarray(disparity, dtype=np.float32)
        if left.shape[:2] != right.shape[:2] or left.shape[:2] != disparity.shape:
            raise ValueError(
                "stereo/disparity geometry mismatch: "
                f"left={left.shape}, right={right.shape}, disparity={disparity.shape}"
            )
        if left.shape[:2] != (self.base_height, self.base_width):
            raise ValueError(
                "resize-aware training expects the canonical view "
                f"{self.base_width}x{self.base_height}, got "
                f"{left.shape[1]}x{left.shape[0]}"
            )

        left_content = cv2.resize(
            left,
            (spec.content_width, spec.content_height),
            interpolation=cv2.INTER_AREA,
        )
        right_content = cv2.resize(
            right,
            (spec.content_width, spec.content_height),
            interpolation=cv2.INTER_AREA,
        )

        # Freeze validity in canonical coordinates before resizing.  Without
        # this step an invalid value just above max_disp could become valid
        # after downscaling and silently contribute to the loss.
        valid = np.isfinite(disparity) & (disparity > 0.0) & (
            disparity < self.max_disp
        )
        safe_disparity = np.where(valid, disparity, 0.0).astype(
            np.float32, copy=False
        )
        disparity_content = cv2.resize(
            safe_disparity,
            (spec.content_width, spec.content_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32, copy=False)
        valid_content = cv2.resize(
            valid.astype(np.uint8),
            (spec.content_width, spec.content_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool, copy=False)
        disparity_content *= np.float32(spec.horizontal_scale)
        disparity_content[~valid_content] = 0.0

        left_tensor = cv2.copyMakeBorder(
            left_content,
            spec.pad_top,
            spec.pad_bottom,
            spec.pad_left,
            spec.pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        right_tensor = cv2.copyMakeBorder(
            right_content,
            spec.pad_top,
            spec.pad_bottom,
            spec.pad_left,
            spec.pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        disparity_tensor = cv2.copyMakeBorder(
            disparity_content,
            spec.pad_top,
            spec.pad_bottom,
            spec.pad_left,
            spec.pad_right,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )
        if left_tensor.shape[:2] != spec.tensor_shape:
            raise AssertionError((left_tensor.shape, spec.tensor_shape))
        if right_tensor.shape[:2] != spec.tensor_shape:
            raise AssertionError((right_tensor.shape, spec.tensor_shape))
        if disparity_tensor.shape != spec.tensor_shape:
            raise AssertionError((disparity_tensor.shape, spec.tensor_shape))

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
    scales: Sequence[float],
    *,
    base_height: int = DEFAULT_BASE_HEIGHT,
    base_width: int = DEFAULT_BASE_WIDTH,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
    max_disp: float = 96.0,
) -> dict[str, object]:
    """Return a serializable configuration for :class:`ResizeAwareStereo`."""

    # Construct once here so invalid experiment configs fail at startup.
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
    "DEFAULT_SIZE_DIVISOR",
    "ResizeAwareSpec",
    "ResizeAwareStereo",
    "build_resize_aware_specs",
    "resize_aware_config",
]

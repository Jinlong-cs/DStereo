"""Fixed-scale resize-aware geometry for the DStereo s0.8 experiment."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

FIXED_RESIZE_SCALE = 0.8
DEFAULT_BASE_HEIGHT = 352
DEFAULT_BASE_WIDTH = 640
DEFAULT_SIZE_DIVISOR = 32


@dataclass(frozen=True)
class ResizeAwareSpec:
    """Concrete content and padded tensor geometry for fixed scale 0.8."""

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


def build_resize_aware_spec(
    *,
    scale: float = FIXED_RESIZE_SCALE,
    base_height: int = DEFAULT_BASE_HEIGHT,
    base_width: int = DEFAULT_BASE_WIDTH,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> ResizeAwareSpec:
    """Build the only supported experiment geometry: fixed scale 0.8."""

    scale = float(scale)
    if not math.isclose(scale, FIXED_RESIZE_SCALE):
        raise ValueError(
            "this profile only supports fixed scale "
            f"{FIXED_RESIZE_SCALE}, got {scale}"
        )
    if base_height <= 0 or base_width <= 0 or size_divisor <= 0:
        raise ValueError("base dimensions and size divisor must be positive")

    content_height = int(round(base_height * scale))
    content_width = int(round(base_width * scale))
    tensor_height = _ceil_to_divisor(content_height, size_divisor)
    tensor_width = _ceil_to_divisor(content_width, size_divisor)
    vertical_padding = tensor_height - content_height
    horizontal_padding = tensor_width - content_width
    pad_top = vertical_padding // 2
    pad_left = horizontal_padding // 2
    return ResizeAwareSpec(
        scale=scale,
        base_height=base_height,
        base_width=base_width,
        size_divisor=size_divisor,
        content_height=content_height,
        content_width=content_width,
        tensor_height=tensor_height,
        tensor_width=tensor_width,
        pad_top=pad_top,
        pad_bottom=vertical_padding - pad_top,
        pad_left=pad_left,
        pad_right=horizontal_padding - pad_left,
    )


class ResizeAwareStereo:
    """Resize a canonical stereo sample without changing its field of view."""

    def __init__(
        self,
        *,
        scale: float = FIXED_RESIZE_SCALE,
        base_height: int = DEFAULT_BASE_HEIGHT,
        base_width: int = DEFAULT_BASE_WIDTH,
        size_divisor: int = DEFAULT_SIZE_DIVISOR,
        max_disp: float = 96.0,
    ) -> None:
        self.spec = build_resize_aware_spec(
            scale=scale,
            base_height=base_height,
            base_width=base_width,
            size_divisor=size_divisor,
        )
        self.max_disp = float(max_disp)
        if self.max_disp <= 0.0:
            raise ValueError("max_disp must be positive")

    @staticmethod
    def _validate_image(name: str, image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} must be an HWC three-channel image")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 BGR, got {image.dtype}")

    def __call__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        disparity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
        self._validate_image("left", left)
        self._validate_image("right", right)
        disparity = np.asarray(disparity, dtype=np.float32)
        expected_shape = (self.spec.base_height, self.spec.base_width)
        if left.shape[:2] != expected_shape:
            raise ValueError(
                "resize-aware s0.8 expects canonical 640x352 input, got "
                f"{left.shape[1]}x{left.shape[0]}"
            )
        if (
            right.shape[:2] != expected_shape
            or disparity.shape != expected_shape
        ):
            raise ValueError(
                "stereo/disparity geometry mismatch: "
                f"left={left.shape}, right={right.shape}, "
                f"disparity={disparity.shape}"
            )

        content_size = (self.spec.content_width, self.spec.content_height)
        left_content = cv2.resize(
            left, content_size, interpolation=cv2.INTER_AREA
        )
        right_content = cv2.resize(
            right, content_size, interpolation=cv2.INTER_AREA
        )

        # Freeze validity before scaling so an originally invalid disparity
        # cannot become valid merely because its value is multiplied by 0.8.
        valid = (
            np.isfinite(disparity)
            & (disparity > 0.0)
            & (disparity < self.max_disp)
        )
        safe_disparity = np.where(valid, disparity, 0.0).astype(
            np.float32, copy=False
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
        disparity_content *= np.float32(self.spec.horizontal_scale)
        disparity_content[~valid_content] = 0.0

        border = (
            self.spec.pad_top,
            self.spec.pad_bottom,
            self.spec.pad_left,
            self.spec.pad_right,
        )
        left_tensor = cv2.copyMakeBorder(
            left_content, *border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        right_tensor = cv2.copyMakeBorder(
            right_content, *border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        disparity_tensor = cv2.copyMakeBorder(
            disparity_content, *border, cv2.BORDER_CONSTANT, value=0.0
        )

        if left_tensor.shape[:2] != self.spec.tensor_shape:
            raise AssertionError((left_tensor.shape, self.spec.tensor_shape))
        if right_tensor.shape[:2] != self.spec.tensor_shape:
            raise AssertionError((right_tensor.shape, self.spec.tensor_shape))
        if disparity_tensor.shape != self.spec.tensor_shape:
            raise AssertionError(
                (disparity_tensor.shape, self.spec.tensor_shape)
            )

        metadata = {
            "resize_scale": self.spec.scale,
            "resize_content_shape": self.spec.content_shape,
            "resize_tensor_shape": self.spec.tensor_shape,
            "resize_padding": self.spec.padding,
            "resize_horizontal_scale": self.spec.horizontal_scale,
            "resize_vertical_scale": self.spec.vertical_scale,
        }
        return (
            np.ascontiguousarray(left_tensor),
            np.ascontiguousarray(right_tensor),
            np.ascontiguousarray(disparity_tensor),
            metadata,
        )


def resize_aware_config(
    *,
    scale: float = FIXED_RESIZE_SCALE,
    base_height: int = DEFAULT_BASE_HEIGHT,
    base_width: int = DEFAULT_BASE_WIDTH,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
    max_disp: float = 96.0,
) -> dict[str, object]:
    """Return the serializable constructor args for the fixed transform."""

    build_resize_aware_spec(
        scale=scale,
        base_height=base_height,
        base_width=base_width,
        size_divisor=size_divisor,
    )
    return {
        "scale": scale,
        "base_height": base_height,
        "base_width": base_width,
        "size_divisor": size_divisor,
        "max_disp": max_disp,
    }


__all__ = [
    "DEFAULT_BASE_HEIGHT",
    "DEFAULT_BASE_WIDTH",
    "DEFAULT_SIZE_DIVISOR",
    "FIXED_RESIZE_SCALE",
    "ResizeAwareSpec",
    "ResizeAwareStereo",
    "build_resize_aware_spec",
    "resize_aware_config",
]

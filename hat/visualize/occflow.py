import math
from typing import Optional, Tuple, Union

import torch

from hat.data.datasets.occflow_dataset import AgentGrids
from hat.utils.tensor_func import divide_no_nan

__all__ = [
    "occupancy_rgb_image",
    "flow_rgb_image",
]


def occupancy_rgb_image(
    agent_grids: AgentGrids,
    roadgraph_image: torch.Tensor,
    gamma: float = 1.6,
) -> torch.Tensor:
    """Visualize predictions or ground-truth occupancy.

    Args:
        agent_grids: AgentGrids object containing optional
            vehicles/pedestrians/cyclists.
        roadgraph_image: Road graph image [batch_size, height, width, 1].
        gamma: Amplify predicted probabilities so that they are easier to see.

    Returns:
        [batch_size, height, width, 3] float32 RGB image.
    """
    zeros = torch.zeros_like(roadgraph_image)
    ones = torch.ones_like(zeros)

    agents = agent_grids
    veh = zeros if agents.vehicles is None else agents.vehicles
    ped = zeros if agents.pedestrians is None else agents.pedestrians
    cyc = zeros if agents.cyclists is None else agents.cyclists

    veh = torch.pow(veh, 1 / gamma)
    ped = torch.pow(ped, 1 / gamma)
    cyc = torch.pow(cyc, 1 / gamma)

    # Convert layers to RGB.
    rg_rgb = torch.cat([zeros, zeros, zeros], dim=-1)
    veh_rgb = torch.cat([veh, zeros, zeros], dim=-1)  # Red.
    ped_rgb = torch.cat([zeros, ped * 0.67, zeros], dim=-1)  # Green.
    cyc_rgb = torch.cat([cyc * 0.33, zeros, zeros * 0.33], dim=-1)  # Purple.
    bg_rgb = torch.cat([ones, ones, ones], dim=-1)  # White background.
    # Set alpha layers over all RGB channels.
    rg_a = torch.cat(
        [roadgraph_image, roadgraph_image, roadgraph_image], dim=-1
    )
    veh_a = torch.cat([veh, veh, veh], dim=-1)
    ped_a = torch.cat([ped, ped, ped], dim=-1)
    cyc_a = torch.cat([cyc, cyc, cyc], dim=-1)
    # Stack layers one by one.
    img, img_a = _alpha_blend(fg=rg_rgb, bg=bg_rgb, fg_a=rg_a)
    img, img_a = _alpha_blend(fg=veh_rgb, bg=img, fg_a=veh_a, bg_a=img_a)
    img, img_a = _alpha_blend(fg=ped_rgb, bg=img, fg_a=ped_a, bg_a=img_a)
    img, img_a = _alpha_blend(fg=cyc_rgb, bg=img, fg_a=cyc_a, bg_a=img_a)
    return img


def flow_rgb_image(
    flow: torch.Tensor,
    roadgraph_image: torch.Tensor,
    agent_trails: torch.Tensor,
) -> torch.Tensor:
    """Convert (dx, dy) flow to RGB image.

    Args:
        flow: [batch_size, height, width, 2] float32 tensor holding (dx, dy)
            values.
        roadgraph_image: Road graph image [batch_size, height, width, 1].
        agent_trails: [batch_size, height, width, 1] float32 tensor containing
            rendered trails for all agents over the past and current time
            frames.

    Returns:
        [batch_size, height, width, 3] float32 RGB image.
    """
    # Swap x, y for compatibilty with published visualizations.
    flow = torch.roll(flow, shifts=1, dims=-1)
    # saturate_magnitude=-1 normalizes highest intensity to largest magnitude.
    flow_image = _optical_flow_to_rgb(flow, saturate_magnitude=-1)
    # Add roadgraph.
    flow_image = _add_grayscale_layer(roadgraph_image, flow_image)  # Black.
    # Overlay agent trails.
    flow_image = _add_grayscale_layer(
        agent_trails * 0.2, flow_image
    )  # 0.2 alpha
    return flow_image


def _add_grayscale_layer(
    fg_a: torch.Tensor,
    scene_rgb: torch.Tensor,
) -> torch.Tensor:
    """Add a black/gray layer using fg_a as alpha over an RGB image."""
    # Create a black layer matching dimensions of fg_a.
    black = torch.zeros_like(fg_a)
    black = torch.cat([black, black, black], dim=-1)
    # Add the black layer with transparency over the scene_rgb image.
    overlay, _ = _alpha_blend(fg=black, bg=scene_rgb, fg_a=fg_a, bg_a=1.0)
    return overlay


def _alpha_blend(
    fg: torch.Tensor,
    bg: torch.Tensor,
    fg_a: Optional[torch.Tensor] = None,
    bg_a: Optional[Union[torch.Tensor, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Overlays foreground and background image with custom alpha values.

    Implements alpha compositing using Porter/Duff equations.
    https://en.wikipedia.org/wiki/Alpha_compositing
    Works with 1-channel or 3-channel images.
    If alpha values are not specified, they are set to the intensity of RGB
    values.

    Args:
        fg: Foreground: float32 tensor shaped
            [batch, grid_height, grid_width, d].
        bg: Background: float32 tensor shaped
            [batch, grid_height, grid_width, d].
        fg_a: Foreground alpha: float32 tensor broadcastable to fg.
        bg_a: Background alpha: float32 tensor broadcastable to bg.

    Returns:
        Output image: torch.float32 tensor shaped
            [batch, grid_height, grid_width, d].
        Output alpha: torch.float32 tensor shaped
            [batch, grid_height, grid_width, d].
    """
    if fg_a is None:
        fg_a = fg
    if bg_a is None:
        bg_a = bg
    eps = 1e-7
    out_a = fg_a + bg_a * (1 - fg_a)
    out_rgb = (fg * fg_a + bg * bg_a * (1 - fg_a)) / (out_a + eps)
    return out_rgb, out_a


def _optical_flow_to_hsv(
    flow: torch.Tensor,
    saturate_magnitude: float = -1.0,
    name: Optional[str] = None,
) -> torch.Tensor:
    """Visualize an optical flow field in HSV colorspace.

    This uses the standard color code with hue corresponding to direction of
    motion and saturation corresponding to magnitude.

    The attr `saturate_magnitude` sets the magnitude of motion (in pixels) at
    which the color code saturates. A negative value is replaced with the
    maximum magnitude in the optical flow field.

    Args:
        flow: A `Tensor` of type `float32`. A 3-D or 4-D tensor storing (a
            batch of) optical flow field(s) as flow([batch,] i, j) = (dx, dy).
            The shape of the tensor is [height, width, 2] or [batch, height,
            width, 2] for the 4-D case.
        saturate_magnitude: An optional `float`. Defaults to `-1`.
        name: A name for the operation (optional).

    Returns:
        An float32 HSV image (or image batch) of size [height, width, 3]
            (or [batch, height, width, 3]) compatible with color conversion
            ops. The hue at each pixel corresponds to direction of motion. The
            saturation at each pixel corresponds to the magnitude of motion
            relative to the `saturate_magnitude` value. Hue, saturation, and
            value are in [0, 1].
    """
    flow_shape = flow.shape
    if len(flow_shape) < 3:
        raise ValueError(
            "flow must be at least 3-dimensional, got" f" `{flow_shape}`"
        )
    if flow_shape[-1] != 2:
        raise ValueError(
            f"flow must have innermost dimension of 2, got" f" `{flow_shape}`"
        )
    height = flow_shape[-3]
    width = flow_shape[-2]
    flow_flat = flow.view(-1, height, width, 2)

    dx = flow_flat[..., 0]
    dy = flow_flat[..., 1]
    # [batch_size, height, width]
    magnitudes = torch.sqrt(torch.square(dx) + torch.square(dy))
    if saturate_magnitude < 0:
        # [batch_size, 1, 1]
        local_saturate_magnitude = torch.amax(
            magnitudes, dim=(1, 2), keepdim=True
        )
    else:
        local_saturate_magnitude = torch.from_numpy(saturate_magnitude)

    # Hue is angle scaled to [0.0, 1.0).
    hue = (torch.remainder(torch.atan2(dy, dx), (2 * math.pi))) / (2 * math.pi)
    # Saturation is relative magnitude.
    relative_magnitudes = divide_no_nan(magnitudes, local_saturate_magnitude)
    saturation = torch.clamp(
        relative_magnitudes, max=1.0  # Larger magnitudes saturate.
    )
    # Value is fixed.
    value = torch.ones_like(saturation)
    hsv_flat = torch.stack((hue, saturation, value), dim=-1)
    return hsv_flat.view(list(flow_shape)[:-1] + [3])


def _optical_flow_to_rgb(
    flow: torch.Tensor,
    saturate_magnitude: float = -1.0,
) -> torch.Tensor:
    """Visualize an optical flow field in RGB colorspace."""
    hsv = _optical_flow_to_hsv(flow, saturate_magnitude)
    return _hsv_to_rgb(hsv)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert HSV image tensor to RGB format."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h = h % 1
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    i = torch.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = torch.stack([r, g, b], dim=-1)
    return rgb

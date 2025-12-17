import torch
import torch.nn as nn

__all__ = ["ScreenCoordinate"]


class ScreenCoordinate(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_millimeters_per_pixel(
        self, lb, lt, rb, pixel_width, pixel_height
    ):
        distance_y = torch.linalg.norm(lb - lt, dim=-1, keepdim=True)  # 1,1
        distance_x = torch.linalg.norm(rb - lb, dim=-1, keepdim=True)  # 1,1
        mpp_x = distance_x / pixel_width  # 1,1
        mpp_y = distance_y / pixel_height  # 1,1
        return torch.cat([-mpp_x, mpp_y], -1)  # left is positive 1,2

    def calc_screen_transform(self, lb, lt, rb):
        # screen coordinate: left is x, down is y, foward is z
        # origin is left top vertex
        down = lb - lt  # 1,3
        down /= torch.linalg.norm(down, dim=-1, keepdim=True)  # 1,3
        right_tmp = rb - lb  # 1,3
        right_tmp /= torch.linalg.norm(right_tmp, dim=-1, keepdim=True)  # 1,3
        forward = torch.cross(-right_tmp, down)  # 1,3
        forward /= torch.linalg.norm(forward, dim=-1, keepdim=True)  # 1,3
        left = torch.cross(down, forward)  # 1,3
        left /= torch.linalg.norm(left, dim=-1, keepdim=True)  # 1,3
        R = torch.stack([left, down, forward], dim=-1)  # 1,3,3
        t = lt[..., None]  # 1,3,1
        return R, t

    def point2d_to_pixel(self, p2d, mpp):
        return torch.round(p2d / mpp)

    def plane_equation(self, R, lt):
        """Given rotation matrix, compute the equation of x-y plane.

        Algorithm:
            The normal vector of the plane is z-axis in rotation matrix.
            And tmat provide on point in the plane.
            It is easy to infer the equation.
        Args:
            R: torch.tensor, the shape is (n,3,3)
            lt: torch.tensor, the shape is (n,3)
        Return:
            n: torch.tensor, the shape is (n,3), containing (a, b, c, d),
            d: torch.tensor, the shape is (n,3), where the equation of plane is
                ax + by + cz = d
        """
        n = R[:, :, 2]  # 1,3
        origin = lt  # 1,3
        d = (n * origin).sum(-1, keepdim=True)  # 1,1
        return n, d

    def gaze_to_scs(self, gaze, origin, R, lt):
        plane_w, plane_b = self.plane_equation(R, lt)

        a1 = torch.stack(
            [gaze[:, 1:2], -gaze[:, 0:1], 0 * gaze[:, 2:3]], dim=2
        )  # 1,1,3
        a2 = torch.stack(
            [0 * gaze[:, 0:1], gaze[:, 2:3], -gaze[:, 1:2]], dim=2
        )  # 1,1,3
        matrix = torch.cat([a1, a2, plane_w.reshape(-1, 1, 3)], dim=1)  # 1,3,3

        b1 = (
            gaze[:, 1:2] * origin[:, 0:1] - gaze[:, 0:1] * origin[:, 1:2]
        )  # 1,1
        b2 = (
            gaze[:, 2:3] * origin[:, 1:2] - gaze[:, 1:2] * origin[:, 2:3]
        )  # 1,1
        bias = torch.stack([b1, b2, plane_b], dim=1)  # 1,3,1

        point = torch.linalg.solve(matrix, bias)  # 1,3,1
        result = torch.linalg.inv(R) @ (point - lt[..., None])  # 1,3,1
        return result[:, :2, 0]

    def forward(
        self,
        gaze: torch.tensor,
        origin: torch.tensor,
        lb: torch.tensor,
        lt: torch.tensor,
        rb: torch.tensor,
        pixel_width: torch.tensor,
        pixel_height: torch.tensor,
    ):
        """Transform gaze vector to gaze point pixel.

        Args:
            gaze: unit vector of visual axis
            origin: corneal center
            lb: left_bottom 3d location of screen
            lt: left_top 3d location of screen
            rb: right_bottom 3d location of screen
            pixel_width: pixels of gaze point img width
            pixel_height: pixels of gaze point img height
        """
        mpp = self.calc_millimeters_per_pixel(
            lb, lt, rb, pixel_width, pixel_height
        )  # 1,2
        R, t = self.calc_screen_transform(lb, lt, rb)
        pt2d = self.gaze_to_scs(gaze, origin, R, lt)  # 1,2
        pixel = self.point2d_to_pixel(pt2d, mpp)  # 1,2
        return pixel

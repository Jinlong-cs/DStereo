from itertools import chain

import numpy as np

__all__ = ["Ellipse"]


# Helper classes
class Ellipse:
    def __init__(self, param):
        self.eps1 = 1e-3
        self.eps2 = 1e-40
        if param is not list:
            self.param = param
            self.mat = self.param2mat(self.param)
            self.quad = self.mat2quad(self.mat)
            # self.Phi = self.recover_Phi()

    def param2mat(self, param):
        cx, cy, a, b, theta = tuple(param)
        h_rot = rotation_2d(-theta)
        h_trans = trans_2d(-cx, -cy)
        scale = np.array([[1 / a ** 2, 0, 0], [0, 1 / b ** 2, 0], [0, 0, -1]])
        mat = h_trans.T @ h_rot.T @ scale @ h_rot @ h_trans
        return mat

    def mat2quad(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.eps1, "Conic form incorrect"
        a, b, c, d, e, f = (
            mat[0, 0],
            2 * mat[0, 1],
            mat[1, 1],
            2 * mat[0, 2],
            2 * mat[1, 2],
            mat[-1, -1],
        )
        return np.array([a, b, c, d, e, f])

    def quad2param(self, quad):
        mat = self.quad2mat(quad)
        param = self.mat2param(mat)
        return param

    def quad2mat(self, quad):
        a, b, c, d, e, f = tuple(quad)
        mat = np.array(
            [[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]]
        )
        return mat

    def mat2param(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.eps1, "Conic form incorrect"
        # Estimate rotation
        theta = self.recover_theta(mat)
        # Estimate translation
        tx, ty = self.recover_center(mat)
        # Invert translation and rotation
        h_rot = rotation_2d(theta)
        h_trans = trans_2d(tx, ty)
        mat_norm = h_rot.T @ h_trans.T @ mat @ h_trans @ h_rot
        major_axis = np.sqrt(1 / mat_norm[0, 0])
        minor_axis = np.sqrt(1 / mat_norm[1, 1])
        area = np.pi * major_axis * minor_axis
        return np.array([tx, ty, major_axis, minor_axis, theta, area])

    def recover_theta(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        # print('a: {}. b: {}. c: {}'.format(a, b, c))
        if abs(b) <= self.eps2 and a <= c:
            theta = 0.0
        elif abs(b) <= self.eps2 and a > c:
            theta = np.pi / 2
        elif abs(b) > self.eps2 and a <= c:
            theta = 0.5 * np.arctan2(b, (a - c))
        elif abs(b) > self.eps2 and a > c:
            # theta = 0.5*(np.pi + np.arctan(b/(a-c)))
            theta = 0.5 * np.arctan2(b, (a - c))
        else:
            print("Unknown condition")
        return theta

    def recover_center(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        tx = (2 * c * d - b * e) / (b ** 2 - 4 * a * c)
        ty = (2 * a * e - b * d) / (b ** 2 - 4 * a * c)
        return (tx, ty)

    def transform(self, h_affine):
        """Given a transformation matrix H, modify the ellipse."""
        mat_trans = (
            np.linalg.inv(h_affine.T) @ self.mat @ np.linalg.inv(h_affine)
        )
        return self.mat2param(mat_trans), self.mat2quad(mat_trans), mat_trans

    def generate_points(self, mode, nums=-1):
        r"""Generate 8 points along the periphery of an ellipse.

        The mode dictates the uniformity between points.

        mode: str
        'equiAngle' - Points along the periphery with angles [0:45:360)
        'equiSlope' - Points along the periphery with tangential slopes [-1:0.5:1)  # noqa
        'random' - Generate points randomly across the ellipse

        """

        a = self.param[2]
        b = self.param[3]

        alpha = (a * np.sin(self.param[-1])) ** 2 + (
            b * np.cos(self.param[-1])
        ) ** 2
        beta = (a * np.cos(self.param[-1])) ** 2 + (
            b * np.sin(self.param[-1])
        ) ** 2
        gamma = (a ** 2 - b ** 2) * np.sin(2 * self.param[-1])

        if mode == "equiSlope":
            slope_list = [1e-6, 1, 1000, -1]
            k_fun = lambda m_i: (m_i * gamma + 2 * alpha) / (
                2 * beta * m_i + gamma
            )

            x_2 = [
                ((a * b) ** 2)
                / (alpha + beta * k_fun(m) ** 2 - gamma * k_fun(m))
                for m in slope_list
            ]

            x = [(+np.sqrt(val), -np.sqrt(val)) for val in x_2]
            y = []
            for i, m in enumerate(slope_list):
                y1 = -x[i][0] * k_fun(m)
                y2 = -x[i][1] * k_fun(m)
                y.append((y1, y2))
            y_r = np.array(list(chain(*y))) + self.param[1]
            x_r = np.array(list(chain(*x))) + self.param[0]
        elif mode == "equiAngle":
            seq = 0.5 * np.pi * np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            nums = len(seq)
            x = self.param[2] * np.cos(seq)
            y = self.param[3] * np.sin(seq)
            h_rot = rotation_2d(self.param[-1])
            relative_coord = h_rot.dot(
                np.stack(
                    [
                        x,
                        y,
                        np.ones(
                            nums,
                        ),
                    ],
                    axis=0,
                )
            )

            x_r = relative_coord[0, :] + self.param[0]
            y_r = relative_coord[1, :] + self.param[1]
        elif mode == "random":
            assert nums != -1, "If mode is `random`, the `nums` is required"
            seq = (
                2
                * np.pi
                * (
                    np.random.rand(
                        nums,
                    )
                    - 0.5
                )
            )
            x = self.param[2] * np.cos(seq)
            y = self.param[3] * np.sin(seq)
            h_rot = rotation_2d(self.param[-1])
            relative_coords = h_rot.dot(
                np.stack(
                    [
                        x,
                        y,
                        np.ones(
                            nums,
                        ),
                    ],
                    axis=0,
                )
            )
            x_r = relative_coords[0, :] + self.param[0]
            y_r = relative_coords[1, :] + self.param[1]
        else:
            raise Exception(
                "Mode undefined,  which can only be \
                    `equiAngle`,`equiSlope`,`random`"
            )
        return x_r, y_r


# Helper functions
def rotation_2d(theta):
    # Return a 2D rotation matrix in the anticlockwise direction
    c, s = np.cos(theta), np.sin(theta)
    h_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])
    return h_rot


def trans_2d(cx, cy):
    h_trans = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1]])
    return h_trans

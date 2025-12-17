import cv2
import numpy as np


class Camera(object):
    def __init__(self):
        self.ipm_size_ = [0, 0]
        self.roi_ = [0, 0, 0, 0]

    # below are privately used functions
    def build_K(self, fu, fv, cu, cv):
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]]).astype(np.float32)

        return K

    def build_R(self, yaw, pitch, roll):
        # yaw, pitch, roll are new camera -> ccs
        # ccs -> new camera
        Rz = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0.0],
                [np.sin(-yaw), np.cos(-yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Ry = np.array(
            [
                [np.cos(-pitch), 0.0, np.sin(-pitch)],
                [0.0, 1.0, 0.0],
                [-np.sin(-pitch), 0.0, np.cos(-pitch)],
            ]
        )
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(-roll), -np.sin(-roll)],
                [0.0, np.sin(-roll), np.cos(-roll)],
            ]
        )
        # ccs -> old camera rotation
        R = Rz.dot(Ry.dot(Rx))

        return R

    def build_t(self, R, tx, ty, tz):
        X = np.array([tx, ty, tz])
        t = -R.dot(X)

        return t

    def build_Rt(self, yaw, pitch, roll, tx, ty, tz):
        # first translate, then rotate
        # R * (x + t)
        R = self.build_R(yaw, pitch, roll)
        t = self.build_t(R, tx, ty, tz)
        Rt = np.zeros([4, 4])
        Rt[0:3, 0:3] = R
        Rt[0:3, 3] = t
        Rt[3, 3] = 1

        return Rt

    def build_T_ipm2vcsgnd(self):
        gnd_pt1 = [self.roi_[1], self.roi_[2]]
        gnd_pt2 = [self.roi_[1], self.roi_[0]]
        gnd_pt3 = [self.roi_[3], self.roi_[2]]
        gnd_pt4 = [self.roi_[3], self.roi_[0]]

        ipm_pt1 = [0, 0]
        ipm_pt2 = [self.ipm_size_[1] - 1, 0]
        ipm_pt3 = [0, self.ipm_size_[0] - 1]
        ipm_pt4 = [self.ipm_size_[1] - 1, self.ipm_size_[0] - 1]

        gnd_region = [gnd_pt1, gnd_pt2, gnd_pt3, gnd_pt4]
        ipm_region = [ipm_pt1, ipm_pt2, ipm_pt3, ipm_pt4]

        trans = cv2.getPerspectiveTransform(
            np.array(ipm_region).astype(np.float32),
            np.array(gnd_region).astype(np.float32),
        )
        return trans

    def build_transform_matrix(
        self,
        fu,
        fv,
        cu,
        cv,
        pitch,
        yaw,
        roll,
        tx,
        ty,
        tz,
        vcs_rotation=None,
        vcs_translation=None,
    ):
        self.camera_rotation = [pitch, yaw, roll]
        self.camera_translation = [tx, ty, tz]
        # old camera -> image
        self.K_ = self.build_K(fu, fv, cu, cv)
        if np.sum(self.roi_) == 0 or np.sum(self.ipm_size_) == 0:
            return

        # ccs to vsc info
        vcs_roll, vcs_pitch, vcs_yaw = vcs_rotation
        vcs_tx, vcs_ty, vcs_tz = vcs_translation

        Rt_ccs2newcam = self.build_Rt(yaw, pitch, roll, tx, ty, tz)
        Rt_vcs2ccs = self.build_Rt(
            vcs_yaw, vcs_pitch, vcs_roll, vcs_tx, vcs_ty, vcs_tz
        )
        Rt_vcs2newcam = Rt_ccs2newcam.dot(Rt_vcs2ccs)
        R_newcam2oldcam = np.linalg.inv(
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        )
        R_vcs2oldcam = R_newcam2oldcam.dot(Rt_vcs2newcam[0:3, :])

        P_vcs2img = self.K_.dot(R_vcs2oldcam)
        Q_vcsgnd2img = P_vcs2img[:, [0, 1, 3]]
        T_ipm2vcsgnd = self.build_T_ipm2vcsgnd()

        self.M_ = Q_vcsgnd2img.dot(T_ipm2vcsgnd)
        self.Minv_ = np.linalg.inv(self.M_)

    def remap_image(self, img, x, y, target_img, u, v, use_opencv=False):
        if use_opencv:
            map1 = np.reshape(x, (target_img.shape[0], target_img.shape[1]))
            map2 = np.reshape(y, (target_img.shape[0], target_img.shape[1]))
            target_img = cv2.remap(
                img,
                map1.astype(np.float32),
                map2.astype(np.float32),
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            valid_index = (
                (x >= 0) * (x < img.shape[1]) * (y >= 0) * (y < img.shape[0])
            )
            x = x[valid_index]
            y = y[valid_index]
            u = u[valid_index]
            v = v[valid_index]
            target_img[v, u, :] = img[y, x, :]

        return target_img

    def warp_image(self, u, v, warp_type):
        UV1 = np.vstack((u, v, np.ones((1, u.shape[0]))))
        if warp_type == "img2ipm":
            XYW = np.dot(self.Minv_, UV1)
        elif warp_type == "ipm2img":
            XYW = np.dot(self.M_, UV1)
        else:
            raise NotImplementedError
        XYW /= XYW[-1, :]
        x = XYW[0, :]
        y = XYW[1, :]

        return x, y

    # below are publicly used functions
    def setIPMInformation(self, roi, ipm_size):
        self.ipm_size_ = ipm_size
        self.roi_ = roi

    def setUndistortInformation(self, is_fisheye, undistorted_scale=1):
        self.is_fisheye = is_fisheye
        if self.is_fisheye:
            self.undistorted_scale = undistorted_scale
            Kscaled = self.K_.copy() * undistorted_scale
            Kscaled[(0, 1), (0, 1)] = (
                Kscaled[(0, 1), (0, 1)] / undistorted_scale
            )
            self.Kscaled = Kscaled
            self.Kscaled_inv = np.linalg.inv(Kscaled)
        else:
            self.Kscaled, self.valid_roi = cv2.getOptimalNewCameraMatrix(
                self.K_,
                self.distort,
                (self.img_size_[1], self.img_size_[0]),
                1,
                (self.img_size_[1], self.img_size_[0]),
            )

    def buildTransformMatrixFromCameraParamDict(self, camera):
        self.distort = np.array(camera["distort"]["param"]).reshape(-1, 1)
        self.build_transform_matrix(
            camera["focal_u"],
            camera["focal_v"],
            camera["center_u"],
            camera["center_v"],
            camera["pitch"],
            camera["yaw"],
            camera["roll"],
            camera["camera_x"],
            camera["camera_y"],
            camera["camera_z"],
            camera["vcs"]["rotation"],
            camera["vcs"]["translation"],
        )

    def getIPMImage(self, img, use_opencv=False):
        # use opencv build-in function for debug
        if use_opencv:
            img_ipm = cv2.warpPerspective(
                img,
                self.Minv_,
                (self.ipm_size_[1], self.ipm_size_[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
        else:
            height = self.ipm_size_[0]
            width = self.ipm_size_[1]
            y, x = np.where(np.ones((height, width)) > -1)
            u, v = self.warp_image(x, y, warp_type="ipm2img")
            img_ipm = self.remap_image(
                img,
                u.astype(int),
                v.astype(int),
                np.zeros((height, width, 3)),
                x,
                y,
                use_opencv=True,
            )

        return img_ipm

    def recoverFromIPMImage(self, img_ipm, use_opencv=False):
        # debug with opencv warpPerspective function
        if use_opencv:
            img = cv2.warpPerspective(
                img_ipm,
                self.M_,
                (self.img_size_[0], self.img_size_[1]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
        else:
            height = self.img_size_[0]
            width = self.img_size_[1]
            v, u = np.where(np.ones((height, width)) > -1)
            x, y = self.warp_image(u, v, warp_type="img2ipm")
            img = self.remap_image(
                img_ipm,
                x.astype(int),
                y.astype(int),
                np.zeros((height, width, 3)),
                u,
                v,
                use_opencv=True,
            )

        return img

    def getUndisortedImage(self, img):
        height = img.shape[0]
        width = img.shape[1]
        y, x = np.where(np.ones((height, width)) > -1)
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        pts = np.concatenate([x, y], axis=1)
        pts_distorted = self.mapPointsFromUndistortedImageToOriginalImage(pts)
        u_d = np.round(pts_distorted[:, 0]).astype(int)
        v_d = np.round(pts_distorted[:, 1]).astype(int)
        img_undistorted = self.remap_image(
            img,
            u_d,
            v_d,
            np.zeros((height, width, 3)),
            x.reshape((-1)),
            y.reshape((-1)),
            use_opencv=True,
        )

        return img_undistorted

    def getRedisortedImage(self, img):
        height = img.shape[0]
        width = img.shape[1]
        y, x = np.where(np.ones((height, width)) > -1)
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        pts = np.concatenate([x, y], axis=1)
        pts_undistorted = self.mapPointsFromOriginalImageToUndistortedImage(
            pts
        )
        u = np.round(pts_undistorted[:, 0]).astype(int)
        v = np.round(pts_undistorted[:, 1]).astype(int)
        img_redistorted = self.remap_image(
            img,
            u,
            v,
            np.zeros((height, width, 3)),
            x.reshape((-1)),
            y.reshape((-1)),
            use_opencv=True,
        )

        return img_redistorted

    def mapPointsFromUndistortedImageToOriginalImage(self, pts):
        if self.is_fisheye:
            XY1 = np.hstack((pts, np.ones((pts.shape[0], 1))))
            UV1 = np.dot(self.Kscaled_inv, XY1.T)
            UV1 /= UV1[2, :]
            pts = UV1[0:2, :].T
            pts = pts[np.newaxis, :, :]
            pts_distorted = cv2.fisheye.distortPoints(
                pts.astype(float), K=self.K_, D=self.distort
            )
            pts_distorted = pts_distorted[0]
        else:
            fx = self.K_[0, 0]
            fy = self.K_[1, 1]
            cx = self.K_[0, 2]
            cy = self.K_[1, 2]

            distort = np.zeros((8,))
            num_params = len(self.distort)
            distort[:num_params] = self.distort[:, 0]
            k1, k2, p1, p2, k3, k4, k5, k6 = distort

            x = (pts[:, 0] - cx) / fx
            y = (pts[:, 1] - cy) / fy
            r2 = x * x + y * y

            coef = (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) / (
                1 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2
            )
            x_dist = x * coef + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
            y_dist = y * coef + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

            x_dist = x_dist * fx + cx
            y_dist = y_dist * fy + cy

            x_dist = x_dist.reshape((-1, 1))
            y_dist = y_dist.reshape((-1, 1))
            pts_distorted = np.hstack((x_dist, y_dist))

        return pts_distorted

    def mapPointsFromOriginalImageToUndistortedImage(self, pts_original):
        pts_original = np.array([pts_original]).astype(float)
        if self.is_fisheye:
            pts_undistorted = cv2.fisheye.undistortPoints(
                pts_original, K=self.K_, D=self.distort, P=self.Kscaled
            )
            pts_undistorted = np.round(pts_undistorted[0]).astype(int)
        else:
            pts_undistorted = cv2.undistortPoints(
                pts_original.reshape((-1, 1, 2)),
                self.K_,
                self.distort,
                None,
                P=self.K_,
            )
            pts_undistorted = np.round(pts_undistorted[:, 0]).astype(int)

        return pts_undistorted

    def getIPMImageFromOriginalImage(self, img):
        height = self.ipm_size_[0]
        width = self.ipm_size_[1]
        y, x = np.where(np.ones((height, width)) > -1)
        u, v = self.warp_image(x, y, warp_type="ipm2img")
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))
        pts = np.concatenate([u, v], axis=1)
        pts_distorted = self.mapPointsFromUndistortedImageToOriginalImage(pts)
        u_d = np.round(pts_distorted[:, 0]).astype(int)
        v_d = np.round(pts_distorted[:, 1]).astype(int)
        img_ipm = self.remap_image(
            img, u_d, v_d, np.zeros((height, width, 3)), x, y, use_opencv=True
        )

        return img_ipm

    def recoverOriginalImageFromIPMImage(self, img_ipm):
        height = self.img_size_[0]
        width = self.img_size_[1]
        v_d, u_d = np.where(np.ones((height, width)) > -1)
        u_d = u_d.reshape((-1, 1))
        v_d = v_d.reshape((-1, 1))
        pts = np.concatenate([u_d, v_d], axis=1)
        pts_undistorted = self.mapPointsFromOriginalImageToUndistortedImage(
            pts
        )
        u = np.round(pts_undistorted[:, 0]).astype(int)
        v = np.round(pts_undistorted[:, 1]).astype(int)

        x, y = self.warp_image(u, v, warp_type="img2ipm")
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        img_restored = self.remap_image(
            img_ipm,
            x,
            y,
            np.zeros((height, width, 3)),
            u_d.reshape((-1)),
            v_d.reshape((-1)),
            use_opencv=True,
        )

        return img_restored

    def mapPointsFromIPMImageToOriginalImage(self, pts_ipm):
        pts_ipm = np.array(pts_ipm)
        x = pts_ipm[:, 0]
        y = pts_ipm[:, 1]
        u, v = self.warp_image(x, y, warp_type="ipm2img")
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))
        pts = np.concatenate([u, v], axis=1)
        pts_distorted = self.mapPointsFromUndistortedImageToOriginalImage(pts)
        u_d = np.round(pts_distorted[:, 0][np.newaxis, :]).astype(int)
        v_d = np.round(pts_distorted[:, 1][np.newaxis, :]).astype(int)
        pts_restored = np.concatenate([u_d, v_d], axis=0).reshape((2, -1)).T

        return pts_restored

    def mapPointsFromOriginalImageToIPMImage(self, pts_original, float_output):
        pts_original = np.array(pts_original)
        pts_undistorted = self.mapPointsFromOriginalImageToUndistortedImage(
            pts_original
        )
        u = np.round(pts_undistorted[:, 0]).astype(int)
        v = np.round(pts_undistorted[:, 1]).astype(int)
        x, y = self.warp_image(u, v, warp_type="img2ipm")
        x = x[np.newaxis, :]
        y = y[np.newaxis, :]
        if float_output is False:
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
        pts_ipm = np.concatenate([x, y], axis=0).reshape((2, -1)).T

        return pts_ipm

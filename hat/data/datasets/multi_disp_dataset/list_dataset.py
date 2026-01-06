#! /usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["ListDataset"]
import OpenEXR
import Imath
import logging
import os.path as osp
import struct
from glob import glob
import os
import cv2
import copy
import re
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from skimage import io, transform
from .readpfm import readPFM
from DStereo.common import depth2disp
import tqdm, random
import torch.nn.functional as F
import torch
from torchvision.transforms import ColorJitter, functional, Compose

logger = logging.getLogger(__name__)


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()["dataWindow"]
    CNum = len(File.header()["channels"].keys())
    if CNum > 1:
        Channels = ["R", "G", "B"]
        CNum = 3
    else:
        Channels = ["G"]
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [
        np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels
    ]
    hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
    if CNum == 1:
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
    else:
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
        hdr[:, :, 1] = np.reshape(Pixels[1], (Size[1], Size[0]))
        hdr[:, :, 2] = np.reshape(Pixels[2], (Size[1], Size[0]))
    return hdr


def img_loader(p, mode="bgr"):
    assert osp.isfile(p), p
    img = cv2.imread(p)[:, :, :3]
    return img


def disp_loader(p):
    assert p.endswith((".tiff", ".png", ".pfm")), p
    if p.endswith(".pfm"):
        disp = readPFM(p)[0].astype(np.float32)
        return disp
    else:
        disp = np.array(Image.open(p))
        if p.endswith(".png"):
            disp = disp / 100.0
        return disp


class AIOTBaseDataset(Dataset):
    def sample_file(self, sample_num=16):
        assert len(self.file_list) > 0
        if self.debug and len(self.file_list) > sample_num:
            self.file_list = random.sample(self.file_list, sample_num)

    def __len__(self):
        return len(self.file_list)

    def statsic_disp(self):
        hist_ = 0
        idx_list = [i for i in range(len(self))]
        if len(self) > 3333:
            idx_list = random.sample(idx_list, 3333)
        for i in tqdm.tqdm(idx_list):
            left, right, disp = self.__getitem__(0)
            hist, bin_edges = np.histogram(a=disp, bins=32, range=(1, 320))
            hist_ += hist
        return hist_, bin_edges

    @staticmethod
    def pad_image_to_multiple_of_32(image, value=(0, 0, 0)):
        height, width = image.shape[:2]
        new_height = (height + 31) // 32 * 32
        new_width = (width + 31) // 32 * 32
        top_pad = (new_height - height) // 2
        bottom_pad = new_height - height - top_pad
        left_pad = (new_width - width) // 2
        right_pad = new_width - width - left_pad
        padded_image = cv2.copyMakeBorder(
            image,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
            value=value,
        )
        return padded_image


class ListDataset(AIOTBaseDataset):
    def __init__(self, file_list, debug=False, img_open_mode="bgr"):
        super().__init__()
        self.name = "ListDataset"  
        logger.info("start loading %s" % (self.name,))
        self.debug = debug
        with open(file_list, "r") as f:
            self.file_list = f.readlines()
            self.file_list = [
                line.replace("/data/horizon_j5/data/", "public/").strip().split(" ")
                for line in self.file_list
                if line.strip()
            ]
        self.root_dir = ""
        self.sample_file(16)
        self.img_open_mode = img_open_mode
        logger.info("len(%s): %d" % (self.name, len(self)))

    def __getitem__(self, index):
        l, r, d = self.file_list[index]
        left, right, disp = (
            img_loader(l, self.img_open_mode),
            img_loader(r, self.img_open_mode),
            disp_loader(d),
        )
        return left, right, disp


class Instereo2KDataset(AIOTBaseDataset):
    def __init__(self, file_list, debug=False, img_open_mode="bgr"):
        super().__init__()
        self.name = "Instereo2KDataset"
        logger.info("start loading %s" % (self.name,))
        with open(file_list, "r") as f:
            self.file_list = f.readlines()
            self.file_list = [
                line.replace(
                    "/data/horizon_j5/data/stereo_data/", "public/Public_Datasets/"
                )
                for line in self.file_list
                if line.strip()
            ]
        self.debug = debug
        self.img_open_mode = img_open_mode
        self.root_dir = "public/Public_Datasets/Instereo2K"
        logger.info("len(%s): %d" % (self.name, len(self)))

    def __getitem__(self, index):
        l, r, d = self.file_list[index].strip().split(" ")
        left, right, disp = (
            img_loader(l, self.img_open_mode),
            img_loader(r, self.img_open_mode),
            disp_loader(d),
        )
        return left, right, disp


class DrivingStereoDataset(AIOTBaseDataset):
    def __init__(self, file_list, debug=False, img_open_mode="rgb"):
        super().__init__()
        self.debug = debug
        self.name = "DrivingStereoDataset"  
        logger.info("start loading %s" % (self.name,))
        self.root_dir = osp.dirname(file_list)
        self.file_list = []
        with open(file_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    l, r, d = line.strip().split(" ")
                    l = osp.join(self.root_dir, l)
                    r = osp.join(self.root_dir, r)
                    d = osp.join(self.root_dir, d)
                    self.file_list.append([l, r, d])
        self.sample_file(len(self))
        self.img_open_mode = img_open_mode
        logger.info("len(%s): %d" % (self.name, len(self)))

    def __getitem__(self, index):
        l, r, d = self.file_list[index]
        left, right, disp = (
            img_loader(l, self.img_open_mode),
            img_loader(r, self.img_open_mode),
            self.disp_loader(d),
        )
        return left, right, disp

    @staticmethod
    def disp_loader(p):
        assert osp.exists(p), p
        disp_img = np.array(Image.open(p), dtype=np.float32)
        full = True if "full" in p else False
        scale = 128.0 if full else 256.0
        disp_img = disp_img / scale
        return disp_img


class ETH3DDataset(DrivingStereoDataset):
    def __init__(self, file_list, debug=False, img_open_mode="rgb"):
        super().__init__(file_list, debug, img_open_mode)
        self.name = "ETH3DDataset"
        logger.info("len(%s): %d" % (self.name, len(self)))

    @staticmethod
    def disp_loader(p):
        assert osp.exists(p), p
        disp_img = readPFM(p)[0].astype(np.float32)
        # remove invalid values
        disp_img[disp_img == np.inf] = 0
        return disp_img


class MiddleburyDataset(ETH3DDataset):
    def __init__(self, file_list, debug=False, img_open_mode="rgb"):
        super().__init__(file_list, debug, img_open_mode)
        self.name = "MiddleburyDataset"
        logger.info("len(%s): %d" % (self.name, len(self)))


class KITTIDataset(DrivingStereoDataset):
    def __init__(self, file_list, debug=False, img_open_mode="rgb"):
        super().__init__(file_list, debug, img_open_mode)
        self.name = "KITTIDataset"
        logger.info("len(%s): %d" % (self.name, len(self)))

    @staticmethod
    def disp_loader(p):
        assert osp.exists(p), p
        return np.array(Image.open(p), dtype=np.float32) / 256.0


class AIOTDataset(AIOTBaseDataset):
    def __init__(self, file_list, debug=False, img_open_mode="rgb"):
        super().__init__()
        self.debug = debug
        self.name = "AIOTDataset"
        logger.info("start loading %s" % (self.name,))
        self.img_open_mode = img_open_mode
        assert os.path.exists(file_list)
        self.root_dir = file_list
        self.file_list = []
        for file in os.listdir(osp.join(file_list, "infra1")):
            self.file_list.append(
                [
                    os.path.join(file_list, file),
                    os.path.join(file_list, file).replace("infra1", "infra2"),
                    os.path.join(file_list, file).replace("infra1", "disp"),
                ]
            )
        logger.info("len(%s): %d" % (self.name, len(self)))

    def __getitem__(self, index):
        l, r, d = self.file_list[index]
        left, right, disp = (
            img_loader(l, self.img_open_mode),
            img_loader(r, self.img_open_mode),
            self.disp_loader(d),
        )
        return left, right, disp

    @staticmethod
    def disp_loader(p):
        assert p.endswith((".tiff", ".png", ".pfm")), p
        disp = cv2.imread(p, -1)
        return disp


class SintelDataset(AIOTBaseDataset):
    def __init__(self, file_list, debug=False, img_open_mode="rgb"):
        super().__init__()
        self.debug = debug
        self.img_open_mode = img_open_mode
        self.name = "SintelDataset"  
        logger.info("start loading %s" % (self.name,))
        self.root_dir = file_list
        self.file_list = []
        for root, dirs, files in os.walk(osp.join(file_list, "clean_left")):
            for file in files:
                if file.endswith(".png"):
                    self.file_list.append(
                        [
                            os.path.join(root, file),
                            os.path.join(root, file).replace(
                                "clean_left", "clean_right"
                            ),
                            os.path.join(root, file).replace(
                                "clean_left", "disparities"
                            ),
                        ]
                    )
        self.sample_file()
        logger.info("len(%s): %d" % (self.name, len(self)))

    def __getitem__(self, index):
        l, r, d = self.file_list[index]
        left, right, disp = (
            img_loader(l, self.img_open_mode),
            img_loader(r, self.img_open_mode),
            self.disp_loader(d),
        )
        return left, right, disp

    @staticmethod
    def disp_loader(p):
        assert osp.exists(p), p
        """ Return disparity read from filename. """
        f_in = np.array(Image.open(p))
        d_r = f_in[:, :, 0].astype("float64")
        d_g = f_in[:, :, 1].astype("float64")
        d_b = f_in[:, :, 2].astype("float64")

        depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
        return depth


class FallingThingsDataset(AIOTBaseDataset):
    def __init__(self, root_dir, debug=False, img_open_mode="rgb"):
        super().__init__()
        self.debug = debug
        self.name = "FallingThingsDataset"  
        logger.info("start loading %s" % (self.name,))
        self.img_open_mode = img_open_mode
        self.root_dir = root_dir
        self.file_list = [i for i in os.listdir(root_dir)]
        self.sample_file(2)
        logger.info("len(%s): %d, root: %s" % (self.name, len(self), root_dir))

    def __getitem__(self, index):
        d = self.file_list[index]
        d_path = os.path.join(self.root_dir, d)
        l_path = os.path.join(
            self.root_dir.replace("_disp_gt", ""), d.replace(".tiff", ".left.jpg")
        ) 
        r_path = os.path.join(
            self.root_dir.replace("_disp_gt", ""), d.replace(".tiff", ".right.jpg")
        )
        left, right, disp = (
            img_loader(l_path, self.img_open_mode),
            img_loader(r_path, self.img_open_mode),
            self.disp_loader(d_path),
        )
        return left, right, disp

    @staticmethod
    def disp_loader(p):
        assert osp.exists(p), p
        return cv2.imread(p, -1)


class SIDODDataset(AIOTBaseDataset):
    def __init__(self, root_dir, debug=False, img_open_mode="rgb"):
        super().__init__()
        self.debug = debug
        self.name = "SIDODDataset"  
        logger.info("start loading %s" % (self.name,))
        self.img_open_mode = img_open_mode
        self.root_dir = root_dir
        self.file_list = [i for i in os.listdir(root_dir)]
        self.sample_file(2)
        logger.info("len(%s): %d, root: %s" % (self.name, len(self), root_dir))

    def __getitem__(self, index):
        d = self.file_list[index]
        d_path = os.path.join(self.root_dir, d)
        l_path = os.path.join(
            self.root_dir.replace("_disp_gt", ""), d.replace(".tiff", ".left.png")
        ) 
        r_path = os.path.join(
            self.root_dir.replace("_disp_gt", ""), d.replace(".tiff", ".right.png")
        )
        left, right, disp = (
            img_loader(l_path, self.img_open_mode),
            img_loader(r_path, self.img_open_mode),
            self.disp_loader(d_path),
        )
        return left, right, disp

    @staticmethod
    def disp_loader(p):
        assert osp.exists(p), p
        return cv2.imread(p, -1)


class TartanAirDataset(SIDODDataset):
    def __init__(self, root_dir, debug=False, img_open_mode="rgb"):
        super(SIDODDataset).__init__()
        self.debug = debug
        self.name = "TartanAirDataset"  
        logger.info("start loading %s" % (self.name,))
        self.img_open_mode = img_open_mode
        self.root_dir = root_dir
        self.file_list = []
        for d in os.listdir(os.path.join(root_dir, "disp_gt")):
            d_path = os.path.join(self.root_dir, "disp_gt", d)
            l_path = os.path.join(
                self.root_dir, "image_left", d.replace(".tiff", "_left.png")
            )  
            r_path = os.path.join(
                self.root_dir, "image_right", d.replace(".tiff", "_right.png")
            )
            self.file_list.append([l_path, r_path, d_path])
       
        self.sample_file(1)
        logger.info("len(%s): %d, root: %s" % (self.name, len(self), root_dir))

    def __getitem__(self, index):
        l_path, r_path, d_path = self.file_list[index]
        left, right, disp = (
            img_loader(l_path, self.img_open_mode),
            img_loader(r_path, self.img_open_mode),
            self.disp_loader(d_path),
        )
        return left, right, disp


class SIRSDataset(AIOTBaseDataset):
    def __init__(self, file_list, debug=False, img_open_mode="rgb"):
        super().__init__()
        self.debug = debug
        self.name = "SIRSDataset"  
        logger.info("start loading %s" % (self.name,))
        self.img_open_mode = img_open_mode
        self.root_dir = osp.dirname(file_list)
        self.file_list = []
        with open(file_list, "r") as f:
            for line in f.readlines():
                img_names = line.rstrip().split()

                img_names[0] = img_names[0].replace("IRSDataset/", "")
                img_names[1] = img_names[1].replace("IRSDataset/", "")
                img_names[2] = img_names[2].replace("IRSDataset/", "")
                img_names[3] = img_names[3].replace("IRSDataset/", "")
                if (
                    "QAOfficeAndSecurityRoom2" in img_names[0]
                    or "OfficeMedley1/" in img_names[0]
                    or "OfficeMedley2/" in img_names[0]
                ):
                    continue
                if (
                    "SCLRestaurant001/" in img_names[0]
                    or "ArchvizKitchen_T2/" in img_names[0]
                    or "ContemporaryRestaurant1_BL2/" in img_names[0]
                ):
                    continue
                if (
                    "SCLRestaurant001_BL2" in img_names[0]
                    or "BistroRestaurantScene" in img_names[0]
                    or "ContemporaryRestaurant2_BL2" in img_names[0]
                    or "OfficeKIT" in img_names[0]
                ):
                    continue
                img_left_name = os.path.join(
                    self.root_dir, img_names[0].split("/")[0], img_names[0]
                )
                img_right_name = os.path.join(
                    self.root_dir, img_names[1].split("/")[0], img_names[1]
                )
                gt_disp_name = os.path.join(
                    self.root_dir, img_names[2].split("/")[0], img_names[2]
                )
                gt_norm_name = os.path.join(
                    self.root_dir, img_names[3].split("/")[0], img_names[3]
                )

                if (
                    not os.path.exists(img_left_name)
                    or not os.path.exists(img_right_name)
                    or not os.path.exists(gt_disp_name)
                    or not os.path.exists(gt_norm_name)
                ):
                    continue
                self.file_list.append(
                    [img_left_name, img_right_name, gt_disp_name, gt_norm_name]
                )
        self.sample_file()
        logger.info("len(%s): %d" % (self.name, len(self)))

    def __getitem__(self, index):
        img_left_name, img_right_name, gt_disp_name, gt_norm_name = self.file_list[
            index
        ]

        def load_rgb(filename):
            assert osp.exists(filename), filename
            img = None
            if filename.find(".npy") > 0:
                img = np.load(filename)
            else:
                img = io.imread(filename)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.pad(img, ((0, 0), (0, 0), (0, 2)), "constant")
                    img[:, :, 1] = img[:, :, 0]
                    img[:, :, 2] = img[:, :, 0]
                h, w, c = img.shape
                if c == 4:
                    img = img[:, :, :3]
            return img

        def load_disp(filename):
            assert osp.exists(filename), filename
            gt_disp = None
            if gt_disp_name.endswith("pfm"):
                gt_disp, scale = self.load_pfm(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith("npy"):
                gt_disp = np.load(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith("exr"):
                gt_disp = self.load_exr(filename)
            else:
                gt_disp = Image.open(gt_disp_name)
                gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32) / 256

            return gt_disp

        def load_norm(filename):
            assert osp.exists(filename), filename
            gt_norm = None
            if filename.endswith("exr"):
                gt_norm = self.load_exr(filename)
                gt_norm = gt_norm * 2.0 - 1.0

            return gt_norm

        img_left = load_rgb(
            img_left_name
        )  
        img_right = load_rgb(img_right_name)
        gt_disp = load_disp(gt_disp_name)
        gt_norm = load_norm(gt_norm_name)
        return img_left, img_right, gt_disp

    @staticmethod
    def load_exr(filename):
        hdr = exr2hdr(filename)
        h, w, c = hdr.shape
        if c == 1:
            hdr = np.squeeze(hdr)
        return hdr

    @staticmethod
    def load_pfm(filename):
        """
        Load a PFM file into a Numpy array. Note that it will have
        a shape of H x W, not W x H. Returns a tuple containing the
        loaded image and the scale factor from the file.
        """
        file = open(filename, "r")
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().rstrip())
        if scale < 0: 
            endian = "<"
            scale = -scale
        else:
            endian = ">"  

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        file.close()
        return np.reshape(data, shape), scale

class DStereoDataset(AIOTBaseDataset):
    """
        d-cloud stereo dataset loading
    """
    def __init__(self, file_list, dataset_name="DStereoDataset", debug=False, img_open_mode='bgr'):
        super().__init__()
        self.debug = debug
        self.name = dataset_name # 436, 1024, 3  float disp
        logger.info("start loading %s" % (self.name,))
        self.img_open_mode = img_open_mode
        self.root_dir = ""
        self.file_list = file_list
        # with open(file_list, 'r') as f:
        #     self.file_list = f.readlines()
        #     # print(self.file_list)
        #     self.file_list = [[os.path.join(self.datasets_path, path) for path in line.strip().split(' ')] for line in self.file_list if line.strip()]
        
        self.sample_file(16)
        logger.info("len(%s): %d" % (self.name, len(self.file_list)))

    
    def __getitem__(self, index):
        l, r, d = self.file_list[index]
        left = self.img_loader(l, self.img_open_mode)
        right = self.img_loader(r, self.img_open_mode)
        disp = self.disp_loader(d)
        return left, right, disp
    
    @staticmethod
    def img_loader(p, mode='bgr'):
        """简化但功能完整的图像读取函数
        
        Args:
            p (str): 图像路径
            mode (str): 颜色模式，支持 'bgr'/'rgb'
            
        Returns:
            np.ndarray: 返回uint8类型的图像数据，shape为(H, W, 3)
        """
        assert os.path.isfile(p), f"图像文件不存在: {p}"
        
        # 优先使用cv2读取
        img = cv2.imread(p)
        
        # 如果cv2读取失败，尝试用PIL读取
        if img is None:
            try:
                img = np.array(Image.open(p))
                # PIL读取的是RGB格式，如果需要BGR则转换
                if mode == 'bgr':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"图像读取失败: {p}")
        
        # 确保是3通道
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
            
        # 如果指定RGB模式且是cv2读取的，需要转换
        if mode == 'rgb' and img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        return img
    
    @staticmethod
    def disp_loader(disp_path):
        """统一的视差图读取函数，支持多种格式
            Args:
                disp_path (str): 视差图路径
                
            Returns:
                np.ndarray: 返回float32类型的视差图数据
                
            Raises:
                AssertionError: 当文件不存在时抛出
                ValueError: 当无法正确读取文件时抛出
        """
        assert os.path.exists(disp_path), f"视差图文件不存在: {disp_path}"
        # 获取文件扩展名
        ext = os.path.splitext(disp_path)[1].lower()
        
        try:
            # 1. PFM格式
            if ext == '.pfm':
                return readPFM(disp_path)[0].astype(np.float32)
            
            # 2. EXR格式
            elif ext == '.exr':
                disp = exr2hdr(disp_path)
                h, w, c = disp.shape
                if c == 1:
                    disp = np.squeeze(disp)  
                elif disp.ndim == 3 and disp.shape[2] > 1:
                    # 如果是多通道，取第一个通道
                    disp = disp[..., 0]
                return disp
            
            # 3. NPY格式
            elif ext == '.npy':
                disp = np.load(disp_path)
                disp = disp[::-1, :]
                return disp
            
            # 4. PNG格式
            elif ext == '.png':
                disp = np.array(Image.open(disp_path), dtype=np.float32)
                # 根据数值范围自动判断缩放因子
                max_val = np.max(disp)
                if max_val > 1000:  # 可能使用了256作为缩放
                    disp = disp / 256.0
                elif max_val > 100:  # 可能使用了100作为缩放
                    disp = disp / 100.0
                return disp
            
            # 5. TIFF/TIF格式
            elif ext in ['.tiff', '.tif']:
                return cv2.imread(disp_path, -1)
            
            # 6. JPEG/JPG格式 (不推荐用于视差图，但有些数据集可能会用)
            elif ext in ['.jpg', '.jpeg']:
                disp = cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE)
                if disp is None:
                    raise ValueError(f"无法读取JPEG文件: {disp_path}")
                return disp.astype(np.float32)
            
            # 7. BIN格式
            elif ext == '.bin':
                with open(disp_path, 'rb') as f:
                    data = f.read()
                # 假设是float32格式存储
                disp = np.frombuffer(data, dtype=np.float32)
                # 需要知道具体的图像尺寸才能reshape
                # disp = disp.reshape((height, width))
                return disp
            
            # 8. 其他格式尝试用PIL打开
            else:
                try:
                    disp = np.array(Image.open(disp_path), dtype=np.float32)
                    return disp
                except Exception as e:
                    raise ValueError(f"不支持的视差图格式或无法读取: {disp_path}, 错误: {str(e)}")
                    
        except Exception as e:
            raise ValueError(f"读取视差图失败: {disp_path}, 错误: {str(e)}")
    
    

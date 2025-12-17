# Copyright (c) Horizon Robotics. All rights reserved.
import re
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from horizon_plugin_pytorch.qtensor import QTensor
from torch.quantization import DeQuantStub, QuantStub

from hat.models.task_modules.bev.spatial_transfomer import (
    SpatialTransfomer,
    SpatialTransfomerFixedOffset,
    SpatialTransfomerWithOffset,
    get_random_idx,
)
from hat.models.task_modules.bev.utils import reorder_first_dim
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, flatten, is_list_of_type

__all__ = ["ANCBEVFusionModule", "ANCBEVMultiFusionModule"]


@OBJECT_REGISTRY.register
class ANCBEVFusionModule(nn.Module):
    r"""
    Input multi-views data and output fused BEV data.

    In general, there are the following situations:
    1.Training stage(float or qat): homography matrix as input to the model.
    Please configure as follows::

        use_homo_offset=False
        homographys=None
        homo_offset=None
        compile_model=False

    2.Training stage(float or qat): homography offset as input to the model.
    Please configure as follows::

        use_homo_offset=True
        homographys=None
        homo_offset=None
        compile_model=False

    3.Int_infer stage: homography offset as model`s parameter and we provide homography matrix.
    Please configure as follows::

        use_homo_offset=False
        homographys='provide homography matrix'
        homo_offset=None
        compile_model=True

    4.Int_infer stage: homography offset as model`s parameter and we provide homography offset.
    Please configure as follows::

        use_homo_offset=False
        homographys=None
        homo_offset='provide homography offset'
        compile_model=True

    5.Int_infer stage: homography offset will be used as input to the model.
    Please configure as follows::

        use_homo_offset=True
        homographys=None
        homo_offset=None
        compile_model=True

    Args:
        ipm_output_size: output size of ipm, (height, width).
        views: view numbers.
        grid_quant_scale: quanti scale of grid for grid_sample.
            NOTE: this value is very important for qat training, must set
            properly. You can get more information from wiki:
            http://wiki.hobot.cc/display/~wenming.meng/grid_sample+op+in+plugin.  # noqa
        use_homo_offset: whether to use homo_offset
            If True, homography offset will as input to the model.
        homographys: homography matrix of each view.usually be used in compiling process.
            NOTE: Setting homographys means we will use it to calculate homo_offset and save homo_offset
            as model`s parameter. So do not setting homographys in training stage.
        homo_offset: homo_offset matrix of each view.usually be used in compiling process.
            NOTE: Setting homo_offset means we will save homo_offset as model`s parameter.
            So do not setting homographys in training stage.
        compile_model: Whether compiling model. Compiling model means we only process single batch.
        random_rotation_cfg: config of random rotaton module which
            will apply random rotation for bev input.
        bev_fusion_input_name: input key name of bev fusion.
        bev_fusion_out_name: out key name of bev fusion.
        vcs_plane_nums: number of multi-height vcs planes.

    """

    def __init__(
        self,
        ipm_output_size: Tuple,
        views: Union[int, Sequence],
        grid_quant_scale: float,
        use_homo_offset: Optional[bool] = False,
        homographys: Optional[torch.Tensor] = None,
        homo_offset: Optional[torch.Tensor] = None,
        compile_model: bool = False,
        random_rotation_cfg: Optional[Mapping] = None,
        drop_view_prob: float = 0.0,
        bev_fusion_input_name: str = "bev_fusion_input",
        bev_fusion_out_name: str = "bev_fusion_out",
        block_warp_padding: list = None,
        vcs_plane_nums: int = 1,
        generate_offset_module: nn.Module = None,
        homo_offset_info_keys: Sequence[str] = None,
        **kwargs,
    ):
        super(ANCBEVFusionModule, self).__init__(**kwargs)
        self.views = _as_list(views)
        self.bev_fusion_input_name = bev_fusion_input_name
        self.bev_fusion_out_name = bev_fusion_out_name
        self.homographys = homographys
        self.homo_offset = homo_offset
        self.use_homo_offset = use_homo_offset
        self.spatial_transformers = nn.ModuleList()
        self.quant = nn.ModuleList()
        self.views_num = sum(_as_list(self.views))
        self.compile_model = compile_model
        self.ipm_output_size = ipm_output_size
        self.vcs_plane_nums = vcs_plane_nums
        assert (
            is_list_of_type(ipm_output_size, int) and len(ipm_output_size) == 2
        )

        if compile_model:
            assert (
                homographys is not None
                or homo_offset is not None
                or use_homo_offset
            )
        else:
            assert (
                homographys is None
            ), "do not setting homographys in training stage"
            assert (
                homo_offset is None
            ), "do not setting homo_offset in training stage"

        if homographys is not None:
            expected_shape = (self.views_num * self.vcs_plane_nums, 3, 3)
            assert (
                homographys.shape == expected_shape
            ), f"shape of homography  provided is not valid.\
            expected: {expected_shape}, now: {homographys.shape}"
            homographys = homographys.split(1, dim=0)
        if homo_offset is not None:
            expected_shape = (
                self.views_num * self.vcs_plane_nums,
                ipm_output_size[0],
                ipm_output_size[1],
                2,
            )
            assert (
                homo_offset.shape == expected_shape
            ), f"shape of homo_offset  provided is not valid.\
            expected: {expected_shape}, now: {homographys.shape}"
            homo_offset = homo_offset.split(1, dim=0)
        for i in range(self.views_num):
            if homographys is None and homo_offset is None:
                if use_homo_offset:
                    st_i = SpatialTransfomerWithOffset(
                        height=ipm_output_size[0],
                        width=ipm_output_size[1],
                        grid_quant_scale=grid_quant_scale,
                        mode="bilinear",
                        padding_mode="zeros",
                        block_warp_padding=block_warp_padding[i]
                        if block_warp_padding is not None
                        else None,
                        compile_model=compile_model,
                        multi_warp_nums=self.vcs_plane_nums,
                    )
                else:
                    st_i = SpatialTransfomer(
                        height=ipm_output_size[0],
                        width=ipm_output_size[1],
                        grid_quant_scale=grid_quant_scale,
                        mode="bilinear",
                        padding_mode="zeros",
                        use_horizon_grid_sample=True,
                        block_warp_padding=block_warp_padding[i]
                        if block_warp_padding is not None
                        else None,
                        multi_warp_nums=self.vcs_plane_nums,
                    )
            else:
                st_i = SpatialTransfomerFixedOffset(
                    height=ipm_output_size[0],
                    grid_quant_scale=grid_quant_scale,
                    width=ipm_output_size[1],
                    homography=None
                    if homographys is None
                    else homographys[
                        i * self.vcs_plane_nums : (i + 1) * self.vcs_plane_nums
                    ],
                    homo_offset=None
                    if homo_offset is None
                    else homo_offset[
                        i * self.vcs_plane_nums : (i + 1) * self.vcs_plane_nums
                    ],
                    mode="bilinear",
                    padding_mode="zeros",
                    multi_warp_nums=self.vcs_plane_nums,
                )
            self.spatial_transformers.append(st_i)
            self.quant.append(QuantStub())
        self.drop_view_prob = drop_view_prob
        self.random_rotation = random_rotation_cfg
        self.adds = nn.ModuleList()
        for _i in range(self.views_num - 1):
            self.adds.append(nn.quantized.FloatFunctional())
        self.generate_offset_module = generate_offset_module
        self.homo_offset_info_keys = homo_offset_info_keys

    def dropout_view(self, views_input, drop_view_idxs=None):
        """
        Dropout views by drop_view_idxs.

        Args:
            views_input: shape is views * [bs,c,h,w]
            drop_view_idxs : shape is bs * [idx (or None)]
        Returns:
            views_input : shape is views * [bs,c,h,w]

        """
        bs = len(drop_view_idxs)
        for bs_i in range(bs):
            cur_drop_view_idx = drop_view_idxs[bs_i]
            if cur_drop_view_idx is not None:
                views_input = list(views_input)
                drop_data = views_input[cur_drop_view_idx][bs_i]
                if type(drop_data) == torch.Tensor:
                    drop_data *= 0
                elif type(drop_data) == QTensor:
                    drop_data = QTensor(
                        drop_data * 0,
                        drop_data.scale.clone(),
                        drop_data.dtype,
                    )
                else:
                    raise TypeError("donot support the type to dropout")
                views_input[cur_drop_view_idx][bs_i] = drop_data
        return views_input

    def parse_bev_data(
        self,
        data: Union[torch.Tensor, QTensor],
        homography_mat: Optional[Sequence[torch.Tensor]] = None,
        homo_offset: Optional[Sequence[torch.Tensor]] = None,
        drop_view_idxs: Sequence[Optional[int]] = (None),
    ) -> torch.Tensor:
        """
        Parse bev data.

        Args:

            data: shape is [(bs*views1,c,h,w),(bs*views1,c,h,w)]
            homography_mat: homograpy matrix shape is (bs, views *
                num_planes, 3, 3)
            homo_offset: homo_offset  shape is (bs * views *
                num_planes, bev_h, bev_w, 2)
            drop_view_idxs: dropout view idx of each sample.
                None means do not dropout.

        Returns:
            bev_input: shape is (bs,c,h,w)

        """
        bs = data[0].shape[0] // self.views[0]
        # homo_offset: (bs*views*num_planes,h,w,2)
        #           -> (views*num_planes*bs,h,w,2)
        if self.use_homo_offset:
            homo_offset = reorder_first_dim(homo_offset, bs)
        else:
            # homography_mat: (bs,views * num_planes, 3, 3)
            #           -> (bs*views*num_planes,3, 3)
            #           -> (views*num_planes*bs,3, 3)
            n, c, h, w = homography_mat.shape
            homography_mat_reshape = homography_mat.reshape(n * c, h, w)
            homography_mat = reorder_first_dim(homography_mat_reshape, bs)
        # data: [(bs*views1,c,h,w),(bs*views2,c,h,w)...]
        #    -> [(views1*bs,c,h,w),(views2*bs,c,h,w)...]
        data_list = []
        for view_data in data:
            data_list.append(reorder_first_dim(view_data, bs))
        # data_batch_list: [(views1*bs,c,h,w),(views2*bs,c,h,w)...]
        #    -> [(bs,c,h,w),(bs,c,h,w),(bs,c,h,w)...]
        data_batch_split = []
        for cur_data_batch in data_list:
            data_batch_split = data_batch_split + list(
                torch.split(cur_data_batch, bs, dim=0)
            )
        if self.use_homo_offset:
            # homo_offset: (views*num_planes*bs,h,w,2)
            #           -> [(bs,h,w,2)，(bs,h,w,2)...]
            batch_homo_offset = torch.split(homo_offset, bs, dim=0)
            cur_bev = self.get_bev_input(
                data_batch_split,
                drop_view_idxs,
                homography_offset=batch_homo_offset,
            )
        else:
            batch_homo_mat = torch.split(homography_mat, bs, dim=0)
            cur_bev = self.get_bev_input(
                data_batch_split,
                drop_view_idxs,
                homography_mat=batch_homo_mat,
            )
        return cur_bev

    def get_bev_input(
        self,
        multi_view_data: Union[torch.Tensor, QTensor],
        drop_view_idxs: Sequence[Optional[int]] = None,
        homography_mat: Optional[Sequence[torch.Tensor]] = None,
        homography_offset: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Get bev input with stage1 output feature and homo_mat or homo_offset.

        Args:
            multi_view_data: input data whose shape
                is views*(bs,c,h,w).
            homography_mat: homograpy matrix whose shape
                is views*num_planes*(bs,3,3).
            homography_offset: homograpy offset with the
                shape of views*num_planes*(bs,h,w,2).
            drop_view_idxs: dropout view idx,
                None means do not dropout.

        Returns:
            bev_input: shape is (bs,c,h,w)

        """
        bevs = []
        for view_idx, each_view in enumerate(multi_view_data):
            if self.homographys is None and self.homo_offset is None:
                # homo matrix or homo offset as input
                if self.use_homo_offset:
                    # homo offset as input
                    bevs.append(
                        self.spatial_transformers[view_idx](
                            self.quant[view_idx](each_view),
                            homography_offset[
                                view_idx
                                * self.vcs_plane_nums : (view_idx + 1)
                                * self.vcs_plane_nums
                            ],
                        )
                    )
                else:
                    # homo matrix as input
                    bevs.append(
                        self.spatial_transformers[view_idx](
                            self.quant[view_idx](each_view),
                            homography_mat[
                                view_idx
                                * self.vcs_plane_nums : (view_idx + 1)
                                * self.vcs_plane_nums
                            ],
                        )[0]
                    )
            else:
                # homo offset has been saved in self.spatial_transformers
                bevs.append(
                    self.spatial_transformers[view_idx](
                        self.quant[view_idx](each_view)
                    )
                )
        bevs = (
            self.dropout_view(bevs, drop_view_idxs=drop_view_idxs)
            if any(_as_list(drop_view_idxs))
            else bevs
        )
        cur_bev = bevs[0]
        for view_idx, one_view in enumerate(bevs[1:]):
            cur_bev = self.adds[view_idx].add(cur_bev, one_view)
        return cur_bev

    def forward(
        self,
        bev_fusion_input,
        meta=None,
    ):
        if self.generate_offset_module is not None:
            assert self.homo_offset_info_keys is not None
            temporal_info = meta["temporal_info"]
            views_type = meta["view"]
            homo_meta_info = {
                k: meta["meta_info"][k] for k in self.homo_offset_info_keys
            }
            homo_offset = self.generate_offset_module(
                homo_meta_info, temporal_info, views_type
            )
        else:
            if self.compile_model:
                homo_offset = []
                for i in range(self.views_num * self.vcs_plane_nums):
                    homo_offset.append(meta[f"homo_offset_{i}"])
            else:
                homo_offset = meta["meta_info"]["homo_offset"]

        if not self.compile_model:
            homography = meta["meta_info"]["homography"]
            # we need to process multi-batch data durning training stage.
            stage1_out = _as_list(bev_fusion_input)
            bev_batchsize = stage1_out[0].shape[0] // self.views[0]
            drop_view_idxs = [None] * bev_batchsize
            if self.drop_view_prob > 0 and self.training:
                drop_view_idxs = [
                    get_random_idx(self.drop_view_prob, n=self.views_num)
                    for _ in range(bev_batchsize)
                ]
            if self.use_homo_offset:
                # homography offset as input to model.
                bev_input = self.parse_bev_data(
                    stage1_out,
                    homo_offset=homo_offset,
                    drop_view_idxs=drop_view_idxs,
                )
            else:
                # homography matrix as input to model.
                bev_input = self.parse_bev_data(
                    stage1_out,
                    homography_mat=homography,
                    drop_view_idxs=drop_view_idxs,
                )
        else:
            # druning compile model stage, only process single batch.
            if self.use_homo_offset:
                # homography offset as input to model.
                bev_input = self.get_bev_input(
                    bev_fusion_input,
                    homography_offset=homo_offset,
                )
            else:
                # homography offset as model` parameters.
                bev_input = self.get_bev_input(bev_fusion_input)

        rot_mat = None
        if self.random_rotation is not None and self.training:
            bev_input, rot_mat = self.random_rotation([bev_input])
            bev_input = bev_input[0]

        return rot_mat, bev_input

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.random_rotation is not None:
            self.random_rotation.set_qconfig()
        for module in list(self.spatial_transformers):
            if module is None:
                continue
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class ANCBEVMultiFusionModule(ANCBEVFusionModule):
    """
    Input multi-views data and output fused BEV data.

    The purpose we introduce a twice fusion:
        e.g., deploy on 2J5 hardware, the view feats need cross-chip transfer.
        we fuse the feats of J51 and J52 separately, then transfer
        only the fused feats. Alleviate the pressure of cross-chip
        feature transfer.

    2 steps to accomplish this purpose:
        (1) fuse the view feats of each chip separately.
        (2) fuse the fused feats of each chip.

    In general, there are the following situations:
        1. Training stage(float or qat):
            fusion_idx_lst='provide fusion_idx_lst'
            dequant_out=False
            compile_model=False

        2. Int_infer stage: only fuse bev feat in J51.
            fusion_idx_lst='provide fusion_idx_lst'
            dequant_out=True
            compile_model=True
            compile_fusion_type='bev_part0_fusion'

        3. Int_infer stage: only fuse bev feat in J52.
            fusion_idx_lst='provide fusion_idx_lst'
            dequant_out=True
            compile_model=True
            compile_fusion_type='bev_part1_fusion'

        4. Int_infer stage: fuse the fused bev feat in J51&2.
            fusion_idx_lst='provide fusion_idx_lst'
            dequant_out=False
            compile_model=True
            compile_fusion_type='bev_part_all_fusion'

    Args:
        fusion_idx_lst: view idx list, e.g
            [[J51 view idxs], [J52 view idxs]].
        compile_fusion_type: fusion type in compile.

    """

    def __init__(
        self,
        fusion_idx_lst: Sequence,
        compile_fusion_type: str = None,
        **kwargs,
    ):
        super(ANCBEVMultiFusionModule, self).__init__(**kwargs)

        self.fusion_idx_lst = fusion_idx_lst
        flatten_fusion_idx_lst = list(flatten(fusion_idx_lst)[0])
        flatten_fusion_idx_lst.sort()
        assert flatten_fusion_idx_lst == list(range(self.views_num))
        self.fusion_quant = nn.ModuleList()
        for _ in range(len(fusion_idx_lst)):
            self.fusion_quant.append(QuantStub())
        self.compile_fusion_type = compile_fusion_type
        self.part_compile_fusion_type = [
            "bev_part%d_fusion" % (i) for i in range(len(fusion_idx_lst))
        ]
        assert self.compile_fusion_type in (
            [None] + self.part_compile_fusion_type + ["bev_part_all_fusion"]
        )
        if (
            self.compile_model
            and self.compile_fusion_type in self.part_compile_fusion_type
        ):
            self.dequant = DeQuantStub()

    def get_bev_input(
        self,
        multi_view_data: Union[torch.Tensor, QTensor],
        drop_view_idxs: Sequence[Optional[int]] = None,
        homography_mat: Optional[Sequence[torch.Tensor]] = None,
        homography_offset: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Get bev input with stage1 output feature and homo_mat or homo_offset.

        Args:
            multi_view_data: input data whose shape
                is views*(bs,c,h,w).
            homography_mat: homograpy matrix whose shape
                is views*num_planes*(bs,3,3).
            homography_offset: homograpy offset with the
                shape of views*num_planes*(bs,h,w,2).
            drop_view_idxs: dropout view idx,
                None means do not dropout.

        Returns:
            bev_input: shape is (bs,c,h,w)

        """
        bevs = []
        for view_idx, each_view in enumerate(multi_view_data):
            if (
                self.compile_model
                and self.compile_fusion_type == "bev_part_all_fusion"
            ):
                bevs.append(self.fusion_quant[view_idx](each_view))
            else:
                homo_offset_idx = view_idx
                if self.compile_model:
                    if (
                        self.compile_fusion_type
                        in self.part_compile_fusion_type
                    ):
                        fusion_id = int(
                            re.findall(r"\d+", self.compile_fusion_type)[0]
                        )
                        fusion_idx_lst = self.fusion_idx_lst[fusion_id]
                    view_idx = fusion_idx_lst[view_idx]

                if self.homographys is None and self.homo_offset is None:
                    # homo matrix or homo offset as input
                    if self.use_homo_offset:
                        # homo offset as input
                        bevs.append(
                            self.spatial_transformers[view_idx](
                                self.quant[view_idx](each_view),
                                homography_offset[
                                    homo_offset_idx
                                    * self.vcs_plane_nums : (
                                        homo_offset_idx + 1
                                    )
                                    * self.vcs_plane_nums
                                ],
                            )
                        )
                    else:
                        # homo matrix as input
                        bevs.append(
                            self.spatial_transformers[view_idx](
                                self.quant[view_idx](each_view),
                                homography_mat[
                                    homo_offset_idx
                                    * self.vcs_plane_nums : (
                                        homo_offset_idx + 1
                                    )
                                    * self.vcs_plane_nums
                                ],
                            )[0]
                        )
                else:
                    # homo offset has been saved in self.spatial_transformers
                    bevs.append(
                        self.spatial_transformers[view_idx](
                            self.quant[view_idx](each_view)
                        )
                    )

        if not self.compile_model:
            bevs = (
                self.dropout_view(bevs, drop_view_idxs=drop_view_idxs)
                if any(_as_list(drop_view_idxs))
                else bevs
            )
            add_idx = 0
            fusion_bev_lst = []
            for i in range(len(self.fusion_idx_lst)):
                fusion_idx_lst = self.fusion_idx_lst[i]
                cur_bev = bevs[fusion_idx_lst[0]]
                for view_idx in fusion_idx_lst[1:]:
                    cur_bev = self.adds[add_idx].add(cur_bev, bevs[view_idx])
                    add_idx += 1
                fusion_bev_lst.append(cur_bev)
            cur_bev = self.fusion_quant[0](fusion_bev_lst[0])
            for i, one_view in enumerate(fusion_bev_lst[1:]):
                cur_bev = self.adds[add_idx].add(
                    cur_bev,
                    self.fusion_quant[i + 1](one_view),
                )
                add_idx += 1
        else:
            if self.compile_fusion_type in self.part_compile_fusion_type:
                fuision_id = int(
                    re.findall(r"\d+", self.compile_fusion_type)[0]
                )
                assert len(self.fusion_idx_lst[fuision_id]) == len(bevs)
                add_idx = (
                    len(flatten(self.fusion_idx_lst[:fuision_id])[0])
                    - fuision_id
                )
            elif self.compile_fusion_type == "bev_part_all_fusion":
                assert len(bevs) == len(self.fusion_idx_lst)
                add_idx = self.views_num - len(self.fusion_idx_lst)
            else:
                raise TypeError(
                    "compile_model must specify compile_fusion_type"
                )

            cur_bev = bevs[0]
            for idx, one_view in enumerate(bevs[1:]):
                cur_bev = self.adds[add_idx + idx].add(cur_bev, one_view)

        return self.dequant(cur_bev) if hasattr(self, "dequant") else cur_bev

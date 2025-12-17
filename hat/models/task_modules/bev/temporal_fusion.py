# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Sequence

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from torch.quantization import DeQuantStub, QuantStub

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.bev.spatial_transfomer import (
    SpatialTransfomer,
    SpatialTransfomerWithOffset,
)
from hat.models.task_modules.bev.temporal_utils import ANCSpatialGRU
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.tensor_func import select_sample


@OBJECT_REGISTRY.register
class ANCBEVFusionTemporalModule(nn.Module):
    """
    Input multi frame bev data and return temporal fused data.

    Args:
        input_shape: input size (height, width) of BEV features.
        in_channels: number of channels for input data.
        fusion_module: fusion module in `.temporal_module.py`.
        out_channels: number of channels for output data.
        temporal_length_each_batch: temporal data length in each batch.
        grid_quant_scale: quanti scale of grid for grid_sample.
            NOTE: this value is very important for qat training, must set
            properly. You can get more information from wiki:
            http://wiki.hobot.cc/display/~wenming.meng/grid_sample+op+in+plugin.  # noqa
        num_extra_encoder_layers: number of layers for extra encoder.
        compile_model: set True when compiling model.

    """  # noqa

    def __init__(
        self,
        input_shape: tuple,
        in_channels: int,
        fusion_module: dict,
        temporal_length_each_batch: int = 2,
        grid_quant_scale: float = 1.0,  # NOTE: set carefuly when qat training.
        num_extra_encoder_layers: int = 2,
        compile_model: bool = False,
        **kwargs,
    ):
        super(ANCBEVFusionTemporalModule, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.temporal_length_each_batch = temporal_length_each_batch
        """
        注意，我们输入进来的feat长度必须跟融合的长度相等，否则会报错
        """

        self.grid_quant_scale = grid_quant_scale
        self.fusion_module = fusion_module
        self.compile_model = compile_model

        if compile_model:
            self.temporal_st = SpatialTransfomerWithOffset(
                height=input_shape[0],
                width=input_shape[1],
                grid_quant_scale=grid_quant_scale,
                mode="bilinear",
                padding_mode="zeros",
                compile_model=True,
            )
        else:
            self.temporal_st = SpatialTransfomer(
                height=input_shape[0],
                width=input_shape[1],
                grid_quant_scale=grid_quant_scale,
                mode="bilinear",
                padding_mode="zeros",
                use_horizon_grid_sample=True,
            )

        self.quant = QuantStub()

        self.dequant = DeQuantStub()

        extra_encoder = nn.ModuleList()
        for _ in range(num_extra_encoder_layers):
            extra_encoder.append(
                BasicResBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    bn_kwargs={},
                )
            )
        self.extra_encoder = nn.Sequential(*extra_encoder)

    def fuse_model(self):
        for module in self.extra_encoder:
            module.fuse_model()
        if hasattr(self.fusion_module, "fuse_model"):
            self.fusion_module.fuse_model()

    def _split_temporal_data(
        self,
        all_frames_data,
        homography_temporal=None,
        homo_offset_temporal=None,
    ):
        """Split temporal data into two parts.

        假设all_frames_data的时序长度是N，那么当homography_temporal不是None的时候，返回
        homography_temporal是n-1个。

        Durning train/val process, all_frames_data is a tensor and it`s shape
            is (b*t,c,h,w), t is temporal_length.
        When compiling model, all_frames_data is a list of tensor and
            each tensor`s shape is (b,c,h,w).
        """
        if self.compile_model:
            cur_frame_data, pre_frames_data = (
                all_frames_data[0],
                all_frames_data[1:],
            )
            return (
                cur_frame_data,
                pre_frames_data,
                homography_temporal,
                homo_offset_temporal,
            )
        else:
            b, _, _, _ = all_frames_data.shape
            cur_frame_data = select_sample(
                all_frames_data,
                list(range(0, b, self.temporal_length_each_batch)),
            )
            pre_frames_data = [
                select_sample(
                    all_frames_data,
                    list(range(idx, b, self.temporal_length_each_batch)),
                )
                for idx in range(1, self.temporal_length_each_batch)
            ]
            if homography_temporal is not None:
                homography_temporal = homography_temporal[
                    :, : self.temporal_length_each_batch - 1
                ]
            return cur_frame_data, pre_frames_data, homography_temporal, None

    def _temporal_warp(
        self,
        pre_frames_data: Sequence,
        homography_temporal: Optional[Sequence],
        homo_offset_temporal: Optional[Sequence],
    ):
        """Warp previous frame datas to current frame.

        Args:
            pre_frames_data: previous frame data to warp.
            homography_temporal: homography matrix
                from current frame to previous, shape is
                (b,temporal_length,3,3). (Used in train/val)
            homo_offset_temporal: homograpy offset
                from current frame to previous. (Used in compiling model)

        """

        cur_frames_warped = []
        if self.compile_model:
            assert len(pre_frames_data) == len(homo_offset_temporal)
            for pre_frame, offset_tempo in zip(
                pre_frames_data, homo_offset_temporal
            ):
                cur_frames_warped.append(
                    self.temporal_st(self.quant(pre_frame), offset_tempo)
                )
        else:
            assert len(pre_frames_data) == homography_temporal.shape[1]
            for frame_idx, pre_frame in enumerate(pre_frames_data):
                cur_frames_warped.append(
                    self.temporal_st(
                        self.quant(pre_frame),
                        homography_temporal[:, frame_idx],
                    )[0]
                )
        return cur_frames_warped

    def forward(
        self,
        all_frames_data,
        homography_temporal=None,
        homo_offset_temporal=None,
    ):

        (
            cur_frame_data,
            pre_frames_data,
            homography_temporal,
            homo_offset_temporal,
        ) = self._split_temporal_data(
            all_frames_data, homography_temporal, homo_offset_temporal
        )

        cur_frames_warped = self._temporal_warp(
            pre_frames_data, homography_temporal, homo_offset_temporal
        )
        pre_frames_data = cur_frames_warped
        frames = [cur_frame_data] + pre_frames_data
        frames = [self.extra_encoder(data) for data in frames]

        fusion_data, _ = self.fusion_module(frames)

        return fusion_data, self.dequant(cur_frame_data)

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()
        self.temporal_st.set_qconfig()


@OBJECT_REGISTRY.register
class ANCBEVFusionRecurrentTemporalModule(ANCBEVFusionTemporalModule):
    r"""Recurrent version of temporal fusion module.

    假设输入的时序长度为3, bs=1, 那么输入的feature shape为(3*1,c,h,w), 顺序是
    [t,t-1,t-2], 输出结果3帧经过融合之后的特征, 用list表示, 顺序是[t-2,t-1,t].

    Training and not continuous frame testing data flow is showed below:

    *  bev_feat(t-2) -> temporal_warp() -> align_feat
    *                        _ _ _ _ _ _ _ _ _ _|
    *                       |
    *                      \|/
    *  bev_feat(t-1) -> fusion_module() -> temporal_warp() -> align_feat
    *                                               _ _ _ _ _ _ _|
    *                                              |
    *                                             \|/
    *  bev_feat(t) -------------------------->fusion_module() -> fused_feat

    Continuous frame testing data flow is showed below:
    *  history_cache_feat ----> temporal_warp() ----> align_feat
    *                                     _ _ _ _ _ _ _ _ _ _ _|
    *                                    |
    *                                   \|/                         (update)
    *  bev_feat(t) -------------> fusion_module() ----> fused_feat----------->history_cache_feat


    Args:
        do_cache_feat: whether cache the last fusioned-feat for next clips.
            (e.g. in test mode, within a pack shuold do so.) Default to false.
        init_zero_his_feat: Only enable in consecutive_frame mode.
            If true, when history_cache_feat is None, use the oldest encoded
            frame feat as history_cache_feat. If false, fill it all 0 when
            history_cache_feat is None.Default to false.
        is_relative_homography: if true, the input homography is relative one
            rather than absolute one.Default to false.
        only_return_latest_feature: if ture, only ouput fusion-feat of the last
            frame(current).Otherwise, output fusion-feat of all frames.(e.g.
            batchsize=1, input feat's shape is 10x3x256x256 [t~t-9]. If true,
            the ouput feat's shape is 1xCx256x256 [t]. Otherwise, the ouput
            feat's shape is 10xCx256x256 [t-9~t]).Default to true.
        force_task_name: force to input the task name.

    """  # noqa

    def __init__(
        self,
        do_cache_feat: bool = False,
        init_zero_his_feat: bool = False,
        is_relative_homography: bool = False,
        use_homo_offset_temporal: bool = False,
        force_task_name: bool = False,
        **kwargs,
    ):
        super(ANCBEVFusionRecurrentTemporalModule, self).__init__(**kwargs)
        if isinstance(self.fusion_module, ANCSpatialGRU):
            """
            When using SpatialGRU and setting multiple layers,
            we will stack the hidden states output by multiple layers in the batch dimension,
            so during inference, we use infer_batch_factor to split the hidden state from the previous frame.
            """  # noqa
            self.infer_batch_factor = self.fusion_module.num_layers
        else:
            self.infer_batch_factor = 1

        if use_homo_offset_temporal:
            self.temporal_st = SpatialTransfomerWithOffset(
                height=kwargs["input_shape"][0],
                width=kwargs["input_shape"][1],
                grid_quant_scale=kwargs["grid_quant_scale"],
                mode="bilinear",
                padding_mode="zeros",
                compile_model=True,
            )

        self.history_cache_feat = {}
        self.init_zero_his_feat = init_zero_his_feat
        self.do_cache_feat = do_cache_feat
        self.is_relative_homography = is_relative_homography
        self.use_homo_offset_temporal = use_homo_offset_temporal
        self.dequant_his_feat = DeQuantStub()
        self.floatmod_mulscalar = FloatFunctional()
        self.fusion_cat = FloatFunctional()
        self.force_task_name = force_task_name

        # 当输入clip长度只为1时，需要打开do_cache_feat
        if self.temporal_length_each_batch == 1:
            assert self.do_cache_feat

    def clear(self, task_name):
        # clear history_cache_feat when switch to another clip or pack.
        self.history_cache_feat[task_name] = None

    def _temporal_warp(
        self,
        pre_frames_data: Sequence,
        homography_temporal: Optional[Sequence],
        homo_offset_temporal: Optional[Sequence],
    ):
        """Warp previous frame datas to current frame.

        Args:
            pre_frames_data: previous frame data to warp.
            homography_temporal: homography matrix
                from current frame to previous, shape is
                (temporal_length,3,3). (Used in train/val)
            homo_offset_temporal: homograpy offset
                from current frame to previous. (Used in compiling model)

        """

        cur_frames_warped = []
        if self.compile_model:
            assert len(pre_frames_data) == len(homo_offset_temporal) == 1
            pre_frame = self.quant(pre_frames_data[0])
            assert pre_frame.shape[0] == self.infer_batch_factor
            # Warp each hidden feature recurrently.
            for j in range(self.infer_batch_factor):
                hidden_state = pre_frame[j : j + 1]
                cur_frames_warped.append(
                    self.temporal_st(hidden_state, homo_offset_temporal[0])
                )
        else:
            pre_frames_data = _as_list(pre_frames_data)
            cur_frames_warped = [
                self.temporal_st(d, homography_temporal)[0]
                for d in pre_frames_data
            ]
        return cur_frames_warped

    def convert_homo_mat_absolute2relative(
        self, homography_temporal: torch.tensor
    ):
        """Convert homography_temporal matrix from absolute to relative.

            before convert: t->t-1, t->t-2, t->t-3
            after convert: t-2->t-3, t-1->t-2, t->t-1

        Args:
            homography_temporal: matrix for temporal align, shape
                is (b,t,3,3). For details, b is batch_size, t is clip_len - 1.
        """
        homography_temporal = torch.flip(homography_temporal, [1])
        # t->t-3,t->t-2, t->t-1

        for frame_idx in range(0, homography_temporal.shape[1] - 1):
            homography_temporal[:, frame_idx] = homography_temporal[
                :, frame_idx
            ] @ torch.linalg.inv(homography_temporal[:, frame_idx + 1])
        # t-2->t-3, t-1->t-2, t->t-1

        return homography_temporal

    def forward(
        self,
        all_frames_data,
        meta,
    ):
        homography_temporal = meta.get("homography_temporal", None)
        homooffset_temporal = meta.get("homooffset_temporal", None)
        homo_offset_temporal = (
            homooffset_temporal
            if homooffset_temporal is not None
            else meta.get("homo_offset_temporal", None)
        )
        clear_cache_feat = False
        only_return_latest_feature = True
        if "temporal_clr_flag" in meta:
            clear_cache_feat = meta["temporal_clr_flag"][0]
        if "return_latest_flag" in meta:
            only_return_latest_feature = meta["return_latest_flag"][0]

        if not self.force_task_name:
            task_name = meta.get("task_name", ["multi_task"])[0]
        else:
            task_name = meta["task_name"][0]

        if clear_cache_feat:
            # clear_cache_feat being true means input data switch to a new
            # clip or pack. Do clear history_cache_feat.
            self.clear(task_name)

        task_history_cache_feat = self.history_cache_feat.get(task_name, None)
        if self.compile_model:
            # just for compile, adaptive for soft code
            assert homo_offset_temporal is not None
            assert "pre_fusion_feat" in meta
            if not isinstance(homo_offset_temporal, Sequence):
                homo_offset_temporal = [homo_offset_temporal]
            all_frames_data = [all_frames_data, meta["pre_fusion_feat"]]
            (
                cur_frame_data,
                pre_frames_data,
                homography_temporal,
                homo_offset_temporal,
            ) = self._split_temporal_data(
                all_frames_data, homography_temporal, homo_offset_temporal
            )

            cur_encoded_frame = self.extra_encoder(cur_frame_data)

            cur_frame_warped = self._temporal_warp(
                pre_frames_data, homography_temporal, homo_offset_temporal
            )
            pre_encoded_frame = cur_frame_warped
            # For inference, hidden features should be parsed to
            # the needed format for each fusion module.
            hidden_feats = self.fusion_module.parse_pre_feats(
                pre_encoded_frame
            )
            fusion_data, hidden_feats = self.fusion_module.forward_once(
                cur_encoded_frame, hidden_feats
            )
            if not only_return_latest_feature:
                fusion_data = self.fusion_cat.cat([fusion_data], dim=0)
        elif self.use_homo_offset_temporal:
            assert homo_offset_temporal is not None
            encoded_all_frames_data = self.extra_encoder(all_frames_data)
            (
                cur_encoded_frame,
                pre_encoded_frame,
                _,
                _,
            ) = self._split_temporal_data(encoded_all_frames_data)

            pre_encoded_frame = pre_encoded_frame[::-1]

            # 将history_cache_feat加到pre_encoded_frame中
            if task_history_cache_feat is not None:
                extra_hist_feat = task_history_cache_feat
            else:
                # 如果 history_cache_feat是None，根据参数用最老时刻的pre_frames_data填充它或者置成全零  # noqa
                if self.init_zero_his_feat:
                    extra_hist_feat = self.floatmod_mulscalar.mul_scalar(
                        cur_encoded_frame, 0
                    )
                else:
                    # 做quant来保证QAT-infer和int-infer的extra_hist_feat和后续帧的hidden_feats的scale一致。  # noqa
                    if len(pre_encoded_frame) == 0:
                        # 在temporal_length_each_batch=1时，第一次forward没有pre_frame,因此用cur_frame构造his_feat  # noqa
                        extra_hist_feat = self.quant(
                            self.dequant_his_feat(cur_encoded_frame)
                        )
                    else:
                        extra_hist_feat = self.quant(
                            self.dequant_his_feat(pre_encoded_frame[0])
                        )
                # "gru" fusion模式下，输入history_cache_feat为list，个数等于gru层数
                if isinstance(self.fusion_module, ANCSpatialGRU):
                    extra_hist_feat = [
                        extra_hist_feat
                    ] * self.fusion_module.num_layers

            pre_encoded_frame = [extra_hist_feat] + pre_encoded_frame
            encoded_frames = pre_encoded_frame + [cur_encoded_frame]

            # 下面开始逐帧融合了
            hidden_feats = encoded_frames[0]
            fusion_data = []
            for frame_idx in range(1, len(encoded_frames)):
                cur_feat = encoded_frames[frame_idx]
                warped_hidden_feats = self.temporal_st(
                    hidden_feats, homo_offset_temporal
                )

                (
                    cur_fusion_data,
                    hidden_feats,
                ) = self.fusion_module.forward_once(
                    cur_feat, warped_hidden_feats
                )
                fusion_data.append(cur_fusion_data)

            # Note: not support GRU at the moment. @Jinqian.Gao
            cur_fusion_data = self.quant(cur_fusion_data)
            if not only_return_latest_feature:
                fusion_data = [f_data.unsqueeze(1) for f_data in fusion_data]
                _, _, c, h, w = fusion_data[0].shape
                fusion_data = self.fusion_cat.cat(fusion_data, dim=1).reshape(
                    (-1, c, h, w)
                )
            else:
                fusion_data = fusion_data[-1]

        else:
            assert homography_temporal is not None

            encoded_all_frames_data = self.extra_encoder(all_frames_data)

            if self.do_cache_feat:
                # do_cache_feat模式下，homography_temporal的时序长度等于历史帧长度+1（history_cache_feat）  # noqa
                # homography_temporal不能过_split_temporal_data函数  # noqa
                (
                    cur_encoded_frame,
                    pre_encoded_frame,
                    _,
                    homo_offset_temporal,
                ) = self._split_temporal_data(
                    encoded_all_frames_data,
                    None,
                    homo_offset_temporal,
                )

                pre_encoded_frame = pre_encoded_frame[::-1]

                # 将history_cache_feat加到pre_encoded_frame中
                if task_history_cache_feat is not None:
                    extra_hist_feat = task_history_cache_feat
                else:
                    # 如果 history_cache_feat是None，根据参数用最老时刻的pre_frames_data填充它或者置成全零  # noqa
                    if self.init_zero_his_feat:
                        extra_hist_feat = self.floatmod_mulscalar.mul_scalar(
                            cur_encoded_frame, 0
                        )
                    else:
                        # 做quant来保证QAT-infer和int-infer的extra_hist_feat和后续帧的hidden_feats的scale一致。  # noqa
                        if len(pre_encoded_frame) == 0:
                            # 在temporal_length_each_batch=1时，第一次forward没有pre_frame,因此用cur_frame构造his_feat  # noqa
                            extra_hist_feat = self.quant(
                                self.dequant_his_feat(cur_encoded_frame)
                            )
                        else:
                            extra_hist_feat = self.quant(
                                self.dequant_his_feat(pre_encoded_frame[0])
                            )
                    # "gru" fusion模式下，输入history_cache_feat为list，个数等于gru层数
                    if isinstance(self.fusion_module, ANCSpatialGRU):
                        extra_hist_feat = [
                            extra_hist_feat
                        ] * self.fusion_module.num_layers

                pre_encoded_frame = [extra_hist_feat] + pre_encoded_frame
            else:
                # do_cache_feat=False时，homography_temporal的时序长度等于历史帧长度  # noqa
                # homography_temporal需要过_split_temporal_data函数  # noqa
                (
                    cur_encoded_frame,
                    pre_encoded_frame,
                    homography_temporal,
                    homo_offset_temporal,
                ) = self._split_temporal_data(
                    all_frames_data,
                    homography_temporal,
                    homo_offset_temporal,
                )

                pre_encoded_frame = pre_encoded_frame[::-1]

            # 历史帧pre_encoded_frame的帧数和homography_temporal的时序个数必然相等
            assert len(pre_encoded_frame) == homography_temporal.shape[1]

            encoded_frames = pre_encoded_frame + [cur_encoded_frame]

            if not self.is_relative_homography:
                # intput: t->t-1, t->t-2, t->t-3, t->hidden
                # output: t-3->hidden, t-2->t-3, t-1->t-2, t->t-1
                homography_temporal = self.convert_homo_mat_absolute2relative(
                    homography_temporal
                )
            else:
                # intput: t->t-1, t-1->t-2, t-2->t-3, t-3->hidden
                # output: t-3->hidden, t-2->t-3, t-1->t-2, t->t-1
                homography_temporal = torch.flip(homography_temporal, [1])
            if self.do_cache_feat and task_history_cache_feat is None:
                # 填充的history_cache_feat, 其relative-homography_temporal需要变成单位阵
                homography_temporal[:, 0] = (
                    torch.eye(3)
                    .view(1, 3, 3)
                    .to(homography_temporal)
                    .repeat(homography_temporal.shape[0], 1, 1)
                )

            # 下面开始逐帧融合了
            hidden_feats = encoded_frames[0]
            fusion_data = []
            for frame_idx in range(1, len(encoded_frames)):
                cur_feat = encoded_frames[frame_idx]
                warped_hidden_feats = self._temporal_warp(
                    hidden_feats,
                    homography_temporal[:, frame_idx - 1],
                    homo_offset_temporal,
                )
                (
                    cur_fusion_data,
                    hidden_feats,
                ) = self.fusion_module.forward_once(
                    cur_feat, warped_hidden_feats
                )
                fusion_data.append(cur_fusion_data)

            # Note: not support GRU at the moment. @Jinqian.Gao
            cur_fusion_data = self.quant(cur_fusion_data)
            if not only_return_latest_feature:
                fusion_data = [f_data.unsqueeze(1) for f_data in fusion_data]
                _, _, c, h, w = fusion_data[0].shape
                fusion_data = self.fusion_cat.cat(fusion_data, dim=1).reshape(
                    (-1, c, h, w)
                )
            else:
                fusion_data = fusion_data[-1]

        if self.do_cache_feat:
            # 需要缓存当前最后一帧的hidden_feats，用于下一个sub-clip.
            if isinstance(hidden_feats, List):
                self.history_cache_feat[task_name] = [
                    tmp.detach() for tmp in hidden_feats
                ]
            else:
                self.history_cache_feat[task_name] = hidden_feats.detach()

        # convert the hidden feats to tensor format.
        hidden_feats = self.fusion_module.to_dumped_feat(hidden_feats)
        return fusion_data, hidden_feats


@OBJECT_REGISTRY.register
class ANCLidarRecurrentTemporalModule(ANCBEVFusionRecurrentTemporalModule):
    r"""Recurrent version of temporal fusion module for LiDAR input feature.

    Different from original module for BEVFusion feature,
    the first frame LiDAR feature is warped to the current timestamp additionally,
    which is specially designed for the case without LiDAR input in the current frame.

    Training and not continuous frame testing data flow is showed below.

    .. code-block:: none

        *  bev_feat(t-2) -> temporal_warp() -> align_feat
        *                        _ _ _ _ _ _ _ _ _ _|
        *                       |
        *                      \|/
        *  bev_feat(t-1) -> fusion_module() -> temporal_warp() -> align_feat
        *                                                        _ _|
        *                                                       |
        *                                                      \|/
        *  bev_feat(t) --------------------------------> fusion_module() -> fused_feat

    suppose frame t has lidar input, frame t+1 and frame t+2 has not lidar
    input, and frame t+3 has lidar input.
    Continuous frame testing data flow is showed below.

    .. code-block:: none

        *     history_cache_lidar ----> temporal_warp() ----> align_feat
        *                                      _ _ _ _ _ _ _ _ _ _|
        *                                     |
        *                                    \|/                         (update)
        *  t:   bev_feat(t) ---------> fusion_module() ----> fused_feat----------->history_cache_lidar  # noqa
        *           |                                             _ _ _ _ _ _ _ __ _ _ _ _ _|
        *           |                                           \|/                         |
        *           |(update)                             temporal_warp()                   |
        *           |                                            |                          |
        *          \|/                                          \|/                         |
        *  t+1: current_cache -------> temporal_warp() -> fusion_module() -> fused_feat     |
        *           |                                            _ _ _ _ _ _ _ _ _ _ _ _ _ _|
        *           |                                          \|/                          |
        *           |                                      temporal_warp()                  |
        *           |                                           |                           |
        *           |                                          \|/                          |
        *  t+2:     -----------------> temporal_warp() -> fusion_module() -> fused_feat     |
        *                                                        _ _ _ _ _ _ _ _ _ _ _ _ _ _|
        *                                                      \|/
        *                                                  temporal_warp()
        *                                                       |
        *                                                      \|/                      (update)
        *  t+3: bev_feat(t+3) --------------------------> fusion_module() -> fused_feat----------->history_cache_lidar  # noqa

    Args:
        share_conv_in_channels: share conv's input channels.
        share_conv_out_channels:share conv's output channels.
        bn_kwargs: share_conv's BatchNorm2d kwargs.
        continuous_frame_test : if use continuous frame test mode.
        temporal_fusion_idxs: temporal indexes to fuse.
    """

    def __init__(
        self,
        share_conv_in_channels: int,
        share_conv_out_channels: int,
        bn_kwargs: dict,
        continuous_frame_test: bool = False,
        temporal_fusion_idxs: tuple = (0, 1),
        **kwargs,
    ):
        super(ANCLidarRecurrentTemporalModule, self).__init__(
            continuous_frame_test, **kwargs
        )
        assert temporal_fusion_idxs[0] == 0
        self.temporal_fusion_idxs = temporal_fusion_idxs
        """
        在训练阶段，我们需要通过temporal_fusion_idxs和temporal_length_each_batch两个参数来
        设置实际的时序融合行为。具体的来说，'接收时序数据的长度' 与'实际时序融合的长度' 可以是不同的。
        比如temporal_length_each_batch=3，并且temporal_fusion_idxs=(0,2)，
        这表示时序融合模块接收到了3帧时序数据，但是只融合index为0,2的两帧。
        这可以使得我们在训练时候的融合逻辑更加灵活。
        """
        self.imitate_recurrent = (
            True if len(self.temporal_fusion_idxs) == 2 else False
        )

        self.history_cache_lidar = None
        self.current_cache = None
        self.continuous_frame_test = continuous_frame_test
        self.share_conv = ConvModule2d(
            share_conv_in_channels,
            share_conv_out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            norm_layer=nn.BatchNorm2d(share_conv_out_channels, **bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.floatmod_mulscalar = FloatFunctional()

    def _temporal_warp(
        self,
        pre_frames_data: Sequence,
        homography_temporal: Optional[Sequence],
        homo_offset_temporal: Optional[Sequence],
    ):
        """Warp previous frame datas to current frame.

        Args:
            pre_frames_data: previous frame data to warp.
            homography_temporal: homography matrix
                from current frame to previous, shape is
                (temporal_length,3,3). (Used in train/val)
            homo_offset_temporal: homograpy offset
                from current frame to previous. (Used in compiling model)

        """

        cur_frames_warped = []
        if self.compile_model:
            # For compile, current frame lidar feature is also warped.
            assert len(pre_frames_data) == len(homo_offset_temporal) == 2

            cur_frame = pre_frames_data[0]  # already QTensor
            cur_frames_warped.append(
                self.temporal_st(cur_frame, homo_offset_temporal[0])
            )

            pre_frame = self.quant(pre_frames_data[1][0])  # hgx
            # from hat.utils.forkedpdb import set_trace
            # set_trace()
            assert pre_frame.shape[0] == self.infer_batch_factor
            # Warp each hidden feature recurrently.
            for j in range(self.infer_batch_factor):
                hidden_state = pre_frame[j : j + 1]
                cur_frames_warped.append(
                    self.temporal_st(hidden_state, homo_offset_temporal[1])
                )
        else:
            cur_frames_warped.append(
                self.temporal_st(pre_frames_data, homography_temporal)[0]
            )
        return cur_frames_warped

    def _split_temporal_data(
        self,
        all_frames_data: Sequence,
        homography_temporal=None,
        homo_offset_temporal=None,
    ):
        """Split temporal data into two parts.

        Durning train/val process, all_frames_data is a tensor and it`s shape
            is (b*t,c,h,w), t is temporal_length.
        When compiling model, all_frames_data is a list of tensor and
            each tensor`s shape is (b,c,h,w).
        """
        if self.compile_model:
            cur_frame_data, pre_frames_data = (
                all_frames_data[0],
                all_frames_data[1:],
            )
            return (
                cur_frame_data,
                pre_frames_data,
                homography_temporal,
                homo_offset_temporal,
            )

        else:
            b, _, _, _ = all_frames_data.shape
            all_frames_data = [
                select_sample(
                    all_frames_data,
                    list(range(idx, b, self.temporal_length_each_batch)),
                )
                for idx in self.temporal_fusion_idxs
            ]
            return (
                all_frames_data[0],
                all_frames_data[1:],
                homography_temporal,
                None,
            )

    def forward(
        self,
        all_frames_data,
        homography_temporal=None,
        homo_offset_temporal=None,
        is_lidar_input=True,
    ):
        # fusion share conv
        if self.compile_model:
            for i in range(len(all_frames_data)):
                all_frames_data[i] = self.share_conv(all_frames_data[i])
        else:
            all_frames_data = self.share_conv(all_frames_data)
        (
            cur_frame_data,
            pre_frames_data,
            homography_temporal,
            homo_offset_temporal,
        ) = self._split_temporal_data(
            all_frames_data, homography_temporal, homo_offset_temporal
        )

        cur_encoded_frame = self.extra_encoder(cur_frame_data)
        if self.imitate_recurrent:
            cur_encoded_frame = self.quant(cur_encoded_frame)
        if self.compile_model:
            # Both current and previous frames should be warped.
            cur_frames_warped = self._temporal_warp(
                [cur_encoded_frame, pre_frames_data],
                homography_temporal,
                homo_offset_temporal,
            )
            cur_encoded_frame = cur_frames_warped[0]
            pre_encoded_frame = cur_frames_warped[1:]
            # For inference, hidden features should be parsed to
            # the needed format for each fusion module.
            hidden_feats = self.fusion_module.parse_pre_feats(
                pre_encoded_frame
            )
            fusion_data, hidden_feats = self.fusion_module.forward_once(
                cur_encoded_frame, hidden_feats
            )
        else:
            if not self.continuous_frame_test:
                # Perform an extra warping for the first frame.
                cur_encoded_frame = self.temporal_st(
                    cur_encoded_frame, homography_temporal[:, 0]
                )[0]
                # For training, each frame should be encoded.
                pre_encoded_frame = [
                    self.extra_encoder(data) for data in pre_frames_data
                ][::-1]
                encoded_frames = pre_encoded_frame + [cur_encoded_frame]

                homography_temporal = torch.flip(homography_temporal, [1])

                # optional (remove the last homography)
                homography_temporal = homography_temporal[:, :-1]

                hidden_feats = encoded_frames[0]
                for frame_idx in range(1, len(encoded_frames)):
                    cur_feat = encoded_frames[frame_idx]
                    warped_hidden_feats = self._temporal_warp(
                        hidden_feats,
                        homography_temporal[:, frame_idx - 1],
                        homo_offset_temporal,
                    )[0]
                    (
                        fusion_data,
                        hidden_feats,
                    ) = self.fusion_module.forward_once(
                        cur_feat, warped_hidden_feats
                    )
                    if not self.imitate_recurrent:
                        fusion_data = self.quant(fusion_data)
            else:
                # For continuous frame testing

                # During continuous frame testing, only one history_cache_lidar is maintained. # noqa
                # The current frame and history_cache_lidar are used as inputs to enter forward_once, output fusion_data. # noqa
                # Update the history_cache_lidar with fusion_data if is_lidar_input. # noqa
                if is_lidar_input:
                    # update current_cache
                    self.current_cache = cur_encoded_frame
                if self.history_cache_lidar is not None:
                    pre_encoded_frame = [self.history_cache_lidar]
                else:
                    pre_encoded_frame = [
                        self.floatmod_mulscalar.mul_scalar(
                            pre_frames_data[0], 0
                        )
                    ]
                encoded_frames = pre_encoded_frame + [cur_encoded_frame]
                homography_temporal = torch.flip(homography_temporal, [1])
                homography_temporal = homography_temporal[:, :-1]
                hidden_feats = encoded_frames[0]
                if is_lidar_input:
                    cur_feat = encoded_frames[1]
                    warped_hidden_feats = self._temporal_warp(
                        hidden_feats,
                        homography_temporal[:, -1],
                        homo_offset_temporal,
                    )[0]
                    (
                        fusion_data,
                        hidden_feats,
                    ) = self.fusion_module.forward_once(
                        cur_feat, warped_hidden_feats
                    )
                    if not self.imitate_recurrent:
                        fusion_data = self.quant(fusion_data)
                    self.history_cache_lidar = fusion_data
                else:
                    if self.current_cache is not None:
                        cur_feat = self.current_cache
                    else:
                        cur_feat = encoded_frames[1]
                    warped_hidden_feats = self._temporal_warp(
                        hidden_feats,
                        homography_temporal[:, -1],
                        homo_offset_temporal,
                    )[0]
                    warped_cur_feats = self._temporal_warp(
                        cur_feat,
                        homography_temporal[:, -1],
                        homo_offset_temporal,
                    )[0]
                    (
                        fusion_data,
                        hidden_feats,
                    ) = self.fusion_module.forward_once(
                        warped_cur_feats, warped_hidden_feats
                    )
                    if not self.imitate_recurrent:
                        fusion_data = self.quant(fusion_data)
                    # update history_cache_lidar

        # convert the hidden feats to tensor format.
        hidden_feats = self.fusion_module.to_dumped_feat(hidden_feats)

        if self.imitate_recurrent:
            return fusion_data, self.dequant(cur_encoded_frame)
        else:
            return fusion_data, self.dequant(hidden_feats)

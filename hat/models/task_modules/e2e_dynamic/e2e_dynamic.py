import logging
from typing import Dict, List

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.qtensor import QTensor as QTensorType
from horizon_plugin_pytorch.quantization import (
    FixedScaleObserver,
    QuantStub,
    get_default_qat_qconfig,
)
from torch import Tensor, nn
from torch.quantization import DeQuantStub

from hat.core.affine import get_vcs2bev_img_mat
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (  # noqa: E501
    FPS_MAX,
    INV_SIGMOID_MAX,
    MLP,
    QINT16_MAX,
    SIGMOID_MAX,
    SPEED_MAX,
    Embedding,
    Linear,
    _get_clones,
    dict_select_keep_dim,
    get_lcf_odo_info,
    inverse_sigmoid,
    kalerman_filter,
    pad_tensors_to_target_shape,
    qat_out_qconfig,
    qint8_qconfig,
    qint16_qconfig,
    update_his_coord,
    update_his_velo,
)
from hat.models.utils import _take_features
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["E2EDynamicModule"]

BACKGROUND_OBJ_IDS = -2


@OBJECT_REGISTRY.register
class E2EDynamicModule(nn.Module):
    """End to end task head.

    An end to end head contains a transformer and multiple subtasks,
    e.g. detection, tracking, velocity and trajectory prediction...

    We define some abbreviations that are frequently used in following
    head, such as lcf: last context frame, odo: odometry, mem: memory,
    traj: trajectory, his: history, pred: prediction, ref: reference ...

    Args:
        transformer: transformer module.
        in_strides: A list contains the strides of feature maps
            from backbone or neck.
        out_strides: A list contains the strides of feature maps
            used in this head.
        decoder_strides: A list contains the strides of feature
            maps used in transformer decoder.
        num_channels: channels of different stride features.
        odometry_channels: channels of odometry features.
        wlh_anchor_size: box wlh anchor size, e.g.{
                "car":[w_0, l_0, h_0],

            }

        bev_size: bev map size, order (h, w).
        vcs_range: visbile range of bev, (bottom, right, top, left) in order.
        default_time_delta: default time delta between frames for qconfig
            setting and the first frame of every clip, unit is seconds.

            .. note:: 目前仅只支持单一帧率的数据集,
                若需支持多个帧率的数据集还需适配.

        subtask_head: dict of output embedding, e.g. class_embed,
            boxes_embed, yaw_embed, zheight_embed.
        extra_subtask_head: dict of extra output embedding, e.g.
            velocity_embed, K_embed, trajpred_embed, trajpred_logits.
        criterion: loss module.
        query_interaction_module: qim module for tracking task.
        memory_bank_module: memory bank module for query's
            temporal and spatial attention.
        track_manager: track manager module in eval mode, used to
            control track start and end in inference.
        feature_name: feature name of the input.
        post_process: post-process module in eval mode, used to convert
            format of output.
        use_auxiliary_loss: whether use auxiliary loss to supervise
            model training.
        use_box_refine: whether to refine box in multi-decoder layer.
        e2e_query_kwargs: parameters for generating queries, contains.

            .. code-block:: none

                num_veh_queries: number of vehicle detection queries.

                num_vru_queries: number of VRU detection queries.

                num_track_queries: number of track queries.

                query_drop_ratio: drop ratio of queries, used to randomly
                    drop some active track queries in training.

                query_fp_ratio: false positive ratio of queries, used to
                    randomly generate some inactive track queries in training.

                num_max_queries: maximum number of queries, used for query
                    quantization scale, usually bigger than actual query number
                    to avoid quantization overflow.

        calibration_model: whether in "calibration" stage.
        compile_model: whether to compile model.
        e2e_out_velocity: whether to output velocity.
        e2e_out_trajectory: whether to output trajectory.
        trajectory_kwargs: parameters for trajectory prediction, contains.

            .. code-block:: none

                num_traj_modal: number of trajectory modal, a modal means
                    a possibility of trajectory prediction, default: 5.
                num_frames_for_pred: number of frames for trajectory
                    prediction, default 60, means 6 seconds in 10 fps data.

        init_query_feat: whether initialize the query. default False
            if True, initial query feat as Embedding.
            if False, generate query feat from bev feat.

            .. note:: param will be remove when all the exp convert to
                init_query_feat=False.

        eval_pure_det: whether eval in pure detection mode. The pure detection
            mode should only use in tasks except detection. It differs from the
            detection task in query num passed into transformer. Pure detection
            mode passes detection queries and complementary track queries into
            transformer, however, detection task only passes detection queries.
    """

    output_names = [
        "bbox_embed_xy",
        "bbox_embed_wh",
        "class_embed",
        "yaw_embed",
        "zheight_embed",
    ]
    extra_output_names = [
        "velocity_embed",
        "K_embed",
        "trajpred_embed",
        "trajpred_logits",
    ]

    def __init__(
        self,
        transformer: nn.Module,
        in_strides: List[int],
        out_strides: List[int],
        decoder_strides: List[int],
        num_channels: List[int],
        odometry_channels: List[int],
        wlh_anchor_size: dict,
        bev_size: List[int],
        vcs_range: List[float],
        default_time_delta: float,
        subtask_head: Dict,
        extra_subtask_head: Dict,
        criterion: nn.Module = None,
        query_interaction_module: nn.Module = None,
        memory_bank_module: nn.Module = None,
        track_manager: nn.Module = None,
        feature_name: str = "bev_stage2_feats",
        post_process: nn.Module = None,
        use_auxiliary_loss: bool = True,
        use_box_refine: bool = False,
        e2e_query_kwargs: dict = None,
        calibration_model: bool = False,
        compile_model: bool = False,
        e2e_out_velocity: bool = False,
        e2e_out_trajectory: bool = False,
        trajectory_kwargs: dict = None,
        init_query_feat: bool = False,
        eval_pure_det: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_box_refine = use_box_refine
        self.vcs_range = vcs_range
        self.bev_size = bev_size
        self.transformer = transformer  # transform head
        self.criterion = criterion  # loss, contains the matcher
        self.query_interaction_module = query_interaction_module  # qim module
        self.memory_bank_module = memory_bank_module  # memory bank module
        self.post_process = post_process
        self.track_manager = track_manager
        self.init_query_feat = init_query_feat
        self.eval_pure_det = eval_pure_det
        for name in E2EDynamicModule.output_names:
            assert name in subtask_head
        # for velocity or trajectory task.
        if len(extra_subtask_head) > 0:
            for name, _ in extra_subtask_head.items():
                assert name in E2EDynamicModule.extra_output_names
        self.subtask_head = subtask_head
        self.e2e_out_velocity = e2e_out_velocity
        self.e2e_out_trajectory = e2e_out_trajectory
        # flatten for forward efficiency
        self.wlh_anchor_size = []
        for key in wlh_anchor_size.keys():
            self.wlh_anchor_size += wlh_anchor_size[key]
        self.wlh_anchor_size = torch.tensor(self.wlh_anchor_size).reshape(
            1, 1, 3, 3
        )
        self.vcs2bev = torch.from_numpy(
            get_vcs2bev_img_mat(vcs_range, bev_size)
        ).float()
        self.vcs2bev_norm = torch.from_numpy(
            get_vcs2bev_img_mat(vcs_range, [1, 1])
        ).float()

        # module parameter for track_instance
        self.num_det_queries = (
            e2e_query_kwargs["num_veh_queries"]
            + e2e_query_kwargs["num_vru_queries"]
        )
        self.num_veh_queries = e2e_query_kwargs["num_veh_queries"]
        self.num_vru_queries = e2e_query_kwargs["num_vru_queries"]
        self.veh_filter_scores = e2e_query_kwargs["veh_filter_scores"]
        self.vru_filter_scores = e2e_query_kwargs["vru_filter_scores"]
        self.num_track_queries = e2e_query_kwargs["num_track_queries"]
        self.query_drop_ratio = e2e_query_kwargs["query_drop_ratio"]
        self.query_fp_ratio = e2e_query_kwargs["query_fp_ratio"]
        # fp ref pts noise setting
        self.fp_rand_ref_pts_value = e2e_query_kwargs["fp_rand_ref_pts_value"]
        self.fp_ref_pts_bias = e2e_query_kwargs["fp_ref_pts_bias"]

        # transformer parameters
        hidden_dim = transformer.embedding_dim
        # features from input
        self.feature_name = feature_name
        self.in_strides = in_strides
        assert len(out_strides) == len(
            num_channels
        ), "num_channels should be equal to the num_out_strides."
        self.out_strides = out_strides
        self.decoder_strides = decoder_strides
        num_backbone_outs = len(in_strides)
        num_feature_levels = len(out_strides)
        assert (
            num_backbone_outs >= num_feature_levels
        ), "Extra feature pyramid is not supported now."
        assert (
            num_feature_levels > 1
        ), "Only multi-level feature pyramid is supported now."

        self._build_anchor_layer(self.num_channels[0], hidden_dim)
        self.dict_active_mask_cat = FF()
        self.dict_qim_embd_cat = FF()
        self.last_timestamp = None
        self.his_time_delta_queue = None
        if self.memory_bank_module:
            self.memory_bank_len = self.memory_bank_module.memory_bank_len
            # odometry quant used in memory bank.
            self.odo_input_mb_quant = QuantStub(None)
            # odometry quant used in state embedding.
            self.mem_bank_quant = QuantStub(None)
            self.mem_padding_mask_quant = QuantStub(1.0)
            self.output_emb_update_dquant = DeQuantStub()
            self.output_emb_dquant = DeQuantStub()
            self.dict_qim_embed_for_mem_cat = FF()
            self.dict_qim_mem_bank_cat = FF()

            self.dict_mem_bank_cat = FF()

        self.calibration_model = calibration_model
        self.compile_model = compile_model

        # 构造query范围空间，用于warp时计算offset, 由于 warp 只支持二维，我们初始化一个二维的空间
        # 尺寸为 (num_max_queries, 2)
        num_max_queries = e2e_query_kwargs["num_max_queries"]
        # 在 x 方向上，我们不需要偏移，所以初始化为 0
        x_range = torch.zeros((num_max_queries, 1), dtype=torch.float32)
        # 在 y 方向上，我们需要偏移，所以初始化为 [0, 1, 2, ..., num_max_queries - 1]
        y_range = np.linspace(
            0, num_max_queries, num_max_queries, endpoint=False
        )
        # 乘以-1, 后续计算offset时，就可以直接相加
        y_range = torch.tensor(y_range, dtype=torch.float32).unsqueeze(1) * (
            -1
        )
        xy_range = torch.cat([y_range, x_range], dim=1).view(
            1, 1, num_max_queries, 2
        )
        self.xy_range = nn.Parameter(xy_range, requires_grad=False)
        self.xy_range_quant = QuantStub(num_max_queries / QINT16_MAX)

        self.input_proj = nn.ModuleList()
        for s in self.decoder_strides:
            in_channels = num_channels[out_strides.index(s)]
            self.input_proj.append(
                ConvModule2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    norm_layer=nn.BatchNorm2d(hidden_dim),
                )
            )

        # 构造各个子任务的mlp层
        num_decoder_layer = transformer.decoder.num_layers
        for out_name, module in subtask_head.items():
            if use_box_refine:
                subtask_module = _get_clones(module, num_decoder_layer)
            else:
                # each level share the same head.
                subtask_module = nn.ModuleList(
                    [module for _ in range(num_decoder_layer)]
                )
            setattr(self, out_name, subtask_module)
            if "bbox_embed" in out_name:
                setattr(self.transformer.decoder, out_name, subtask_module)
        for out_name, module in extra_subtask_head.items():
            setattr(self, out_name, module)
        self.pred_boxes_xy_dequant_list = nn.ModuleList()
        self.pred_boxes_wh_dequant_list = nn.ModuleList()
        self.pred_logits_dequants = nn.ModuleList()
        self.pred_yaws_dequants = nn.ModuleList()
        self.pred_zheights_dequants = nn.ModuleList()
        for _ in range(num_decoder_layer):
            self.pred_boxes_xy_dequant_list.append(DeQuantStub())
            self.pred_boxes_wh_dequant_list.append(DeQuantStub())
            self.pred_logits_dequants.append(DeQuantStub())
            self.pred_yaws_dequants.append(DeQuantStub())
            self.pred_zheights_dequants.append(DeQuantStub())

        self.last_track_instances = None
        self.ref_pts_quant = QuantStub(INV_SIGMOID_MAX / QINT16_MAX)
        self.output_embedding_dequant = DeQuantStub()

        self.track_query_quant = QuantStub()
        self.query_embed_dequant = DeQuantStub()
        self.query_embed_cat = FF()
        self.track_output_embedding_quant = QuantStub()

        # all mask use the same quant op
        self.active_mask_quant = QuantStub(1.0)
        self.active_mask_dequant = DeQuantStub()

        self.calib_tmp_quant = QuantStub()
        self.track_ref_pts_sigmoid = nn.Sigmoid()

        self.feat_quant = nn.ModuleList()
        for _ in range(num_feature_levels):
            self.feat_quant.append(QuantStub())

        self.default_time_delta = default_time_delta
        self.trk_query_pos_cat = FF()
        self.time_mask_mul = FF()

        if self.e2e_out_velocity or self.e2e_out_trajectory:
            self.fps_mul = FF()
            # lcf: last context frame (current frame).
            self.lcf_his_xy_cat = FF()
            self.lcf_x_muls = FF()
            self.lcf_x_adds = FF()
            self.lcf_y_muls = FF()
            self.lcf_y_adds = FF()
            self.lcf_xy_cat = FF()
            self.his_coord_sub = FF()
            self.memory_coord_quant = QuantStub(None)
            assert (
                self.default_time_delta >= 1.0 / FPS_MAX
            ), f"fps_quant support {FPS_MAX} or below, please set default_time_delta >= {1. / FPS_MAX}, now {self.default_time_delta}"  # noqa
            self.fps_quant = QuantStub(scale=FPS_MAX / QINT16_MAX)
            self.boxes_sigmoid = nn.Sigmoid()
            self.all_query_mask = torch.nn.Parameter(
                torch.ones(
                    (1, 1, (self.num_det_queries + self.num_track_queries), 1),
                ),
                requires_grad=False,
            )
            self.memory_padding_mask_cat = FF()

            self.his_velo_mask_mul = FF()
            self.odometry_channels = odometry_channels
            self.trajpred_odo_proj = MLP(
                self.odometry_channels[0],
                self.odometry_channels[1],
                self.odometry_channels[2],
                2,
            )
            # encode history position difference to predict velocity and
            # future position, self.memory_bank_len is the length of history
            # position, thus channel length is self.memory_bank_len - 1.
            self.trajpred_encode = MLP(
                self.odometry_channels[2] * (self.memory_bank_len - 1),
                self.odometry_channels[2] * (self.memory_bank_len - 1) // 2,
                self.odometry_channels[2],
                2,
            )
            self.embedding_cat = FF()

        if self.e2e_out_velocity:
            self.velocities_dequant = DeQuantStub()
            self.qmat_dquant = DeQuantStub()
        if self.e2e_out_trajectory:
            self.num_traj_modal = trajectory_kwargs["num_traj_modal"]
            self.trajpred_length = trajectory_kwargs["num_frames_for_pred"]
            self.modal_embeding = Embedding(
                (
                    1,
                    self.num_traj_modal,
                    1,
                    hidden_dim + self.odometry_channels[2],
                )
            )
            self.embedding_add = FF()
            self.traj_encode_dequant = DeQuantStub()
            self.traj_logits_dequant = DeQuantStub()

        if self.init_query_feat:
            self.object_query = Embedding(
                (
                    1,
                    1,
                    self.num_det_queries,
                    hidden_dim,
                )
            )

    def _build_anchor_layer(self, in_dim, embed_dim):
        self.max_bev_size = max(self.bev_size)
        # 用于计算warp offset 的 scale
        self.heatmap_grid_quant = QuantStub(self.max_bev_size / QINT16_MAX)
        self.generate_pos_embed = Linear(2, embed_dim)
        self.heatmap_cat = FF()
        self.coord_mul1 = FF()
        self.coord_mul2 = FF()
        self.coord_cat = FF()
        self.bev_norm_coef = [
            float(1 / self.bev_size[0]),
            float(1 / self.bev_size[1]),
        ]
        if not self.init_query_feat:
            # generate query feat from stride0 feat by gridsample
            self.heatmap_add_offsets = FF()
            self.heatmap_grid_sample = hnn.GridSample(
                mode="bilinear", padding_mode="zeros"
            )
            self.generate_query_embed = Linear(in_dim, embed_dim)

    def _init_instances(self, num_det_queries, device, dim=256) -> dict:
        """Generate common empty instance."""
        empty_instance = {
            "ref_pts": torch.zeros(
                (1, 1, num_det_queries, 2), dtype=torch.float, device=device
            ),
            "query_embed": torch.zeros(
                (1, 1, num_det_queries, dim * 2),
                dtype=torch.float,
                device=device,
            ),
            "output_embedding": self.track_output_embedding_quant(
                torch.zeros(
                    (1, 1, num_det_queries, dim),
                    dtype=torch.float,
                    device=device,
                )
            ),
            "obj_idxes": torch.full(
                (num_det_queries,), -1, dtype=torch.long, device=device
            ),
            "matched_gt_idxes": torch.full(
                (num_det_queries,), -1, dtype=torch.long, device=device
            ),
            "appear_time": torch.zeros(
                (num_det_queries,), dtype=torch.long, device=device
            ),
            "disappear_time": torch.zeros(
                (num_det_queries,), dtype=torch.long, device=device
            ),
            "scores": torch.zeros(
                (num_det_queries,), dtype=torch.float, device=device
            ),
            "pred_boxes": torch.zeros(
                (num_det_queries, 4), dtype=torch.float, device=device
            ),
            "save_period": torch.zeros(
                (num_det_queries,), dtype=torch.float, device=device
            ),
            "active_mask": torch.ones(
                (1, 1, num_det_queries, 1), dtype=torch.float, device=device
            ),
            "pred_velo_for_ref_pts": torch.zeros(
                (1, 1, num_det_queries, 3), dtype=torch.float, device=device
            ),
        }
        if self.memory_bank_module is not None:
            empty_instance.update(
                {
                    "mem_bank": self.mem_bank_quant(
                        torch.zeros(
                            (1, self.memory_bank_len, num_det_queries, dim),
                            dtype=torch.float32,
                            device=device,
                        )
                    ),
                    "mem_padding_mask": torch.zeros(
                        (1, self.memory_bank_len, num_det_queries, 1),
                        dtype=torch.float32,
                        device=device,
                    ),
                    "mem_coord": torch.zeros(
                        (1, self.memory_bank_len, num_det_queries, 4),
                        dtype=torch.float32,
                        device=device,
                    ),
                    # TODO: remove z velocity
                    "mem_velo": torch.zeros(
                        (1, self.memory_bank_len, num_det_queries, 3),
                        dtype=torch.float32,
                        device=device,
                    ),
                    # TODO: update pred_accelerations to new version
                    "mem_acce": torch.zeros(
                        (num_det_queries, 2), dtype=torch.float, device=device
                    ),
                }
            )
        return empty_instance

    def _random_drop_instances(
        self, track_instances: dict, drop_probability: float
    ):
        assert (
            "scores" in track_instances
        ), "`scores` have to be one of the key in the tracking_instances"
        if drop_probability > 0 and len(track_instances["scores"]) > 0:
            keep_idxes = (
                torch.rand_like(track_instances["scores"]) > drop_probability
            )
            track_instances = dict_select_keep_dim(track_instances, keep_idxes)
        return track_instances

    def _add_fp_instances(
        self, track_instances: dict, active_track_instances: dict
    ):
        # add fp for each active track in a specific probability.
        fp_prob = (
            torch.ones_like(active_track_instances["scores"])
            * self.query_fp_ratio
        )
        selected_active_track_instances = dict_select_keep_dim(
            active_track_instances,
            torch.bernoulli(fp_prob).bool(),
        )

        num_active_tracks = len(active_track_instances)
        num_fp_tracks = min(
            len(selected_active_track_instances["scores"]),
            self.num_track_queries - num_active_tracks,
        )
        if num_fp_tracks > 0:
            fp_track_instances = selected_active_track_instances
            rand_noise = (
                torch.randn_like(fp_track_instances["ref_pts"])
                * self.fp_rand_ref_pts_value
            )
            fp_track_instances["ref_pts"] = inverse_sigmoid(
                fp_track_instances["ref_pts"].sigmoid()
                + rand_noise
                + torch.sign(rand_noise) * self.fp_ref_pts_bias
            )
            fp_ref_pts = self.ref_pts_quant(fp_track_instances["ref_pts"])
            fp_pos_embed = self.query_embed_dequant(
                self.generate_pos_embed(self.track_ref_pts_sigmoid(fp_ref_pts))
            )
            fp_track_instances["query_embed"] = torch.cat(
                (
                    fp_pos_embed,
                    fp_track_instances["query_embed"][..., 256:],
                ),
                dim=-1,
            )
            fp_track_instances[
                "obj_idxes"
            ] = BACKGROUND_OBJ_IDS * torch.ones_like(
                fp_track_instances["obj_idxes"]
            )

            merged_track_instances = self._concat_instances(
                active_track_instances,
                fp_track_instances,
            )
            return merged_track_instances
        return active_track_instances

    def generate_det_instances(self, feat=None, anchor=None, xy_range=None):
        """Genrate det_instances for train/val."""
        device = xy_range.device
        det_instances = self._init_instances(self.num_det_queries, device)
        query_embed = self._generate_query_from_anchor(feat, anchor, xy_range)
        ref_pts = anchor["det_ref_pts"]
        det_instances["active_mask"] = anchor["det_active_mask"]
        det_instances["query_embed"] = query_embed
        det_instances["ref_pts"] = ref_pts
        return det_instances

    def _concat_instances(self, first_instances, second_instances):
        if second_instances is None:
            return first_instances
        key_union = set(first_instances.keys()) | set(second_instances.keys())
        for k in key_union:
            if k not in first_instances or k not in second_instances:
                continue
            # track instance 中存在两种典型维度，两维和四维，其常用含义如下：
            # dim==4: [batch_size, temporal_length, num_queries, feat_channel]
            # dim==2: [num_queries, feat_channel]
            # 两种维度的cat方式不同，需要分别处理
            if first_instances[k].dim() == 4:
                cat_dim = 2
            else:
                cat_dim = 0
            if isinstance(first_instances[k], QTensorType):
                assert first_instances[k].dim() == 4
                if "query_embed" in k:
                    second_instances[k] = self.query_embed_cat.cat(
                        [first_instances[k], second_instances[k]], dim=cat_dim
                    )
                elif "output_embedding_for_update" in k:
                    second_instances[k] = self.dict_qim_embed_for_mem_cat.cat(
                        [first_instances[k], second_instances[k]], dim=cat_dim
                    )
                elif "active_mask" in k:
                    second_instances[k] = self.dict_active_mask_cat.cat(
                        [first_instances[k], second_instances[k]], dim=cat_dim
                    )
                elif "mem_bank" in k:
                    second_instances[k] = self.dict_qim_mem_bank_cat.cat(
                        [first_instances[k], second_instances[k]], dim=cat_dim
                    )
                elif "output_embedding" == k:
                    second_instances[k] = self.dict_qim_embd_cat.cat(
                        [first_instances[k], second_instances[k]], dim=cat_dim
                    )
                else:
                    logger.error("key not found!")
            else:
                second_instances[k] = torch.cat(
                    (first_instances[k], second_instances[k]), dim=cat_dim
                )
        return second_instances

    def clear(self):
        if self.track_manager:
            self.track_manager.clear()
        self.last_track_instances = None
        self.his_time_delta_queue = None
        self.last_timestamp = None

    @torch.jit.unused
    def _set_auxiliary_loss(
        self,
        outputs_class,
        outputs_coord,
        outputs_yaw=None,
        outputs_zheight=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        out_list = []
        for idx in range(len(outputs_class) - 1):
            pred_dict = {
                "pred_logits": outputs_class[idx],
                "pred_boxes": outputs_coord[idx],
                "pred_yaws": self.pred_yaws_dequants[idx](outputs_yaw[idx]),
                "pred_zheights": outputs_zheight[idx],
            }
            out_list.append(pred_dict)
        return out_list

    def _select_topk_instances(self, track_instances: dict, target_num):
        with torch.no_grad():
            scores = track_instances["scores"]
            _, indices_query = torch.topk(scores, target_num)
            mask = torch.zeros_like(scores)
            mask = torch.scatter(mask, 0, indices_query, 1)
        track_instances = dict_select_keep_dim(track_instances, mask > 0.5)
        return track_instances

    def _process_odo_info(self, odo_info, time_delta):
        """Process odometry information to get odo_input_mb.

        Returns:
            odo_input_mb: odo_velo and odo_yaw_rate of the past memory_bank_len
                frame and current frame, shape=[1, 11, 1, 4].
        """

        odo_xy = odo_info[:, :2]
        odo_yaw = odo_info[:, 2]
        odo_vel = odo_xy[1:] - odo_xy[:-1]
        odo_yawrate = odo_yaw[1:] - odo_yaw[:-1]
        odo_input_mb = torch.cat(
            (
                odo_vel,
                torch.sin(odo_yawrate).unsqueeze(1),
                torch.cos(odo_yawrate).unsqueeze(1),
            ),
            dim=1,
        )
        odo_input_mb = odo_input_mb.float().unsqueeze(0).unsqueeze(2)
        odo_input_mb = odo_input_mb / time_delta

        return odo_input_mb  # [1, 11, 1, 4]

    def _update_memory_bank_use_kalerman(
        self, track_instances, pred_boxes, pred_velo, pred_acce
    ):
        embed = track_instances.pop("output_embedding_for_update")
        track_instances["mem_bank"] = self.dict_mem_bank_cat.cat(
            [track_instances["mem_bank"][:, 1:, ...], embed], dim=1
        )

        track_instances["mem_padding_mask"][:, :-1, ...] = track_instances[
            "mem_padding_mask"
        ][:, 1:, ...].clone()
        track_instances["mem_padding_mask"][:, -1, ...] = torch.ones_like(
            track_instances["mem_padding_mask"][:, -1, ...]
        )

        track_instances["mem_coord"][:, :-1, ...] = track_instances[
            "mem_coord"
        ][:, 1:, ...].clone()
        track_instances["mem_coord"][:, -1, ...] = pred_boxes.detach().clone()

        track_instances["mem_velo"][:, :-1, ...] = track_instances["mem_velo"][
            :, 1:, ...
        ].clone()
        track_instances["mem_velo"][:, -1, ...] = pred_velo.detach().clone()
        track_instances["mem_acce"] = pred_acce

    def _bbox_parser(
        self,
        xy: Tensor,
        wl: Tensor,
        zheights: Tensor,
        labels: Tensor,
        use_sigmoid: bool = True,
    ) -> Tensor:
        """Parse bbox prediction.

        Args:
            xy: Bbox prediction of center point. [1, 1, N, 2].
            wl: Bbox prediction of width, length. [1, 1, N, 2].
            zheights: Bbox prediction of center z and h. [1, 1, N, 2].
            labels: Classification prediction result. [1, 1, N].
            use_sigmoid: Whether to use sigmoid. decoder output except
                last layer is activated by sigmoid in module, so we set true
                when last layer, otherwise false.
        Returns:
            coord: Parsed bbox prediction. [1, 1, N, 4].
        """
        wlh_anchor_size = self.wlh_anchor_size.to(wl)
        labels = labels.squeeze(0).squeeze(0)
        if use_sigmoid:
            xy = xy.sigmoid()
        # 网络输出是编码后的wlh，即 pred -> log (\frac{whl_{gt}}{range})
        # 因此需要对应解码
        wl = torch.exp(wl) * wlh_anchor_size[:, :, labels, :2]
        coord = torch.cat((xy, wl), dim=-1)
        zhs = torch.cat(
            (
                zheights[..., :1],
                torch.exp(zheights[..., 1:])
                * wlh_anchor_size[:, :, labels, 2:3],
            ),
            dim=-1,
        )
        return coord, zhs

    def _traj_pred_parser(
        self,
        traj_regression: Tensor,
        traj_logits: Tensor,
        use_momentum: bool = True,
    ):
        """Parse trajectory prediction.

        Args:
            traj_regression: trajectory regression outputs.
            traj_logits: trajectory classification outputs.
            use_momentum: whether to use momentum to cumsum
                trajectory regression.

        """
        traj_regression = traj_regression.view(
            1, self.num_traj_modal, -1, self.trajpred_length, 2
        )
        if use_momentum:
            traj_regression = torch.cumsum(traj_regression, dim=3)

        return traj_regression, traj_logits

    def _get_time_mask(self, mem_padding_mask):
        """Get valid time mask.

        Since we use the interpolation of position to encode, a valid
        time mask means the two positions in both adjacent frames are valid.
        However, whether current frame's positions valid is unavailable,
        thus we define all current frames' mask valid.

        Args:
            mem_padding_mask: valid position mask from t-memory_bank_len-1
                to t-1, [1, 1, N, T]
        Returns:
            time_mask: valid time mask from t-memory_bank_len+1 to t,
                [1, N, T-1, 1]
        """
        cur_qat_padding_mask = self.active_mask_quant(
            self.all_query_mask
        )  # [1, 1, 360, 1]
        valid_query_shape = mem_padding_mask.shape[2]
        cur_qat_padding_mask = cur_qat_padding_mask[:, :valid_query_shape]

        time_mask = self.memory_padding_mask_cat.cat(
            [mem_padding_mask[:, 1:, :, :], cur_qat_padding_mask], dim=1
        )
        time_mask = self.time_mask_mul.mul(
            time_mask[:, 1:, :, :], time_mask[:, :-1, :, :]
        )
        return time_mask

    def _get_his_velo(self, lcf_history_xy, bbox, current_fps, time_mask):
        """Get history velocity.

        Calculate velocity According to history bbox diffenences
        and current bbox.
        """
        # 当前帧BEV坐标转VCS坐标 ##
        sigmoid_boxes = self.boxes_sigmoid(bbox.detach())
        cur_lcf_x = sigmoid_boxes[..., 1:2]  # [1, num_q, 1, 1]
        cur_lcf_y = sigmoid_boxes[..., 0:1]  # [1, num_q, 1, 1]

        cur_lcf_x = self.lcf_x_muls.mul_scalar(
            cur_lcf_x, float(self.vcs_range[0] - self.vcs_range[2])
        )  # 乘以-bev_h
        cur_lcf_x = self.lcf_x_adds.add_scalar(cur_lcf_x, self.vcs_range[2])

        cur_lcf_y = self.lcf_y_muls.mul_scalar(
            cur_lcf_y, float(self.vcs_range[1] - self.vcs_range[3])
        )  # 乘以-bev_w
        cur_lcf_y = self.lcf_y_adds.add_scalar(cur_lcf_y, self.vcs_range[3])

        lcf_vcs_sigmoid_xy = self.lcf_xy_cat.cat(
            [cur_lcf_x, cur_lcf_y], dim=-1
        )  # [1, num_q, 1, 2]

        tmp_his_coord_xy = self.lcf_his_xy_cat.cat(
            [lcf_history_xy[:, 1:, :, :], lcf_vcs_sigmoid_xy], dim=1
        )  # [1, mem_Len, num_q, 2]

        his_coord_xy_diff = self.his_coord_sub.sub(
            tmp_his_coord_xy[:, 1:, :, :], tmp_his_coord_xy[:, :-1, :, :]
        )
        # 将 mask 提前乘到 his_coord_xy_diff 而非 his_velo 上，his_velo 是
        # his_coord_xy_diff 的 current_fps 倍（大于1），因此提前 mask 可使
        # 部分 his_velo 的值变为0，从而减小统计的scale，避免影响精度。
        his_coord_xy_diff = self.his_velo_mask_mul.mul(
            his_coord_xy_diff, time_mask
        )

        his_velo = self.fps_mul.mul(his_coord_xy_diff, current_fps)
        return his_velo

    def _get_lcf_his_xy(self, his_coord, odo_info):
        his_x = his_coord[..., 1:2]
        his_y = his_coord[..., 0:1]
        his_vcs_x = self.vcs_range[2] - his_x * (
            self.vcs_range[2] - self.vcs_range[0]
        )
        his_vcs_y = self.vcs_range[3] - his_y * (
            self.vcs_range[3] - self.vcs_range[1]
        )
        odo_info = odo_info.view((1, odo_info.shape[0], 1, odo_info.shape[1]))
        cos_yaw = torch.cos(odo_info[:, 1:-1, :, 2:3])
        sin_yaw = torch.sin(odo_info[:, 1:-1, :, 2:3])
        his_lcf_x = (
            his_vcs_x * cos_yaw
            - his_vcs_y * sin_yaw
            + odo_info[:, 1:-1, :, 0:1]
        )
        his_lcf_y = (
            his_vcs_x * sin_yaw
            + his_vcs_y * cos_yaw
            + odo_info[:, 1:-1, :, 1:2]
        )
        his_lcf_xy = torch.cat([his_lcf_x, his_lcf_y], dim=-1)
        return his_lcf_xy

    def generate_kalerman_input(
        self, his_bbox, his_velo, his_acce, pred_boxes, pred_velo, odo_info
    ):
        # convert to BEV coordinate
        his_bbox = update_his_coord(
            his_bbox.squeeze(0), odo_info[-2], self.vcs_range
        )
        his_velo = update_his_velo(
            his_velo.squeeze(0), odo_info[-2], self.vcs_range
        )
        his_acce = update_his_velo(his_acce, odo_info[-2], self.vcs_range)
        # generate state vector
        state_vector = torch.cat([his_bbox, his_velo, his_acce], dim=-1)
        # obser_bbox = track_instances["pred_boxes"]

        obser_bbox = pred_boxes
        last_outputs_velocities = pred_velo
        # convert to BEV coordinate
        obser_velo = -last_outputs_velocities[:, [1, 0]]
        obser_velo[:, 0] /= self.vcs_range[3] - self.vcs_range[1]
        obser_velo[:, 1] /= self.vcs_range[2] - self.vcs_range[0]
        # generate observe vector
        obser_state = torch.cat([obser_bbox, obser_velo], dim=-1)
        return state_vector, obser_state  # [num_queries, 8], [num_queries, 6]

    def generate_kalerman_output(self, output_state):
        pred_boxes = F.relu(output_state[:, :4]).clone()
        # transform to [-bev_y, -bev_x]
        pred_velo = -output_state[:, [5, 4]]
        pred_velo[:, 0] *= self.vcs_range[2] - self.vcs_range[0]
        pred_velo[:, 1] *= self.vcs_range[3] - self.vcs_range[1]
        # transform to [-bev_y, -bev_x]
        pred_acce = -output_state[:, [7, 6]]
        pred_acce[:, 0] *= self.vcs_range[2] - self.vcs_range[0]
        pred_acce[:, 1] *= self.vcs_range[3] - self.vcs_range[1]
        # pad z velocity, will be remove in the future.
        pred_velo_pad = torch.zeros((pred_velo.shape[0], 1)).to(pred_velo)
        pred_velo = torch.cat([pred_velo, pred_velo_pad], dim=1)
        return (
            pred_boxes,
            pred_velo,
            pred_acce,
        )  # [num_queries, 4], [num_queries, 3], [num_queries, 2]

    def _update_ref_pts_with_velo(
        self,
        track_instances: dict,
        odo_info: torch.Tensor,
        cur_time_delta: torch.Tensor,
    ):
        reference_points = track_instances["ref_pts"].sigmoid()
        # convert from previous bev norm coordinate to previous vcs coordinate
        reference_points = torch.cat(
            (
                reference_points,
                torch.ones(1, 1, reference_points.shape[2], 1).to(
                    reference_points
                ),
            ),
            dim=3,
        )
        reference_points = (
            reference_points
            @ torch.linalg.inv(self.vcs2bev_norm.to(reference_points)).T
        )
        # update ref_pts with velo
        velo_pred = track_instances["pred_velo_for_ref_pts"]
        reference_points[..., :2] = (
            reference_points[..., :2] + velo_pred[..., :2] * cur_time_delta
        )
        # convert from previous vcs coordinate to current vcs coordinate
        odo_x, odo_y, odo_yaw = odo_info
        odo_cosyaw = torch.cos(odo_yaw)
        odo_sinyaw = torch.sin(odo_yaw)
        odo_mat = torch.tensor(
            [
                [odo_cosyaw, -odo_sinyaw, odo_x],
                [odo_sinyaw, odo_cosyaw, odo_y],
                [0, 0, 1],
            ]
        ).to(odo_info)
        reference_points = reference_points @ odo_mat.T
        # convert from current vcs coordinate to current bev norm coordinate
        reference_points = (
            reference_points @ self.vcs2bev_norm.to(reference_points).T
        )
        track_instances["ref_pts"] = inverse_sigmoid(reference_points[..., :2])

    def _forward_each_frame(
        self,
        features: List[torch.Tensor],
        anchors: dict,
        odos: List[torch.Tensor],
        track_instances: dict = None,
        time_delta_queue: torch.Tensor = None,
        state_info: dict = None,
    ):
        """Forward each image in clip feats.

        Args:
            features: single image's multi-level feats.
            anchors: use for generate det_query.
            odos: odometry info of current frame and
                last frame.
            track_instances: track_instances of last frame.
                Defaults to None.
            time_delta_queue: time delta list of history frame.
            state_info: state_info of history frames, including:
                lcf_history_xy: [1, memory_bank_len, num_q, 2]
                fps_queue: [1, memory_bank_len + 1, 1, 1]
                only use in compile_model.
        """
        if time_delta_queue is not None:
            cur_time_delta = time_delta_queue[0, -1, 0, 0]
        else:
            cur_time_delta = self.default_time_delta
            time_delta_queue = self.default_time_delta

        if not self.training and self.eval_pure_det:
            track_instances = None

        if self.compile_model:
            odo_input_mb = odos[0]
        else:
            odo_info = get_lcf_odo_info(odos)
            odo_input_mb = self._process_odo_info(
                odo_info,
                time_delta_queue,
            )

        features = [
            feat_quant(feat) if not isinstance(feat, QTensorType) else feat
            for (feat_quant, feat) in zip(self.feat_quant, features)
        ]
        xy_range = self.xy_range_quant(self.xy_range)
        # prepare input of qim module for temporal task, conduct on cpu.
        if self.query_interaction_module is not None:
            is_first_frame = False
            if not self.compile_model:
                # for the first frame
                if track_instances is None:
                    cur_track_num_query = 0
                    is_first_frame = True
                else:
                    cur_track_num_query = len(track_instances["obj_idxes"])
                if cur_track_num_query < self.num_track_queries:
                    # pad track_query to fixed num_track_query
                    pad_num = int(self.num_track_queries - cur_track_num_query)
                    pad_instances = self._init_instances(
                        pad_num, xy_range.device
                    )
                    pad_instances["active_mask"] = torch.zeros_like(
                        pad_instances["active_mask"]
                    )
                    track_instances = self._concat_instances(
                        pad_instances, track_instances
                    )  # all type is tensor
                elif cur_track_num_query > self.num_track_queries:
                    # select fixed num_track_query of track_query
                    # according to score ranking.
                    track_instances = self._select_topk_instances(
                        track_instances, self.num_track_queries
                    )

            track_instances["active_mask"] = self.active_mask_quant(
                track_instances["active_mask"]
            )  # (1, 60, 1, 1)
            active_mask_for_fix_trk_query = track_instances["active_mask"]

            if not (self.calibration_model and is_first_frame):
                track_instances["query_embed"] = self.track_query_quant(
                    track_instances["query_embed"]
                )

            # forward qim module for temporal task,
            # update query_embed of track_query

            if not self.compile_model:
                with torch.no_grad():
                    self._update_ref_pts_with_velo(
                        track_instances, odo_info[-2], cur_time_delta
                    )

            if not is_first_frame:
                output_embedding = self.track_output_embedding_quant(
                    track_instances.pop("output_embedding")
                )
                track_instances["query_embed"] = self.query_interaction_module(
                    query_embed=track_instances["query_embed"],
                    output_embedding=output_embedding,
                    active_mask=active_mask_for_fix_trk_query,
                )
                if self.compile_model:
                    all_ref_pts = self.ref_pts_quant(
                        track_instances["ref_pts"]
                    )
                    trk_ref_pts = all_ref_pts[:, :, self.num_det_queries :, :]
                    trk_ref_pts = self.track_ref_pts_sigmoid(trk_ref_pts)
                    pos_emb = self.generate_pos_embed(trk_ref_pts)
                    track_instances[
                        "query_embed"
                    ] = self.trk_query_pos_cat.cat(
                        [pos_emb, track_instances["query_embed"][..., 256:]],
                        dim=-1,
                    )
                else:
                    trk_ref_pts = self.ref_pts_quant(
                        track_instances["ref_pts"]
                    )
                    trk_ref_pts = self.track_ref_pts_sigmoid(trk_ref_pts)
                    pos_emb = self.generate_pos_embed(trk_ref_pts)
                    track_instances[
                        "query_embed"
                    ] = self.trk_query_pos_cat.cat(
                        [pos_emb, track_instances["query_embed"][..., 256:]],
                        dim=-1,
                    )
        if self.init_query_feat:
            sample_feat = None
            if not self.compile_model:
                features.pop(0)
        else:
            sample_feat = features.pop(0)

        if self.compile_model:
            det_query_pos = self._generate_query_from_anchor(
                sample_feat, anchors, xy_range
            )
            active_mask_for_fix_det_query = self.active_mask_quant(
                anchors["det_active_mask"]
            )
            track_instances["query_embed"] = self.query_embed_cat.cat(
                [det_query_pos, track_instances["query_embed"]], dim=2
            )  # cat 初始的query_pos, 变成[360, 512]的##
            active_mask_for_fix_all_query = self.dict_active_mask_cat.cat(
                [active_mask_for_fix_det_query, active_mask_for_fix_trk_query],
                dim=2,
            )
        else:
            # generate det_instances for each frame individually
            # using the first stride feature.
            det_instances = self.generate_det_instances(
                sample_feat, anchors, xy_range
            )
            det_instances["active_mask"] = self.active_mask_quant(
                det_instances["active_mask"]
            )
            if self.calibration_model and is_first_frame:
                track_instances = det_instances
            else:
                track_instances = self._concat_instances(
                    det_instances,
                    track_instances,
                )
            active_mask_for_fix_all_query = track_instances["active_mask"]

        track_instances["ref_pts"] = self.ref_pts_quant(
            track_instances["ref_pts"]
        )

        srcs_feats = []
        for lvl, feat in enumerate(features):  # forward each level's feat
            srcs_feats.append(self.input_proj[lvl](feat))
        # NOTE: all the data in out should be convert to nchw to the model
        # unsqueeze tensor: ori_dim -> (1,1,ori_dim)
        (
            inter_embedding,
            inter_references_xy,
            inter_references_wh,
        ) = self.transformer(
            srcs=srcs_feats,
            query_embed=track_instances["query_embed"],
            ref_pts=track_instances["ref_pts"],
            xy_range=xy_range,
            active_mask_for_fix_all_query=active_mask_for_fix_all_query,
        )
        track_instances.update({"output_embedding": inter_embedding[-1]})

        outputs_classes = []
        outputs_coords = []
        outputs_zheights = []
        outputs_yaws = []
        # Get the output of the last layer
        last_outputs_class = self.pred_logits_dequants[-1](
            self.class_embed[-1](inter_embedding[-1])
        )
        last_outputs_coords_xy = self.pred_boxes_xy_dequant_list[-1](
            inter_references_xy[-1]
        )
        last_outputs_coords_wh = self.pred_boxes_wh_dequant_list[-1](
            inter_references_wh[-1]
        )
        _, all_label = last_outputs_class.max(-1)

        last_outputs_zhs = self.pred_zheights_dequants[-1](
            self.zheight_embed[-1](inter_embedding[-1])
        )
        last_outputs_coords, last_outputs_zheights = self._bbox_parser(
            last_outputs_coords_xy,
            last_outputs_coords_wh,
            last_outputs_zhs,
            all_label,
            use_sigmoid=True,
        )

        last_outputs_yaws = self.pred_yaws_dequants[-1](
            self.yaw_embed[-1](inter_embedding[-1])
        )

        # forward memory_bank module for temporal task,
        # update output_embedding of all query
        if self.memory_bank_module is not None:
            odo_input_mb = self.odo_input_mb_quant(odo_input_mb)
            mem_padding_mask = self.mem_padding_mask_quant(
                track_instances["mem_padding_mask"]
            )
            dim = track_instances["output_embedding"].shape[-1]
            query_pos = track_instances["query_embed"][:, :, :, :dim]

            if self.compile_model:
                mem_bank = self.mem_bank_quant(track_instances["mem_bank"])
                mem_bank = mem_bank.transpose(1, 2)
                mem_padding_mask = mem_padding_mask.transpose(1, 2)
                odo_input_mb = odo_input_mb.transpose(1, 2)
                # fps_queue shape: [1, memory_bank_len + 1, 1, 1]
                fps_queue = state_info["fps_queue"]
            else:
                mem_bank = track_instances["mem_bank"]
                # fps_queue shape: [1, memory_bank_len + 1, 1, 1]
                fps_queue = (1.0 / time_delta_queue).float()
            fps_queue = self.fps_quant(fps_queue)
            (
                track_instances["output_embedding"],
                track_instances["output_embedding_for_update"],
                track_scores,
            ) = self.memory_bank_module(
                query_pos=query_pos,
                output_embedding=track_instances["output_embedding"],
                mem_bank=mem_bank,
                mem_padding_mask=mem_padding_mask,
                odo_input=odo_input_mb,
                fps_queue_input=fps_queue,
                active_mask=active_mask_for_fix_all_query,
            )

            if not self.compile_model:
                # unify the scale of mem_bank and output_embedding_for_update
                # to avoid extra quant and dequant in software
                track_instances[
                    "output_embedding_for_update"
                ] = self.mem_bank_quant(
                    track_instances["output_embedding_for_update"]
                )

        if (
            self.calibration_model
            and track_instances["mem_coord"].abs().sum() == 0
        ):
            with torch.no_grad():
                maxV, _ = torch.max(last_outputs_class, -1)
                minMaxIdx = torch.argmin(maxV)
                track_instances["mem_coord"][0, 0, minMaxIdx, :] = (
                    torch.rand(size=(1, 4)).to(maxV) * 0.001
                )

        # generate embed for velocity head and trajectory head.
        if self.e2e_out_velocity or self.e2e_out_trajectory:
            if self.compile_model:
                lcf_history_xy = state_info["lcf_history_xy"]
            else:
                lcf_history_xy = self._get_lcf_his_xy(
                    track_instances["mem_coord"], odo_info
                )  # [1, memory_bank_len, num_q, 2]
            lcf_history_xy = self.memory_coord_quant(lcf_history_xy)
            if self.compile_model:
                lcf_history_xy = lcf_history_xy.transpose(1, 2)

            time_mask = self._get_time_mask(mem_padding_mask)
            velo_embed = self._get_his_velo(
                lcf_history_xy,
                inter_references_xy[-1],
                fps_queue[:, 2:, :, :],  # [1, memory_bank_len - 1, 1, 1]
                time_mask,
            )  # [1, memory_bank_len-1, num_q, 2]
            if self.calibration_model and is_first_frame:
                velo_embed = torch.ones(
                    velo_embed.dequantize().shape,
                    device=velo_embed.device,
                )
                velo_embed = self.calib_tmp_quant(velo_embed)
            velo_embed = self.trajpred_odo_proj(velo_embed)
            cur_traj_embed = self.trajpred_encode(
                velo_embed.permute(0, 2, 1, 3).reshape(
                    1, 1, velo_embed.shape[2], -1
                )
            )
            cur_embed = self.embedding_cat.cat(
                [track_instances["output_embedding"], cur_traj_embed], dim=3
            )  # [1, 1, 360, 288]

        if self.e2e_out_velocity:
            last_outputs_velocities = self.velocity_embed(cur_embed)
            last_outputs_velocities = self.velocities_dequant(
                last_outputs_velocities
            )

            K_out_mat = self.K_embed(cur_embed)
            K_mat = self.qmat_dquant(K_out_mat)

        if self.e2e_out_trajectory:
            modal_hh_embeding = self.modal_embeding()
            cur_embed_add_anchor_embed = self.embedding_add.add(
                cur_embed, modal_hh_embeding
            )  # [1, num_modal, num_q, 288]
            tmp_encode_trajs = self.trajpred_embed(
                cur_embed_add_anchor_embed
            )  # [1, num_modal, num_q, 120]
            tmp_logits_trajs = self.trajpred_logits(
                cur_embed_add_anchor_embed
            )  # [1, num_modal, num_q, 1]

            tmp_encode_trajs = self.traj_encode_dequant(tmp_encode_trajs)
            tmp_logits_trajs = self.traj_logits_dequant(tmp_logits_trajs)

        track_instances.update(
            {
                "query_embed": self.query_embed_dequant(
                    track_instances["query_embed"]
                )
            }
        )

        if self.compile_model:
            out_for_compile = []
            out_for_compile.append(track_instances["query_embed"])
            out_for_compile.append(last_outputs_class)
            out_for_compile.append(last_outputs_coords_xy)
            out_for_compile.append(last_outputs_coords_wh)
            out_for_compile.append(last_outputs_yaws)
            out_for_compile.append(last_outputs_zhs)
            if self.memory_bank_module is not None:
                out_for_compile.append(
                    self.output_emb_update_dquant(
                        track_instances["output_embedding_for_update"]
                    )
                )
                out_for_compile.append(
                    self.output_embedding_dequant(
                        track_instances["output_embedding"]
                    )
                )
                out_for_compile.append(track_scores)
            if self.e2e_out_velocity:
                out_for_compile.append(last_outputs_velocities)
                out_for_compile.append(K_mat)
            if self.e2e_out_trajectory:
                out_for_compile.append(tmp_encode_trajs)
                out_for_compile.append(tmp_logits_trajs)
            return out_for_compile
        else:
            track_instances.update(
                {
                    "output_embedding": track_instances["output_embedding"],
                    "ref_pts": last_outputs_coords_xy.detach(),
                }
            )

            if self.use_auxiliary_loss:
                for lvl in range(len(inter_embedding) - 1):
                    outputs_classes.append(
                        self.pred_logits_dequants[lvl](
                            self.class_embed[lvl](inter_embedding[lvl])
                        )
                    )  # (1,num_det_queries,c)
                    outputs_coord_xy = self.pred_boxes_xy_dequant_list[lvl](
                        inter_references_xy[lvl]
                    )
                    outputs_coord_wh = self.pred_boxes_xy_dequant_list[lvl](
                        inter_references_wh[lvl]
                    )
                    _, all_label = outputs_classes[lvl].max(-1)

                    outputs_zh = self.pred_zheights_dequants[lvl](
                        self.zheight_embed[lvl](inter_embedding[lvl])
                    )
                    outputs_coord, outputs_zheight = self._bbox_parser(
                        outputs_coord_xy,
                        outputs_coord_wh,
                        outputs_zh,
                        all_label,
                        use_sigmoid=False,
                    )
                    outputs_coords.append(outputs_coord)
                    # outputs_coords.append(new_reference_points[lvl])
                    outputs_yaws.append(
                        self.yaw_embed[lvl](inter_embedding[lvl])
                    )
                    outputs_zheights.append(outputs_zheight)
            outputs_classes.append(last_outputs_class)  # has't pass sigmoid
            outputs_coords.append(last_outputs_coords)
            outputs_yaws.append(last_outputs_yaws)
            outputs_zheights.append(last_outputs_zheights)

            # active_mask: 0 or 1, 1 means valid instances.
            track_instances["active_mask"] = self.active_mask_dequant(
                track_instances["active_mask"]
            ).clone()
            select_mask = track_instances["active_mask"].squeeze(-1) >= 0.5
            track_instances = dict_select_keep_dim(
                track_instances, select_mask
            )
            outputs_classes = [tmp[select_mask] for tmp in outputs_classes]
            outputs_coords = [tmp[select_mask] for tmp in outputs_coords]
            outputs_yaws = [tmp[select_mask] for tmp in outputs_yaws]
            outputs_zheights = [tmp[select_mask] for tmp in outputs_zheights]

            # out contains the multi-lvl output, track_instance
            # only contains the last lvl out
            out = {
                "pred_logits": outputs_classes[-1],
                "pred_boxes": outputs_coords[-1],
                "pred_yaws": outputs_yaws[-1],
                "pred_zheights": outputs_zheights[-1],
            }
            if self.memory_bank_module is not None:
                track_scores = track_scores[select_mask]
                out.update({"pred_track_scores": track_scores})

            if self.e2e_out_velocity:
                K_mat = K_mat.view(-1, 8, 6)
                # [1, 1, 360, 3] -> [active_num, 3],  origin [1, 300, 3]
                last_outputs_velocities = last_outputs_velocities[select_mask]
                # [360, 8, 6] -> [active, 8, 6]
                K_mat = K_mat[select_mask.squeeze()]

                (state_vector, obser_vector,) = self.generate_kalerman_input(
                    # mv _update_memory_bank backward, origin -2
                    his_bbox=track_instances["mem_coord"][:, -1, ...].clone(),
                    his_velo=track_instances["mem_velo"][:, -1, :, :2].clone(),
                    his_acce=track_instances["mem_acce"].clone(),
                    pred_boxes=out["pred_boxes"],
                    pred_velo=last_outputs_velocities,
                    odo_info=odo_info,
                )
                output_state = kalerman_filter(
                    state_vector, obser_vector, K_mat, cur_time_delta
                )  # [num_queries, 8]

                (
                    pred_boxes,
                    pred_velo,
                    pred_acce,
                ) = self.generate_kalerman_output(output_state)

                replace_index = (
                    track_instances["mem_padding_mask"][:, -3, :, :] == 0
                )
                replace_index = replace_index.squeeze(0).squeeze(-1)
                pred_boxes[replace_index, :] = out["pred_boxes"][
                    replace_index, :
                ]
                pred_velo[replace_index, :] = last_outputs_velocities[
                    replace_index, :
                ]
                track_instances["ref_pts"][
                    :, :, ~replace_index, :
                ] = inverse_sigmoid(pred_boxes[~replace_index, :2])

                self._update_memory_bank_use_kalerman(
                    track_instances, pred_boxes, pred_velo, pred_acce
                )

                out.update(
                    {
                        "pred_velocities": last_outputs_velocities,
                        "pred_boxes_kalerman": pred_boxes,
                        "pred_velocities_kalerman": pred_velo,
                        "replace_index": replace_index,
                    }
                )
                if "pred_velocities_kalerman" in out:
                    track_instances["pred_velo_for_ref_pts"] = (
                        out["pred_velocities_kalerman"]
                        .clone()
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                else:
                    track_instances["pred_velo_for_ref_pts"] = (
                        out["pred_velocities"]
                        .clone()
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )

            if self.e2e_out_trajectory:
                # [1, 5, 360, 120] -> [1, 5, 300, 120]
                tmp_encode_trajs = tmp_encode_trajs[
                    :, :, select_mask.squeeze()
                ]
                # [1, 5, 360, 1] -> [1, 5, 300, 1]
                tmp_logits_trajs = tmp_logits_trajs[
                    :, :, select_mask.squeeze()
                ]
                (
                    last_encode_trajs,
                    last_logits_trajs,
                ) = self._traj_pred_parser(
                    tmp_encode_trajs, tmp_logits_trajs, True
                )
                out["pred_TrajRegs"] = last_encode_trajs
                out["pred_TrajLogits"] = last_logits_trajs

            if self.use_auxiliary_loss and self.training:
                out["aux_outputs"] = self._set_auxiliary_loss(
                    outputs_classes,
                    outputs_coords,
                    outputs_yaws,
                    outputs_zheights,
                )
            with torch.no_grad():
                scores = out["pred_logits"].sigmoid()
                scores, classes = torch.max(scores, dim=-1, keepdim=False)
            track_instances["scores"] = scores
            track_instances["pred_boxes"] = out["pred_boxes"].clone()
            if not self.training:  # inference mode
                track_instances["pred_yaws"] = out["pred_yaws"].clone()
                track_instances["classes"] = classes
                track_instances, out = self.track_manager(track_instances, out)
            else:  # train mode
                # update 'obj_idxes' according to gt match and
                # generate output for loss calculation.
                (
                    # prev_match_index,
                    track_instances,
                    output_for_losses,
                ) = self.criterion.match_for_single_frame(out, track_instances)

                out["output_for_losses"] = output_for_losses

            if self.memory_bank_module is not None:
                track_instances["mem_padding_mask"][:, -1, :, :] = (
                    track_instances["obj_idxes"] >= 0
                ).unsqueeze(1)
        return track_instances, out

    def _select_active_instances(self, track_instances):
        active_idxes = track_instances["obj_idxes"] >= 0
        active_track_instances = dict_select_keep_dim(
            track_instances, active_idxes
        )
        if self.training:
            # select active track instances, for next frame.
            active_track_instances = self._random_drop_instances(
                active_track_instances, self.query_drop_ratio
            )
            if self.query_fp_ratio > 0:
                active_track_instances = self._add_fp_instances(
                    track_instances, active_track_instances
                )
        return active_track_instances

    def _forward_each_iter(
        self,
        iter_feats: List[torch.Tensor],
        iter_anchors: List[dict],
        iter_odos: torch.Tensor,
        iter_cur_timestamp: torch.Tensor,
        track_instances: dict = None,
        time_delta_queue: torch.Tensor = None,
        last_timestamp: torch.Tensor = None,
    ):
        """Forward each clip feats.

        Args:
            iter_feats: iter中各帧的多尺度特征.
            iter_anchors: iter各帧的anchor.
            iter_odos: iter中各帧中用到的odo信息.
            iter_cur_timestamp: iter中各帧的timestamp.
            track_instances: iter中输入的track_instance.
            time_delta_queue: iter中输入的历史帧time_delta序列.
            last_timestamp: 当前iter上一帧的timestamp,用来计算时间间隔.
        """
        outputs = {
            # "track_instances": [],
            "output_for_losses": {},
            "clip_num_samples": 0,
        }
        assert len(iter_feats) == len(iter_anchors)
        # forward feat of each clip
        for _idx in range(len(iter_feats)):
            if self.e2e_out_velocity or self.e2e_out_trajectory:
                time_delta_queue = self.update_time_delta_queue(
                    time_delta_queue,
                    iter_cur_timestamp[_idx],
                    last_timestamp,
                )
            else:
                time_delta_queue = None
            last_timestamp = iter_cur_timestamp[_idx]
            (track_instances, output,) = self._forward_each_frame(
                iter_feats[_idx],
                iter_anchors[_idx],
                iter_odos[_idx],
                track_instances,
                time_delta_queue,
            )
            track_instances = self._select_active_instances(track_instances)

            self.criterion._step()
            assert "output_for_losses" in output

            tmp_for_loss = output["output_for_losses"]
            if "aux_outputs_for_loss" in outputs["output_for_losses"].keys():
                aux_for_losses = tmp_for_loss.pop("aux_outputs_for_loss", None)
            else:
                aux_for_losses = None
            outputs["output_for_losses"].update(output["output_for_losses"])
            if aux_for_losses is not None:
                outputs["output_for_losses"]["aux_outputs_for_loss"].update(
                    aux_for_losses
                )
        return track_instances, time_delta_queue, outputs

    def _take_features_by_clip(self, feats):
        """Stack data according to views.

        Args:
            data (Sequence): data need to stack, e.g. imgs, seg. shape=
        Returns:
            [torch.Tensor]: list of stacked data by frames.
        """
        feats = _take_features(feats, self.in_strides, self.out_strides)
        ret_feats = [[] for _ in range(feats[0].shape[0])]
        for feat in feats:
            assert len(feat.shape) == 4, "feat's shape should be (n,c,h,w)"
            for _idx, _feat in enumerate(feat.split(1)):
                ret_feats[_idx].append(_feat)
        return ret_feats

    def _get_anchors_by_clip(self, clip_datas, clip_length):
        veh_score = clip_datas[
            "bev_stage2_3d_vehicle_head_predict_bev3d_score"
        ]
        vru_score = clip_datas[
            "bev_stage2_3d_vrumerge_head_predict_bev3d_score"
        ]

        veh_vcs_coord = clip_datas[
            "bev_stage2_3d_vehicle_head_predict_bev3d_ct"
        ].float()
        vru_vcs_coord = clip_datas[
            "bev_stage2_3d_vrumerge_head_predict_bev3d_ct"
        ].float()

        anchors = []
        veh_vcs_coord = torch.cat(
            (
                veh_vcs_coord,
                torch.ones(
                    vru_vcs_coord.shape[0], vru_vcs_coord.shape[1], 1
                ).to(vru_vcs_coord),
            ),
            dim=2,
        )
        vru_vcs_coord = torch.cat(
            (
                vru_vcs_coord,
                torch.ones(
                    vru_vcs_coord.shape[0], vru_vcs_coord.shape[1], 1
                ).to(vru_vcs_coord),
            ),
            dim=2,
        )
        veh_bev_coord = (veh_vcs_coord @ self.vcs2bev.to(veh_vcs_coord).T)[
            :, :, :2
        ]
        vru_bev_coord = (vru_vcs_coord @ self.vcs2bev.to(vru_vcs_coord).T)[
            :, :, :2
        ]

        def _get_ref_pts_from_coord(coord, bev_size):
            """Convert the bev coord to ref points."""
            u = coord[:, 0] / (bev_size[1])
            v = coord[:, 1] / (bev_size[0])
            bev_coord_center = torch.stack(
                [u, v], dim=1
            )  # (u,v) = (x,y)  (300,2)
            init_ref_point = inverse_sigmoid(bev_coord_center)
            return init_ref_point

        for b in range(clip_length):
            veh_frame_score = veh_score[b]
            vru_frame_score = vru_score[b]
            veh_frame_coord = veh_bev_coord[b]
            vru_frame_coord = vru_bev_coord[b]

            veh_filter_index = torch.nonzero(
                torch.ge(veh_frame_score, self.veh_filter_scores)
            )
            vru_filter_index = torch.nonzero(
                torch.ge(vru_frame_score, self.vru_filter_scores)
            )
            veh_frame_coord = veh_frame_coord[veh_filter_index].squeeze(1)
            vru_frame_coord = vru_frame_coord[vru_filter_index].squeeze(1)

            veh_filter_ref_pt = _get_ref_pts_from_coord(
                veh_frame_coord, self.bev_size
            )
            vru_filter_ref_pt = _get_ref_pts_from_coord(
                vru_frame_coord, self.bev_size
            )

            veh_pad_list, veh_mask = pad_tensors_to_target_shape(
                [veh_frame_coord, veh_filter_ref_pt],
                target_num=self.num_veh_queries,
                dim=0,
            )
            veh_coord, veh_ref_pt = veh_pad_list

            vru_pad_list, vru_mask = pad_tensors_to_target_shape(
                [vru_frame_coord, vru_filter_ref_pt],
                target_num=self.num_vru_queries,
                dim=0,
            )
            vru_coord, vru_ref_pt = vru_pad_list

            heatmap_sampling_grids = torch.cat(
                (veh_coord, vru_coord), dim=0
            ).float()
            init_ref_point = torch.cat((veh_ref_pt, vru_ref_pt), dim=0)
            det_active_mask = torch.cat([veh_mask, vru_mask], dim=0)

            heatmap_sampling_grids = heatmap_sampling_grids.unsqueeze(
                0
            ).unsqueeze(0)
            init_ref_point = init_ref_point.unsqueeze(0).unsqueeze(0)
            det_active_mask = (
                det_active_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            )

            anchors.append(
                {
                    "sample_grid": heatmap_sampling_grids,
                    "det_ref_pts": init_ref_point,
                    "det_active_mask": det_active_mask,
                }
            )
        return anchors

    def _generate_query_from_anchor(self, sample_feature, anchor, xy_range):
        sample_grids = anchor["sample_grid"]
        sample_grids = self.heatmap_grid_quant(sample_grids)
        if self.init_query_feat:
            object_query = self.object_query()
        else:
            sample_offsets = self.heatmap_add_offsets.add(
                sample_grids, xy_range[:, :, : self.num_det_queries, :]
            )
            # generate query_feat
            object_query = self.heatmap_grid_sample(
                sample_feature.detach(), sample_offsets
            ).permute(0, 2, 3, 1)
            object_query = self.generate_query_embed(object_query)

        # generate query_pos
        sampling_grids_x = sample_grids[:, :, :, 0:1]
        sampling_grids_y = sample_grids[:, :, :, 1:2]
        bev_coord_center_x = self.coord_mul1.mul_scalar(
            sampling_grids_x, self.bev_norm_coef[1]
        )
        bev_coord_center_y = self.coord_mul2.mul_scalar(
            sampling_grids_y, self.bev_norm_coef[0]
        )
        bev_coord_center = self.coord_cat.cat(
            [bev_coord_center_x, bev_coord_center_y], dim=-1
        )
        query_pos = self.generate_pos_embed(bev_coord_center)

        query_embed = self.heatmap_cat.cat((query_pos, object_query), -1)
        return query_embed

    def _detach(self, track_instances):
        trk_ins = {}
        for k, v in track_instances.items():
            if isinstance(v, QTensorType):
                trk_ins[k] = v.detach()
            else:
                trk_ins[k] = v.detach().clone()
        return trk_ins

    def update_time_delta_queue(
        self, his_time_delta_queue, cur_timestamp, last_timestamp
    ):
        """历史时间间隔序列更新函数.

        根据当前帧的timestamp以及上一帧的timestamp，计算时间间隔.并对历史时间间隔序列进行更新.
        当last_timestamp为None时,时间间隔采用default_time_delta.

        Args:
            his_time_delta_queue: 存储历史帧之间间隔的序列，shape：[1, his_len, 1, 1]，
                当前时刻为T时：
                his_time_delta_queue[0, -1, 0, 0] 表示 T-1 到T-2的时间间隔，
                his_time_delta_queue[0, -2, 0, 0] 表示 T-2 到T-3的时间间隔，
            cur_timestamp: 当前帧的timestamp。
            last_timestamp: 上一帧的timestamp。
        """

        if last_timestamp is None:
            cur_time_delta = self.default_time_delta * torch.ones(1).to(
                cur_timestamp
            )
        else:
            cur_time_delta = (cur_timestamp - last_timestamp).unsqueeze(-1)
        cur_time_delta = (
            cur_time_delta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        )
        if his_time_delta_queue is None:
            his_time_delta_queue = (
                torch.ones(
                    (1, self.memory_bank_len + 1, 1, 1),
                    dtype=torch.float32,
                ).to(cur_time_delta)
                * self.default_time_delta
            )
        time_delta_queue = torch.cat(
            [his_time_delta_queue[:, 1:, :, :], cur_time_delta], dim=1
        )
        return time_delta_queue

    def forward(self, clip_datas: dict):
        """Forward all the clip_datas."""
        if self.compile_model:
            feats = _take_features(
                clip_datas[self.feature_name],
                self.in_strides,
                self.out_strides,
            )
            anchors = {
                "sample_grid": clip_datas["beve2e_trackernet_inputs"][9],
                "det_active_mask": clip_datas["beve2e_trackernet_inputs"][10],
            }
            track_instances = {
                "active_mask": clip_datas["beve2e_trackernet_inputs"][2],
                "ref_pts": clip_datas["beve2e_trackernet_inputs"][3],
                "query_embed": clip_datas["beve2e_trackernet_inputs"][0],
                "output_embedding": clip_datas["beve2e_trackernet_inputs"][1],
                "mem_bank": clip_datas["beve2e_trackernet_inputs"][4],
                "mem_padding_mask": clip_datas["beve2e_trackernet_inputs"][5],
            }
            odos = [
                clip_datas["beve2e_trackernet_inputs"][6],
            ]

            state_info = {
                "fps_queue": clip_datas["beve2e_trackernet_inputs"][7],
                # history position x, y converted to last context frame.
                "lcf_history_xy": clip_datas["beve2e_trackernet_inputs"][8],
            }
            frame_res = self._forward_each_frame(
                feats,
                anchors,
                odos,
                track_instances=track_instances,
                state_info=state_info,
            )
            return frame_res
        else:
            # clear the track_instances in the beginning of a clip or a pack
            if (
                "temporal_clr_flag" in clip_datas
                and clip_datas["temporal_clr_flag"][0]
            ):  # noqa: E501
                self.clear()

            clip_length = clip_datas[self.feature_name][0][0].size(0)
            clip_feats = self._take_features_by_clip(
                clip_datas[self.feature_name][0]
            )
            clip_odos = clip_datas["odo_info"]
            clip_anchors = self._get_anchors_by_clip(clip_datas, clip_length)
            clip_timestamp = clip_datas["timestamp"]
            assert len(clip_feats) == len(clip_anchors)

            batch_output = []

            if self.training:
                batch_size = len(clip_datas["motr_targets"])
                iter_length = clip_length // batch_size
                if self.last_track_instances is None:
                    self.last_track_instances = [None] * batch_size
                if self.his_time_delta_queue is None:
                    self.his_time_delta_queue = [None] * batch_size
                if self.last_timestamp is None:
                    self.last_timestamp = [None] * batch_size
                for _bidx in range(batch_size):
                    clip_target = clip_datas["motr_targets"][_bidx]
                    self.criterion.initialize_for_single_clip(clip_target)
                    iter_feats = clip_feats[
                        _bidx * iter_length : (_bidx + 1) * iter_length
                    ]
                    iter_anchors = clip_anchors[
                        _bidx * iter_length : (_bidx + 1) * iter_length
                    ]
                    iter_odos = clip_odos[
                        _bidx * iter_length : (_bidx + 1) * iter_length
                    ]
                    iter_timestamp = clip_timestamp[_bidx]
                    (
                        track_instances,
                        time_delta_queue,
                        outputs,
                    ) = self._forward_each_iter(
                        iter_feats,
                        iter_anchors,
                        iter_odos,
                        iter_timestamp,
                        track_instances=self.last_track_instances[_bidx],
                        time_delta_queue=self.his_time_delta_queue[_bidx],
                        last_timestamp=self.last_timestamp[_bidx],
                    )
                    self.last_track_instances[_bidx] = self._detach(
                        track_instances
                    )
                    if self.e2e_out_velocity or self.e2e_out_trajectory:
                        self.his_time_delta_queue[
                            _bidx
                        ] = time_delta_queue.detach()
                        self.last_timestamp[_bidx] = iter_timestamp[-1]
                    batch_output.append(outputs)

                if self.query_interaction_module is None:  # for detection task
                    self.last_track_instances = [None] * batch_size
                return batch_output
            else:
                for _clp in range(len(clip_feats)):
                    if self.e2e_out_velocity or self.e2e_out_trajectory:
                        time_delta_queue = self.update_time_delta_queue(
                            self.his_time_delta_queue,
                            clip_timestamp[0][_clp],
                            self.last_timestamp,
                        )
                        self.last_timestamp = clip_timestamp[0][_clp]
                    else:
                        time_delta_queue = None
                    (track_instances, outputs,) = self._forward_each_frame(
                        clip_feats[_clp],
                        clip_anchors[_clp],
                        clip_odos[_clp],
                        track_instances=self.last_track_instances,
                        time_delta_queue=time_delta_queue,
                    )
                    if "aux_outputs" in outputs:
                        outputs.pop("aux_outputs")
                    tmp_track_instances = {
                        key: val
                        for key, val in track_instances.items()
                        if key
                        in [
                            "obj_idxes",
                            "appear_time",
                            "disappear_time",
                            "scores",
                        ]
                    }
                    outputs.update(tmp_track_instances)

                    # select active track instances, for next frame.
                    active_idxes = track_instances["obj_idxes"] >= 0
                    track_instances = dict_select_keep_dim(
                        track_instances, active_idxes
                    )
                    if (
                        self.query_interaction_module is None
                    ):  # for detection task
                        self.last_track_instances = None
                    else:
                        self.last_track_instances = self._detach(
                            track_instances
                        )
                    if self.e2e_out_velocity or self.e2e_out_trajectory:
                        self.his_time_delta_queue = time_delta_queue.detach()
                    outputs = dict_select_keep_dim(outputs, active_idxes)
                    outputs = self.post_process(outputs)
                    batch_output.append(outputs)

                return batch_output

    def fuse_model(self):
        self.transformer.fuse_model()
        for m in self.input_proj:
            m.fuse_model()
        self.transformer.reference_points_xy.fuse_model()
        self.transformer.reference_points_wh.fuse_model()
        for m in self.zheight_embed:
            m.fuse_model()
        for m in self.bbox_embed_xy:
            m.fuse_model()
        for m in self.bbox_embed_wh:
            m.fuse_model()
        if self.query_interaction_module is not None:
            self.query_interaction_module.fuse_model()
        if self.memory_bank_module is not None:
            self.memory_bank_module.fuse_model()
        if self.e2e_out_trajectory or self.e2e_out_velocity:
            self.trajpred_odo_proj.fuse_model()
            self.trajpred_encode.fuse_model()
        if self.e2e_out_velocity:
            self.velocity_embed.fuse_model()
            self.K_embed.fuse_model()
        if self.e2e_out_trajectory:
            self.trajpred_embed.fuse_model()
            self.trajpred_logits.fuse_model()

    def set_qconfig(self):
        self.qconfig = qint16_qconfig()

        if not self.init_query_feat:
            self.heatmap_add_offsets.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": max(self.max_bev_size, self.num_det_queries)
                    / QINT16_MAX,
                },
            )

            self.heatmap_grid_sample.qconfig = get_default_qat_qconfig(
                dtype="qint8"
            )

        self.track_ref_pts_sigmoid.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": SIGMOID_MAX / QINT16_MAX,
            },
        )

        if self.e2e_out_velocity or self.e2e_out_trajectory:
            self.boxes_sigmoid.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": SIGMOID_MAX / QINT16_MAX,
                },
            )
            self.lcf_x_muls.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": float(self.vcs_range[2] - self.vcs_range[0])
                    / QINT16_MAX,
                },
            )
            self.lcf_x_adds.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": float(self.vcs_range[2]) / QINT16_MAX,
                },
            )
            self.lcf_y_muls.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": float(self.vcs_range[3] - self.vcs_range[1])
                    / QINT16_MAX,
                },
            )
            self.lcf_y_adds.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": float(self.vcs_range[3]) / QINT16_MAX,
                },
            )
            self.lcf_xy_cat.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": float(max(self.vcs_range[2], self.vcs_range[3]))
                    / QINT16_MAX,
                },
            )
            self.lcf_his_xy_cat.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": float(
                        max(self.vcs_range[2], self.vcs_range[3])
                        + (
                            SPEED_MAX
                            * self.default_time_delta
                            * self.memory_bank_len
                        )
                    )
                    / QINT16_MAX,
                },
            )
            self.memory_coord_quant.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": float(
                        max(self.vcs_range[2], self.vcs_range[3]) + 40.0
                    )
                    / QINT16_MAX,
                },
            )
            # # velo setting
            self.fps_mul.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": SPEED_MAX / QINT16_MAX,
                },
            )
            self.odo_input_mb_quant.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": SPEED_MAX / QINT16_MAX,
                },
            )

        for layer in self.feat_quant:
            layer.qconfig = get_default_qat_qconfig(dtype="qint8")

        for layer in self.class_embed:
            layer.qconfig = qat_out_qconfig()
        for layer in self.yaw_embed:
            layer.qconfig = qat_out_qconfig()

        for layer in self.input_proj:
            layer.qconfig = qint8_qconfig()

        self.transformer.set_qconfig()
        self.time_mask_mul.qconfig = qint8_qconfig()
        self.track_output_embedding_quant.qconfig = qint8_qconfig()

        for layer in self.zheight_embed:
            layer.layers[-1].qconfig = qat_out_qconfig()

        if self.query_interaction_module is not None:
            self.query_interaction_module.set_qconfig()
            self.track_query_quant.qconfig = qint8_qconfig()
            self.trk_query_pos_cat.qconfig = qint8_qconfig()
        self.dict_qim_embd_cat.qconfig = qint8_qconfig()
        if self.memory_bank_module is not None:
            self.memory_bank_module.set_qconfig()
            self.mem_bank_quant.qconfig = qint8_qconfig()
            self.mem_padding_mask_quant.qconfig = qint8_qconfig()
            self.dict_qim_embed_for_mem_cat.qconfig = qint8_qconfig()
            self.dict_qim_mem_bank_cat.qconfig = qint8_qconfig()
            self.dict_mem_bank_cat.qconfig = qint8_qconfig()
            self.query_embed_cat.qconfig = qint8_qconfig()

        if self.e2e_out_velocity:
            self.velocity_embed.layers[-1].qconfig = qat_out_qconfig()
            self.K_embed.layers[-1].qconfig = qat_out_qconfig()
        if self.e2e_out_trajectory:
            self.trajpred_embed.layers[-1].qconfig = qat_out_qconfig()
            self.trajpred_logits.layers[-1].qconfig = qat_out_qconfig()

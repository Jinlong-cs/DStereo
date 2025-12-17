import os

from aidisdk.utils import running_in_cluster

IS_LOCAL_TRAIN = not running_in_cluster()
BUCKET_ROOT = "/horizon-bucket" if IS_LOCAL_TRAIN else "/bucket/input"
MANO_ROOT = os.path.abspath(
    os.path.join(
        BUCKET_ROOT,
        "MultiMode_2/dynamic_gesture/training_details/mano_params/",
    )
)

VIRTUAL_CAMERA_FOCAL = 600
VIRTUAL_IMAGE_HW = [1080, 1920, 3]
VIRTUAL_CROP_SIZE = 480
VIRTUAL_NORM_RATIO = 1.2
ENABLE_INSHAPE_UNIFORM = False

ENABLE_GRID_SAMPLE = True
ENABLE_HEATMAP_HEAD = False
ENABLE_RENDER_HEAD = False
ENABLE_JOINTS_25D = False

INPUT_IMAGE_SIZE = 160
INPUT_CHANNELS = 5 if ENABLE_GRID_SAMPLE else 3
BIFPN_CHANNELS = 64
ENCODING_CHANNELS = BIFPN_CHANNELS * 4

MANO_HAND_SIDE = "right"
MANO_CENTER_IDX = 0
FLAT_HAND_MEAN = True
# fmt: off
SMPL_HANDS_MEAN = [
    -0.12307574, -0.08413376, 0.00970373, 0.07512784, -0.02228195,
    0.1428208, -0.02180195, -0.10839321, 0.04418178, -0.29530047,
    -0.00820144, 1.25067314, -0.17040829, -0.01297567, 1.41953006,
    -0.092699, 0.05311598, 0.60319288, -1.18973491, 0.4340069,
    1.12807981, -0.58122936, -0.10665728, 0.79413231, -0.69380294,
    -0.0149339, 0.6391199, -0.14385047, 0.46268508, 1.315784,
    -0.47935597, 0.58838911, 1.21133691, -0.12685085, 0.40946304,
    0.82403714, 0.55878933, -0.55092484, 0.28625131, -0.30980729,
    -0.17909153, -0.249276, 0.74102683, -0.353278, 0.34527757,
]
# fmt: on


def get_model(
    input_image_size=256,
    input_channels=4,
    bifpn_channels=64,
    encoding_channels=256,
    enable_grid_sample=True,
    enable_heatmap_head=False,
    enable_render_head=False,
    enable_joints_25d=False,
    mano_root=None,
    mano_hand_side="right",
    mano_center_idx=0,
    flat_hand_mean=True,
    smpl_hands_mean=None,
):

    loss = dict(
        type="H3DLossStucture",
        loss_class_init_list=[],
        loss_forward_list=[
            dict(
                loss_name="l_tvec",
                forward_func="smoothl1",
                weight=200,
                forward_params=dict(
                    pred="output_decoder['pred_trans']",
                    target="data['gt_root']",
                    normlier="data['ldmk3d_vis']",
                    sigma=5,
                ),
            ),
            dict(
                loss_name="l_mano_ldmk3d",
                forward_func="smoothl1",
                weight=20000,
                forward_params=dict(
                    pred="output_decoder['pred_ldmk3d_relat']",
                    target="data['ldmk3d_relat']",
                    normlier="data['ldmk3d_vis']",
                    sigma=2.5,
                ),
            ),
            dict(
                loss_name="l_mano_ldmk3d_cam",
                forward_func="smoothl1",
                weight=500,
                forward_params=dict(
                    pred="output_decoder['pred_ldmk3d']",
                    target="data['gt_ldmk3d']",
                    normlier="data['ldmk3d_vis']",
                    sigma=2.5,
                ),
            ),
            dict(
                loss_name="l_reproj_gt",
                forward_func="smoothl1",
                weight=2,
                forward_params=dict(
                    pred="output_decoder['pred_ldmk_proj']",
                    target="data['gt_ldmk']",
                    normlier="data['ldmk_vis']",
                    sigma=2.5,
                ),
            ),
            dict(
                loss_name="l_ldmk2d_gt",
                forward_func="smoothl1",
                weight=3,
                forward_params=dict(
                    pred="output_decoder['pred_ldmk']",
                    target="data['gt_ldmk']",
                    normlier="data['ldmk_vis']",
                    sigma=2.5,
                ),
            ),
            dict(
                loss_name="l_edge",
                forward_func="mesh_edge",
                weight=1,
                forward_params=dict(
                    rerender_mesh="output_decoder['pred_mesh']",
                ),
            ),
            dict(
                loss_name="l_normal",
                forward_func="mesh_normal_consistency",
                weight=1,
                forward_params=dict(
                    rerender_mesh="output_decoder['pred_mesh']",
                ),
            ),
            dict(
                loss_name="l_laplacian",
                forward_func="mesh_laplacian_smoothing",
                weight=1,
                forward_params=dict(
                    rerender_mesh="output_decoder['pred_mesh']",
                ),
            ),
            dict(
                loss_name="l_chamfer",
                forward_func="mesh_chamfer_distance",
                weight=1,
                forward_params=dict(
                    verts_pred="output_decoder['pred_verts']",
                    verts_gt="data['gt_verts']",
                    normlier="data['vert3d_vis']",
                ),
            ),
        ],
    )
    if enable_heatmap_head:
        raise NotImplementedError("TODO: add heatmap loss")

    if enable_render_head:
        loss["loss_forward_list"].append(
            dict(
                loss_name="l_render_image",
                forward_func="ms_ssim_l1",
                weight=1,
                forward_params=dict(
                    pred="output_decoder['render_images']",
                    target="data['img']",
                    foreground_mask="output_decoder['render_images']",
                    normlier=1.0,
                    sigma=0.25,
                ),
            ),
        )

    if enable_joints_25d:
        loss["loss_forward_list"].append(
            dict(
                loss_name="l_j2d_repro",
                forward_func="smoothl1",
                weight=1,
                forward_params=dict(
                    pred="output_decoder['pred_ldmk3d']",
                    target="output_decoder['pred_ldmk25d']",
                    normlier=1.0,
                    sigma=2.5,
                ),
            ),
            dict(
                loss_name="l_joints25d",
                forward_func="smoothl1",
                weight=1,
                forward_params=dict(
                    # torch.Size([16, 21, 3])
                    pred="output_decoder['pred_ldmk25d']",
                    target="data['gt_ldmk3d']",
                    normlier="data['ldmk3d_vis']",
                    sigma=2.5,
                ),
            ),
        )

    model = dict(
        type="H3DStructure",
        backbone=dict(
            type="SNDRMobileNetV2",
            num_classes=0,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            in_chls=[
                [32],
                [16, 24],
                [24, 32, 32],
                [32, 64, 64, 64, 64, 96, 96],
                [96, 160, 160, 160],
            ],
            out_chls=[
                [16],
                [24, 24],
                [32, 32, 32],
                [64, 64, 64, 64, 96, 96, 96],
                [160, 160, 160, 320],
            ],
            expand_ratio=6,
            input_channels=input_channels,
            alpha=1.0,
            bias=True,
            include_top=False,
        ),
        neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[16, 24, 32, 96, 320],
            out_strides=[8, 16, 32],
            out_channels=[64, 128, 192],
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            node_name="fpn_neck",
        ),
        encoder=dict(
            type="H3DFuseHighLowEncoder",
            in_channels=192,
            bifpn_channels=bifpn_channels,
            encoding_channels=encoding_channels,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            heatmap_encoder=dict(
                type="H3DHeatMapResEncoder",
                in_channels=bifpn_channels,
                out_channels=encoding_channels,
                bn_kwargs=dict(eps=1e-5, momentum=0.01),
            ),
            global_avg_size=3,
        ),
        head=dict(
            type="H3DManoMultiFcHead",
            encoding_channel=encoding_channels,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            scale_neurons=(256, 128, 1),
            pose_neurons=(256, 128, 90),
            shape_neurons=(256, 10),
            camera_base_neurons=(256,),
            rot_neurons=(256, 6),
            trans_neurons=(256, 3),
            text_fc_neurons=(256, 256, 128),
            text_reg_neurons=(1024, 778, 778),
            head_heatmap=None,
            head_heatmap_latent=True,
            enable_render_head=False,
            enable_handscale_head=False,
            num_joints=21,
            deploy=False,
        ),
        decoder=dict(
            type="H3DManoDecoder",
            intput_shape=(input_image_size, input_image_size),
            heatmap_stride=4,
            trans_coeff=2.5,
            min_distance=0.001,
            hand_scale_coeff=1,
            mano_layer=dict(
                type="MANOLayer",
                model_path=mano_root,
                is_rhand=mano_hand_side == "right",
                center_idx=mano_center_idx,
                flat_hand_mean=flat_hand_mean,
                smpl_hands_mean=smpl_hands_mean,
            ),
            point_nerf=dict(
                type="PointNeRF",
                intput_shape=(input_image_size, input_image_size),
                point_cloud_radius=0.025,
                points_per_pixel=50,
                text_feat_length=128,
                position_encoding_length=4,
                density_feat_length=12,
            ),
            enable_grid_sample=enable_grid_sample,
            enable_heatmap_head=enable_heatmap_head,
            enable_render_head=enable_render_head,
            enable_handscale_head=False,
            enable_subdivide_points=False,
            infer_mode=False,
        ),
        loss=loss,
        enable_grid_sample=enable_grid_sample,
    )
    return model

import os
import uuid

import numpy as np
import pytest
from aidisdk.model import DeviceMeta
from hatbc.message import CameraFrame, Image
from PIL import Image as pil_image

from hat.utils.apply_func import _as_list
from projects.cloudmodel.tools.instance_engine import CloudModelInference
from projects.cloudmodel.tools.visualizer.class_metas.cloudmodel.class_meta_cloudmodel import (
    ClassMetaCloudModel,
)
from projects.cloudmodel.tools.visualizer.render_base import InstanceConfigs
from projects.cloudmodel.tools.visualizer.render_camera_frame import (
    RenderCameraFrame,
)


class TestCloudModelInference:
    @staticmethod
    def make_test_images(img_path, img_scale=None, raw_byte=False):
        example_input = []
        for img_path_i in _as_list(img_path):
            if raw_byte:
                assert (
                    img_scale is None
                ), "raw byte import do not support import"
                with open(img_path_i, "rb") as rf:
                    img = rf.read()
                layout = None
                color_space = None
            else:
                img = pil_image.open(img_path_i).convert("RGB")
                if img_scale is not None:
                    img = img.resize(img_scale)
                img = np.array(img, dtype=np.uint8)
                color_space = "rgb"
                layout = "hwc"

            frame = CameraFrame(
                image=Image(data=img, layout=layout, color_space=color_space)
            )
            example_input.append(frame)

        return example_input

    def apply_render_result(self, results, render_root="tmp"):
        visualizer = RenderCameraFrame(
            image_scale=1.0,
            instance_config=InstanceConfigs(instance_threshold=0.3),
        )
        class_meta = ClassMetaCloudModel()
        for res_i in results:
            render_res = visualizer(res_i, metadata=class_meta)
            save_dir = os.path.join(
                render_root, res_i.topic, f"{uuid.uuid4().hex}.png"
            )
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            render_res.save(save_dir)

    def compare2results(self, res1, res2):
        # TODO impl me
        return True

    @pytest.mark.parametrize(
        [
            "model_name",
            "model_version",
            "test_aidi_infer",
            "test_local_infer",
            "test_img_path",
            "render_res_root",
        ],
        [
            pytest.param(
                "cloudmodel_singletask_adb_light_detection_swinb_w12_pafpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_traffic_sign_euro_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/dataset/20230322/641aac07e6107ae52340b0dd/20211108T082334_8N1430_1636359883.013937.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_cyclist_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_person_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_rear_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_road_arrow_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_2/adas_data_raw/pack/white_changan/DG202_20210626_D/ADAS_20210626-114108_016_5/ADAS_20210626-114108_016_5__324945_1624679121346_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_traffic_cone_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_traffic_light_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_traffic_sign_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_vehicle_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_multitask_2dets_ped_cyc_detection_swinb_w12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_multitask_8dets_detection_swinb_w12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                [
                    "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                    "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                    "/horizon-bucket/auto_image_2/adas_data_raw/pack/white_changan/DG202_20210626_D/ADAS_20210626-114108_016_5/ADAS_20210626-114108_016_5__324945_1624679121346_0.jpg",
                ],
                None,
            ),
            pytest.param(
                "cloudmodel_multitask_8dets_mmefficientnetb5_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                [
                    "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                    "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                    "/horizon-bucket/auto_image_2/adas_data_raw/pack/white_changan/DG202_20210626_D/ADAS_20210626-114108_016_5/ADAS_20210626-114108_016_5__324945_1624679121346_0.jpg",
                ],
                None,
            ),
            pytest.param(
                "cloudmodel_multitask_sd_cone_detection_swinb_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                None,
            ),
            pytest.param(
                "cloudmodel_singletask_vehicle_side_detection_swinl_w12_bifpn_fcos",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_2/adas_data_raw/pack/white_changan/DG202_20210626_D/ADAS_20210626-114108_016_5/ADAS_20210626-114108_016_5__324945_1624679121346_0.jpg",
                None,
            ),
        ],
    )
    def test_detection_model_card(
        self,
        model_name,
        model_version,
        test_aidi_infer,
        test_local_infer,
        test_img_path,
        render_res_root,
    ):
        cfg = dict(
            model_name=model_name,
            model_version=model_version,
        )
        if test_local_infer:
            cfg.update(dict(run_device=DeviceMeta("gpu", 0)))
        example_input = self.make_test_images(test_img_path)
        # test run remote
        res_aidi_infer = None
        res_local_infer = None
        if test_aidi_infer:
            cfg.update(
                model_serving_config=dict(
                    service_path=f"{model_name}:CloudModelInferTest-{model_version}",
                    project_id="PD20230003",
                    resource_pool="model-deploy-titanx",
                    instance_min=0,
                    instance_max=1,
                )
            )

            inference = CloudModelInference(
                infer_mode=CloudModelInference.InferenceMode.AidiInferServiceRemote,
                init_config=cfg,
            )
            res_aidi_infer = [inference([i])[0] for i in example_input]
            if render_res_root:
                self.apply_render_result(
                    inference.result2cameraframe(
                        res_aidi_infer, example_input
                    ),
                    render_root=os.path.join(
                        render_res_root, inference.infer_mode.value
                    ),
                )
        if test_local_infer:
            inference = CloudModelInference(
                infer_mode=CloudModelInference.InferenceMode.InferModelRemote,
                init_config=cfg,
            )
            res_local_infer = [inference([i])[0] for i in example_input]
            if render_res_root:
                self.apply_render_result(
                    inference.result2cameraframe(
                        res_local_infer, example_input
                    ),
                    render_root=os.path.join(
                        render_res_root, inference.infer_mode.value
                    ),
                )

    @pytest.mark.parametrize(
        [
            "model_name",
            "model_version",
            "test_aidi_infer",
            "test_local_infer",
            "test_img_path",
            "render_res_root",
            "img_scale",
        ],
        [
            pytest.param(
                "cloudmodel_lane_instanceseg_swins_bifpn_solov2_prelabel",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                None,
                None,
            ),
            pytest.param(
                "cloudmodel_lane_instanceseg_swins_bifpn_solov2_datamining",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                None,
                None,
            ),
            pytest.param(
                "cloudmodel_parsing40cls_swinb_bifpn_semanticfpn",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                None,
                (2048, 1024),
            ),
            pytest.param(
                "cloudmodel_parsing40cls_swinb_bifpn_semanticfpn_polygon",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                None,
                (2048, 1024),
            ),
            pytest.param(
                "cloudmodel_lane_parsing_7cls_segmentation",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/adas_data_raw/pack/white_changan/UT126_20220820_D/ADAS_20220820-121239_985_5/ADAS_20220820-121239_985_5__833733_1660968859967_0.jpg",
                None,
                None,
            ),
            pytest.param(
                "cloudmodel_lane_instanceseg_badcase_parsing_13cls_segmentation",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_2/adas_data_raw/pack/white_changan/DG202_20210626_D/ADAS_20210626-114108_016_5/ADAS_20210626-114108_016_5__324945_1624679121346_0.jpg",
                None,
                None,
            ),
        ],
    )
    def test_segmentation_model_card(
        self,
        model_name,
        model_version,
        test_aidi_infer,
        test_local_infer,
        test_img_path,
        render_res_root,
        img_scale,
    ):
        cfg = dict(
            model_name=model_name,
            model_version=model_version,
        )
        if test_local_infer:
            cfg.update(dict(run_device=DeviceMeta("gpu", 0)))
        example_input = self.make_test_images(test_img_path, img_scale)
        # test run remote
        res_aidi_infer = None
        res_local_infer = None
        if test_aidi_infer:
            cfg.update(
                model_serving_config=dict(
                    service_path=f"{model_name}:CloudModelInferTest-{model_version}",
                    project_id="PD20230003",
                    resource_pool="model-deploy-titanx",
                    instance_min=0,
                    instance_max=1,
                )
            )

            inference = CloudModelInference(
                infer_mode=CloudModelInference.InferenceMode.AidiInferServiceRemote,
                init_config=cfg,
            )
            res_aidi_infer = [inference([i])[0] for i in example_input]
            if render_res_root:
                self.apply_render_result(
                    inference.result2cameraframe(
                        res_aidi_infer, example_input
                    ),
                    render_root=os.path.join(
                        render_res_root, inference.infer_mode.value
                    ),
                )
        if test_local_infer:
            inference = CloudModelInference(
                infer_mode=CloudModelInference.InferenceMode.InferModelRemote,
                init_config=cfg,
            )
            res_local_infer = [inference([i])[0] for i in example_input]
            if render_res_root:
                self.apply_render_result(
                    inference.result2cameraframe(
                        res_local_infer, example_input
                    ),
                    render_root=os.path.join(
                        render_res_root, inference.infer_mode.value
                    ),
                )

    @pytest.mark.parametrize(
        [
            "base_model_name",
            "base_model_version",
            "model_name",
            "model_version",
            "test_aidi_infer",
            "test_local_infer",
            "test_img_path",
            "render_res_root",
            "raw_byte",
        ],
        [
            pytest.param(
                "cloudmodel_vehicle_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_vehicle_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_rear_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_vehicle_light_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_rear_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_vehicle_rear_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_traffic_sign_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_traffic_sign_sub_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_traffic_sign_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_traffic_sign_medium_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_multitask_2dets_ped_cyc_detection_swinb_w12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_person_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_road_arrow_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_road_arrow_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_2/adas_data_raw/pack/white_changan/DG202_20210626_D/ADAS_20210626-114108_016_5/ADAS_20210626-114108_016_5__324945_1624679121346_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_traffic_cone_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_traffic_cone_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_traffic_light_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_traffic_light_primary_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_multitask_8dets_detection_swinb_w12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_traffic_light_secondary_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_traffic_sign_euro_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_traffic_sign_euro_medium_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/dataset/20230322/641aac07e6107ae52340b0dd/20211108T082334_8N1430_1636359883.013937.jpg",
                None,
                False,
            ),
            pytest.param(
                "cloudmodel_traffic_sign_euro_detection_swinbw_12_bifpn_fcos",
                "v0.7.0",
                "cloudmodel_traffic_sign_euro_sub_attribute",
                "v0.7.0",
                True,
                True,
                "/horizon-bucket/auto_image_6/dataset/20230322/641aac07e6107ae52340b0dd/20211108T082334_8N1430_1636359883.013937.jpg",
                None,
                False,
            ),
            pytest.param(
                None,
                None,
                "cloudmodel_work_condition_RN50",
                "v0.7.0",
                True,
                True,
                [
                    "/horizon-bucket/adas/lina.shen/hdflow_pack_image/7864/ADAS_20210628-205629_679_0_3071/ADAS_20210628-205629_679_0_3071__58796_1624885158637_0.jpg",
                    "/horizon-bucket/auto_jenkins_test/hdflow_workspace/test_modelzoo/ADAS_20220315-142130_023_4__29219_1647325476006_0.jpg",
                    "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20210304-165302_633_0__93449_1614848188663_0.jpg",
                ],
                None,
                True,
            ),
        ],
    )
    def test_classification_model_card(
        self,
        base_model_name,
        base_model_version,
        model_name,
        model_version,
        test_aidi_infer,
        test_local_infer,
        test_img_path,
        render_res_root,
        raw_byte,
    ):
        # run base detection model to generate rois
        cfg = dict(
            model_name=base_model_name,
            model_version=base_model_version,
            model_serving_config=dict(
                service_path=f"{base_model_name}:CloudModelInferTest-{base_model_version}",
                project_id="PD20230003",
                resource_pool="model-deploy-titanx",
                instance_min=0,
                instance_max=1,
            ),
        )
        example_input = self.make_test_images(test_img_path, raw_byte=raw_byte)
        if base_model_name and base_model_version:
            inference = CloudModelInference(
                infer_mode=CloudModelInference.InferenceMode.AidiInferServiceRemote,
                init_config=cfg,
            )
            res_detection = [inference([i])[0] for i in example_input]
            classification_inputs = inference.result2cameraframe(
                res_detection, example_input
            )
        else:
            classification_inputs = example_input

        cfg = dict(
            model_name=model_name,
            model_version=model_version,
        )
        if test_local_infer:
            cfg.update(dict(run_device=DeviceMeta("gpu", 0)))
        # test run remote
        res_aidi_infer = None
        res_local_infer = None
        if test_aidi_infer:
            cfg.update(
                model_serving_config=dict(
                    service_path=f"{model_name}:CloudModelInferTest-{model_version}",
                    project_id="PD20230003",
                    resource_pool="model-deploy-titanx",
                    instance_min=0,
                    instance_max=1,
                )
            )
            inference = CloudModelInference(
                infer_mode=CloudModelInference.InferenceMode.AidiInferServiceRemote,
                init_config=cfg,
            )
            res_aidi_infer = [inference([i])[0] for i in classification_inputs]
            if render_res_root:
                self.apply_render_result(
                    inference.result2cameraframe(
                        res_aidi_infer, classification_inputs
                    ),
                    render_root=os.path.join(
                        render_res_root, inference.infer_mode.value
                    ),
                )
        if test_local_infer:
            inference = CloudModelInference(
                infer_mode=CloudModelInference.InferenceMode.InferModelRemote,
                init_config=cfg,
            )
            res_local_infer = [
                inference([i])[0] for i in classification_inputs
            ]
            if render_res_root:
                self.apply_render_result(
                    inference.result2cameraframe(
                        res_local_infer, classification_inputs
                    ),
                    render_root=os.path.join(
                        render_res_root, inference.infer_mode.value
                    ),
                )


if __name__ == "__main__":
    pytest.main(["-s", "-x", f"{__file__}"])

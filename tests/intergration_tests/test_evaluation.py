import json
import os
import shutil

import cv2
import pytest
import yaml

from hat.evaluation.classification import evaluate as evaluate_cls
from hat.evaluation.detection2d.fp import evaluate as evaluate_fp
from hat.evaluation.detection2d.single import evaluate as evaluate_single
from hat.evaluation.detection3d.det3d import evaluate as evaluate_3d
from hat.evaluation.detection_bev.evaluate import evaluate as evaluate_bev_det
from hat.evaluation.detection_bev_rotbox import evaluate as evaluate_bev_rotbox
from hat.evaluation.semantic_segmentation import evaluate as seg_evaluate
from hat.visualize.detection2d.draw_fp_samples import (
    draw_sample,
    sort_samples_by_fp,
)
from hat.visualize.detection3d.draw_samples import (
    draw_sample as draw_det3d_sample,
)
from hat.visualize.detection3d.draw_samples import list_failure_samples
from hat.visualize.detection_bev.draw_samples import (
    draw_sample as draw_bev_det_sample,
)
from hat.visualize.detection_bev.draw_samples import (
    list_samples_order_by_drot,
    list_samples_order_by_dxyp,
    list_samples_order_by_fn,
    list_samples_order_by_fp,
    list_samples_order_by_tp,
)
from hat.visualize.detection_bev_rotbox.draw_samples import (
    draw_sample as draw_bev_rotbox_sample,
)
from hat.visualize.detection_bev_rotbox.draw_samples import (
    list_samples_order_by_drot as list_drot,
)
from hat.visualize.detection_bev_rotbox.draw_samples import (
    list_samples_order_by_dxy as list_dxy,
)
from hat.visualize.detection_bev_rotbox.draw_samples import (
    list_samples_order_by_fn as list_fn,
)
from hat.visualize.detection_bev_rotbox.draw_samples import (
    list_samples_order_by_fp as list_fp,
)
from hat.visualize.detection_bev_rotbox.draw_samples import (
    list_samples_order_by_tp as list_tp,
)
from hat.visualize.semantic_segmentation import render_image
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH
from tests.utils import execute_cmd


def reduce_samples_to_draw(samples):
    """Reduce samples numbers to draw.

    Note: The original samples may take too much time in pytest pipeline,
    this function reduces the numbers of samples to save testing time.
    Temporary handling method, later organize test cases of Evaluation.

    Args:
        samples: Original samples.

    Returns:
        reduced_samples: Reduced samples.
    """
    if len(samples) <= 0:
        return samples

    num = min(10, len(samples))
    return samples[:num]


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
class TestDetection2DEval:
    def setup_class(self):
        cmd = """
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/traffic_sign_mini_50.tgz . # noqa: E501
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/traffic_sign_smalltest_num1000.tar.gz . # noqa: E501
        tar xzf traffic_sign_mini_50.tgz
        tar xzf traffic_sign_smalltest_num1000.tar.gz
        rm traffic_sign_mini_50.tgz
        rm traffic_sign_smalltest_num1000.tar.gz
        """
        execute_cmd(cmd)

    def test_single(self):
        gt_file = "traffic_sign_mini_50/gt.json"
        det_file = "traffic_sign_mini_50/predict.json"
        config_file = "traffic_sign_mini_50/config_test_sep.yaml"
        output_dir = "output/traffic_sign_mini_sep_single/"
        summary = evaluate_single(gt_file, det_file, config_file, output_dir)
        assert abs(summary["mAP"] - 0.17746719361722782) < 1e-6

    def test_detection_mode(self):
        gt_file = "traffic_sign_smalltest_num1000/gt.json"
        det_file = "traffic_sign_smalltest_num1000/predict.json"

        test_examples = {
            "DET_CLS_SEP": [
                {
                    "gt_file": gt_file,
                    "det_file": det_file,
                    "config_file": "traffic_sign_smalltest_num1000/cnts-det-cls258-eval-profile-det-cls-all-sep-template.yaml",  # noqa E501,
                    "output_dir": "output/traffic_sign_det_cls_all_sep/",
                    "results": {
                        "mAP": 0.290470587983939,
                        "mRecall": 0.2907162001277866,
                        "mPrecision": 0.3096507583340558,
                    },
                },
                {
                    "gt_file": gt_file,
                    "det_file": det_file,
                    "config_file": "traffic_sign_smalltest_num1000/cnts-det-cls258-eval-profile-det-cls-sep-template.yaml",  # noqa E501,
                    "output_dir": "output/traffic_sign_det_cls_sep/",
                    "results": {
                        "mAP": 0.290470587983939,
                        "mRecall": 0.2907162001277866,
                        "mPrecision": 0.3096507583340558,
                    },
                },
            ],
            "DET_CLS_SINGLE": [
                {
                    "gt_file": gt_file,
                    "det_file": det_file,
                    "config_file": "traffic_sign_smalltest_num1000/cnts-det-cls258-eval-profile-det-cls-all-single-template.yaml",  # noqa E501,
                    "output_dir": "output/traffic_sign_det_cls_all_single/",
                    "results": {"AP": 0.9624999999975482, "AR": 0},
                },
                {
                    "gt_file": gt_file,
                    "det_file": det_file,
                    "config_file": "traffic_sign_smalltest_num1000/cnts-det-cls258-eval-profile-det-cls-single-template.yaml",  # noqa E501,
                    "output_dir": "output/traffic_sign_det_cls_single/",
                    "results": {"AP": 0.9624999999975482, "AR": 0},
                },
            ],
            "DET_CLS_ALL": [
                {
                    "gt_file": gt_file,
                    "det_file": det_file,
                    "config_file": "traffic_sign_smalltest_num1000/cnts-det-cls258-eval-profile-det-cls-all-template.yaml",  # noqa E501,
                    "output_dir": "output/traffic_sign_det_cls_all/",
                    "results": {
                        "AP": 0.8125639097918721,
                        "AR": 0.260266126438668,
                    },
                }
            ],
            "DET": [
                {
                    "gt_file": gt_file,
                    "det_file": det_file,
                    "config_file": "traffic_sign_smalltest_num1000/cnts-det-cls258-eval-profile-det-template.yaml",  # noqa E501,
                    "output_dir": "output/traffic_sign_det/",
                    "results": {
                        "AP": 0.8919094615695543,
                        "AR": 0.5354456035155177,
                    },
                }
            ],
        }

        for mode, examples in test_examples.items():
            for kwargs in examples:
                results = kwargs.pop("results")
                print(kwargs)
                summary = evaluate_single(**kwargs)
                print(summary)
                if mode == "DET_CLS_SEP":
                    assert abs(summary["mAP"] - results["mAP"]) < 1e-6
                    assert abs(summary["mRecall"] - results["mRecall"]) < 1e-6
                    assert (
                        abs(summary["mPrecision"] - results["mPrecision"])
                        < 1e-6
                    )  # noqa E501
                else:
                    assert abs(summary["AP"] - results["AP"]) < 1e-6
                    assert abs(summary["AR"] - results["AR"]) < 1e-6


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
class TestDetection3DEval:
    def setup_class(self):
        cmd = """
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/det3d_example_data.tar.gz . # noqa: E501
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/det3d_example_data_w_tag.tar.gz . # noqa: E501
        tar xzf det3d_example_data.tar.gz
        tar xzf det3d_example_data_w_tag.tar.gz
        rm det3d_example_data.tar.gz
        rm det3d_example_data_w_tag.tar.gz
        """
        execute_cmd(cmd)

    def test_detection3d(self):
        gt_file = "example_data/gt.json"
        det_file = "example_data/pred.json"
        config_file = "example_data/setting.yaml"
        output_dir = "output/vehicle_3d_detection/"

        # test evaluate
        summary = evaluate_3d(gt_file, det_file, config_file, output_dir)
        print(summary)
        assert abs(summary[0]["AP_3D"] - round(0.27037030197532885, 4)) < 1e-6
        assert abs(summary[0]["AR_3D"] - round(0.03902230631450243, 4)) < 1e-6
        assert abs(summary[1]["AP"] - round(0.9999998666666803, 4)) < 1e-6
        assert abs(summary[1]["AR"] - round(0.20068610753050778, 4)) < 1e-6

        # test draw sample
        samples = list_failure_samples(
            open("output/vehicle_3d_detection/all.json")
        )
        all_results = json.load(open("output/vehicle_3d_detection/all.json"))

        for img_path, sample in zip(all_results["images"], samples):
            img_path = os.path.join("example_data/data", sample["image_key"])
            img = cv2.imread(img_path)
            img = draw_det3d_sample(img, sample)
            cv2.imwrite(
                os.path.join(
                    "output/vehicle_3d_detection", sample["image_key"]
                ),
                img,
            )

    def test_detection3d_with_tag(self):
        gt_file = "example_data/gt_w_tag.json"
        det_file = "example_data/pred.json"
        config_file = "example_data/setting_w_tag.yaml"
        output_dir = "output/vehicle_3d_detection_with_tag/"

        # test evaluate
        summary = evaluate_3d(gt_file, det_file, config_file, output_dir)
        print(summary)
        assert abs(summary[0]["AP_3D"] - round(0.1667, 4)) < 1e-6
        assert abs(summary[0]["AR_3D"] - round(0.0154, 4)) < 1e-6
        assert abs(summary[1]["AP"] - round(1.0, 4)) < 1e-6
        assert abs(summary[1]["AR"] - round(0.1003, 4)) < 1e-6

        # test draw sample
        samples = list_failure_samples(open(f"{output_dir}/all.json"))
        all_results = json.load(open(f"{output_dir}/all.json"))

        for img_path, sample in zip(all_results["images"], samples):
            img_path = os.path.join("example_data/data", sample["image_key"])
            img = cv2.imread(img_path)
            img = draw_det3d_sample(img, sample)
            cv2.imwrite(
                os.path.join(output_dir, sample["image_key"]),
                img,
            )


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
class TestBEVDetectionEval:
    def setup_class(self):
        cmd = """
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/bev_det_example_data.tar.gz . # noqa: E501
        tar xzf bev_det_example_data.tar.gz
        rm bev_det_example_data.tar.gz
        """
        execute_cmd(cmd)

    def test_bev_detection(self):
        img_dir = "bev3d_aidieval_example/data"
        gt_file = "bev3d_aidieval_example/mini_gt.json"
        det_file = "bev3d_aidieval_example/mini_pred.json"

        config_file = "bev3d_aidieval_example/setting.yaml"
        output_dir = "output/vehicle_bev_detection/"
        all_gt_results = "output/vehicle_bev_detection/gts_result.json"
        all_pred_results = "output/vehicle_bev_detection/preds_result.json"

        # test evaluate
        summary = evaluate_bev_det(gt_file, det_file, config_file, output_dir)
        print(summary)

        assert abs(summary["AP_3D"] - 0.7172) < 1e-3
        assert abs(summary["APH_3D"] - 0.6668) < 1e-3
        assert abs(summary["AR_3D"] - 0.1218) < 1e-3

        assert abs(summary["mAP_3D"] - 0.0553) < 1e-3
        assert abs(summary["mAPH_3D"] - 0.1143) < 1e-3
        assert abs(summary["mAR_3D"] - 0.0041) < 1e-3

        assert abs(summary["temporal_dx"] - (0.0347)) < 1e-3
        assert abs(summary["temporal_dy"] - (-0.0034)) < 1e-3

        # test draw sample
        def get_multi_img(data):
            img_dict = {}
            img_meta = data["img_meta"]
            for key in img_meta.keys():
                img_path = os.path.join(img_dir, key + ".jpg")
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img_dict[key] = image
            return img_dict

        for func in [
            list_samples_order_by_tp,
            list_samples_order_by_fp,
            list_samples_order_by_fn,
            list_samples_order_by_drot,
            list_samples_order_by_dxyp,
        ]:
            samples = func(all_gt_results, all_pred_results)

            # Note: reduce samples to reduce test time.
            samples = reduce_samples_to_draw(samples)
            for sample in samples:
                draw_bev_det_sample(get_multi_img(sample), sample)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
class TestBEVRotBoxDectionEval:
    def test_bev_detection(self):
        img_dir = os.path.join(
            HAT_BUCKET_PATH,
            "pipeline_test_data/fsd/bev_rotbox/bev_rotbox_example_data/data",  # noqa
        )
        gt_file = os.path.join(
            HAT_BUCKET_PATH,
            "pipeline_test_data/fsd/bev_rotbox/bev_rotbox_example_data/gt.json",  # noqa
        )
        det_file = os.path.join(
            HAT_BUCKET_PATH,
            "pipeline_test_data/fsd/bev_rotbox/bev_rotbox_example_data/pred.json",  # noqa
        )
        config_file = os.path.join(
            HAT_BUCKET_PATH,
            "pipeline_test_data/fsd/bev_rotbox/bev_rotbox_example_data/setting.yaml",  # noqa
        )
        output_dir = "output/bev_rotbox_detection/"
        all_results = "output/bev_rotbox_detection/result.json"

        summary = evaluate_bev_rotbox(
            gt_file, det_file, config_file, output_dir
        )
        print(summary)
        assert abs(summary["AP"] - 0.924) < 1e-3

        # test draw sample
        def get_multi_img(data):
            img_dict = {}
            img_meta = data["img_meta"]
            for key in img_meta.keys():
                img_path = os.path.join(img_dir, key + ".jpg")
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img_dict[key] = image
            return img_dict

        for func in [
            list_tp,
            list_fp,
            list_fn,
            list_drot,
            list_dxy,
        ]:
            samples = func(all_results)
            # Note: reduce samples to reduce test time.
            samples = reduce_samples_to_draw(samples)
            for sample in samples:
                draw_bev_rotbox_sample(get_multi_img(sample), sample)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
class TestSemanticSegEval:
    def setup_class(self):
        cmd = """
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/seg_example_data.tar.gz . # noqa: E501
        tar -xzf seg_example_data.tar.gz
        rm -rf seg_example_data.tar.gz
        """
        execute_cmd(cmd)

    def test_seg_metric(self):
        test_dict = {
            "outfile_dir": "example_result/segmentation_res_full_region_freespace",  # noqa: E501
            "gt_dir": "seg_example_data/example_data/test_set/labels",
            "images_dir": "seg_example_data/example_data/test_set/images",
            "pred_dir": "seg_example_data/example_data/preds",
            "config_file": "seg_example_data/example_data/parsing_cls125_freespace_eval.yaml",  # noqa: E501
        }
        gt_dir = test_dict["gt_dir"]
        pred_dir = test_dict["pred_dir"]
        config_file = test_dict["config_file"]
        outfile_dir = test_dict["outfile_dir"]
        images_dir = test_dict["images_dir"]
        if os.path.exists(outfile_dir):
            shutil.rmtree(outfile_dir)
        os.makedirs(outfile_dir)
        if "images_json" in test_dict.keys():
            result = seg_evaluate(
                images_dir,
                gt_dir,
                pred_dir,
                config_file,
                outfile_dir,
                images_json=test_dict["images_json"],
            )
        else:
            result = seg_evaluate(
                images_dir, gt_dir, pred_dir, config_file, outfile_dir
            )
        assert abs(result[0]["value"] - 0.5718) < 1e-3
        assert abs(result[1]["value"] - 0.9834) < 1e-3
        render_output = os.path.join(outfile_dir, "render_result")
        if not os.path.isdir(render_output):
            os.makedirs(render_output)
        labels_info = json.load(
            open(os.path.join(outfile_dir, "labels.json"))
        )["labels"]
        config_dict = yaml.load(open(config_file, "r"))
        add_vis_info = config_dict.get("add_vis", None)
        with open(os.path.join(outfile_dir, "images.json"), "r") as f:
            lines = f.readlines()
            for line in lines:
                sample_eval = json.loads(line.strip())
                image_path = os.path.join(
                    outfile_dir, "res_image", sample_eval["image_name"]
                )
                diff_path = os.path.join(
                    outfile_dir,
                    "res_diff",
                    os.path.splitext(sample_eval["image_name"])[0] + ".png",
                )
                gt_label_path = os.path.join(
                    outfile_dir,
                    "res_gt_label",
                    os.path.splitext(sample_eval["image_name"])[0] + ".png",
                )
                pred_label_path = os.path.join(
                    outfile_dir,
                    "res_pred_label",
                    os.path.splitext(sample_eval["image_name"])[0] + ".png",
                )
                image_map = cv2.imread(image_path, -1)
                diff_map = cv2.imread(diff_path, -1)
                gt_label_map = cv2.imread(gt_label_path, -1)
                pred_label_map = cv2.imread(pred_label_path, -1)
                gt_freespace_line, pred_freespace_line = sample_eval.get(
                    "gt_freespace_line"
                ), sample_eval.get("pred_freespace_line")
                rendered_image = render_image(
                    image_map,
                    diff_map,
                    gt_label_map,
                    pred_label_map,
                    gt_freespace_line,
                    pred_freespace_line,
                    labels_info,
                    image_map,
                    add_vis_info,
                )
                cv2.imwrite(
                    os.path.join(render_output, sample_eval["image_name"]),
                    rendered_image,
                )


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
class TestDetection2DFPEval:
    def setup_class(self):
        cmd = """
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/person_posneg_classification_pytest.tar.gz . # noqa: E501
        tar xzf person_posneg_classification_pytest.tar.gz
        rm person_posneg_classification_pytest.tar.gz
        """
        execute_cmd(cmd)

    def test_single(self):
        gt_file = "data/person_posneg_classification/gt.json"
        det_file = "data/person_posneg_classification/pred.json"
        config_file = "data/person_posneg_classification/all.yaml"
        output_dir = "outputs/person_posneg_classification/"

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        summary = evaluate_fp(gt_file, det_file, config_file, output_dir)
        assert abs(summary["micro FP"] - 0.08672829833907866) < 1e-6
        assert abs(summary["macro FP"] - 0.12390023198258851) < 1e-6

        # test draw sample
        with open(f"{output_dir}/samples.json") as fin:
            result = json.load(fin)

        samples = sort_samples_by_fp(result, fp_type="other")
        top_3_samples = samples[:3]

        for sample in top_3_samples:
            image_key = sample["image_key"]
            image = cv2.imread(
                os.path.join("data/person_posneg_classification", image_key)
            )
            draw_sample(image, sample)
            output_image_path = os.path.join(
                output_dir, image_key.replace("/", "__")
            )
            cv2.imwrite(output_image_path, image)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
class TestClassificationEval:
    def setup_class(self):
        cmd = """
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/evaluation/person_orientation_classification_pytest.tar.gz . # noqa: E501
        tar xzf person_orientation_classification_pytest.tar.gz
        rm person_orientation_classification_pytest.tar.gz
        """
        execute_cmd(cmd)

    def test_cls(self):
        gt_file = "data/person_orientation_classification/gt.json"
        pred_file = "data/person_orientation_classification/pred.json"

        test_examples = {
            "CLS": [
                {
                    "gt_path": gt_file,
                    "pred_path": pred_file,
                    "config_path": "data/person_orientation_classification/All_0-110m.yaml",  # noqa E501,
                    "output_dir": "outputs/all/",
                    "image_dir": "outputs/all/",
                    "results": {
                        "mPre": 0.6991380772871387,
                        "mRec": 0.7057925402873582,
                        "ACC": 0.7062722276107339,
                    },
                },
                {
                    "gt_path": gt_file,
                    "pred_path": pred_file,
                    "config_path": "data/person_orientation_classification/All_0-110m_fuzzy.yaml",  # noqa E501,
                    "output_dir": "outputs/all_fuzzy/",
                    "image_dir": "outputs/all_fuzzy/",
                    "results": {
                        "mPre": 0.9320008567859043,
                        "mRec": 0.9343282858642079,
                        "ACC": 0.9354186873585516,
                    },
                },
            ],
        }

        outfile_dir = "outputs"
        if os.path.exists(outfile_dir):
            shutil.rmtree(outfile_dir)
        os.makedirs(outfile_dir)

        EPS = 1e-6
        for _, examples in test_examples.items():
            for kwargs in examples:
                results = kwargs.pop("results")
                print(kwargs)
                summary = evaluate_cls(**kwargs)
                print(summary)

                assert abs(summary["mPre"] - results["mPre"]) < EPS
                assert abs(summary["mRec"] - results["mRec"]) < EPS
                assert abs(summary["ACC"] - results["ACC"]) < EPS

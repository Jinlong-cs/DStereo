import copy
import json
import os.path as osp
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from hat.metrics.lidar_eval_utils import (
    HTMLGenerator,
    IoUMatching,
    dump_log_file,
    filter_by_range,
)
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

__all__ = ["LidarDetEval"]


def lidar_dist_reduce_fx(x: List[Dict]) -> List[Dict]:
    """Lidar eval result dist reduce function."""

    work_size = len(x[0]["token"])
    reduce_results = []
    for elem_dict in x:
        for i in range(work_size):
            token = elem_dict["token"][i].cpu().numpy().tobytes()
            box3d_lidar = elem_dict["box3d_lidar"][i]
            scores = elem_dict["scores"][i]
            label_preds = elem_dict["label_preds"][i]
            result = {
                "token": token,
                "box3d_lidar": box3d_lidar.detach().cpu().numpy(),
                "scores": scores.detach().cpu().numpy(),
                "label_preds": label_preds.detach().cpu().numpy(),
            }
            reduce_results.append(result)
    return reduce_results


@OBJECT_REGISTRY.register_module
class LidarDetEval(EvalMetric):
    """Lidar evaluation metric."""

    def __init__(
        self,
        id2name: Dict[int, str],
        gt_classes_map: Dict[str, str],
        gt_pkl: str,
        work_dir: str,
        matching: Dict[str, Any],
        workflow: List[Dict[str, Any]],
        all_range_limit: List[int] = (-1000, -1000, 1000, 1000),
        save_log: bool = False,
        print_result: bool = True,
        dump_result: bool = False,
        print_fp_analysis: bool = False,
        generate_html: bool = False,
        html_content: Iterable = None,
        test_orientation: Optional[Tuple[str]] = ("all",),
        test_rear_range: Optional[List[List]] = None,
        use_bev_matching: bool = False,
        save_pkl_path: Optional[str] = None,
    ):
        """Initialize a LidarDetEval callback.

        Args:
            id2name (Dict[int, str]): a dictionary that maps integer label to
                string names.
            gt_classes_map (Dict[str, str]): a dictionary that maps string
                class names to another set of names. Usually used to merge
                classes.
            gt_pkl (str): ground-truth pickle file path.
            work_dir (str): working directory. Evaluation results will be saved
                there.
            matching (Dict[str, Any]): a dictionary config for the matching
                module.
            workflow (List[Dict[str, Any]]): a list of dictionaries, each
                configures a evaluation workflow module.
            all_range_limit (List[int], optional): pc range limit.
                Defaults to (-1000, -1000, 1000, 1000).
            save_log (bool, optional): whether to save log. Defaults to False.
            print_result (bool, optional): whether to print result in console.
                Defaults to True.
            dump_result (bool,optionak): whether to generate result file.
                Defaults to False.
            print_fp_analysis (bool, optional): whether to print fp analysis.
                Defaults to False.
            generate_html (bool, optional): whether to generate an HTML report.
                Defaults to False.
            html_content (Iterable, optional): A sequence of HTML content
                config. Defaults to None.
            test_orientation: test orientation.
                Can choose one or more of ('rear','front','all'),defaults to ('all',).
                if front,Only select targets and predictions with x greater than 0 when testing.
                if rear,Only select targets and predictions with x less than 0 when testing.
                if all,select all targets and predicionts when testing.
            test_rear_range: rear test range,
                if test_orientation is rear,should use the parameter.
            use_bev_matching: use bev iou matching if True,otherwise use 3d iou matching.
            save_pkl_path: if save_pkl_path is not None,
                infer pkl will save in save_pkl_path.
        """  # noqa
        super(LidarDetEval, self).__init__("LidarDetEval")
        self.id2name = id2name
        self.gt_pkl = gt_pkl
        self.class_names = list(id2name.values())
        self.gt_classes_map = gt_classes_map
        self.matching_module = IoUMatching(
            matching["class_names"], matching["iou_thresholds"]
        )
        self.all_range_limit = all_range_limit
        self.work_dir = work_dir
        self.test_orientation = test_orientation
        self.test_rear_range = test_rear_range
        # Build evaluation pipeline
        self.eval_pipes = []
        self.eval_pipes_rear = []
        for eval_pipe in workflow:
            self.eval_pipes.append(eval_pipe)
        if use_bev_matching:
            workflow[0]["bev_metric"] = True
        if "rear" in test_orientation:
            workflow_rear = copy.deepcopy(workflow)
            workflow_rear[0]._range = self.test_rear_range
            for eval_pipe in workflow_rear:
                self.eval_pipes_rear.append((eval_pipe))

        # Output options
        self.epoch_id = 0
        self.save_log = save_log
        self.print_result = print_result
        self.dump_result = dump_result
        self.print_fp_analysis = print_fp_analysis
        self.generate_html = generate_html
        self.html_content = html_content
        self.save_pkl_path = save_pkl_path
        self.reset()
        self.gt_raw = []

    def _init_states(self):

        self.add_state(
            "all_predictions", default=[], dist_reduce_fx=lidar_dist_reduce_fx
        )

    def reset(self):
        super().reset()
        self.epoch_id = 0

    def update(self, model_outs: Dict, batch: Dict, task_name="out"):
        """Save batch predictions and gt during batch end."""
        # Store each batch output and gt
        for output in model_outs:
            out_dict = output._asdict()  # namedtuple to dict

            token = out_dict[f"{task_name}_0_token"]
            bbox_ret = out_dict[f"{task_name}_0_box3d_lidar"]
            scores = out_dict[f"{task_name}_0_scores"]
            label_ret = out_dict[f"{task_name}_0_label_preds"]

            filename_token = token.split("/")[-1].split(".")[0]
            token_tensors = torch.ByteTensor(
                list(bytes(filename_token, "utf8"))
            ).to(bbox_ret.device)
            result = {
                "token": token_tensors,
                "box3d_lidar": bbox_ret,
                "scores": scores,
                "label_preds": label_ret,
            }
        self.all_predictions.append(result)

        gt = batch[0]["annos_dict"]
        gt.update(lidar_path=batch[0]["object_token"][0])
        self.gt_raw.append(gt)

    def compute(self):
        """Perform evaluation on epoch end."""
        if self.save_pkl_path is not None:
            with open(self.save_pkl_path, "wb") as f:
                pickle.dump(self.all_predictions, f)
        for orient in self.test_orientation:
            # Evaluation happens at end of each epoch
            self.results = {}
            all_predictions = self._reformat_prediction(
                test_orientation=orient
            )
            all_gts = self._reformat_gt(test_orientation=orient)
            # initial filter given overall range
            all_gts, all_predictions = filter_by_range(
                all_gts, all_predictions, None, self.all_range_limit
            )

            (
                all_gts,
                all_predictions,
            ) = self.matching_module.prediction_matching(
                all_gts, all_predictions
            )

            # if orient is rear,use eval_pipe_rear,otherwise use eval_pipe.
            if orient == "rear":
                for eval_pipe_rear in self.eval_pipes_rear:
                    result = eval_pipe_rear.run(all_gts, all_predictions)
                    if eval_pipe_rear.result_name is not None:
                        self.results[eval_pipe_rear.result_name] = result
            else:
                for eval_pipe in self.eval_pipes:
                    result = eval_pipe.run(all_gts, all_predictions)
                    if eval_pipe.result_name is not None:
                        self.results[eval_pipe.result_name] = result

            # Dump result
            if "aph_result" in self.results and self.dump_result:
                self._dump_result(self.epoch_id)
                self.epoch_id = self.epoch_id + 1
            if self.generate_html:
                self._save_to_html()
            if self.print_result:
                self._print_result()
            if self.save_log:
                self._save_log_file()

    @staticmethod
    def load_pickle(pkl_path: str):
        """Load a pickle file."""
        with open(pkl_path, "rb") as f:
            info = pickle.load(f)
        return info

    def _reformat_prediction(self, test_orientation: str = "all"):
        """Reformat cached predictions.

        Args:
            test_orientation: test orientation.
                if front,Only select targets and  with x greater than 0 when testing.
                if rear,Only select targets and predictions with x less than 0 when testing.
                if all,select all targets and predicionts when testing.
        """  # noqa

        pred_raw = sorted(self.all_predictions, key=lambda x: x["token"][0])
        predictions = []
        for j in range(len(pred_raw)):
            for i in range(len(pred_raw[j]["box3d_lidar"])):
                # for fast compute
                if pred_raw[j]["scores"][i] <= 0.1:
                    continue
                cur_box = pred_raw[j]["box3d_lidar"][i]
                x = cur_box[0]
                if test_orientation == "front":
                    if x <= 0:
                        continue
                elif test_orientation == "rear":
                    if x > 0:
                        continue
                name = self.id2name[pred_raw[j]["label_preds"][i]]
                if name.lower() != "skip":
                    predictions.append(
                        {
                            "token": pred_raw[j]["token"],
                            "box": cur_box,
                            "name": name,
                            "score": pred_raw[j]["scores"][i],
                        }
                    )
        return predictions

    def _reformat_gt(self, test_orientation: str = "all"):
        """Load and reformat ground-truth pickle.

        Args:
            test_orientation: test orientation.
                if front,Only select targets and  with x greater than 0 when testing.
                if rear,Only select targets and predictions with x less than 0 when testing.
                if all,select all targets and predicionts when testing.
        """  # noqa
        gts = []
        gt_raw = self.gt_raw
        for j in range(len(gt_raw)):
            if gt_raw[j]["gt_boxes"] is None:
                continue
            for i in range(len(gt_raw[j]["gt_boxes"])):
                cur_box = gt_raw[j]["gt_boxes"][i]
                class_name = gt_raw[j]["gt_names"][i]
                if class_name not in self.gt_classes_map:
                    continue
                x = cur_box[0]
                if test_orientation == "front":
                    if x <= 0:
                        continue
                elif test_orientation == "rear":
                    if x > 0:
                        continue
                token = gt_raw[j]["lidar_path"]
                filename_token = token.split("/")[-1].split(".")[0]
                token_bytes = (
                    torch.ByteTensor(list(bytes(filename_token, "utf8")))
                    .numpy()
                    .tobytes()
                )
                gts.append(
                    {
                        "token": token_bytes,
                        "box": cur_box,
                        "name": self.gt_classes_map[class_name],
                    }
                )
        return gts

    def _dump_result(self, global_id: int):
        """Dump epoch evaluation result to a json file.

        Args:
            output_path (str): output directory.
            global_id (int): epoch id for the result.
        """
        output_path = osp.join(
            self.work_dir, "det_result_epoch{}.json".format(global_id)
        )
        with open(output_path, "w") as f:
            json.dump(self.results["aph_result"][1], f, indent=4)

    def _save_to_html(self):
        """Save analysis reports to an HTML file."""
        assert self.html_content is not None
        # Generate HTML
        html_path = osp.join(self.work_dir, "report.html")
        html_generator = HTMLGenerator(html_path)
        for html_element in self.html_content:
            element, *content = html_element
            if element == "style":
                html_generator.add_style(*content)
            elif element == "heading":
                html_generator.add_heading(*content)
            elif element == "table":
                try:
                    result_key = content[0].split("#")
                    if len(result_key) == 1:
                        content[0] = self.results[result_key]
                    else:
                        content[0] = self.results[result_key[0]][
                            int(result_key[1]) - 1
                        ]
                    html_generator.add_table(*content)
                except (KeyError, TypeError):
                    print(
                        f"Table {content[0]} is skipped as it is not in "
                        f"results."
                    )
            elif element == "image":
                html_generator.add_imgs(*content)

        html_generator.write()
        print(f"HTML report has been generated to {html_path}.")

    def _print_result(self):
        """Print result to console."""
        result = self.results["aph_result"][1]
        for range_key in result:
            print(
                "***************For range {}***************".format(range_key)
            )
            for key in result[range_key]:
                if (key in ["fp_metrics"]) and self.print_fp_analysis:
                    print(result[range_key][key])
                else:
                    print(
                        "{}: {}, {}".format(
                            key,
                            result[range_key][key][0] * 100,
                            result[range_key][key][1] * 100,
                        )
                    )

    def _save_log_file(self):
        """Save log file."""
        # currently only implemented for horizon 8 class
        if len(self.id2name) != 8:
            return

        output_file_name = "python_result_{}.log".format(
            "_".join(str(i) for i in self.all_range_limit)
        )
        output_log_file = osp.join(self.work_dir, "eval_log", output_file_name)
        dump_log_file(
            output_log_file, self.results["aph_result"][1]["0m~1000m"]
        )

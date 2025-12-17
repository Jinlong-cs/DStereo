import os
import random
import shutil
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from hatbc.utils.utils import _as_list
from prettytable import PrettyTable
from tqdm import tqdm

from hat.data.datasets.densebox_dataset import DenseboxDataset
from projects.cloudmodel.configs.perception_2d.data.datasets import (
    all_datasets,
)
from projects.cloudmodel.configs.perception_2d.data.structures import (
    CloudModelDatasetItem,
    DataSplit,
    DenseboxSource,
    LMDBSource,
    SourceDataProject,
)
from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import (
    CloudModelTask,
)


class HorizonDataset(object):
    """This class takes project and task infos as input, and can be used to
    get training datasets, val datasets and test datasets from different
    projects.

    Args:
        projects: project names, e.g., "mono", "pilot" or ("mono", "pilot").
        tasks: tasks defined in cloudmodel. Type CloudModelTask e.g.,
            "cyclist_detection", "person_detection",
            "semantic_parsing_40cls_segmentation", or "lane_instanceseg". These
            names have a general format that is "class_task", so please write
            your task_names in such a format.
        source_dataset_items: a list of CloudModelDatasetItem that contains
            all useful infos for each train data(e.g. a rec/lmdb).
    """

    def __init__(
        self,
        tasks: Union[CloudModelTask, List[CloudModelTask]],
        projects: Union[str, List[str]] = SourceDataProject.mono.value,
        source_dataset_items: List[CloudModelDatasetItem] = all_datasets,
    ):
        projects = _as_list(projects)
        tasks = _as_list(tasks)
        assert all(
            [i in SourceDataProject._member_names_ for i in projects]
        ), f"Invalid dataset projects: {projects} all valid projects: {SourceDataProject._member_names_}"  # noqa
        self.source_dataset_items = source_dataset_items
        self.projects = sorted(set(projects), key=len)
        self.tasks = list(set(tasks))
        # filter dataset items from all datasets by given task and projects
        self.source_dataset_items = self.filter_dataset_item(
            tasks=tasks,
            projects=projects,
        )

    @staticmethod
    def category_from_task(tasks: List[CloudModelTask]):
        #  in a detection head.
        instance_ids = list({j for i in tasks for j in i.class_id})
        ins_id2category = {i_id: idx for idx, i_id in enumerate(instance_ids)}
        return ins_id2category

    def filter_dataset_item(
        self,
        tasks: Optional[List[CloudModelTask]] = None,
        projects: Optional[List[str]] = None,
        mode: Optional[List[str]] = None,
    ) -> List[CloudModelDatasetItem]:
        def _check_valid_query():
            if tasks:
                assert all(
                    [i in self.tasks for i in tasks]
                ), f"task: {tasks} is not supported!, all supported:{self.tasks}"  # noqa
            if projects:
                assert all(
                    [i in self.projects for i in projects]
                ), f"project: {projects} is not supported!, all supported:{self.projects}"  # noqa
            if mode:
                assert all(
                    [i in DataSplit._member_names_ for i in mode]
                ), f"mode: {mode} is not supported!, all supported mode:{DataSplit._member_names_}"  # noqa

        _check_valid_query()

        valid_datasets: List[
            CloudModelDatasetItem
        ] = self.source_dataset_items  # noqa
        # filter by given info
        if tasks:
            valid_datasets = [i for i in valid_datasets if i.task in tasks]
        if projects:
            valid_datasets = [
                i for i in valid_datasets if i.project.value in projects
            ]
        if mode:
            assert all(
                [i in DataSplit._member_names_ for i in mode]
            ), f"mode: {mode} is not supported!, all supported mode:{DataSplit._member_names_}"  # noqa
            valid_datasets = [i for i in valid_datasets if i.split in mode]

        return valid_datasets

    def build_dataset(
        self,
        mode: str = "train",
        projects: Union[str, List[str]] = None,
        tasks: Union[CloudModelTask, List[CloudModelTask]] = None,
        to_rgb: bool = True,
        use_ignore: bool = True,
        remove_det_duplicate: bool = True,
        abandon_other_category: bool = True,
        transforms: List[Dict] = None,
        fast_debug: int = None,
        use_all_data: bool = True,
    ):
        """This function builds datasets for training, validation and testing.

        Args:
            mode: "Train", "val" or "test". It works with `projects` to access
                proj_mode data in self.data_paths.
            projects: Which projects' data you want to use, e.g. ["mono"].
            tasks: What task data you want to fetch, e.g. cloudmodel.structures.AllTasks.person_detection
            to_rgb: Whether to convert image to the "rgb" format.
            use_ignore: Whether to fetch ig_bboxes.
            remove_det_duplicate: Whether to remove duplicate gts.
            abandon_other_category: Whether to throw away unused categories.
            transforms: Data transforms, e.g. resize, crop.
            fast_debug: If you want to do fast debugging, you can just use the
                number of `fast_debug` datasets.
            use_all_data: Return all filtered dataset.
                Note. If set to true, fast_debug will be shortcut.
        Returns:
            A list of datasets.

        """  # noqa
        projects = _as_list(projects) if projects else self.projects
        tasks = _as_list(tasks) if tasks else self.tasks
        dataset_items = self.filter_dataset_item(
            tasks=tasks, projects=projects, mode=[mode]
        )
        # build AIDI Eval datasets
        if mode == "test":
            datasets = [
                i.get_default_dataset_config(
                    transforms=transforms,
                    to_rgb=to_rgb,
                )
                for i in dataset_items
            ]
            return datasets
        ins_id2category = self.category_from_task(tasks)

        # build detection, semantic segmentation or instance segmentation
        # training or validation datasets
        if not use_all_data:
            # ## fast debugging with the first fast_debug datasets
            if fast_debug is not None and fast_debug > 0:
                i = min(len(dataset_items), fast_debug)
                dataset_items = dataset_items[:i]

        datasets = [
            item.get_default_dataset_config(
                transforms=transforms,
                ins_id2category=ins_id2category,
                to_rgb=to_rgb,
                use_ignore=use_ignore,
                remove_det_duplicate=remove_det_duplicate,
                abandon_other_category=abandon_other_category,
            )
            for item in dataset_items
        ]

        return datasets

    def get_val_coco_anno_path(
        self,
        task: Union[CloudModelTask, List[CloudModelTask]],
        project: Union[str, List[str]],
    ):
        tasks = _as_list(task)
        projects = _as_list(project)
        assert all(
            [i in self.tasks for i in tasks]
        ), f"The datasets support task : {self.tasks} but no: {tasks}."
        assert all(
            [i in self.projects for i in projects]
        ), f"The datasets support projects: {self.projects} but not: {projects}."  # noqa
        task_dataset_item: List[
            CloudModelDatasetItem
        ] = self.filter_dataset_item(
            tasks=tasks,
            projects=projects,
            mode=["val"],
        )
        coco_anno_jsons = [
            t.coco_anno_json
            for t in task_dataset_item
            if t.coco_anno_json is not None
        ]

        return coco_anno_jsons

    def __str__(self):
        return f"[CloudModel]HorizonDataset contains projects:\n{self.projects}\ntasks:\n{self.tasks}"  # noqa


class StatisticDatasets:
    def __init__(self, dataset: HorizonDataset):
        self.dataset = dataset

    def print_dataset_length_statistics(
        self,
        mode,
        tasks: List[CloudModelTask] = None,
        projects: List[str] = None,
    ):
        projects = projects if projects else self.dataset.projects
        tasks = tasks if tasks else self.dataset.tasks
        statistic_table = PrettyTable()
        statistic_table.field_names = [
            "task_name",
            "class_name",
            *[f"num_{i}" for i in projects],
            "num_all",
        ]
        # fill items
        table_rows = list()
        for task in tasks:
            table_row = [task.task_name, ",".join(task.class_name)]
            for project in projects:
                dataset_items = self.dataset.filter_dataset_item(
                    mode=[mode],
                    tasks=[task],
                    projects=[project],
                )
                table_row.append(
                    sum(i.get_single_dataset_length() for i in dataset_items)
                )
            dataset_items = self.dataset.filter_dataset_item(
                mode=[mode],
                tasks=[task],
            )
            table_row.append(
                sum(i.get_single_dataset_length() for i in dataset_items)
            )
            table_rows.append(table_row)
        statistic_table.add_rows(table_rows)
        print(statistic_table)

    def visualize_sample(
        self,
        mode: str = DataSplit.train.value,
        projects: List[str] = None,
        tasks: List[CloudModelTask] = None,
        vis_len: int = 1,
        with_label=True,
        save_root=None,
        shuffle=False,
    ):
        assert mode in (
            "train",
            "val",
        ), f"Don't support to visualize {mode} images."
        projects = _as_list(projects) if projects else self.dataset.projects
        tasks = _as_list(tasks) if tasks else self.dataset.tasks
        for project in projects:
            for task in tasks:
                dataset_items = self.dataset.filter_dataset_item(
                    mode=[mode], tasks=[task], projects=[project]
                )
                for item_index, dataset_item in enumerate(dataset_items):
                    curr_path = os.path.abspath(os.path.dirname(__file__))
                    if save_root is None:
                        save_path = os.path.join(
                            curr_path, project, task.task_name, str(item_index)
                        )
                    else:
                        save_path = os.path.join(
                            save_root, project, task.task_name, str(item_index)
                        )
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    else:
                        print(f"Delete {save_path}")
                        shutil.rmtree(save_path)
                        os.makedirs(save_path)
                    dataset = self.build_dataset(dataset_item)

                    if shuffle:
                        indices = random.sample(range(len(dataset)), vis_len)
                    else:
                        indices = range(vis_len)

                    tqdm_bar = tqdm(indices)
                    for idx in tqdm_bar:
                        tqdm_bar.set_description(
                            f"[project]{project}[task]{task.task_name}[data_idx]{item_index}"  # noqa
                        )
                        self.visualize_single(
                            data=dataset[idx],
                            task_type=dataset_item.task.task_type.value,
                            save_path=save_path,
                            source_type=type(dataset_item.data_source),
                            with_label=with_label,
                        )
                    print(f"Visualization results are saved in '{save_path}'")

    def build_dataset(self, dataset_item: CloudModelDatasetItem):
        if isinstance(dataset_item.data_source, DenseboxSource):
            rec_path = dataset_item.data_source.rec_path
            anno_path = dataset_item.data_source.anno_path
            ins_id2category = self.dataset.category_from_task(
                [dataset_item.task]
            )
            densebox_dataset = DenseboxDataset(
                data_path=rec_path,
                anno_path=anno_path,
                task_type=dataset_item.task.task_type.value,
                ignore_hard=False,
                class_id=list(ins_id2category.keys())[0],
                category=list(ins_id2category.values())[0],
            )
            return densebox_dataset
        elif isinstance(dataset_item, LMDBSource):
            raise NotImplementedError("LMDBSource Do not supported yet.")
        else:
            raise ValueError("Datasource do not supported.")

    def visualize_single(
        self,
        data,
        task_type,
        save_path,
        source_type,
        with_label,
    ) -> np.ndarray:
        if source_type in [DenseboxSource]:
            return self.render_densebox_data(
                data, task_type, save_path, with_label
            )

    @staticmethod
    def render_densebox_data(
        data,
        task_type,
        save_path,
        with_label: Optional[bool] = False,
    ):
        img = data["img"]
        if with_label:
            if task_type == "detection":
                gt_bboxes = data["gt_bboxes"]
                gt_classes = data["gt_classes"]
                for gt_class, bbox in zip(gt_classes, gt_bboxes):
                    # Only draw those bboxes with gt_class >= 0
                    if gt_class >= 0:
                        img = cv2.rectangle(
                            img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color=(0, 255, 0),
                            thickness=2,
                        )
                    # draw those bboxes with gt_class < 0
                    if gt_class < 0:
                        img = cv2.rectangle(
                            img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            color=(0, 0, 255),
                            thickness=2,
                        )
            elif task_type == "segmentation":
                gt_seg = data["gt_seg"]
                gt_mask = gt_seg.astype(np.bool_)
                color_mask = np.array([0, 0, 255], dtype=np.uint8)
                img[gt_mask] = color_mask
            else:
                raise ValueError(
                    f"Currently, don't support " f"{task_type} task."
                )
        filename = data["img_name"]
        file_path = os.path.join(save_path, filename)
        cv2.imwrite(file_path, img)
        return img


if __name__ == "__main__":
    from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import (
        AllTasks,
    )

    # test read datasets
    def _test_read_dataset():
        data = HorizonDataset(
            projects=["mono", "pilot"], tasks=[AllTasks.person_detection]
        )
        train_datasets = data.build_dataset(mode="train")
        val_datasets = data.build_dataset(mode="val")
        test_dataset = data.build_dataset(mode="test")
        coco_anno_json = data.get_val_coco_anno_path(
            task=AllTasks.person_detection,
            project=SourceDataProject.mono.value,
        )
        return train_datasets, val_datasets, test_dataset, coco_anno_json

    # test statistic
    def _test_statistic():
        data = HorizonDataset(
            projects=["mono", "pilot"], tasks=AllTasks.all_tasks()
        )
        static_ins = StatisticDatasets(dataset=data)
        static_ins.print_dataset_length_statistics(mode=DataSplit.train.value)
        # Statistics dataset length
        # dataset length statistics example:
        """
+-------------------------------------+--------------------------------------+----------+-----------+---------+
|              task_name              |              class_name              | num_mono | num_pilot | num_all |
+-------------------------------------+--------------------------------------+----------+-----------+---------+
|         road_arrow_detection        |              road_arrow              |  333804  |     0     |  333804 |
|        vehicle_rear_detection       |          vehicle_rear,rear           | 4190184  |   782078  | 4972262 |
|       vehicle_light_detection       |            vehicle_light             |  308345  |   354801  |  663146 |
|         person_ped_detection        |                person                | 3450689  |  1406741  | 4857430 |
|           cycle_detection           |                cycle                 |  210801  |     0     |  210801 |
|           lane_instanceseg          |                 lane                 |  587497  |   965735  | 1553232 |
|       vehicle_plate_detection       |            vehicle_plate             |  232187  |     0     |  232187 |
|        parking_lock_detection       |             parking_lock             |    0     |     0     |    0    |
|        traffic_sign_detection       |             traffic_sign             | 1159316  |     0     | 1159316 |
|           person_detection          |                person                | 3450689  |  1406741  | 4857430 |
| semantic_parsing_40cls_segmentation |        semantic_parsing_40cls        |  139586  |   187276  |  326862 |
|          vehicle_detection          |               vehicle                | 1143864  |  1872568  | 3016432 |
|    traffic_light_shell_detection    |         traffic_light_shell          |    0     |     0     |    0    |
|        person_head_detection        |             person_head              |  555195  |   183103  |  738298 |
|       vehicle_wheel_detection       |            vehicle_wheel             |  421999  |     0     |  421999 |
|        vehicle_2pe_detection        |               vehicle                | 1143864  |  1872568  | 3016432 |
|       traffic_light_detection       |            traffic_light             |    91    |   997126  |  997217 |
|          cyclist_detection          |               cyclist                | 2650628  |   967136  | 3617764 |
|        person_face_detection        |             person_face              |  52957   |   47511   |  100468 |
|       cyclist_wheel_detection       |            cyclist_wheel             |  50469   |     0     |  50469  |
|        traffic_cone_detection       |             traffic_cone             |  531938  |   29877   |  561815 |
|     traffic_light_len_detection     | traffic_light_len,traffic_light_lens |  645456  |   161731  |  807187 |
|        cyclist_2pe_detection        |               cyclist                | 2650628  |   967136  | 3617764 |
+-------------------------------------+--------------------------------------+----------+-----------+---------+
        """  # noqa

    def _test_visualize():
        # visualize images and labels in dataset, and results are saved in
        # save_root
        data = HorizonDataset(
            projects=["mono", "pilot"], tasks=[AllTasks.person_detection]
        )
        static_ins = StatisticDatasets(dataset=data)
        static_ins.visualize_sample(
            mode="train",
            projects=["mono", "pilot"],
            tasks=AllTasks.person_detection,
            vis_len=10,
            with_label=True,
            shuffle=True,
            # save_root="/home/users/hao.chen/codes/HAT/projects/cloudmodel/output",
        )

    _test_read_dataset()
    _test_statistic()
    _test_visualize()

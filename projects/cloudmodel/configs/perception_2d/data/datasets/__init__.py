from typing import Dict

from projects.cloudmodel.configs.perception_2d.data.structures import (
    AidiEval2DSource,
    CloudModelDatasetItem,
    DataSplit,
    DenseboxSource,
    LMDBSource,
    MulticlassDenseboxSource,
    SourceDataProject,
)
from projects.cloudmodel.configs.perception_2d.tasks.all_tasks import AllTasks

__all__ = [
    "all_datasets",
]


def _extract_dataset_item_from_project_format(
    project_dataset: Dict,
    project: SourceDataProject,
    source_type: str,
    collect_split: tuple = ("train", "val"),
):
    def _get_source(data_i):
        if source_type == "densebox":
            return DenseboxSource(
                rec_path=data_i["rec_path"],
                anno_path=data_i["anno_path"],
                rec_path_md5sum=data_i.get("rec_path_md5sum", None),
                anno_path_md5sum=data_i.get("anno_path_md5sum", None),
                roi_list=data_i.get("roi_list", None),
            )
        elif source_type == "multiclass_densebox":
            return MulticlassDenseboxSource(
                rec_path=data_i["rec_path"],
                anno_path=data_i["anno_path"],
                rec_path_md5sum=data_i.get("rec_path_md5sum", None),
                anno_path_md5sum=data_i.get("anno_path_md5sum", None),
                roi_list=data_i.get("roi_list", None),
            )
        elif source_type == "lmdb":
            return LMDBSource(data_path=data_i["data_path"])
        elif source_type == "aidieval_2d":
            return AidiEval2DSource(
                leaderboard_id=int(data_i["leaderboard_id"]),
                root_dir=data_i["root_dir"],
            )

    assert project.value in SourceDataProject._member_names_
    dataset_item_list = list()
    class_name2tasks = AllTasks.class_name2tasks()
    for class_name, task_datasets in project_dataset.items():
        # cleanup classnames. hardcode here.
        class_name = (
            class_name[: -len("_detection")]
            if class_name.endswith("_detection")
            else class_name
        )
        class_name = (
            class_name[: -len("_classification")]
            if class_name.endswith("_classification")
            else class_name
        )
        class_name = (
            class_name[: -len("_segmentation")]
            if class_name.endswith("_segmentation")
            else class_name
        )
        class_name = (
            class_name[: -len("_instanceseg")]
            if class_name.endswith("_instanceseg")
            else class_name
        )
        if class_name in class_name2tasks:
            respect_tasks = class_name2tasks[class_name]
        else:
            respect_tasks = []
            print(f"class_name:{class_name} not defined.")

        # train set
        if "train" in collect_split:
            for train_data_i in task_datasets.get("train_data_paths", []):
                for task_i in respect_tasks:
                    dataset_item_list.append(
                        CloudModelDatasetItem(
                            data_source=_get_source(data_i=train_data_i),
                            project=project,
                            task=task_i,
                            split=DataSplit.train.value,
                            sample_weight=train_data_i.get(
                                "sample_weight", None
                            ),
                            length=train_data_i.get("length", None),
                            img_shape=train_data_i.get("img_shape", None),
                            tags=train_data_i.get("tags", []),
                        )
                    )
        # val set
        if "val" in collect_split:
            for eval_data_i in task_datasets.get("val_data_paths", []):
                for task_i in respect_tasks:
                    dataset_item_list.append(
                        CloudModelDatasetItem(
                            data_source=_get_source(data_i=eval_data_i),
                            project=project,
                            task=task_i,
                            split=DataSplit.val.value,
                            sample_weight=eval_data_i.get(
                                "sample_weight", None
                            ),
                            length=eval_data_i.get("length", None),
                            img_shape=eval_data_i.get("img_shape", None),
                            tags=eval_data_i.get("tags", []),
                            coco_anno_json=eval_data_i.get(
                                "coco_anno_json", None
                            ),
                        )
                    )
        # test set
        if "test" in collect_split:
            for leaderboard_id, root_dir in task_datasets.items():
                for task_i in respect_tasks:
                    dataset_item_list.append(
                        CloudModelDatasetItem(
                            data_source=_get_source(
                                data_i=dict(
                                    leaderboard_id=leaderboard_id,
                                    root_dir=root_dir,
                                ),
                            ),
                            project=project,
                            task=task_i,
                            split=DataSplit.test.value,
                        )
                    )

    return dataset_item_list


def _all_datasets():
    """Find all dataset items defined in current folder."""
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_adb_light_train_set import (
        adb_light_train_data_paths,
        adb_light_val_data_paths,
    )
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_aidi_eval import (
        mono_aidi_eval_dataset_paths,
        pilot_aidi_eval_dataset_paths,
    )
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_attribute_trainval_set import (
        datapaths as attribute_data_pathes,
    )
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_canoo_train_set import (
        canoo_data_paths,
    )
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_mono_train_set import (
        mono_data_paths,
    )
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_mono_val_set import (
        mono_val_data_paths,
    )
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_pilot_train_set_lmdb import (
        pilot_data_paths_lmdb,
    )
    from projects.cloudmodel.configs.perception_2d.data.datasets.horizon_pilot_val_set import (
        pilot_val_data_paths,
    )

    datasets = list()
    print(
        f"All supported class names:{list(AllTasks.class_name2tasks().keys())}"
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            canoo_data_paths,
            SourceDataProject.canno,
            source_type="densebox",
            collect_split=("train",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            mono_data_paths,
            SourceDataProject.mono,
            source_type="densebox",
            collect_split=("train",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            mono_val_data_paths,
            SourceDataProject.mono,
            source_type="densebox",
            collect_split=("val",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            pilot_val_data_paths,
            SourceDataProject.pilot,
            source_type="densebox",
            collect_split=("val",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            pilot_data_paths_lmdb,
            SourceDataProject.pilot,
            source_type="lmdb",
            collect_split=("train",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            mono_aidi_eval_dataset_paths,
            SourceDataProject.mono,
            source_type="aidieval_2d",
            collect_split=("test",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            pilot_aidi_eval_dataset_paths,
            SourceDataProject.pilot,
            source_type="aidieval_2d",
            collect_split=("test",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            adb_light_train_data_paths,
            SourceDataProject.unknown,
            source_type="multiclass_densebox",
            collect_split=("train",),
        )
    )
    datasets.extend(
        _extract_dataset_item_from_project_format(
            adb_light_val_data_paths,
            SourceDataProject.unknown,
            source_type="densebox",
            collect_split=("val",),
        )
    )
    # append attribute datasets
    datasets.extend(
        _extract_dataset_item_from_project_format(
            attribute_data_pathes,
            SourceDataProject.unknown,
            source_type="densebox",
            collect_split=("train", "val"),
        )
    )
    return datasets


all_datasets = _all_datasets()

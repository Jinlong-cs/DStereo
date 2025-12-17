from hat.utils.apply_func import _as_list


# Version Train Project
def merge_datapaths(*source_datapaths):
    result_datapaths = dict()
    deduplication_set = set()
    # Traverse all datapaths
    for source_datapath in source_datapaths:
        for data_name, dataset in source_datapath.items():
            if data_name not in result_datapaths:
                result_datapaths[data_name] = dict(
                    train_data_paths=[],
                )

            for data_path_item in dataset.get("train_data_paths", []):
                is_dup_item = True
                # Judge whether there is duplicate data
                for data_path in _as_list(data_path_item.get("data_path", [])):
                    if data_path not in deduplication_set:
                        deduplication_set.add(data_path)
                        is_dup_item = False

                # Save if it is not exactly duplicated
                if not is_dup_item:
                    result_datapaths[data_name]["train_data_paths"].append(
                        data_path_item
                    )

    return result_datapaths

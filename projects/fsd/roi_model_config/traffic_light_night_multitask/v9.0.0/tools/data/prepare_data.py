import os

from hat.utils import Config

ds_path = os.path.join(
    os.path.dirname(__file__), "../../config/datasets/train_datasets.py"
)
dst_file = os.path.join(os.path.dirname(__file__), "prepare_file.lst")
fp = open(dst_file, "w")

if os.path.exists(ds_path):
    datapaths = Config.fromfile(ds_path).datapaths

    for _, v in datapaths.items():
        rec_paths = [
            d["rec_path"].replace("/horizon-bucket", "dmpv2:/")
            for d in v["train_data_paths"]
        ]
        pb_rec_paths = [
            d["anno_path"].replace("/horizon-bucket", "dmpv2:/")
            for d in v["train_data_paths"]
        ]
        anno_idx_paths = [
            rec_p.replace(".rec", ".rec.idx").replace(
                "/horizon-bucket", "dmpv2:/"
            )
            for rec_p in rec_paths
        ]

        for rec, pb_rec, anno in zip(rec_paths, pb_rec_paths, anno_idx_paths):
            fp.writelines(rec + "\n")
            fp.writelines(pb_rec + "\n")
            fp.writelines(anno + "\n")
        print(rec_paths, pb_rec_paths, anno_idx_paths)

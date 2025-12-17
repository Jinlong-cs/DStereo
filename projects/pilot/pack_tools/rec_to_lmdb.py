import argparse

from matrix_gluon.data.dataset.common import DefaultBufWithAnnoDataset

from hat.registry import build_from_registry


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_rec_path", type=str, required=True, help="configure file"
    )
    parser.add_argument(
        "--anno_rec_path", type=str, required=True, help="configure file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="configure file"
    )

    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    img_rec_path = args.img_rec_path
    anno_rec_path = args.anno_rec_path
    rec_datasets = []
    output_dir = args.output_dir
    rec_datasets.append(
        DefaultBufWithAnnoDataset(
            img_rec_path, anno_rec_path, decode_img=True, decode_anno=True
        )
    )

    packer_config = {
        "type": "RecToLmdbPacker",
        "rec_datasets": rec_datasets,
        "output_dir": output_dir,
        "num_workers": 32,
    }

    packer = build_from_registry(packer_config)

    print("~~~Packing index data for task %s~~~")
    packer.pack_idx()
    print("~~~Packing image data for task %s~~~")
    packer.pack_img()
    print("~~~Packing annotation data for task %s~~~")
    packer.pack_anno()


if __name__ == "__main__":
    main()

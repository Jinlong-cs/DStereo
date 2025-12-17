import argparse
import os

from redis import Redis
from tqdm import tqdm

from hat.utils.filesystem import get_filesystem
from projects.halo.avspeech.utils.tendis import TENDIS_KWARGS
from projects.halo.avspeech.utils.wav_write import save_tendis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write Splited Wav File to Tendis"
    )
    pa = parser.add_argument
    pa(
        "--lst",
        help="The list file of the wav files. which line format is '{key} {wav_name}\n', or '{wav_name}\n' only.",  # noqa: E501
        required=True,
        type=str,
    )
    pa(
        "--zfill",
        help="The zfill number of the idx.",
        default=6,
        type=int,
    )
    pa(
        "--key_pref",
        help="The v_Key2Waveform in tendis of the wav datas.",
        default="multimodal-Key2Waveform_",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    zfill = args.zfill
    key_pref = args.key_pref
    client = Redis(**TENDIS_KWARGS)
    lst_fs = get_filesystem(args.lst)

    with lst_fs.open(args.lst, "r", encoding="utf-8") as fr:
        for line in tqdm(fr, ncols=120, desc="Saving..."):
            # 解析 lst file
            lsp = line.strip().split()
            if len(lsp) == 2:
                wav_key, wav_file = lsp
            elif len(lsp) == 1:
                wav_file = lsp
                wav_key, _ = os.path.splitext(os.path.basename(wav_file))
            else:
                raise TypeError(f"UnSupported Format: {line}")
            save_tendis(client, wav_file, wav_key, key_pref, zfill)

import logging
import pickle

import soundfile
from redis import Redis

from hat.utils.filesystem import get_filesystem

SUBTYPE_2_DTYPE = {
    "PCM_16": "int16",
    "PCM_S8": "int8",
    "PCM_32": "float32",
    "DOUBLE": "float64",
}


def get_dataset_attrs(audio_info: soundfile._SoundFileInfo):
    ret = {
        "endian": audio_info.endian,
        "channels": audio_info.channels,
        "duration": audio_info.duration,
        # "extra_info": audio_info.extra_info,
        "format": audio_info.format,
        # "format_info": audio_info.format_info,
        "frames": audio_info.frames,
        "samplerate": audio_info.samplerate,
        "sections": audio_info.sections,
        "subtype": audio_info.subtype,
        # "subtype_info": audio_info.subtype_info,
    }
    return ret


def save_tendis(
    client: Redis,
    wav_file: str,
    wav_key: str,
    key_pref: str,
    zfill_num: int,
):
    fs = get_filesystem(wav_file)
    if not fs.exists(wav_file):
        logging.info(f"{wav_file} file not exists.")
        return

    if not isinstance(wav_key, str):
        raise ValueError(
            f"Illegal wav_key: type: {type(wav_key)}, value：{wav_key}"
        )

    batch = {}
    # 读取音频信息
    wav_fs = get_filesystem(wav_file)
    with wav_fs.open(wav_file, mode="rb") as fr:
        audio_info = soundfile.info(fr)
        dtype = SUBTYPE_2_DTYPE[audio_info.subtype]
        attrs = get_dataset_attrs(audio_info)
        attrs["dtype"] = dtype
        attrs_key = f"{key_pref}{wav_key}_attrs"
        batch[attrs_key] = pickle.dumps(attrs)
        fr.seek(0)
        audio_data, framerate = soundfile.read(fr, dtype=dtype)
        frame_num = audio_data.shape[0]
        seconds = frame_num // framerate
        other = frame_num % framerate

        count = 0
        for idx in range(seconds):
            sec_data = audio_data[idx * framerate : (idx + 1) * framerate]
            sec_key = f"{key_pref}{wav_key}_{idx :>0{zfill_num}d}"
            batch[sec_key] = sec_data.tobytes()
            count += 1
            if count % 1000 == 0:
                client.mset(batch)
                count = 0
                batch.clear()

        if other != 0:
            other_data = audio_data[(-1) * other :]
            other_key = f"{key_pref}{wav_key}_{seconds :>0{zfill_num}d}"
            batch[other_key] = other_data.tobytes()

        if len(batch) > 0:
            client.mset(batch)

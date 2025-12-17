import pickle

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from redis import Redis
except ImportError:
    Redis = None

from hat.data.datasets.avspeech.audio import (
    DirectWaveformReader,
    HDF5WaveformReader,
    TendisWaveformReader,
)
from tests import HAT_BUCKET_PATH

tendis_kwargs = dict(
    host="aidi-kv-cluster-01.hogpu.cc",
    port=7616,
    db=0,
    socket_connect_timeout=5000,
    password="pB9cA2eN1eD1lJ0",
    socket_keepalive=True,
)


def get_dataset_attrs(audio_info):
    ret = {
        "endian": audio_info.endian,
        "channels": audio_info.channels,
        "duration": audio_info.duration,
        "format": audio_info.format,
        "frames": audio_info.frames,
        "samplerate": audio_info.samplerate,
        "sections": audio_info.sections,
        "subtype": audio_info.subtype,
    }
    return ret


def save_tendis(client, wav_path):
    batch = {}
    keys = []
    key_pref = "multimodal-Key2Waveform_audio_unit_test"
    wav_data, samplerate = sf.read(wav_path, dtype="int16")
    wav_info = sf.info(wav_path)
    attrs = get_dataset_attrs(wav_info)
    dtype = "int16"
    attrs["dtype"] = dtype
    attrs_key = f"{key_pref}_attrs"
    batch[attrs_key] = pickle.dumps(attrs)
    keys.append(attrs_key)
    frame_num = wav_data.shape[0]
    seconds = frame_num // samplerate
    other = frame_num % samplerate

    for idx in range(seconds):
        sec_data = wav_data[idx * samplerate : (idx + 1) * samplerate]
        sec_key = f"{key_pref}_{idx :>0{6}d}"
        batch[sec_key] = sec_data.tobytes()
        keys.append(sec_key)

    if other != 0:
        other_data = wav_data[(-1) * other :]
        other_key = f"{key_pref}_{seconds :>0{6}d}"
        batch[other_key] = other_data.tobytes()
        keys.append(other_key)

    client.mset(batch)
    return keys


def check_data(wav_path, wav_hdf5):
    wav_data, samplerate = sf.read(wav_path, dtype="int16")
    # 测试HDF5WaveformReader
    audio_reader = HDF5WaveformReader(
        hdf5_file=wav_hdf5,
        ret_dtype="int16",
        channel_first=False,
    )
    waveform_hdf5, framerate_hdf5 = audio_reader.read_seg_audio(
        utt="audio_unit_test",
        seg_beg=0.0,
        seg_end=wav_data.shape[0] / (samplerate * 1.0),
    )

    # 测试TendisWaveformReader
    audio_reader = TendisWaveformReader(
        zfill_num=6,
        ret_dtype="int16",
        channel_first=False,
        tendis_kwargs=tendis_kwargs,
    )
    waveform_tendis, framerate_tendis = audio_reader.read_seg_audio(
        utt="audio_unit_test",
        seg_beg=0.0,
        seg_end=wav_data.shape[0] / (samplerate * 1.0),
    )

    # 测试DirectWaveformReader
    audio_reader = DirectWaveformReader(
        ret_dtype="int16",
        channel_first=False,
    )
    waveform_file, framerate_file = audio_reader.read_seg_audio(wav_path)

    assert (
        (waveform_hdf5 == waveform_tendis).all()
        and waveform_tendis == waveform_file
    ).all()
    assert (
        framerate_hdf5 == framerate_tendis
        and framerate_tendis == framerate_file
    )


def test_audio():
    wav_path = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/datasets/unit_test.wav"  # noqa: E501
    wav_hdf5 = f"{HAT_BUCKET_PATH}/unit_test_data/avspeech/data/datasets/unit_test.hdf5"  # noqa: E501

    if sf is not None and Redis is not None:
        client = Redis(**tendis_kwargs)
        keys = save_tendis(client, wav_path)

        check_data(wav_path, wav_hdf5)
        client.delete(*keys)

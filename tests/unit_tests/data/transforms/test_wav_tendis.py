import os
import tempfile

import numpy as np
import pytest

from hat.data.datasets.avspeech.audio import TendisWaveformReader

try:
    import soundfile
    from redis import Redis

    from projects.halo.avspeech.utils import wav_write
    from projects.halo.avspeech.utils.tendis import TENDIS_KWARGS
except ImportError:
    Redis = None
    soundfile = None


@pytest.mark.skipif(
    Redis is None or soundfile is None, reason="need redis, soundfile"
)
@pytest.mark.parametrize(
    ["wav_key", "key_pref", "zfill_num"],
    [pytest.param("Test_tendis_mic1", "multimodal-Key2Waveform_", 6)],
)
def test_save_and_read(wav_key, key_pref, zfill_num):
    with tempfile.TemporaryDirectory() as local_dir:
        CLIENT = Redis(**TENDIS_KWARGS)
        samplerate = 16000
        wav_data = np.random.randint(
            -100,
            100,
            size=(samplerate * 5),
            dtype=np.int16,
        )
        wav_file = os.path.join(local_dir, "tmp.wav")
        soundfile.write(wav_file, wav_data, samplerate, "PCM_16")
        wav_data = wav_data.reshape(1, -1)
        # 存音频到tendis
        wav_write.save_tendis(
            CLIENT,
            wav_file,
            wav_key,
            key_pref,
            zfill_num,
        )
        # 从tendis读音频
        tendis_reader = TendisWaveformReader(
            zfill_num=zfill_num,
            ret_dtype="int16",
            tendis_kwargs=TENDIS_KWARGS,
        )
        waveform, framerate = tendis_reader.read_seg_audio(wav_key, 0.0, 5.0)

        assert (waveform == wav_data).all()
        assert samplerate == framerate

        # 临时音频数据从tendis删除
        keys = []
        attrs_key = f"{key_pref}{wav_key}_attrs"
        keys.append(attrs_key)
        for idx in range(5):
            sec_key = f"{key_pref}{wav_key}_{idx :>0{zfill_num}d}"
            keys.append(sec_key)
        CLIENT.delete(*keys)


if __name__ == "__main__":
    pytest.main(["-s", __file__])

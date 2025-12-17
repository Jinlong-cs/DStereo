import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from jiwer.measures import compute_measures

from hat.callbacks import CallbackMixin
from hat.data.datasets.avspeech.audio import BaseWaveformReader
from hat.data.datasets.avspeech.image import BaseImageReader
from hat.data.datasets.avspeech.info import UttParser
from hat.data.datasets.avspeech_dataset import CocktailDatasetV2
from projects.halo.avspeech.utils.data_check import make_segment_check_data


class DumpBadCase(CallbackMixin):
    """Dump bad case to disk."""

    def __init__(
        self,
        image_reader: Optional[BaseImageReader],
        audio_reader: Optional[BaseWaveformReader],
        out_root: str,
        wer_threshold: float = 0.5,
        prefix: str = "",
    ):
        """Dump bad case to disk.

        Args:
            image_reader: Image reader.
            audio_reader: Audio reader.
            out_root: Output root.
            wer_threshold: WER threshold. Defaults to 0.5.
            prefix: Prefix. Defaults to "".
        """
        self.image_reader = image_reader
        self.audio_reader = audio_reader
        self.out_root = Path(out_root)
        self.wer_threshold = wer_threshold
        self.prefix = prefix
        self.utt_parser = UttParser()

        os.makedirs(self.out_root, exist_ok=True)

    def on_batch_end(self, batch, model_outs, train_metrics, **kwargs):
        loss, loss_att, loss_ctc, att_acc, ctc_out = model_outs
        for i, token in enumerate(batch["tokens"]):
            key = batch["key"][i]
            seg_beg = batch["info"]["seg_beg"][i]
            seg_end = batch["info"]["seg_end"][i]
            match = self.utt_parser.match(key)
            snt_utt = f"{match['dataset']}_{match['batch']}-{match['snt']}"
            pred = ctc_out[i]
            label = " ".join(token)
            wer = self._wer(pred, label)
            if wer > self.wer_threshold:
                make_segment_check_data(
                    snt_utt=snt_utt,
                    cam_id=match["cam"],
                    mic_id=match["mic"],
                    seg_beg=seg_beg,
                    seg_end=seg_end,
                    out_root=self.out_root,
                    image_reader=self.image_reader,
                    audio_reader=self.audio_reader,
                    prefix=self.prefix,
                    suffix=f"-{''.join(token)}-{''.join(pred)}-{wer:.2f}",
                    quiet=True,
                )

    def _wer(self, pred, ref):
        res = compute_measures(ref, pred)

        error = res["substitutions"] + res["deletions"] + res["insertions"]
        total = res["substitutions"] + res["deletions"] + res["hits"]
        wer = error * 1.0 / total
        return wer


class DumpJsonBadCase(CallbackMixin):
    """Dump bad case to disk for json dataset.

    Args:
        dataset: Dataset.
        out_root: Output root.
        wer_threshold: WER threshold. Defaults to 0.5.
        prefix: Prefix. Defaults to "".
    """

    def __init__(
        self,
        dataset: CocktailDatasetV2,
        out_root: str,
        wer_threshold: float = 0.5,
        prefix: str = "",
    ):
        self.info_dict = self.get_info_dict(dataset)
        self.out_root = Path(out_root)
        self.wer_threshold = wer_threshold
        self.prefix = prefix

        os.makedirs(self.out_root, exist_ok=True)

    def get_info_dict(self, data):
        infos = data.info_list.infos
        info_dict = {}
        for info in infos:
            _key = info["wav_path"].rsplit("/", 1)[-1][:-4]
            key = f"{_key}_{_key}"
            info_dict[key] = info
        return info_dict

    def on_batch_end(self, batch, model_outs, train_metrics, **kwargs):
        loss, loss_att, loss_ctc, att_acc, ctc_out = model_outs
        for i, key in enumerate(batch["key"]):
            info = self.info_dict[key]
            mp4_path = info["mp4_path"]
            wav_path = info["wav_path"]
            label = info["text"]
            mp4_file = os.path.split(mp4_path)[-1]
            mp4_name = os.path.splitext(mp4_file)[0]
            pred = ctc_out[i]
            label = " ".join(label)
            wer = self._wer(pred, label)

            # label_ = label.replace(" ", "").replace("/", "\\").replace("")
            label_ = re.sub("\W*", "", label)  # noqa: W605
            pred_ = re.sub("\W*", "", pred)  # noqa: W605
            # pred_ = pred.replace(" ", "").replace("/", "\\")
            out_video_path = os.path.join(
                self.out_root,
                f"{self.prefix}-{mp4_name}-{label_}-{pred_}-{wer}.mp4",
            )

            if wer > self.wer_threshold:
                cmd = (
                    f"ffmpeg -hide_banner -loglevel error -y -i {mp4_path} -i "
                    f"{wav_path} -c:v copy -c:a libmp3lame -map 0:v:0 -map 1:a:0 "  # noqa: E501
                    f"{out_video_path} </dev/null"
                )
                subprocess.check_call(cmd, shell=True)

    def _wer(self, pred, ref):
        res = compute_measures(ref, pred)

        error = res["substitutions"] + res["deletions"] + res["insertions"]
        total = res["substitutions"] + res["deletions"] + res["hits"]
        wer = error * 1.0 / total
        return wer


class DumpResult(CallbackMixin):
    """Dump result to disk.

    Args:
        out_root: Output root.
        prefix: Prefix. Defaults to "".
    """

    def __init__(self, out_root: str, prefix: str):
        if not os.path.exists(out_root):
            os.makedirs(out_root, exist_ok=True)

        self.out_path = os.path.join(out_root, f"{prefix}.txt")
        self.lines = []

    def on_batch_end(self, batch, model_outs, train_metrics, **kwargs):
        _, _, _, _, ctc_out = model_outs
        for i, key in enumerate(batch["key"]):
            pred = ctc_out[i]
            label = " ".join(batch["tokens"][i])
            res = compute_measures(label, pred)

            error = res["substitutions"] + res["deletions"] + res["insertions"]
            total = res["substitutions"] + res["deletions"] + res["hits"]
            wer = error * 1.0 / total
            sub = res["substitutions"]
            ins = res["insertions"]
            dele = res["deletions"]
            line = f"{key} label:{label.replace(' ', '')} pred:{pred.replace(' ', '')} wer:{wer:.2f} S:{sub} D:{dele} I:{ins}"  # noqa: E501
            self.lines.append(line)

    def on_epoch_end(self, **kwargs):
        with open(self.out_path, "w") as fw:
            for line in self.lines:
                fw.write(line + "\n")
        self.lines = []

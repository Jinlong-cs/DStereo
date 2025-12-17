# Copyright (c) Horizon Robotics. All rights reserved.
import argparse
import os
import re
import subprocess
import tempfile

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool to anaylse log with tensorboard"
    )

    pa = parser.add_argument
    pa("--log_url", type=str, required=True)
    pa("--output_dir", type=str, default=None)

    args = parser.parse_args()
    return args


def parse_log(log_path, writer):

    train_beg = "Begin =================================================="
    train_end = "End   =================================================="
    val_beg = "train `model` as val model"
    state = None
    f = "{name}\[(?P<{name}>\d+(\.\d+)?)\]".format  # noqa W605
    train_metric_reg = re.compile(
        f".* {f(name='Epoch')} {f(name='Step')} {f(name='GlobalStep')} (?P<prefix>.*): {f(name='loss')} {f(name='loss_att')} {f(name='loss_ctc')} {f(name='att_acc')} {f(name='wer')}"  # noqa E501
    )
    train_lr_reg = re.compile(
        f".* {f(name='Epoch')} {f(name='Step')} {f(name='GlobalStep')} lr=(?P<lr>\d+\.?\d+)"  # noqa E501
    )
    val_metric_reg = re.compile(
        f".* {f(name='Epoch')} (?P<prefix>.*): {f(name='loss')} {f(name='loss_att')} {f(name='loss_ctc')} {f(name='att_acc')} {f(name='wer')}"  # noqa E501
    )

    val_stat_dict = {}
    with open(log_path, "r") as fr:
        for line in tqdm(fr, desc="Parsing log"):
            line = line.strip()
            if state is None:
                if train_beg not in line:
                    continue
                else:
                    state = "train"
                    continue
            elif state == "train":
                if train_end in line:
                    state = "train_end"
                    continue
                else:
                    lr_match = train_lr_reg.match(line)
                    if lr_match:
                        global_step = int(lr_match["GlobalStep"])
                        lr = float(lr_match["lr"])
                        writer.add_scalar("train/lr", lr, global_step)
                        continue
                    else:
                        train_metric_match = train_metric_reg.match(line)
                        if train_metric_match:
                            loss = float(train_metric_match["loss"])
                            loss_att = float(train_metric_match["loss_att"])
                            loss_ctc = float(train_metric_match["loss_ctc"])
                            att_acc = float(train_metric_match["att_acc"])
                            wer = float(train_metric_match["wer"])
                            prefix = train_metric_match["prefix"]
                            global_step = float(
                                train_metric_match["GlobalStep"]
                            )
                            loss_dict = {
                                f"{prefix}_total_loss": loss,
                                f"{prefix}_loss_att": loss_att,
                                f"{prefix}_loss_ctc": loss_ctc,
                            }
                            acc_dict = {f"{prefix}_att_acc": att_acc}
                            wer_dict = {f"{prefix}_wer": wer}
                            writer.add_scalars(
                                "train/loss", loss_dict, global_step
                            )
                            writer.add_scalars(
                                "train/att_acc", acc_dict, global_step
                            )

                            writer.add_scalars(
                                "train/wer", wer_dict, global_step
                            )
                            continue
            elif state == "train_end":
                if val_beg in line:
                    state = "val"
                continue
            elif state == "val":
                if train_beg in line:
                    epoch = val_stat_dict.pop("epoch")
                    for val_set, state_dict in val_stat_dict.items():
                        writer.add_scalars(f"val/{val_set}", state_dict, epoch)

                    state = "train"
                    continue
                else:
                    val_metric_match = val_metric_reg.match(line)
                    if val_metric_match:
                        loss = float(val_metric_match["loss"])
                        loss_att = float(val_metric_match["loss_att"])
                        loss_ctc = float(val_metric_match["loss_ctc"])
                        att_acc = float(val_metric_match["att_acc"])
                        wer = float(val_metric_match["wer"])
                        val_set = val_metric_match["prefix"]
                        epoch = int(val_metric_match["Epoch"])
                        if "total_loss" not in val_stat_dict:
                            val_stat_dict = {
                                "total_loss": {},
                                "loss_att": {},
                                "loss_ctc": {},
                                "att_acc": {},
                                "wer": {},
                            }
                        val_stat_dict["total_loss"][val_set] = loss
                        val_stat_dict["loss_att"][val_set] = loss_att
                        val_stat_dict["loss_ctc"][val_set] = loss_ctc
                        val_stat_dict["att_acc"][val_set] = att_acc
                        val_stat_dict["wer"][val_set] = wer
                        val_stat_dict["epoch"] = epoch


def main(output_dir, log_urls):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(output_dir)
    for log_url in log_urls:
        log_name = os.path.split(log_url)[-1]
        log_path = os.path.join(output_dir, log_name)
        cmd = f"wget -O {log_path} {log_url}"
        subprocess.check_call(cmd, shell=True)

        parse_log(log_path, writer)

    cmd = f"tensorboard --logdir {output_dir} --bind_all"
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    args = parse_args()
    log_urls = args.log_url.split(",")
    if args.output_dir is None:
        with tempfile.TemporaryDirectory() as output_dir:
            main(output_dir, log_urls)
    else:
        main(args.output_dir, log_urls)

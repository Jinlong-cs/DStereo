import argparse
import glob
import logging
import os
import re
import sys
import traceback

import torch

from hat.utils.config import Config
from hat.utils.setup_env import setup_args_env


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s %(message)s",
    level=logging.INFO,
)


_PATTERN_DEFAULT = "*.pth.tar,*.pth,*.pt,*.ckpt"
_STEP_RE = re.compile(r"(?:step|iter|iteration|epoch)[-_]?(\d+)", re.IGNORECASE)


def _import_export_onnx():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from export_onnx import export_onnx_from_cfg  # type: ignore

    return export_onnx_from_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory containing checkpoint files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for onnx files. Default: <ckpt_dir>/onnx",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=_PATTERN_DEFAULT,
        help="Comma-separated glob patterns for checkpoint files.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip exporting if target onnx already exists.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Override stage name for output naming.",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def _parse_iter_from_name(path):
    base = os.path.basename(path)
    match = _STEP_RE.findall(base)
    if match:
        return int(match[-1])
    nums = re.findall(r"(\d+)", base)
    if nums:
        return int(nums[-1])
    return None


def _parse_iter_from_ckpt(path):
    try:
        state = torch.load(path, map_location="cpu")
    except Exception:
        return None
    for key in ("iteration", "iter", "step", "global_step", "epoch"):
        if key in state and isinstance(state[key], int):
            return int(state[key])
    return None


def _collect_ckpts(ckpt_dir, pattern):
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(ckpt_dir, pat)))
    ckpts = sorted({os.path.abspath(p) for p in files if os.path.isfile(p)})
    return ckpts


def _set_ckpt_path(onnx_cfg, ckpt_path):
    pipeline = onnx_cfg.get("model_convert_pipeline")
    if pipeline is None:
        return None, None, False
    pipelines = pipeline if isinstance(pipeline, (list, tuple)) else [pipeline]
    for pl in pipelines:
        converters = pl.get("converters", [])
        for converter in converters:
            if not isinstance(converter, dict):
                continue
            if converter.get("type") == "LoadCheckpoint":
                had_key = "checkpoint_path" in converter
                old_val = converter.get("checkpoint_path")
                converter["checkpoint_path"] = ckpt_path
                return converter, old_val, had_key
    return None, None, False


def _stage_name(cfg, override=None):
    if override:
        return override
    onnx_cfg = cfg.onnx_cfg if hasattr(cfg, "onnx_cfg") else cfg.get("onnx_cfg", {})
    return onnx_cfg.get("stage", "float")


def main():
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)

    export_onnx_from_cfg = _import_export_onnx()
    cfg = Config.fromfile(args.config)

    ckpt_dir = args.ckpt_dir or cfg.get("ckpt_dir", ".")
    ckpt_dir = os.path.abspath(ckpt_dir)
    out_dir = args.out_dir or os.path.join(ckpt_dir, "onnx")
    out_dir = os.path.abspath(out_dir)

    ckpts = _collect_ckpts(ckpt_dir, args.pattern)
    if not ckpts:
        logger.error("No checkpoints found in %s", ckpt_dir)
        return 1

    logger.info("Found %d checkpoints in %s", len(ckpts), ckpt_dir)
    os.makedirs(out_dir, exist_ok=True)

    success = []
    skipped = []
    failed = []
    stage = _stage_name(cfg, args.stage)

    for idx, ckpt_path in enumerate(ckpts):
        iter_id = _parse_iter_from_name(ckpt_path)
        if iter_id is None:
            iter_id = _parse_iter_from_ckpt(ckpt_path)
        if iter_id is None:
            iter_id = idx
            logger.warning(
                "Cannot parse iteration from %s, fallback to index %d",
                os.path.basename(ckpt_path),
                iter_id,
            )
        onnx_name = f"{stage}_{iter_id}.onnx"
        onnx_path = os.path.join(out_dir, onnx_name)
        if args.skip_existing and os.path.exists(onnx_path):
            skipped.append((ckpt_path, onnx_path))
            continue

        converter, old_val, had_key = _set_ckpt_path(cfg.onnx_cfg, ckpt_path)
        if converter is None:
            failed.append((ckpt_path, "LoadCheckpoint not found in onnx_cfg"))
            continue

        try:
            export_onnx_from_cfg(
                cfg,
                out_dir=out_dir,
                stage_override=stage,
                filename=onnx_name,
            )
            success.append((ckpt_path, onnx_path))
        except Exception as exc:
            failed.append((ckpt_path, str(exc)))
            logger.error("Failed on %s\n%s", ckpt_path, traceback.format_exc())
        finally:
            if converter is not None:
                if had_key:
                    converter["checkpoint_path"] = old_val
                else:
                    converter.pop("checkpoint_path", None)

    logger.info("Export summary: success=%d skipped=%d failed=%d", len(success), len(skipped), len(failed))
    if success:
        logger.info("Success list:")
        for ckpt_path, onnx_path in success:
            logger.info("  %s -> %s", ckpt_path, onnx_path)
    if skipped:
        logger.info("Skipped list:")
        for ckpt_path, onnx_path in skipped:
            logger.info("  %s -> %s", ckpt_path, onnx_path)
    if failed:
        logger.info("Failed list:")
        for ckpt_path, reason in failed:
            logger.info("  %s : %s", ckpt_path, reason)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())


# python3 tools/deploy/export_onnx_batch.py -c DStereo/DStereoPlus.py --ckpt_dir work_dirs/tmp_models_szp1/DStereoV23_60000 --out_dir work_dirs/tmp_models_szp1/DStereoV23_60000_onnx/ --skip_existing

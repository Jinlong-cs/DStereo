import torch
from horizon_plugin_pytorch.dtype import qint8, qint16

from hat.utils import qconfig_manager

__all__ = []


def qint8_qconfig():
    return qconfig_manager.get_qconfig(
        activation_qat_observer="min_max",
        weight_qat_observer="min_max",
        activation_calibration_observer="mix",
        weight_calibration_observer="min_max",
        activation_qat_qkwargs={
            "dtype": qint8,
        },
        weight_qat_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "dtype": qint8,
            "ch_axis": 0,
        },
        activation_calibration_qkwargs={
            "dtype": qint8,
        },
        weight_calibration_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "dtype": qint8,
            "ch_axis": 0,
        },
    )


def qint16_qconfig():
    return qconfig_manager.get_qconfig(
        activation_qat_observer="min_max",
        weight_qat_observer="min_max",
        activation_calibration_observer="mix",
        weight_calibration_observer="min_max",
        activation_qat_qkwargs={
            "dtype": qint16,
        },
        weight_qat_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
        activation_calibration_qkwargs={
            "dtype": qint16,
        },
        weight_calibration_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
    )


def qint8_freezescale_qconfig():
    return qconfig_manager.get_qconfig(
        activation_qat_observer="min_max",
        weight_qat_observer="min_max",
        activation_calibration_observer="mix",
        weight_calibration_observer="min_max",
        activation_qat_qkwargs={
            "averaging_constant": 0.0,
            "dtype": qint8,
        },
        weight_qat_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "dtype": qint8,
            "ch_axis": 0,
        },
        activation_calibration_qkwargs={
            "dtype": qint8,
        },
        weight_calibration_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "dtype": qint8,
            "ch_axis": 0,
        },
    )


def qint16_freezescale_qconfig():
    return qconfig_manager.get_qconfig(
        activation_qat_observer="min_max",
        weight_qat_observer="min_max",
        activation_calibration_observer="mix",
        weight_calibration_observer="min_max",
        activation_qat_qkwargs={
            "averaging_constant": 0.0,
            "dtype": qint16,
        },
        weight_qat_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
        activation_calibration_qkwargs={
            "dtype": qint16,
        },
        weight_calibration_qkwargs={
            "qscheme": torch.per_channel_symmetric,
            "ch_axis": 0,
            "dtype": qint8,
        },
    )

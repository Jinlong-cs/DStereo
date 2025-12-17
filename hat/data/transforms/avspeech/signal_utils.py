# mypy: allow-untyped-defs

import math
import random
import subprocess
import tempfile
from os.path import join

import numpy as np
from scipy import signal

from hat.utils.package_helper import require_packages

try:
    import soundfile as sf
except ImportError:
    pass

try:
    import mxnet as mx
except ImportError:
    pass

try:
    import librosa
except ImportError:
    pass

EPS = np.finfo(np.float32).eps


def butter_highpass(cutoff_freq, samplerate, order=5):
    """Design a Butterworth highpass filter.

    Args:
        cutoff_freq: the cutoff frequency in Hz.
        samplerate: the sample rate of the signal.
        order: the order of the Butterworth filter.

    Returns:
        b, a: the numerator and denominator coefficients of
            the filter transfer function.
    """

    nyq = 0.5 * samplerate
    normal_cutoff = cutoff_freq / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff_freq, samplerate, order=5):
    """Filter the data with a Butterworth highpass filter.

    Args:
        data: the input data.
        cutoff_freq: the cutoff frequency in Hz.
        samplerate: the sample rate of the signal.
        order: the order of the Butterworth filter.

    Returns:
        y: the filtered data with the same type as the input data.
    """

    b, a = butter_highpass(cutoff_freq, samplerate, order=order)
    data_shape_restore = False
    mx_nd_restore = False
    if type(data) == mx.nd.ndarray.NDArray:
        data = data.asnumpy()
        mx_nd_restore = True
    if len(data.shape) == 2 and data.shape[1] == 1:
        data = np.squeeze(data)
        data_shape_restore = True
    y = signal.filtfilt(b, a, data, axis=0)
    if data_shape_restore:
        y = np.squeeze(y)
        y = np.expand_dims(y, axis=1)
    if mx_nd_restore:
        y = mx.nd.array(y)
    return y


def get_raw_audio_snr(signal, vad_info_list, audio_sample_rate=16000):
    """Calculate the snr from the raw signal.

    Args:
        signal (numpy.ndarray): one dimensional data with waveform data.
        vad_info_list (list): list of pairs of (vad_start, vad_end).
        audio_sample_rate (int): audio sample rate (default: 16000).

    Returns:
        snr (int): the snr value of signal.
    """

    assert type(signal) == np.ndarray
    # assert len(signal.shape) == 1
    if len(signal.shape) == 2:
        # we assume the signal channel num is less than 10
        assert signal.shape[1] < 10
    else:
        assert len(signal.shape) == 1
    assert len(vad_info_list) != 0
    assert type(audio_sample_rate) == int

    mean_voice_power = get_signal_segments_average_power(
        signal, vad_info_list, audio_sample_rate
    )
    silence_segments = get_complement_segments(vad_info_list)
    mean_noise_power = get_signal_segments_average_power(
        signal, silence_segments, audio_sample_rate
    )

    return 10 * np.log10(mean_voice_power / (mean_noise_power + EPS))


def get_signal_segments_average_power(
    signal, segment_list, audio_sample_rate=16000
):
    """Calculate the power level within segments.

    Args:
        signal (numpy.ndarray): one dimensional data with waveform data.
        segment_list (list): list of pair of (start, end) where start and end
            are float/int values in second indicating the start and end time
            of the segment.
        audio_sample_rate (int): audio sample rate.

    Returns:
        mean_power (float): mean power level of all segments.
    """

    assert type(signal) == np.ndarray
    # assert len(signal.shape) == 1
    if len(signal.shape) == 2:
        # we assume the signal channel num is less than 10
        assert signal.shape[1] < 10
    else:
        assert len(signal.shape) == 1
    assert len(segment_list) != 0

    power_level_list = []
    for start, end in segment_list:
        assert isinstance(start, (float, int))
        assert type(end) == float
        assert start <= end
        start_idx = int(start * audio_sample_rate)
        end_idx = int(end * audio_sample_rate)
        assert start_idx < signal.shape[0]
        if end_idx > signal.shape[0]:
            continue
        assert end_idx <= signal.shape[0]
        # print('----------')
        # print(start, end)
        # print(start_idx, end_idx)
        signal_segment = signal[start_idx:end_idx]
        power_level_list.append(np.mean(signal_segment ** 2))
    # print(power_level_list)
    return np.mean(power_level_list)


def get_complement_segments(segment_list):
    """
    Get complement segments of given segment list.

    The complement segment is the part of the full time range
    that is not contained in the segments.

    Args:
        segment_list (list): list of segments represented as a pair of
            start and end times.

    Returns:
        new_segment_list (list): complement segments.
    """

    assert len(segment_list) > 1
    new_segment_list = []
    last_start, last_end = None, None
    for start, end in segment_list:
        if last_start is None and last_end is None:
            last_start = start
            last_end = end

        new_segment_list.append((last_end, start))
    return new_segment_list


def get_signal_average_power(signal):
    """Calculate the average power of the input signal.

    Args:
        signal (numpy.ndarray): One dimensional numpy array with waveform data.

    Returns:
        float: average power level of the input signal.
    """

    return np.mean(signal ** 2)


def match_signal_to_target_power(src_signal, target_power, src_vad_list=None):
    """Match the power of the input signal to a target power level.

    Args:
        src_signal (numpy.ndarray): one dimensional numpy array representing
            the input signal.
        target_power (float): target power level to match the input signal to.
        src_vad_list (list, optional): list of pair of (start, end) where start
            and end are float/int values in second indicating the start and end
            time of the valid audio segment. Defaults to None.
    Returns:
        numpy.ndarray: one dimensional numpy array representing the adjusted
            input signal.
    """

    if src_vad_list is None:
        src_power = get_signal_average_power(src_signal)
    else:
        src_power = get_signal_segments_average_power(src_signal, src_vad_list)
    src_signal = src_signal * np.sqrt(target_power / src_power + EPS)
    return src_signal


def match_signal_to_target_signal_power(
    src_signal, target_signal, src_vad_list=None, target_vad_list=None
):
    """
    Match the power of `src_signal` to `target_signal`.

    Args:
        src_signal (numpy.ndarray): The source signal whose power needs
            to be matched.
        target_signal (numpy.ndarray): The target signal whose power the
            source signal needs to match.
        src_vad_list (list, optional): The voice activity detection (VAD)
            list for the source signal.
        target_vad_list (list, optional): The voice activity detection (VAD)
            list for the target signal.
    Returns:
        numpy.ndarray: The source signal whose power has been matched to that
            of the target signal.
    """

    if target_vad_list is None:
        tgt_power = get_signal_average_power(target_signal)
    else:
        tgt_power = get_signal_segments_average_power(
            target_signal, target_vad_list
        )
    return match_signal_to_target_power(src_signal, tgt_power, src_vad_list)


def spectrum_extraction(signal, stft_param):
    """Extract the complex-valued spectrogram from the signal.

    Args:
        signal (numpy.ndarray): 1-dimensional audio signal
        stft_param (dict): A dictionary containing the parameters for STFT,
            including:
                "fft_size": FFT window size
                "hop_len": number of samples between STFT columns
                "win_len": length of the analysis window
                "center": whether to center the frame data

    Returns:
        numpy.ndarray: 2-dimensional complex-valued spectrogram.
    """

    stft_result = librosa.core.spectrum.stft(
        signal,
        stft_param["fft_size"],
        stft_param["hop_len"],
        stft_param["win_len"],
        center=stft_param["center"],
    )
    return np.stack([stft_result.real, stft_result.imag], axis=-1)


def norm_audio_signal(signal):
    """Normalize audio signal.

    Args:
        signal (numpy.ndarray): audio signal.

    Returns:
        tuple: (normalized audio signal, maximum value of audio signal)
    """

    s = signal - np.mean(signal)
    max_value = np.max(np.abs(s))
    s /= max_value + EPS
    return s, max_value


@require_packages("soundfile")
def norm_audio_signal_by_decibel(audio_data, norm):
    """Normalize audio signal by decibel.

    Args:
        audio_data (numpy.ndarray): audio signal.
        norm (float): decibel level.

    Returns:
        numpy.ndarray: normalized audio signal.
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_wav_path = join(tmpdirname, "tmp.wav")
        sf.write(tmp_wav_path, audio_data, samplerate=16000)
        tmp_wav2_path = join(tmpdirname, "tmp2.wav")
        cmd = f"sox {tmp_wav_path} {tmp_wav2_path} norm {norm}"
        subprocess.check_call(cmd, shell=True, env=None)
        signal_out, sample_rate = sf.read(tmp_wav2_path)
        return signal_out


def power_scaling(data, ratio=0.3):
    """Power scaling for audio signal.

    Args:
        data (numpy.ndarray or mxnet.ndarray.NDArray): audio signal.
        ratio (float, optional): scaling ratio. Defaults to 0.3.

    Returns:
        numpy.ndarray or mxnet.ndarray.NDArray: power scaled audio signal.
    """

    if isinstance(data, mx.nd.NDArray):
        data = data.asnumpy()
    sign = np.sign(data)
    data_cp = sign * data
    data_cp = np.power(data_cp, ratio)
    return sign * data_cp


def power_scale_cartesian_complex_data_with_accurate_phase_info(
    data, ratio=0.3, axis=2
):
    """
    Scale complex data in cartesian format with accurate phase information.

    Args:
        data (np.ndarray or mx.nd.NDArray): Complex data in cartesian format.
        ratio (float, optional): Power scaling ratio. Default is 0.3.
        axis (int, optional): Axis along which to perform scaling.
            Default is 2.

    Returns:
        np.ndarray or mx.nd.NDArray: Scaled complex data in cartesian format.
    """

    assert data.shape[axis] == 2
    if isinstance(data, mx.nd.NDArray):
        real, imag = mx.nd.split(
            data, axis=axis, num_outputs=2, squeeze_axis=1
        )
        mag = mx.nd.sqrt(real ** 2 + imag ** 2 + EPS)

        mag_norm = mx.nd.power(mag, ratio)
        scaling_ratio = mag_norm / (mag + EPS)
        return mx.nd.stack(
            real * scaling_ratio, imag * scaling_ratio, axis=axis
        )

    else:
        real, imag = np.split(data, 2, axis=axis)
        real = np.squeeze(real, axis=axis)
        imag = np.squeeze(imag, axis=axis)
        mag = np.sqrt(real ** 2 + imag ** 2 + EPS)
        mag_norm = np.power(mag, ratio)
        scaling_ratio = mag_norm / (mag + EPS)
        return np.stack(
            [real * scaling_ratio, imag * scaling_ratio], axis=axis
        )


def log_scale_cartesian_complex_data_with_accurate_phase_info(
    data,
    base=10,
    axis=2,
):
    """
    Scale the cartesian complex data by logarithmically \
    scaling the magnitude.

    Args:
        data (ndarray or mx.nd.NDArray): The input data to be scaled
        base (int or float, optional): The base of logarithm. Can be
            10, 'e' or 2. (default: 10)
        axis (int, optional): The axis along which the complex numbers
            are stored. (default: 2)

    Returns:
        ndarray or mx.nd.NDArray: The logarithmically scaled cartesian
            complex data.
    """

    assert data.shape[axis] == 2
    if isinstance(data, mx.nd.NDArray):
        real, imag = mx.nd.split(data, axis=axis, num_outputs=2)
        F = mx.nd
    else:
        real, imag = np.split(data, 2, axis=axis)
        F = np

    mag = F.sqrt(real ** 2 + imag ** 2)
    if base == 10:
        mag_norm = F.log10(1 + mag)
    elif base == "e":
        mag_norm = F.log(1 + mag)
    elif base == 2:
        mag_norm = F.log2(1 + mag)
    else:
        mag_norm = F.log2(1 + mag) / F.log2(base)

    scaling_ratio = mag_norm / (mag + EPS)
    return data * scaling_ratio


def log_unscale_cartesian_complex_data_with_accurate_phase_info(
    data, base=10, axis=2
):
    """Revert the log-scale of complex data with accurate phase information.

    Args:
        data: The input complex data, with shape (..., 2),
            where the last axis holds the real and imaginary parts.
        base: The logarithmic base to use, can be either 10, "e",
            2 or a custom base.
        axis: The axis along which the data is split into real
            and imaginary parts.

    Returns:
        np.ndarray or mx.nd.NDArray: The unscaled complex data, with the
            same shape as the input.
    """

    assert data.shape[axis] == 2
    if isinstance(data, mx.nd.NDArray):
        real, imag = mx.nd.split(data.detach(), axis=axis, num_outputs=2)
        F = mx.nd
    else:
        real, imag = np.split(data, 2, axis=axis)
        F = np

    mag = F.sqrt(real ** 2 + imag ** 2)
    if base == "e":
        uncompress_mag = F.exp(mag) - 1
    else:
        uncompress_mag = F.power(base, mag) - 1

    scaling_ratio = uncompress_mag / (mag + EPS)
    return data * scaling_ratio


def linear_fuse_signal(
    left_signal, right_signal, overlap_time, audio_sampling_rate
):
    """
    Fuse two audio signals linearly at their overlapping part.

    Args:
        left_signal: The left audio signal to be fused.
        right_signal: The right audio signal to be fused.
        overlap_time: The duration of the overlapping part in seconds.
        audio_sampling_rate: The audio sampling rate in Hz.

    Returns:
        np.ndarray: The fused audio signal.
    """

    num_overlap_samples = int(overlap_time * audio_sampling_rate)

    left_clean_sample = left_signal[:-num_overlap_samples]

    left_fusion_prob = (
        np.arange(num_overlap_samples).astype("float32") / num_overlap_samples
    )
    left_fusion_samples = left_signal[-num_overlap_samples:] * left_fusion_prob

    right_fusion_samples = right_signal[:num_overlap_samples] * (
        1 - left_fusion_prob
    )
    right_clean_sample = right_signal[num_overlap_samples:]

    fusion_samples = left_fusion_samples + right_fusion_samples
    return np.concatenate(
        [left_clean_sample, fusion_samples, right_clean_sample], axis=0
    )


def concat_signal(
    left_signal, right_signal, overlap_time, audio_sampling_rate
):
    """
    Concatenate two signals.

    Args:
        left_signal: The left signal to be concatenated.
        right_signal: The right signal to be concatenated.
        overlap_time: The time of the overlap between the two signals.
        audio_sampling_rate: The audio sample rate.

    Returns:
        np.ndarray: The concatenated signal.
    """

    num_overlap_samples = int(overlap_time * audio_sampling_rate)
    right_clean_sample = right_signal[num_overlap_samples:]
    return np.concatenate([left_signal, right_clean_sample], axis=0)


def randomly_cut_signal_to_target_length(signal, target_length):
    """
    Randomly cut the signal to the target length.

    Args:
        signal: The signal to be cut.
        target_length: The target length of the cut signal.

    Returns:
        np.ndarray: The cut signal.
    """

    assert signal.shape[0] > target_length
    signal_length = signal.shape[0]
    length_diff = signal_length - target_length
    start_idx = random.randint(0, length_diff - 1)
    return signal[start_idx : start_idx + target_length]


def randomly_cut_signal_and_vad_to_target_length(
    signal, target_length, vad_segment_list, sample_rate
):
    """
    Randomly cut the signal and the vad segment list to the target length.

    Args:
        signal: The signal to be cut.
        target_length: The target length of the cut signal.
        vad_segment_list: The list of the vad segments.
        sample_rate: The sample rate of the signal.

    Returns:
        np.ndarray: The cut signal.
        list: The cut vad segment list.
    """

    assert signal.shape[0] > target_length
    signal_length = signal.shape[0]
    length_diff = signal_length - target_length
    start_idx = random.randint(0, length_diff - 1)

    shift_time = -start_idx / sample_rate
    max_time = target_length / sample_rate
    shifted_vad_segment_list = shift_vad_segment(
        vad_segment_list, shift_time, max_time
    )

    return (
        signal[start_idx : start_idx + target_length],
        shifted_vad_segment_list,
    )


def shift_vad_segment(vad_segment_list, shift_time, max_time, min_time=0):
    """Shift VAD segments by a given time.

    Args:
        vad_segment_list (list of tuple): list of VAD segments, where each
            segment is represented by a tuple of start and end time.
        shift_time (float): time shift in seconds.
        max_time (float): maximum possible time of a VAD segment end in
            seconds.
        min_time (float, optional): minimum possible time of a VAD segment
            start in seconds (default 0).

    Returns:
        shifted_segment_list (list of tuple): list of shifted VAD segments.
    """

    shifted_segment_list = []
    for vad_segment in vad_segment_list:
        shifted_vad_beg = vad_segment[0] + shift_time
        shifted_vad_end = vad_segment[1] + shift_time
        shifted_vad_segment = (shifted_vad_beg, shifted_vad_end)
        if shifted_vad_beg < min_time or shifted_vad_end > max_time:
            continue

        shifted_segment_list.append(shifted_vad_segment)
    return shifted_segment_list


def randomly_pad_signal_and_vad_to_target_length(
    signal, target_length, vad_segment_list, sample_rate
):
    """
    Pad a given signal and its VAD segments to a target length randomly.

    Args:
        signal (ndarray): audio signal.
        target_length (int): target length of the padded signal
            in number of samples.
        vad_segment_list (list of tuple): list of VAD segments,
            where each segment is represented by a tuple of start and end time.
        sample_rate (int): sample rate of the audio signal.

    Returns:
        padded_signal (ndarray): padded audio signal
        shifted_vad_segment_list (list of tuple): list of shifted VAD
            segments to match the padded signal
    """

    assert signal.shape[0] < target_length
    signal_length = signal.shape[0]
    length_diff = target_length - signal_length
    left_pad = random.randint(0, length_diff)
    right_pad = length_diff - left_pad
    padded_signal = np.pad(
        signal, ((left_pad, right_pad), (0, 0)), mode="constant"
    )

    shift_time = left_pad / sample_rate
    max_time = target_length / sample_rate
    shifted_vad_segment_list = shift_vad_segment(
        vad_segment_list, shift_time, max_time
    )

    return padded_signal, shifted_vad_segment_list


def random_align_signal_to_target_length(
    signal, target_length, vad_segment_list, sample_rate
):
    """Align the signal to the target length by either \
       padding or cutting randomly.

    Args:
        signal (ndarray): The audio signal.
        target_length (int): The target length of the audio signal.
        vad_segment_list (list): List of VAD segments
            in the format [(start, end)].
        sample_rate (int): The sample rate of the audio signal.

    Returns:
        ndarray: The padded or cut audio signal.
        list: List of VAD segments in the format [(start, end)].
    """

    signal_length = len(signal)
    if signal_length < target_length:
        return randomly_pad_signal_and_vad_to_target_length(
            signal, target_length, vad_segment_list, sample_rate
        )
    elif signal_length > target_length:
        return randomly_cut_signal_and_vad_to_target_length(
            signal, target_length, vad_segment_list, sample_rate
        )
    else:
        return signal, vad_segment_list


def get_bandpass_bin_index_from_hz(
    lowpass, highpass, fft_size=512, audio_sampling_rate=16000
):
    """Get the bin index of the audio signal that corresponds \
       to the lowpass and highpass frequencies.

    Args:
        lowpass (int): The lowpass frequency in Hz.
        highpass (int): The highpass frequency in Hz.
        fft_size (int, optional): The size of the FFT. Defaults to 512.
        audio_sampling_rate (int, optional): The sample rate of
            the audio signal. Defaults to 16000.

    Returns:
        tuple: The bin index of the lowpass frequency and the bin
            index of the highpass frequency.
    """

    freq_grid = np.linspace(0, audio_sampling_rate, fft_size)[
        0 : np.int64(fft_size / 2) + 1
    ]
    hz_low_index = np.argmin(np.abs(freq_grid - lowpass))
    hz_high_index = np.argmin(np.abs(freq_grid - highpass))
    return hz_low_index, hz_high_index


def mix_signals(
    premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None, n_itf=None
):
    """Adjust SNR and SIR, used as callback mix function for pyroomacoustics.

    Args:
        premix (numpy.ndarray): [n_src, mic_chns, wav_len]
            dimensional waveform data
        snr (int): ref signal to background noises ratio in decibels
        sir (int): ref signal to interferers ratio in decibels
        ref_mic (int): one mic siganl as reference
        n_src: num of total source signals
        n_tgt: num of target signals
        n_itf: num of interferers

    Returns:
        mix (numpy.ndarray): mix of the adjusted target,
            interferer and noise signals
    """

    # first normalize all separate recording to have unit
    # power at microphone one
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None] + EPS
    if n_itf is not None and n_itf > 0:
        # now compute the power of interference signal needed
        # to achieve desired SIR
        sigma_i = np.sqrt(10 ** (-sir / 10) / (n_itf + EPS))
        premix[n_tgt : n_tgt + n_itf, :, :] *= sigma_i
    # compute noise variance
    sigma_n = np.sqrt(10 ** (-snr / 10) / (n_src - n_tgt - n_itf + EPS))
    premix[n_tgt + n_itf : n_src, :, :] *= sigma_n

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src, :], axis=0)

    return mix


def calculate_doa(mic_pos, source_pos):
    """
    Calculate the direction of arrival (DOA) of a sound source \
    using the positions of the microphone and the source.

    Args:
        mic_pos (tuple): A tuple representing the (x, y)
            position of the microphone.
        source_pos (tuple): A tuple representing the (x, y)
            position of the sound source.

    Returns:
        float: A float representing the DOA in degrees.
    """

    x = source_pos[0] - mic_pos[0]
    y = source_pos[1] - mic_pos[1]
    if x >= 0 and y >= 0:
        angle = math.degrees(
            math.atan(math.fabs(y) / (math.fabs(x) + 1e-8))
        )  # first quadrant
    if x < 0 and y >= 0:
        # second quadrant
        angle = 180 - math.degrees(
            math.atan(math.fabs(y) / (math.fabs(x) + 1e-8))
        )
    if x <= 0 and y < 0:
        angle = (
            math.degrees(math.atan(math.fabs(y) / (math.fabs(x) + 1e-8))) + 180
        )  # third quadrant
    if x > 0 and y < 0:
        # fourth quadrant
        angle = 360 - math.degrees(
            math.atan(math.fabs(y) / (math.fabs(x) + 1e-8))
        )
    return angle


def calculate_steer_vector(topo, doa, num_bins=257, c=340, sample_rate=16000):
    """
    Calculate the steer vector for a given direction of arrival (DOA).

    Args:
        topo (np.ndarray): A numpy array representing the topography
            of the microphone array.
        doa (float): A float representing the DOA in degrees.
        num_bins (int): An integer representing the number of frequency bins.
            Defaults to 257.
        c (float): A float representing the speed of sound.
            Defaults to 340 m/s.
        sample_rate (int): An integer representing the sample rate in Hz.
            Defaults to 16000.

    Returns:
        np.ndarray: A numpy array representing the steer vector.
    """

    omega = np.pi * np.arange(num_bins) * sample_rate / (num_bins - 1)
    # convert to radians
    dist = np.cos(math.radians(doa)) * topo
    time_delay = dist / c
    steer_vector = np.exp(-1j * np.outer(omega, time_delay)) / len(topo)
    return steer_vector


def apply_beamforming(mixed_signal, steer_vector):
    """
    Apply beamforming to a mixed audio signal using a given steer vector.

    Args:
        mixed_signal (np.ndarray): A numpy array representing
            the mixed audio signal.
        steer_vector (np.ndarray): A numpy array representing
            the steer vector.

    Returns:
        np.ndarray: A numpy array representing the estimated source signal.
    """

    stft = [
        librosa.core.spectrum.stft(mixed_signal[:, i], 512, 100, 512)
        for i in range(mixed_signal.shape[1])
    ]
    stft = np.transpose(np.array(stft), (1, 0, 2))
    stft_enh = np.einsum("...n,...nt->...t", steer_vector.conj(), stft)
    esti_signal = librosa.core.spectrum.istft(stft_enh, 100, 512)
    return esti_signal


def is_empty_signal(signal):
    """
    Check if the input signal is empty or not.

    Args:
        signal (numpy.ndarray): Input signal data.

    Returns:
        bool: True if the signal is empty, False otherwise.

    Raises:
        NotImplementedError: If the input signal data type is not supported.
    """

    if type(signal) == mx.nd.ndarray.NDArray:
        raise NotImplementedError
    elif type(signal) == np.ndarray:
        return np.max(signal) == 0 and np.mean(signal) == 0
    else:
        raise NotImplementedError


@require_packages("soundfile")
def get_audio_length(audio_path):
    """
    Get the length of the audio file.

    Args:
        audio_path (str): Path of the audio file.

    Returns:
        float: The length of the audio file in seconds.
    """

    return sf.info(audio_path).duration


def multichannel_homogeneous_noise_generation(
    signal, mic_num, fft_size=512, hop_len=256, win_len=512
):
    """
    Generate multi-channel homogeneous noise for the input signal.

    Args:
        signal (numpy.ndarray): Input signal data.
        mic_num (int): The number of microphones.
        fft_size (int): The size of FFT. Default is 512.
        hop_len (int): The hop length for FFT. Default is 256.
        win_len (int): The window length for FFT. Default is 512.

    Returns:
        list of numpy.ndarray: A list of multi-channel noisy signals.

    Raises:
        TypeError: If the input signal is not mono.
    """

    if len(signal.shape) > 1:
        raise TypeError("input must be mono")
    x_freq = librosa.core.spectrum.stft(signal, fft_size, hop_len, win_len)
    x_mag = np.abs(x_freq)
    stft = []
    for _i in range(mic_num):
        ang = np.pi * np.random.uniform(-1, 1, x_freq.shape)
        s = x_mag * np.exp(1j * ang)
        stft.append(s)
    return stft


def linear_array_mixing_matrix_generation(
    mic_num, mic_distance, fft_size=512, sample_rate=16000, c=340
):
    """
    Generate the linear array mixing matrix for the specified \
    microphone array configuration.

    Args:
        mic_num (int): The number of microphones.
        mic_distance (float): The distance between microphones in meters.
        fft_size (int): The size of FFT. Default is 512.
        sample_rate (int): The sampling rate in Hz. Default is 16000.
        c (float): The speed of sound in meters per second. Default is 340.

    Returns:
        numpy.ndarray: The mixing matrix for the specified microphone
            array configuration.
    """

    # sound_speed = 340
    nfreq = np.int64(fft_size / 2 + 1)
    f = np.linspace(0, sample_rate / 2, nfreq, endpoint=True)
    tau_0 = mic_distance / c
    omega = 2 * np.pi * f
    spatial_coherence_matrix = np.zeros(
        (nfreq, mic_num, mic_num), dtype=np.float64
    )
    for i in range(mic_num):
        for j in range(mic_num):
            spatial_coherence_matrix[:, i, j] = np.sinc(
                omega * (j - i) * tau_0 / np.pi
            )

    eigen_value, eigen_vector = np.linalg.eigh(spatial_coherence_matrix)
    mixing_matrix = (
        eigen_value[:, :, None].astype(np.complex64)
    ) ** 0.5 * np.conj(eigen_vector.transpose(0, 2, 1))
    return mixing_matrix


def multichannel_homogeneous_noise(
    signal,
    mic_num=2,
    mic_distance=0.11,
    c=340,
    sample_rate=16000,
    fft_size=512,
    hop_len=256,
    win_len=512,
):
    """
    Generate multi-channel homogeneous noise based on the input signal.

    Args:
        signal (np.ndarray): Input audio signal.
        mic_num (int): Number of microphones in the linear array.
        mic_distance (float): Distance between adjacent microphones in meters.
        c (float): Speed of sound in meters per second.
        sample_rate (int): Sampling rate of the input signal in Hz.
        fft_size (int): Size of the FFT.
        hop_len (int): Number of samples between adjacent STFT columns.
        win_len (int): Length of the STFT window.

    Returns:
        y (np.ndarray): The noisy audio signal with shape (n_samples, mic_num).
    """

    mixing_matrix = linear_array_mixing_matrix_generation(
        mic_num, mic_distance, fft_size, sample_rate
    )

    homogeneous_noise_stft = multichannel_homogeneous_noise_generation(
        signal, mic_num, fft_size, hop_len, win_len
    )

    X = np.array(homogeneous_noise_stft).transpose(
        1, 0, 2
    )  # nfreq * nmic * nframe
    Y = np.matmul(np.conj(mixing_matrix.transpose(0, 2, 1)), X).transpose(
        1, 2, 0
    )
    xlen = len(signal)
    y = np.zeros((xlen, mic_num), dtype=np.float32)

    for i in range(mic_num):

        y[:, i] = librosa.core.spectrum.istft(
            Y[i].transpose(1, 0),
            hop_length=hop_len,
            win_length=win_len,
            length=xlen,
        )
    return y

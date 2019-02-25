import numpy as np
from scipy.signal import stft
from mir_eval.separation import bss_eval_sources


def sdr_batch_eval(target_sources, noisy_sources, estimated_sources, sample_rate=16e3, step_size=10, sequence_lengths=None):
    sdr_list = []
    sir_list = []
    sar_list = []

    n_samples_frame = int(step_size / 1e3 * sample_rate)
    for i, (target, noisy, estimated) in enumerate(zip(target_sources, noisy_sources, estimated_sources)):
        if sequence_lengths is not None:
            target = target[: sequence_lengths[i] * n_samples_frame]
            noisy = noisy[: len(target)]
            estimated = estimated[: len(target)]

        # Skip evaluation if estimated sources is all-zero vector
        if np.any(estimated):
            ref_sources = np.vstack([target, noisy])
            est_sources = np.vstack([estimated, np.ones_like(estimated)])                    
            sdr, sir, sar, _ = bss_eval_sources(ref_sources, est_sources, compute_permutation=False)
            sdr_list.append(sdr[0])
            sir_list.append(sir[0])
            sar_list.append(sar[0])
                             
    return np.array(sdr_list), np.array(sir_list), np.array(sar_list)


def sdr_batch_eval_ss(target_source, estimated_source, sample_rate=16e3, step_size=10, sequence_lengths=None):
    """
    Single source version of SDR, SIR and SDR computation
    """
    sdr_list = []
    sir_list = []
    sar_list = []
    
    n_samples_frame = int(step_size / 1e3 * sample_rate)
    for i, (target, estimated) in enumerate(zip(target_source, estimated_source)):
        if sequence_lengths is not None:
           target = target[: sequence_lengths[i] * n_samples_frame]
           estimated = estimated[: len(target)]

        # Skip evaluation if estimated sources is all-zero vector
        if np.any(estimated):
            sdr, sir, sar, _ = bss_eval_sources(np.array([target]), np.array([estimated]), compute_permutation=False)
            sdr_list.append(sdr[0])
            sir_list.append(sir[0])
            sar_list.append(sar[0])
                             
    return np.array(sdr_list), np.array(sir_list), np.array(sar_list)


def l2_batch_eval(target_sources, estimated_sources, sequence_lengths=None, sample_rate=16e3, window_size=25, overlap_size=15):
    l2_list = []

    window_frame_len = int(window_size / 1e3 * sample_rate)
    overlap_frame_len = int(overlap_size / 1e3 * sample_rate) 
    n_samples_frame = window_frame_len - overlap_frame_len
    for i, (target, estimated) in enumerate(zip(target_sources, estimated_sources)):
        if sequence_lengths is not None:
           target = target[: sequence_lengths[i] * n_samples_frame]
           estimated = estimated[: len(target)]

        freqs, times, target_spec = stft(target, nperseg=window_frame_len, noverlap=overlap_frame_len, nfft=512)
        freqs, times, estimated_spec = stft(estimated, nperseg=window_frame_len, noverlap=overlap_frame_len, nfft=512)

        l2_list.append(np.square(np.subtract(np.abs(target_spec) ** 0.3, np.abs(estimated_spec) ** 0.3)).sum())

    return np.array(l2_list)


def snr_batch_eval(target_sources, estimated_sources, sample_rate=16e3, step_size=10, sequence_lengths=None):
    snr_list = []

    n_samples_frame = int(step_size / 1e3 * sample_rate)
    for i, (target, estimated, seq_len) in enumerate(zip(target_sources, estimated_sources, sequence_lengths)):
        target = target[: seq_len * n_samples_frame]
        estimated = estimated[: len(target)]
        snr_list.append(10 * np.log10(np.sum(target ** 2) / np.sum((target - estimated) ** 2)))

    return np.array(snr_list)

import os
from os.path import join
from glob import glob
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import math
from audio_features import downsampling

# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF constants definition
TF_INTRA_OP_PT = int(os.getenv('TF_INTRA_OP', 0))
TF_INTER_OP_PT = int(os.getenv('TF_INTER_OP', 0))


def ltass_speaker(audio_folder, sample_rate, max_audio_length=48000, window_size=25, step_size=10, n_samples=1000):
    """
    Compute the speaker Long-Term Average Speech Spectrum of a speaker.
    """
    audio_filenames = sorted(glob(join(audio_folder, '*.wav')))
    
    audio_samples = np.zeros((len(audio_filenames), max_audio_length))
    num_frames = np.zeros(len(audio_filenames), dtype=np.int32)
    
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    
    for i, wav_file in enumerate(audio_filenames):
        rate, samples = wavfile.read(wav_file)
        samples = downsampling(samples, rate, sample_rate)
        audio_samples[i, :len(samples)] = samples
        num_frames[i] = len(samples) // step_frame_size
    
    # Create Graph
    with tf.Graph().as_default():
        samples_tensor = tf.constant(audio_samples, dtype=tf.float32)
        # Compute STFT
        specs_tensor = tf.contrib.signal.stft(samples_tensor, frame_length=window_frame_size, frame_step=step_frame_size, pad_end=False)
        # Apply power-law compression
        specs_tensor = tf.abs(specs_tensor) ** 0.3
        
        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT)) as sess:
            specs = sess.run(specs_tensor)

    spectrograms = []
    for spec, nf in zip(specs, num_frames):
        spectrograms.append(spec[:nf])
    
    flat_specs = np.vstack(spectrograms)
    mean_spec = flat_specs.mean(axis=0)
    stdev_spec = flat_specs.std(axis=0)
    
    return mean_spec, stdev_spec, specs


def compute_tbm(audio_folder, mask_threshold, max_audio_length=48000, sample_rate=16e3, n_fft=512, window_size=25, step_size=10):
    """
    Compute TBMs using LTASS.
    """
    audio_filenames = sorted(glob(join(audio_folder, '*.wav')))
    num_frames = np.zeros(len(audio_filenames), dtype=np.int32)
    
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    
    audio_samples = np.zeros((len(audio_filenames), max_audio_length + n_fft//2))
    
    for i, wav_file in enumerate(audio_filenames):
        rate, samples = wavfile.read(wav_file)
        samples = downsampling(samples, rate, sample_rate)
        audio_samples[i, n_fft//2: len(samples) + n_fft//2] = samples
        num_frames[i] = math.ceil(float(len(samples) + n_fft//2) / step_frame_size)
    
    # Create Graph
    with tf.Graph().as_default():
        samples_tensor = tf.constant(audio_samples, dtype=tf.float32)
        # Compute STFT
        specs_tensor = tf.contrib.signal.stft(samples_tensor, frame_length=window_frame_size, frame_step=step_frame_size,
                                              fft_length=n_fft, pad_end=True)
        specs_tensor = tf.abs(specs_tensor)

        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT)) as sess:
            specs = sess.run(specs_tensor)
            
    masks = specs > mask_threshold
    
    return audio_filenames, masks, num_frames


def save_target_binary_masks_speaker(audio_folder, mask_folder, mask_factor=0.5, sample_rate=16e3, max_audio_length=48000, ltass_samples=1000):
    # Create destination directory if not exists
    if not os.path.isdir(mask_folder):
        os.makedirs(mask_folder)
    
    # Compute thresholds and spectrograms
    print('Computing LTASS threshold...')
    threshold_mean, threshold_std, _ = ltass_speaker(audio_folder, sample_rate, max_audio_length=max_audio_length, n_samples=ltass_samples)
    # Denormalize
    threshold_freq = (threshold_mean + threshold_std * mask_factor) ** (1 / 0.3)
    print('done.')
    print('Threshold shape:', threshold_freq.shape)
    
    # Compute binary masks
    audio_filenames, masks, num_frames = compute_tbm(audio_folder, threshold_freq, max_audio_length=max_audio_length)

    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    for a_file, mask, nf in zip(audio_filenames, masks, num_frames):
        s_file = os.path.join(mask_folder, os.path.basename(a_file).replace('.wav', '.npy'))
        np.save(s_file, mask[:nf])
            
    print('Done. Target Binary Masks generated:', len(audio_filenames))


def save_target_binary_masks(dataset_path, list_of_speakers, audio_folder, dest_dir, mask_factor=0.5, sample_rate=16e3, max_audio_length=48000, ltass_samples=1000):
    for s in list_of_speakers:
        print('Computing Target Binary Masks of speaker {:d}...'.format(s))
        audio_path = os.path.join(dataset_path, 's' + str(s), audio_folder)
        dest_path = os.path.join(dataset_path, 's' + str(s), dest_dir)
        
        save_target_binary_masks_speaker(audio_path, dest_path, mask_factor, sample_rate, max_audio_length, ltass_samples)

        print('Speaker {:d} completed.'.format(s))
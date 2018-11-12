import os
from os.path import join
from glob import glob
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import math

# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF constants definition
TF_INTRA_OP_PT = int(os.getenv('TF_INTRA_OP', 0))
TF_INTER_OP_PT = int(os.getenv('TF_INTER_OP', 0))


def downsampling(samples, sample_rate, downsample_rate):
    secs = len(samples) / float(sample_rate)
    num_samples = int(downsample_rate * secs)

    return signal.resample(samples, num_samples)


def compute_spectrograms(audio_folder, max_audio_length=48000, sample_rate=16e3, n_fft=512, window_size=25, step_size=10):
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
        # Apply power-law compression
        specs_tensor = tf.abs(specs_tensor) ** 0.3
    
        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT)) as sess:
            specs = sess.run(specs_tensor)

    return audio_filenames, specs, num_frames


def save_spectrograms_speaker(audio_folder, dest_folder, sample_rate=16e3, max_audio_length=48000):
    # Create destination directory if not exists
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    
    audio_filenames, specs, num_frames = compute_spectrograms(audio_folder, sample_rate=sample_rate, max_audio_length=max_audio_length)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for a_file, spec, nf in zip(audio_filenames, specs, num_frames):
        s_file = os.path.join(dest_folder, os.path.basename(a_file).replace('.wav', '.npy'))
        np.save(s_file, spec[:nf])
            
    print('Done. Spectrogram generated: %d.' % len(audio_filenames))


def save_spectrograms(dataset_path, list_of_speakers, audio_dir, dest_dir, sample_rate=16e3, max_audio_length=48000):
    for s in list_of_speakers:
        print('Computing spectrograms of speaker {:d}...'.format(s))
        audio_path = os.path.join(dataset_path, 's' + str(s), audio_dir)
        dest_path = os.path.join(dataset_path, 's' + str(s), dest_dir)
        
        save_spectrograms_speaker(audio_path, dest_path, sample_rate, max_audio_length)

        print('Speaker {:d} completed.'.format(s))
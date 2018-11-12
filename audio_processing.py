import tensorflow as tf


def compute_stft(sources, window_size=25, step_size=10, out_shape=None):
    """Compute STFT"""
    # Explicit padding at start for correct reconstruction
    paddings = [[0, 0], [256, 0]]
    sources_pad = tf.pad(sources, paddings)
    
    window_frame_size = int(round(window_size / 1e3 * 16e3))
    step_frame_size = int(round(step_size  / 1e3 * 16e3))
        
    # Compute STFTs
    specs = tf.contrib.signal.stft(sources_pad, frame_length=window_frame_size,
                                   frame_step=step_frame_size, pad_end=True)
    
    if out_shape is not None:
        specs = tf.slice(specs, begin=[0, 0, 0], size=out_shape)
    
    return specs


def reconstruct_sources(specs, num_samples=48000, sample_rate=16e3, window_size=25, step_size=10):
    """Compute inverse STFT"""
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    reconstructed_sources = tf.contrib.signal.inverse_stft(specs, frame_length=window_frame_size,
                                                           frame_step=step_frame_size,
                                                           window_fn=tf.contrib.signal.inverse_stft_window_fn(step_frame_size))

    if num_samples > 0:
        return tf.slice(reconstructed_sources, begin=[0,0], size=[tf.shape(specs)[0], num_samples])
    else:
        return reconstructed_sources


def get_sources(enh_mag_specs, rec_ang_specs, num_samples=48000, sample_rate=16e3, window_size=25, step_size=10):
    """Get waveform from magnitude and phase of STFT"""
    enh_specs = tf.complex(real=enh_mag_specs * tf.cos(rec_ang_specs), imag=enh_mag_specs * tf.sin(rec_ang_specs))
    
    return reconstruct_sources(enh_specs, num_samples, sample_rate, window_size, step_size)


def get_oracle_iam(target_stft, mixed_stft, clip_value=10):
    """Get oracle Ideal Amplitude Mask (IAM) from target and mixed spectrograms"""
    target_specs_mag = tf.abs(target_stft) ** 0.3
    mixed_specs_mag = tf.abs(mixed_stft) ** 0.3

    return tf.cast(tf.clip_by_value(target_specs_mag / mixed_specs_mag, 0, clip_value), tf.float64)
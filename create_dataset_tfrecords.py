from glob import glob
import os
import random
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy import signal
import time
from av_sync import sync_audio_visual_features
from audio_features import downsampling


def adjust_samples(reference_len, adjusted):
    if len(adjusted) < reference_len:
        adjusted_samples = np.zeros(reference_len)
        shift = (reference_len - len(adjusted)) // 2
        adjusted_samples[shift: shift + len(adjusted)] = adjusted
        return adjusted_samples
    else:
        return adjusted[: reference_len]


def get_video_filenames(data_path, video_folder, speaker, n_samples=1000):
    features_file_list = []

    speaker_file_list = glob(os.path.join(data_path, 's' + str(speaker), video_folder, '*.txt'))
    random.shuffle(speaker_file_list)
        
    return sorted(speaker_file_list[:n_samples])


def get_mix_filenames(data_path, mix_audio_folder, speaker, n_samples):
    file_list = []

    speaker_file_list = glob(os.path.join(data_path, 's' + str(speaker), mix_audio_folder, '*.wav'))
    random.shuffle(speaker_file_list)
        
    return sorted(speaker_file_list[:n_samples])


def video_normalization(features_file_list, delta=0):
    print('Total number of files = {}'.format(len(features_file_list)))

    features = []

    for file_index, txt_file in enumerate(features_file_list):
        data = np.loadtxt(txt_file)
        if len(data.shape) == 2:
            if delta > 0:
                data = data[1:] - data[:-1]
                if delta == 2:
                    data = data[1:] - data[:-1]
            features.append(data)

    features = np.vstack(features)
    print(features.shape)

    dataset_mean = np.mean(features, axis=0)
    dataset_stdev = np.std(features, axis=0)

    print('Mean:')
    print(dataset_mean.shape)
    print(dataset_mean)
    print('Standard deviation:')
    print(dataset_stdev.shape)
    print(dataset_stdev)

    return dataset_mean, dataset_stdev


def audio_normalization(features_file_list):
    print('Total number of files = {}'.format(len(features_file_list)))

    features = []

    for file_index, wav_file in enumerate(features_file_list):
        npy_file = wav_file.replace('.wav', '.npy')
        data = np.load(npy_file)
        if len(data.shape) == 2:
            features.append(data)

    features = np.vstack(features)
    print(features.shape)

    dataset_mean = np.mean(features, axis=0)
    dataset_stdev = np.std(features, axis=0)

    print('Mean:')
    print(dataset_mean.shape)
    print(dataset_mean)
    print('Standard deviation:')
    print(dataset_stdev.shape)
    print(dataset_stdev)

    return dataset_mean, dataset_stdev

def serialize_sample_fixed(video_sequence, tbm, audio_samples, audio_paths, mix_spec_norm):
    # Video_sequence and audio_sequence must have the same length
    # The object we return
    example = tf.train.SequenceExample()

    # Non-sequential features of our example
    example.context.feature['sequence_length'].int64_list.value.append(len(video_sequence))
    example.context.feature['base_audio_wav'].float_list.value.extend(audio_samples[0])
    example.context.feature['other_audio_wav'].float_list.value.extend(audio_samples[1])
    example.context.feature['mix_audio_wav'].float_list.value.extend(audio_samples[2])
    example.context.feature['base_audio_path'].bytes_list.value.append(audio_paths[0].encode())
    example.context.feature['other_audio_path'].bytes_list.value.append(audio_paths[1].encode())
    example.context.feature['mix_audio_path'].bytes_list.value.append(audio_paths[2].encode())

    # Feature lists for the sequential features of our example
    fl_tbm = example.feature_lists.feature_list['tbm']
    fl_video_feat = example.feature_lists.feature_list['video_features']
    fl_mix_spec = example.feature_lists.feature_list['mixed_spectrogram']

    for tbm_feat in tbm:
        fl_tbm.feature.add().float_list.value.extend(tbm_feat)
    
    for video_feat in video_sequence:
        fl_video_feat.feature.add().float_list.value.extend(video_feat)

    for mix_spec_feat in mix_spec_norm:
        fl_mix_spec.feature.add().float_list.value.extend(mix_spec_feat)
    
    return example


def serialize_sample_var(video_sequence, tbm, audio_samples, audio_paths, mix_spec_norm):
    # Video_sequence and audio_sequence must have the same length
    # The object we return
    example = tf.train.SequenceExample()
    
    # Non-sequential features of our example
    example.context.feature['sequence_length'].int64_list.value.append(len(video_sequence))
    
    # Feature lists for the sequential features of our example
    fl_tbm = example.feature_lists.feature_list['tbm']
    fl_video_feat = example.feature_lists.feature_list['video_features']
    fl_mix_spec = example.feature_lists.feature_list['mixed_spectrogram']
    fl_base_audio_wav = example.feature_lists.feature_list['base_audio_wav']
    fl_other_audio_wav = example.feature_lists.feature_list['other_audio_wav']
    fl_mix_audio_wav = example.feature_lists.feature_list['mix_audio_wav']
    fl_base_audio_path = example.feature_lists.feature_list['base_audio_path']
    fl_other_audio_path = example.feature_lists.feature_list['other_audio_path']
    fl_mix_audio_path = example.feature_lists.feature_list['mix_audio_path']
    
    for tbm_feat in tbm:
        fl_tbm.feature.add().float_list.value.extend(tbm_feat)
    
    for video_feat in video_sequence:
        fl_video_feat.feature.add().float_list.value.extend(video_feat)
    
    for mix_spec_feat in mix_spec_norm:
        fl_mix_spec.feature.add().float_list.value.extend(mix_spec_feat)
    
    for base_audio_sample in audio_samples[0]:
        fl_base_audio_wav.feature.add().float_list.value.append(base_audio_sample)
    
    for other_audio_sample in audio_samples[1]:
        fl_other_audio_wav.feature.add().float_list.value.append(other_audio_sample)

    for mix_audio_sample in audio_samples[2]:
        fl_mix_audio_wav.feature.add().float_list.value.append(mix_audio_sample)
    
    for base_path_feat in audio_paths[0]:
        fl_base_audio_path.feature.add().int64_list.value.append(ord(base_path_feat))
    
    for other_path_feat in audio_paths[1]:
        fl_other_audio_path.feature.add().int64_list.value.append(ord(other_path_feat))

    for mix_path_feat in audio_paths[2]:
        fl_mix_audio_path.feature.add().int64_list.value.append(ord(mix_path_feat))
    
    return example


def get_filenames_2spk(data_path, mix_audio_folder, base_audio_folder, video_folder, tbm_folder):
    speaker = os.path.basename(mix_audio_folder)
    mix_audio_file_list = glob(os.path.join(data_path, mix_audio_folder, '*.wav'))
    base_audio_file_list = []
    other_audio_file_list = []
    video_file_list = []
    tbm_file_list = []

    for mix_file in mix_audio_file_list:
        mix_split = os.path.basename(mix_file).split('_')
        base_audio_file_list.append(os.path.join(data_path, speaker, base_audio_folder, mix_split[0] + '.wav'))
        tbm_file_list.append(os.path.join(data_path, speaker, tbm_folder, mix_split[0] + '.npy'))
        video_file_list.append(os.path.join(data_path, speaker, video_folder, mix_split[0] + '.txt'))
        other_audio_file_list.append(os.path.join(data_path, mix_split[2], base_audio_folder, mix_split[3]))

    return mix_audio_file_list, base_audio_file_list, tbm_file_list, video_file_list, other_audio_file_list


def get_filenames_3spk(data_path, mix_audio_folder, base_audio_folder, video_folder, tbm_folder):
    speaker = os.path.basename(mix_audio_folder)
    mix_audio_file_list = glob(os.path.join(data_path, mix_audio_folder, '*.wav'))
    base_audio_file_list = []
    other1_audio_file_list = []
    other2_audio_file_list = []
    video_file_list = []
    tbm_file_list = []

    for mix_file in mix_audio_file_list:
        mix_split = os.path.basename(mix_file).split('_')
        base_audio_file_list.append(os.path.join(data_path, speaker, base_audio_folder, mix_split[0] + '.wav'))
        tbm_file_list.append(os.path.join(data_path, speaker, tbm_folder, mix_split[0] + '.npy'))
        video_file_list.append(os.path.join(data_path, speaker, video_folder, mix_split[0] + '.txt'))
        other1_audio_file_list.append(os.path.join(data_path, mix_split[2], base_audio_folder, mix_split[3] + '.wav'))
        other2_audio_file_list.append(os.path.join(data_path, mix_split[4], base_audio_folder, mix_split[5]))

    return mix_audio_file_list, base_audio_file_list, tbm_file_list, video_file_list, other1_audio_file_list, other2_audio_file_list


def create_tfrecords_speaker_2spk(data_path, mix_audio_folder, video_folder, tbm_folder, base_audio_folder, dest_dir,
                                  delta, dataset_norm_folder, tfrecord_mode='fixed', file_counter=0):
    # get speaker code
    speaker = os.path.basename(mix_audio_folder)
    
    # get associated files from file_list
    mix_audio_file_list, base_audio_file_list, tbm_file_list, video_file_list, other_audio_file_list = \
        get_filenames_2spk(data_path, mix_audio_folder, base_audio_folder, video_folder, tbm_folder)
    
    # normalization data filenames
    dataset_video_mean_file = os.path.join(dataset_norm_folder, speaker + '_video_mean.npy')
    dataset_video_std_file = os.path.join(dataset_norm_folder, speaker + '_video_std.npy')
    dataset_audio_mean_file = os.path.join(dataset_norm_folder, speaker + '_audio_mean.npy')
    dataset_audio_std_file = os.path.join(dataset_norm_folder, speaker + '_audio_std.npy')

    # load dataset video mean and std
    try:
        dataset_video_mean = np.load(dataset_video_mean_file)
        dataset_video_std = np.load(dataset_video_std_file)
    except IOError:
        print('Computing mean and std of video features of speaker %s...' % speaker)
        dataset_video_mean, dataset_video_std = video_normalization(video_file_list, delta=delta)
        np.save(dataset_video_mean_file, dataset_video_mean)
        np.save(dataset_video_std_file, dataset_video_std)

    # load dataset audio mean and std
    try:
        dataset_audio_mean = np.load(dataset_audio_mean_file)
        dataset_audio_std = np.load(dataset_audio_std_file)
    except IOError:
        print('Computing mean and std of audio features of speaker %s...' % speaker)
        dataset_audio_mean, dataset_audio_std = audio_normalization(mix_audio_file_list)
        np.save(dataset_audio_mean_file, dataset_audio_mean)
        np.save(dataset_audio_std_file, dataset_audio_std)

    
    for file_video, file_tbm, file_base_audio, file_mix_audio, file_other_audio in zip(video_file_list, tbm_file_list, base_audio_file_list, mix_audio_file_list, other_audio_file_list):
        # get sync features
        tbm, features_video = sync_audio_visual_features(file_tbm, file_video)
        mix_spec = np.load(file_mix_audio.replace('.wav', '.npy'))

        # Check return value
        if features_video is not None:
            print('TBM: {:s} - Video: {:s} - Audio: {:s}'.format(file_tbm, file_video, file_mix_audio))

            # Normalize video features
            if not delta:
                features_video = np.subtract(features_video, dataset_video_mean) / dataset_video_std
            else:
                # Add delta video features
                delta_features_video = np.zeros_like(features_video)
                delta_features_video[1:] = features_video[1:] - features_video[:-1]
                if delta == 1:
                    delta_features_video_norm = np.subtract(delta_features_video, dataset_video_mean) / dataset_video_std
                elif delta == 2:
                    delta_features_video[2:] = delta_features_video[2:] - delta_features_video[1:-1]
                    delta_features_video_norm = np.subtract(delta_features_video, dataset_video_mean) / dataset_video_std
                features_video = delta_features_video_norm
        
            # Normalize spectrograms
            mix_spec_norm = np.subtract(mix_spec, dataset_audio_mean) / dataset_audio_std
  
            # read audio files
            sample_rate, base_audio_wav = wavfile.read(file_base_audio)
            _, other_audio_wav = wavfile.read(file_other_audio)
            _, mix_audio_wav = wavfile.read(file_mix_audio)
            # downsampling
            base_audio_wav = downsampling(base_audio_wav, sample_rate, 16e3)
            other_audio_wav = downsampling(other_audio_wav, sample_rate, 16e3)
            mix_audio_wav = downsampling(mix_audio_wav, sample_rate, 16e3)
            # adjust samples to match audio lengths
            base_audio_wav = base_audio_wav[: len(mix_audio_wav)]
            other_audio_wav = adjust_samples(len(mix_audio_wav), other_audio_wav)

            # get filenames relative paths
            base_audio_path = file_base_audio.replace(data_path, '')
            other_audio_path = file_other_audio.replace(data_path, '')
            mix_audio_path = file_mix_audio.replace(data_path, '')

            # create TFRecord
            sample_file = os.path.join(dest_dir, 'sample_{:05d}.tfrecords'.format(file_counter))
            file_counter += 1

            with open(sample_file, 'w') as fp:
                writer = tf.python_io.TFRecordWriter(fp.name)
                audio_wavs = [base_audio_wav, other_audio_wav, mix_audio_wav]
                audio_paths = [base_audio_path, other_audio_path, mix_audio_path]
                if tfrecord_mode == 'fixed':
                    serialized_sample = serialize_sample_fixed(features_video, tbm, audio_wavs, audio_paths, mix_spec_norm)
                elif tfrecord_mode == 'var':
                    serialized_sample = serialize_sample_var(features_video, tbm, audio_wavs, audio_paths, mix_spec_norm)
                writer.write(serialized_sample.SerializeToString())
                writer.close()
        else:
            print('Skipped -> TBM: {:s} - Video: {:s}'.format(file_tbm, file_video))

    return file_counter


def create_tfrecords_speaker_3spk(data_path, mix_audio_folder, video_folder, tbm_folder, base_audio_folder, dest_dir,
                                  delta, dataset_norm_folder, tfrecord_mode='fixed', file_counter=0):
    # get speaker code
    speaker = os.path.basename(mix_audio_folder)
    
    # get associated files from file_list
    mix_audio_file_list, base_audio_file_list, tbm_file_list, video_file_list, other1_audio_file_list, other2_audio_file_list = \
        get_filenames_3spk(data_path, mix_audio_folder, base_audio_folder, video_folder, tbm_folder)

    # normalization data filenames
    dataset_video_mean_file = os.path.join(dataset_norm_folder, speaker + '_video_mean.npy')
    dataset_video_std_file = os.path.join(dataset_norm_folder, speaker + '_video_std.npy')
    dataset_audio_mean_file = os.path.join(dataset_norm_folder, speaker + '_audio_mean.npy')
    dataset_audio_std_file = os.path.join(dataset_norm_folder, speaker + '_audio_std.npy')

    # load dataset video mean and std
    try:
        dataset_video_mean = np.load(dataset_video_mean_file)
        dataset_video_std = np.load(dataset_video_std_file)
    except IOError:
        print('Computing mean and std of video features of speaker %s...' % speaker)
        dataset_video_mean, dataset_video_std = video_normalization(video_file_list, delta=delta)
        np.save(dataset_video_mean_file, dataset_video_mean)
        np.save(dataset_video_std_file, dataset_video_std)

    # load dataset audio mean and std
    try:
        dataset_audio_mean = np.load(dataset_audio_mean_file)
        dataset_audio_std = np.load(dataset_audio_std_file)
    except IOError:
        print('Computing mean and std of audio features of speaker %s...' % speaker)
        dataset_audio_mean, dataset_audio_std = audio_normalization(mix_audio_file_list)
        np.save(dataset_audio_mean_file, dataset_audio_mean)
        np.save(dataset_audio_std_file, dataset_audio_std)


    for file_video, file_tbm, file_base_audio, file_mix_audio, file_other_audio1, file_other_audio2 in zip(video_file_list, tbm_file_list, base_audio_file_list, mix_audio_file_list, other1_audio_file_list, other2_audio_file_list):
        # get sync features
        tbm, features_video = sync_audio_visual_features(file_tbm, file_video)
        mix_spec = np.load(file_mix_audio.replace('.wav', '.npy'))

        # Check return value
        if features_video is not None:
            print('TBM: {:s} - Video: {:s} - Audio: {:s}'.format(file_tbm, file_video, file_mix_audio))

            # Normalize video features
            if not delta:
                features_video = np.subtract(features_video, dataset_video_mean) / dataset_video_std
            else:
                # Add delta video features
                delta_features_video = np.zeros_like(features_video)
                delta_features_video[1:] = features_video[1:] - features_video[:-1]
                if delta == 1:
                    delta_features_video_norm = np.subtract(delta_features_video, dataset_video_mean) / dataset_video_std
                elif delta == 2:
                    delta_features_video[2:] = delta_features_video[2:] - delta_features_video[1:-1]
                    delta_features_video_norm = np.subtract(delta_features_video, dataset_video_mean) / dataset_video_std
                features_video = delta_features_video_norm
        
            # Normalize spectrograms
            mix_spec_norm = np.subtract(mix_spec, dataset_audio_mean) / dataset_audio_std
  
            # read audio files
            sample_rate, base_audio_wav = wavfile.read(file_base_audio)
            _, other1_audio_wav = wavfile.read(file_other_audio1)
            _, other2_audio_wav = wavfile.read(file_other_audio2)
            _, mix_audio_wav = wavfile.read(file_mix_audio)
            # downsampling
            base_audio_wav = downsampling(base_audio_wav, sample_rate, 16e3)
            other1_audio_wav = downsampling(other1_audio_wav, sample_rate, 16e3)
            other2_audio_wav = downsampling(other2_audio_wav, sample_rate, 16e3)
            mix_audio_wav = downsampling(mix_audio_wav, sample_rate, 16e3)
            # adjust samples to match audio lengths
            base_audio_wav = base_audio_wav[: len(mix_audio_wav)]
            other1_audio_wav = adjust_samples(len(mix_audio_wav), other1_audio_wav)
            other2_audio_wav = adjust_samples(len(mix_audio_wav), other2_audio_wav)
            other_audio_wav = other1_audio_wav + other2_audio_wav

            # get filenames relative paths
            base_audio_path = file_base_audio.replace(data_path, '')
            other_audio_path = '_'.join(os.path.basename(file_mix_audio).split('_')[2:])[:-4]
            mix_audio_path = file_mix_audio.replace(data_path, '')

            # create TFRecord
            sample_file = os.path.join(dest_dir, 'sample_{:05d}.tfrecords'.format(file_counter))
            file_counter += 1

            with open(sample_file, 'w') as fp:
                writer = tf.python_io.TFRecordWriter(fp.name)
                audio_wavs = [base_audio_wav, other_audio_wav, mix_audio_wav]
                audio_paths = [base_audio_path, other_audio_path, mix_audio_path]
                if tfrecord_mode == 'fixed':
                    serialized_sample = serialize_sample_fixed(features_video, tbm, audio_wavs, audio_paths, mix_spec_norm)
                elif tfrecord_mode == 'var':
                    serialized_sample = serialize_sample_var(features_video, tbm, audio_wavs, audio_paths, mix_spec_norm)
                writer.write(serialized_sample.SerializeToString())
                writer.close()
        else:
            print('Skipped -> TBM: {:s} - Video: {:s}'.format(file_tbm, file_video))

    return file_counter


def create_tfrecords(data_path, n_speakers, mix_audio_folder, video_folder, tbm_folder, base_audio_folder, dest_dir,
                     delta, dataset_norm_folder, tfrecord_mode='fixed'):
    file_counter = 0
    for mix_speaker_dir in glob(os.path.join(data_path, mix_audio_folder, '*')):
        if n_speakers == 2:
            file_counter = create_tfrecords_speaker_2spk(data_path, mix_speaker_dir, video_folder, tbm_folder, base_audio_folder,
                                                         dest_dir, delta, dataset_norm_folder, tfrecord_mode, file_counter)
        elif n_speakers == 3:
            file_counter = create_tfrecords_speaker_3spk(data_path, mix_speaker_dir, video_folder, tbm_folder, base_audio_folder,
                                                         dest_dir, delta, dataset_norm_folder, tfrecord_mode, file_counter)
    return file_counter


def create_dataset(dataset_dir, n_speakers, video_dir, tbm_dir, base_audio_dir, mix_audio_dir, norm_data_dir, save_dir, tfrecord_mode='fixed', delta=1):
    train_dir = os.path.join(dataset_dir, save_dir, 'TRAINING_SET')
    val_dir = os.path.join(dataset_dir, save_dir, 'VALIDATION_SET')
    test_dir = os.path.join(dataset_dir, save_dir, 'TEST_SET')

    # create tfrecord directories if they not exist
    for folder in (train_dir, val_dir, test_dir):
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # create normalization folder if not exists
    if not os.path.exists(norm_data_dir):
        os.makedirs(norm_data_dir)
    
    # generate TFRecords
    print('Creating training TFRecords...')
    train_mix_path = os.path.join(mix_audio_dir, 'TRAINING_SET')
    num_train_samples = create_tfrecords(dataset_dir, n_speakers, train_mix_path, video_dir, tbm_dir, base_audio_dir,
                                         train_dir, delta, norm_data_dir, tfrecord_mode)

    print('Creating validation TFRecords...')
    val_mix_path = os.path.join(mix_audio_dir, 'VALIDATION_SET')
    num_val_samples = create_tfrecords(dataset_dir, n_speakers, val_mix_path, video_dir, tbm_dir, base_audio_dir,
                                       val_dir, delta, norm_data_dir, tfrecord_mode)

    print('Creating test TFRecords...')
    test_mix_path = os.path.join(mix_audio_dir, 'TEST_SET')
    num_test_samples = create_tfrecords(dataset_dir, n_speakers, test_mix_path, video_dir, tbm_dir, base_audio_dir,
                                        test_dir, delta, norm_data_dir, tfrecord_mode)

    print('')
    print('Samples successfully generated:')
    print('-> Training:', num_train_samples)
    print('-> Validation:', num_val_samples)
    print('-> Test:', num_test_samples)
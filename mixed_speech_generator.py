import os
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment


def two_files_audio_sum(file_1_path, file_2_path, file_sum_name, volume_reduction=0):
    s1 = AudioSegment.from_file(file_1_path)
    s2 = AudioSegment.from_file(file_2_path) - volume_reduction # volume_reduction in dB

    s2_shift = (len(s1)-len(s2)) / 2 if len(s1) > len(s2) else 0
    
    audio_sum = s1.overlay(s2, position=s2_shift)
    audio_sum.export(file_sum_name, format='wav')

    return np.array(audio_sum.get_array_of_samples())


def three_files_audio_sum(file_1_path, file_2_path, file_3_path, file_sum_name, volume_reduction=0):
    s1 = AudioSegment.from_file(file_1_path)
    s2 = AudioSegment.from_file(file_2_path) - volume_reduction # volume_reduction in dB
    s3 = AudioSegment.from_file(file_3_path) - volume_reduction

    s2_shift = (len(s1)-len(s2)) / 2 if len(s1) > len(s2) else 0
    s3_shift = (len(s1)-len(s3)) / 2 if len(s1) > len(s3) else 0

    audio_sum = s1.overlay(s2, position=s2_shift)
    audio_sum = audio_sum.overlay(s2, position=s3_shift)
    audio_sum.export(file_sum_name, format='wav')

    return np.array(audio_sum.get_array_of_samples())


def random_files_selector(folders, n_file=1, exclude_files=[]):
    dir_file_list = []
    for dir in folders:
        dir_file_list.append((dir, glob(os.path.join(dir, '*.wav'))))
    
    selected_files = []
    for i in range(n_file):
        if exclude_files:
            condition = False
            while not condition:
                condition = False
                random_folder_files = random.choice(dir_file_list)
                # Avoid repeating same mixing speaker
                s1 = os.path.basename(os.path.dirname(random_folder_files[0].replace('audio', '')))
                s2 = os.path.basename(os.path.dirname(exclude_files[i].replace('audio', '')))
                if s1 != s2:
                    condition = True
        else:
            random_folder_files = random.choice(dir_file_list)
        is_not_append = True
        while is_not_append:
            random_file = random.choice(random_folder_files[1])
            if random_folder_files not in selected_files:
                selected_files.append(random_file)
                is_not_append = False
    return selected_files


def compare_lengths(file_1_path, file_2_path, max_duration_diff=1000):
    # max_duration_diff in milliseconds
    s1 = AudioSegment.from_file(file_1_path)
    s2 = AudioSegment.from_file(file_2_path)
   
    return len(s1) - len(s2) < max_duration_diff


def create_mixed_tracks_speaker(base_audio_dir, noisy_audio_dirs, dest_dir, n_samples=0, n_mix_per_sample=1, n_mix_speakers=1):
    """
    This function generate a mixed audio track for each sample in base_audio_dir.
    """
    # Create destination directory if not exists
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    base_speech_list = glob(os.path.join(base_audio_dir, '*.wav'))
    
    # if num_samples is > 0 we choice num_samples random samples from base_speech_list
    if n_samples > 0:
        random.seed(30)
        random.shuffle(base_speech_list)
        base_speech_list = base_speech_list[:n_samples]
    
    # now, for each speech we have to create n_mix_per_sample combinations:
    for base_speech in base_speech_list:
        condition = False
        # check if noisy utterance is not 2 sec shorter than base utterance, otherwise we have to re-perform the selection.
        while not condition:
            condition = True
            # to create combinations we have to select n_mix_per_samples from noisy speaker
            speechs_to_combine_1 = random_files_selector(noisy_audio_dirs, n_file=n_mix_per_sample)
            for speech_to_combine in speechs_to_combine_1:
                if not compare_lengths(base_speech, speech_to_combine, max_duration_diff=2000):
                    condition = False
                
        if n_mix_speakers == 2:
            condition = False
            while not condition:
                condition = True
                speechs_to_combine_2 = random_files_selector(noisy_audio_dirs, n_file=n_mix_per_sample, exclude_files=speechs_to_combine_1)
                for speech_to_combine in speechs_to_combine_2:
                    if not compare_lengths(base_speech, speech_to_combine, max_duration_diff=2000):
                        condition = False

        if n_mix_speakers == 1:
            for speech_to_combine in speechs_to_combine_1:
                # create a mix
                speaker = os.path.basename(os.path.dirname(os.path.dirname(speech_to_combine)))
                mix_name = os.path.splitext(os.path.basename(base_speech))[0] + '_with_' + speaker + '_' + \
                           os.path.basename(speech_to_combine)
                two_files_audio_sum(base_speech, speech_to_combine, os.path.join(dest_dir, mix_name))
        elif n_mix_speakers == 2:
            for speech_to_combine_1, speech_to_combine_2 in zip(speechs_to_combine_1, speechs_to_combine_2):
                # create a mix
                speaker_1 = os.path.basename(os.path.dirname(os.path.dirname(speech_to_combine_1)))
                speaker_2 = os.path.basename(os.path.dirname(os.path.dirname(speech_to_combine_2)))
                mix_name = os.path.splitext(os.path.basename(base_speech))[0] + '_with_' + \
                                            speaker_1 + '_' + os.path.splitext(os.path.basename(speech_to_combine_1))[0] + '_' + \
                                            speaker_2 + '_' + os.path.basename(speech_to_combine_2)
                three_files_audio_sum(base_speech, speech_to_combine_1, speech_to_combine_2, os.path.join(dest_dir, mix_name))


def create_mixed_tracks_data(dataset_dir, base_speakers, noisy_speakers=[], audio_dir='audio', dest_dir='mixed', n_samples=0, n_mix_per_sample=1, n_mix_speakers=1):
    if not noisy_speakers:
        noisy_speakers = base_speakers

    for s in base_speakers:
        print('Creating mixed tracks of speaker {:d}...'.format(s))
        base_audio_dir = os.path.join(dataset_dir, 's' + str(s), 'audio')
        noisy_speakers_dirs = [os.path.join(dataset_dir, 's' + str(n_s), 'audio') for n_s in noisy_speakers if n_s != s]
        mixed_dest_dir = os.path.join(dataset_dir, dest_dir, 's' + str(s))
        
        create_mixed_tracks_speaker(base_audio_dir, noisy_speakers_dirs, mixed_dest_dir, n_samples, n_mix_per_sample, n_mix_speakers)
        
        print('Speaker {:d} completed.'.format(s))
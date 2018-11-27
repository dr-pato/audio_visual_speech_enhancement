import tensorflow as tf
from os import listdir
from os.path import isfile, join
import random
import numpy as np


class DataManager:
    """Utilities to read TFRecords"""

    def __init__(self, single_audio_frame_size, single_video_frame_size, num_audio_samples=48000, buffer_size=1000, mode='fixed'):
        self.num_audio_samples = num_audio_samples
        self.single_audio_frame_size = single_audio_frame_size
        self.single_video_frame_size = single_video_frame_size
        self.buffer_size = buffer_size
        self.mode = mode


    def get_dataset(self, folder_path):
        file_list = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
        random.shuffle(file_list)
        dataset = tf.data.TFRecordDataset(file_list)

        if self.mode == 'fixed':
            dataset = dataset.map(self.read_data_format_fixed)
        elif self.mode == 'var':
            dataset = dataset.map(self.read_data_format_var)

        n_samples = len(file_list)
        
        return dataset, n_samples


    def get_iterator(self, dataset, batch_size=16, n_epochs=None, train=True):
        if train:
            dataset = dataset.shuffle(self.buffer_size)
        
        dataset = dataset.repeat(n_epochs)
        
        if self.mode == 'fixed':
            batch_dataset = dataset.batch(batch_size)
        elif self.mode == 'var':
            batch_dataset = dataset.padded_batch(batch_size, padded_shapes=([], [None, None], [None, None], [None, None], [None], [None], [None], [None], [None], [None]))
             
        iterator = batch_dataset.make_initializable_iterator()
        
        return batch_dataset, iterator


    def read_data_format_fixed(self, sample):
        context_parsed, sequence_parsed = \
                    tf.parse_single_sequence_example(sample,
                                                     context_features={
                                                      'sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
                                                      'base_audio_wav': tf.FixedLenFeature([self.num_audio_samples], dtype=tf.float32),
                                                      'other_audio_wav': tf.FixedLenFeature([self.num_audio_samples], dtype=tf.float32),
                                                      'mix_audio_wav': tf.FixedLenFeature([self.num_audio_samples], dtype=tf.float32),
                                                      'base_audio_path': tf.VarLenFeature(dtype=tf.string),
                                                      'other_audio_path': tf.VarLenFeature(dtype=tf.string),
                                                      'mix_audio_path': tf.VarLenFeature(dtype=tf.string)
                                                      },
                                                     sequence_features={
                                                      'tbm': tf.FixedLenSequenceFeature([self.single_audio_frame_size], dtype=tf.float32),
                                                      'video_features': tf.FixedLenSequenceFeature([self.single_video_frame_size], dtype=tf.float32),
                                                      'mixed_spectrogram': tf.FixedLenSequenceFeature([self.single_audio_frame_size], dtype=tf.float32)
                                                      })

        #sequence_parsed['mixed_spectrogram'],
        return tf.to_int32(context_parsed['sequence_length']), \
               sequence_parsed['tbm'], sequence_parsed['video_features'], sequence_parsed['mixed_spectrogram'], \
               context_parsed['base_audio_wav'], context_parsed['other_audio_wav'], context_parsed['mix_audio_wav'], \
               context_parsed['base_audio_path'], context_parsed['other_audio_path'], context_parsed['mix_audio_path']
    

    def read_data_format_var(self, sample):
        context_parsed, sequence_parsed = \
                    tf.parse_single_sequence_example(sample,
                                                     context_features={
                                                      'sequence_length': tf.FixedLenFeature([], dtype=tf.int64)
                                                      },
                                                     sequence_features={
                                                      'tbm': tf.FixedLenSequenceFeature([self.single_audio_frame_size], dtype=tf.float32),
                                                      'video_features': tf.FixedLenSequenceFeature([self.single_video_frame_size], dtype=tf.float32),
                                                      'mixed_spectrogram': tf.FixedLenSequenceFeature([self.single_audio_frame_size], dtype=tf.float32),
                                                      'base_audio_wav': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                                                      'other_audio_wav': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                                                      'mix_audio_wav': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                                                      'base_audio_path': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                      'other_audio_path': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                      'mix_audio_path': tf.FixedLenSequenceFeature([], dtype=tf.int64)
                                                     })

        return tf.to_int32(context_parsed['sequence_length']), \
               sequence_parsed['tbm'], sequence_parsed['video_features'], sequence_parsed['mixed_spectrogram'], \
               sequence_parsed['base_audio_wav'], sequence_parsed['other_audio_wav'], sequence_parsed['mix_audio_wav'], \
               sequence_parsed['base_audio_path'], sequence_parsed['other_audio_path'], sequence_parsed['mix_audio_path']
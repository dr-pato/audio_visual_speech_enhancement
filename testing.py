from __future__ import division

import os
import numpy as np
import itertools
from scipy.io import wavfile
import tensorflow as tf
from dataset_reader import DataManager
from eval_metrics import snr_batch_eval, sdr_batch_eval, l2_batch_eval

# Avoid printing tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF constants definition
TF_INTRA_OP_PT = int(os.getenv('TF_INTRA_OP', 0))
TF_INTER_OP_PT = int(os.getenv('TF_INTER_OP', 0))


def test(data_path, data_path_test, exp_num, ckp_step, video_feat_dim, audio_feat_dim, tfrecord_mode, num_audio_samples,
         mask_threshold=-1, mix_eval=True, save_rec_dir='', save_out_dir=''):
    data_path_test = os.path.join(data_path, data_path_test)

    checkpoints_dir = os.path.join(data_path, 'logs', 'checkpoints', 'exp_' + exp_num)
    graph_checkpoint = os.path.join(checkpoints_dir, 'model_epoch_' + str(ckp_step) + '.ckpt.meta')
    params_checkpoint = os.path.join(checkpoints_dir, 'model_epoch_' + str(ckp_step) + '.ckpt')

    print('Loading model:')
    saver = tf.train.import_meta_graph(graph_checkpoint)
    print('done.\n')
    graph = tf.get_default_graph()
    
    # Create the DataManager that reads TFRecords.
    with tf.name_scope('test_batch'):
        test_data_manager = DataManager(audio_feat_dim, video_feat_dim, num_audio_samples, mode=tfrecord_mode)
        test_dataset, num_examples_test = test_data_manager.get_dataset(data_path_test)
        test_dataset, test_it = test_data_manager.get_iterator(test_dataset, batch_size=256,
                                                               n_epochs=1, train=False)
        next_test_batch = test_it.get_next()
    
    # Placeholders
    inputs = graph.get_tensor_by_name('placeholder/input_video:0')
    sequence_lengths = graph.get_tensor_by_name('placeholder/sequence_lengths:0')
    targets = graph.get_tensor_by_name('placeholder/tbm:0')
    input_mixed_specs = graph.get_tensor_by_name('placeholder/input_mixed_specs:0')
    mixed_sources = graph.get_tensor_by_name('placeholder/mixed_sources:0')
    target_sources = graph.get_tensor_by_name('placeholder/target_sources:0')
    keep_prob = graph.get_tensor_by_name('placeholder/keep_prob:0')

    # The inizializer operation.
    init_op = tf.group(test_it.initializer)
        
    # Start session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                          inter_op_parallelism_threads=TF_INTER_OP_PT)) as sess:
        sess.run(init_op)

        # Load model weigths
        print('Restore weigths:')
        saver.restore(sess, params_checkpoint)
        print('done.\n')

        # Get enhanced sources tensor op
        enhanced_sources_tensor = graph.get_tensor_by_name('model/enhanced_sources:0')

        # Get output tensor op
        output_tensor = graph.get_tensor_by_name('model/prediction:0')
        
        # Get loss tensor op
        loss_tensor = graph.get_tensor_by_name('model/loss:0')

        try:
            test_cost = 0
            test_sequence_lengths_list = []
            test_output_list = []
            test_base_audio_list = []
            test_other_audio_list = []
            test_mixed_audio_list = []
            test_enhanced_audio_list = []
            test_base_paths_list = []
            test_mixed_paths_list = []
            
            print('Starting dataset inference...')
            while True:
                # Fetch test samples batch.
                test_length, test_tbm, test_video_feature, test_mixed_specs, test_base_audio, test_other_audio, \
                   test_mixed_audio, test_base_paths, test_other_paths, test_mixed_paths = sess.run(next_test_batch)
                
                # Mask thresholding (useful if you are using estimated TBMs)
                if mask_threshold != -1:
                    test_tbm = (test_tbm > mask_threshold).astype(np.float32)

                # Compute validation loss and enhanced sources
                cost, test_output, test_enhanced_audio = sess.run(fetches=[loss_tensor, output_tensor, enhanced_sources_tensor],
                                                                  feed_dict={
                                                                      inputs: test_video_feature,
                                                                      sequence_lengths: test_length,
                                                                      targets: test_tbm,
                                                                      input_mixed_specs: test_mixed_specs,
                                                                      mixed_sources: test_mixed_audio,
                                                                      target_sources: test_base_audio,
                                                                      keep_prob: 1.0})
                
                test_cost += cost
                test_sequence_lengths_list.append(test_length)
                test_output_list.append(test_output)
                test_base_audio_list.append(test_base_audio)
                test_other_audio_list.append(test_other_audio)
                test_mixed_audio_list.append(test_mixed_audio)
                test_enhanced_audio_list.append(test_enhanced_audio)
                test_base_paths_list.append(test_base_paths.values if tfrecord_mode == 'fixed' else test_base_paths)
                test_mixed_paths_list.append(test_mixed_paths.values if tfrecord_mode == 'fixed' else test_mixed_paths)
        except tf.errors.OutOfRangeError:
            print('End dataset inference.')
            test_sequence_lengths_list = np.concatenate(test_sequence_lengths_list)
            test_output_list  = list(itertools.chain(*test_output_list)) 
            test_base_audio_list  = list(itertools.chain(*test_base_audio_list)) 
            test_other_audio_list = list(itertools.chain(*test_other_audio_list))
            test_mixed_audio_list = list(itertools.chain(*test_mixed_audio_list))
            test_enhanced_audio_list = list(itertools.chain(*test_enhanced_audio_list))
            test_base_paths_list = np.concatenate(test_base_paths_list) if tfrecord_mode == 'fixed' else list(itertools.chain(*test_base_paths_list))
            test_mixed_paths_list = np.concatenate(test_mixed_paths_list) if tfrecord_mode == 'fixed' else list(itertools.chain(*test_mixed_paths_list))
            
        # Evaluate enhanced and mixed sources
        print('')
        print('Computing dataset evaluation metrics...')
        snr_enhanced = snr_batch_eval(test_base_audio_list, test_enhanced_audio_list, sequence_lengths=test_sequence_lengths_list)
        sdr_enhanced, sir_enhanced, sar_enhanced = sdr_batch_eval(test_base_audio_list, test_other_audio_list, test_enhanced_audio_list, sequence_lengths=test_sequence_lengths_list)
        l2_enhanced = l2_batch_eval(test_base_audio_list, test_enhanced_audio_list, sequence_lengths=test_sequence_lengths_list)

        if mix_eval:
            snr_mixed = snr_batch_eval(test_base_audio_list, test_mixed_audio_list, sequence_lengths=test_sequence_lengths_list)
            sdr_mixed, sir_mixed, sar_mixed = sdr_batch_eval(test_base_audio_list, test_other_audio_list, test_mixed_audio_list, sequence_lengths=test_sequence_lengths_list)
            l2_mixed = l2_batch_eval(test_base_audio_list, test_mixed_audio_list, sequence_lengths=test_sequence_lengths_list)
        
        print('done.')

        # Print results
        print('')
        print('Experiment number: {:s}'.format(exp_num))
        print('Epoch (step) checkpoint: {:s} ({:s})'.format(*ckp_step.split('_')))
        print('Test set evaluation: {:s}'.format(data_path_test))
        print('Size: {:d}'.format(len(test_base_audio_list)))
        print('Loss per sample: {:.6f}'.format(test_cost / num_examples_test))
        print('')
        print('Enhanced SDR: {:.5f} [{:.5f}]'.format(sdr_enhanced.mean(), sdr_enhanced.std()))
        print('Enhanced SIR: {:.5f} [{:.5f}]'.format(sir_enhanced.mean(), sir_enhanced.std()))
        print('Enhanced SAR: {:.5f} [{:.5f}]'.format(sar_enhanced.mean(), sar_enhanced.std()))
        print('Enhanced L2 (spectrogram): {:.5f} [{:.5f}]'.format(l2_enhanced.mean(), l2_enhanced.std()))
        print('Enhanced SNR: {:.5f} [{:.5f}]'.format(snr_enhanced.mean(), snr_enhanced.std()))
        if mix_eval:
            print('')
            print('Mixed SDR: {:.5f} [{:.5f}]'.format(sdr_mixed.mean(), sdr_mixed.std()))
            print('Mixed SIR: {:.5f} [{:.5f}]'.format(sir_mixed.mean(), sir_mixed.std()))
            print('Mixed SAR: {:.5f} [{:.5f}]'.format(sar_mixed.mean(), sar_mixed.std()))
            print('Mixed L2 (spectrogram): {:.5f} [{:.5f}]'.format(l2_mixed.mean(), l2_mixed.std()))
            print('Mixed SNR: {:.5f} [{:.5f}]'.format(snr_mixed.mean(), snr_mixed.std()))

        # Save audio samples: target, mixed and enhanced samples are saved in three
        # subdirectories of "save_samples_dir".
        if save_rec_dir:
            print('\nSaving audio samples...')
            target_dir = os.path.join(data_path, save_rec_dir, 'target')
            mixed_dir = os.path.join(data_path, save_rec_dir, 'mixed')
            enhanced_dir = os.path.join(data_path, save_rec_dir, 'enhanced')

            if not os.path.exists(save_rec_dir):
                os.makedirs(save_rec_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if not os.path.exists(mixed_dir):
                os.makedirs(mixed_dir)
            if not os.path.exists(enhanced_dir):
                os.makedirs(enhanced_dir)

            for base, mixed, enhanced, filename, seq_len in zip(test_base_audio_list, test_mixed_audio_list, test_enhanced_audio_list, test_mixed_paths_list, test_sequence_lengths_list):
                if tfrecord_mode == 'fixed':
                    filename = filename.decode()
                else:
                    filename = ''.join([chr(x) for x in np.trim_zeros(filename)])
                speaker_dir, filename = os.path.split(filename)
                num_wav_samples = seq_len * 160
                filename = os.path.basename(speaker_dir) + '_' + filename
                wavfile.write(os.path.join(target_dir, filename), 16000, base[:num_wav_samples].astype(np.int16))
                wavfile.write(os.path.join(mixed_dir, filename), 16000, mixed[:num_wav_samples].astype(np.int16))
                wavfile.write(os.path.join(enhanced_dir, filename), 16000, enhanced[:num_wav_samples].astype(np.int16))

            print('done.')


        # Save outputs: each element is saved in dir "s(n_speaker)/(output_dir)" where
        # (n_speaker) is the id of speaker to whom base belongs and (output_dir)
        # is the value in variable "save_out_dir".
        if save_out_dir:
            print('\nSaving outputs...')
            for output, filename, seq_len in zip(test_output_list, test_base_paths_list, test_sequence_lengths_list):
                if tfrecord_mode == 'fixed':
                    filename = filename.decode()
                else:
                    filename = ''.join([chr(x) for x in np.trim_zeros(filename)])
                dirname, basename = os.path.split(filename)
                basename = os.path.splitext(basename)[0] + '.npy'
                dirname = os.path.basename(dirname.replace('/audio', ''))
                dirname = os.path.join(data_path, dirname, save_out_dir)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
           
                np.save(os.path.join(dirname, basename), output[:seq_len])
                
            print('done.')


if __name__ == '__main__':
    data_path = os.getenv('DATA_PATH', 'C:/Users/Giovanni Morrone/Documents/Dottorato di Ricerca/Speech Recognition/datasets/GRID')
    data_path_test = os.getenv('TEST_DIR', 'TFRecords/grid_test_2spk/TRAINING_SET')

    exp_num = os.getenv('EXP_NUM', 'VL2M_TEST')
    video_feat_dim = int(os.getenv('VIDEO_FEAT_DIM', 136))
    audio_feat_dim = int(os.getenv('AUDIO_FEAT_DIM', 257))
    mask_threshold = float(os.getenv('MASK_THRESHOLD', -1)) # -1: no thresholding
    tfrecord_mode = os.getenv('TFRECORD_MODE', 'fixed')
    num_audio_samples = int(os.getenv('NUM_AUDIO_SAMPLES', 48000)) if tfrecord_mode == 'fixed' else 0 # If "tfrecord_mode" is "var" this value is ignored.
    max_batch_size = int(os.getenv('BATCH_SIZE', 10))
    
    restored_epoch = os.getenv('EPOCH', '0_7')
    mix_eval = int(os.getenv('MIX_EVAL', 1)) # Evaluation of mixed samples. 0: no - 1: yes.
    save_rec_dir = os.getenv('SAVE_SAMPLES_DIR', 'samples_si/test_grid') # Save target, mixed, enhanced samples
    save_out_dir = os.getenv('SAVE_OUTPUT_DIR', 'mask_test') # Save estimated masks (spectrograms for "av_concat_spec" model)

    eval(data_path, data_path_test, exp_num, restored_epoch, video_feat_dim, audio_feat_dim, tfrecord_mode, num_audio_samples, mask_threshold,mix_eval, save_rec_dir, save_out_dir)
    
    
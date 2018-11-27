from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf
from time import time

from dataset_reader import DataManager
from enhancement_model import VL2M, VL2MRef, AudioVisualConcatMask, AudioVisualConcatSpec
from eval_metrics import snr_batch_eval, sdr_batch_eval, l2_batch_eval

# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF constants definition
TF_INTRA_OP_PT = int(os.getenv('TF_INTRA_OP', 0))
TF_INTER_OP_PT = int(os.getenv('TF_INTER_OP', 0))


class Configuration(object):
    def __init__(self, lr, us, ld, kp, bs, opt, video_dim, audio_dim, a_samples, n_epochs, n_hidden, n_layer, reg, mask_thresh):
        self.learning_rate = lr
        self.updating_step = us
        self.learning_decay = ld
        self.keep_prob = kp
        self.batch_size = bs 
        self.optimizer_choice = opt
        self.num_epochs = n_epochs

        self.video_feat_dim = video_dim
        self.audio_feat_dim = audio_dim
        self.num_audio_samples = a_samples

        self.n_hidden = n_hidden
        self.num_layers = n_layer
        self.reg = reg
        self.mask_thresh = mask_thresh


def train(model_selection, data_path, data_path_train, data_path_val, config, exp_num='0', tfrecord_mode='fixed'):
    """
    Train the audio-visual speech enhancement model.
    """
    data_path_train = os.path.join(data_path, data_path_train)
    data_path_val = os.path.join(data_path, data_path_val)

    checkpoints_dir = os.path.join(data_path, 'logs', 'checkpoints', 'exp_' + exp_num)
    tensorboard_dir = os.path.join(data_path, 'logs', 'tensorboard', 'exp_' + exp_num)
    traininglog_dir = os.path.join(data_path, 'logs', 'training_logs')

    # Training Graph
    with tf.Graph().as_default():

        # Create the DataManager that reads TFRecords.
        with tf.name_scope('train_batch'):
            train_data_manager = DataManager(config.audio_feat_dim, config.video_feat_dim, config.num_audio_samples, buffer_size=4000, mode=tfrecord_mode)
            train_dataset, num_examples_train = train_data_manager.get_dataset(data_path_train)
            train_dataset, train_it = train_data_manager.get_iterator(train_dataset, batch_size=config.batch_size,
                                                                      n_epochs=config.num_epochs, train=True)
            next_train_batch = train_it.get_next()

        with tf.name_scope('validation_batch'):
            val_data_manager = DataManager(config.audio_feat_dim, config.video_feat_dim, config.num_audio_samples, mode=tfrecord_mode)
            val_dataset, num_examples_val = val_data_manager.get_dataset(data_path_val)
            val_dataset, val_it = val_data_manager.get_iterator(val_dataset, batch_size=num_examples_val,
                                                                n_epochs=1, train=False)
            next_val_batch = val_it.get_next()
    
        # Placeholders.
        with tf.name_scope('placeholder'):
            input_video = tf.placeholder(tf.float64, shape=[None, None, config.video_feat_dim], name='input_video')
            sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
            tbm = tf.placeholder(tf.float64, shape=[None, None, config.audio_feat_dim], name='tbm')
            input_mixed_specs = tf.placeholder(tf.float64, shape=[None, None, config.audio_feat_dim], name='input_mixed_specs')
            mixed_sources = tf.placeholder(tf.float32, shape=[None, None], name='mixed_sources')
            target_sources = tf.placeholder(tf.float32, shape=[None, None], name='target_sources')
            keep_prob = tf.placeholder(tf.float64, name='keep_prob')
        
        # Graph building and definition.
        print('Building model:') 
        with tf.variable_scope('model'):
            if model_selection == 'vl2m':
                model = VL2M(input_video, sequence_lengths, tbm, mixed_sources, target_sources, keep_prob, config)
            elif model_selection == 'vl2m_ref':
                model = VL2MRef(input_video, sequence_lengths, tbm, mixed_sources, target_sources, input_mixed_specs, keep_prob, config)
            elif model_selection == 'av_concat_mask':
                model = AudioVisualConcatMask(input_video, sequence_lengths, tbm, mixed_sources, target_sources, input_mixed_specs, keep_prob, config)
            elif model_selection == 'av_concat_spec':
                model = AudioVisualConcatSpec(input_video, sequence_lengths, tbm, mixed_sources, target_sources, input_mixed_specs, keep_prob, config)
            else:
                print('Model selection must be "vl2m", "vl2m_ref", "av_concat_mask" or "av_concat_spec". Closing...')
                sys.exit()
        
            model.create_graph()
        print('done.')

        # The inizializer operation.
        init_op = tf.group(train_it.initializer, val_it.initializer, tf.global_variables_initializer())
        # save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=10)

        # create log directories if not exist.
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        if not os.path.exists(traininglog_dir):
            os.makedirs(traininglog_dir)

        trainingLogFile = open(os.path.join(traininglog_dir, 'TrainingExperiment_' + str(exp_num) + '.txt'), 'a')

        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT)) as sess:

            # FileWriter to save TensorBoard summary
            tb_writer = tf.summary.FileWriter(tensorboard_dir, graph=sess.graph)

            # Initialize the variables and the batch iterator.
            sess.run(init_op)
        
            n_steps_epoch = int(num_examples_train / config.batch_size)
            n_steps = int(config.num_epochs * n_steps_epoch)
             
            # Fetch validation samples batch.
            val_length, val_tbm, val_video_feature, val_mixed_specs, val_base_audio, val_other_audio, \
               val_mixed_audio, val_base_paths, val_other_paths, val_mixed_paths = sess.run(next_val_batch)
            
            # restore variables
            last_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
            if last_checkpoint:
                last_checkpoint_path = saver.restore(sess, last_checkpoint)
                print('\nVariables restored. Global step: {:d}'.format(sess.run(model.global_step)))
            else:
                trainingLogFile.write('+-- EXPERIMENT NUMBER - {:s} --+\n'.format(exp_num))
                trainingLogFile.write('## optimizer: {:s}\n'.format(config.optimizer_choice))
                trainingLogFile.write('## number of hidden layers (other): {:d}\n'.format(config.num_layers))
                trainingLogFile.write('## number of hidden units: {:d}\n'.format(config.n_hidden))
                trainingLogFile.write('## initial learning rate: {:.6f}\n'.format(config.learning_rate))
                if config.optimizer_choice != 'adam':
                    trainingLogFile.write('## learning rate update steps: {:d}\n'.format(config.updating_step))
                    trainingLogFile.write('## learning rate decay: {:.6f}\n'.format(config.learning_decay))
                trainingLogFile.write('## regularization: {:.6f}\n'.format(config.reg))
                trainingLogFile.write('## dropout keep probability (no dropout if 1): {:.6f}\n'.format(config.keep_prob))
                trainingLogFile.write('## training size: {:d}\n'.format(num_examples_train))
                trainingLogFile.write('## validation size: {:d}\n'.format(num_examples_val))
                trainingLogFile.write('## batch size: {:d}\n'.format(config.batch_size))
                trainingLogFile.write('## approx number of steps: {:d}\n'.format(n_steps))
                trainingLogFile.write('## approx number of steps per epoch: {:d}\n'.format(n_steps_epoch))
                trainingLogFile.write('\nEpoch\tLR\tTrain[Cost|L2-SPEC|SNR|SDR|SIR|SAR]\tVal[Cost|L2-SPEC|SNR|SDR|SIR|SAR]\n')
            
            # Print training details.
            print('')
            print('+--- EXPERIMENT NUMBER - {:s} ---+'.format(exp_num))
            print('## optimizer: {:s}'.format(config.optimizer_choice))
            print('## number of hidden layers (other): {:d}'.format(config.num_layers))
            print('## number of hidden units: {:d}'.format(config.n_hidden))
            print('## initial learning rate: {:.6f}'.format(config.learning_rate))
            if config.optimizer_choice != 'adam':
                print('## learning rate update steps: {:d}'.format(config.updating_step))
                print('## learning rate decay: {:.6f}'.format(config.learning_decay))
            print('## regularization: {:.6f}'.format(config.reg))
            print('## dropout keep probability (no dropout if 1): {:.6f}'.format(config.keep_prob))
            print('## training size: {:d}'.format(num_examples_train))
            print('## validation size: {:d}'.format(num_examples_val))
            print('## batch size: {:d}'.format(config.batch_size))
            print('## approx number of steps: {:d}'.format(n_steps))
            print('## approx number of steps per epoch: {:d}'.format(n_steps_epoch))
            print('')


            try:
                step = sess.run(model.global_step)
                epoch_counter = int(step / n_steps_epoch)
                epoch_cost = 0
                epoch_snr = []
                epoch_sdr = []
                epoch_sar = []
                epoch_sir = []
                epoch_l2_spec = []

                epoch_start_time = time()
                
                while True:
                    step += 1
                    if (step - 1) % n_steps_epoch == 0:
                        print('-> Epoch {:d}'.format(epoch_counter))
                    
                    # Fetch training batch.
                    length_batch, tbm_batch, video_feature_batch, mixed_specs_batch, base_audio_batch, other_audio_batch, \
                        mixed_audio_batch, base_paths_batch, other_paths_batch, mixed_paths_batch = sess.run(next_train_batch)
                    
                    _, cost, lr = sess.run(fetches=[model.train_op, model.loss, model.learning_rate],
                                           feed_dict={
                                            input_video: video_feature_batch,
                                            sequence_lengths: length_batch,
                                            tbm: tbm_batch,
                                            input_mixed_specs: mixed_specs_batch,
                                            mixed_sources: mixed_audio_batch,
                                            target_sources: base_audio_batch,
                                            keep_prob: config.keep_prob})
                    
                    # Enhanced sources w/o dropout
                    enhanced_audio_batch = sess.run(fetches=model.enhanced_sources,
                                                  feed_dict={
                                                    input_video: video_feature_batch,
                                                    sequence_lengths: length_batch,
                                                    tbm: tbm_batch,
                                                    input_mixed_specs: mixed_specs_batch,
                                                    mixed_sources: mixed_audio_batch,
                                                    target_sources: base_audio_batch,
                                                    keep_prob: 1.0})
                  
                    if np.isnan(cost):
                        print('GOT INSTABILITY: cost is NaN. Leaving...')
                        sys.exit()
                   
                    if np.isinf(cost):
                        print('GOT INSTABILITY: cost is inf. Leaving...')
                        sys.exit()

                    snr_batch = snr_batch_eval(base_audio_batch, enhanced_audio_batch, sequence_lengths=length_batch)
                    sdr_batch, sir_batch, sar_batch = sdr_batch_eval(base_audio_batch, other_audio_batch, enhanced_audio_batch, sequence_lengths=length_batch)
                    l2_spec_batch = l2_batch_eval(base_audio_batch, enhanced_audio_batch, sequence_lengths=length_batch)

                    epoch_cost += cost
                    epoch_snr = np.append(epoch_snr, snr_batch)
                    epoch_sdr = np.append(epoch_sdr, sdr_batch)
                    epoch_sir = np.append(epoch_sir, sir_batch)
                    epoch_sar = np.append(epoch_sar, sar_batch)
                    epoch_l2_spec = np.append(epoch_l2_spec, l2_spec_batch)
                    
                    #Print loss value fairly often.
                    if step % (n_steps_epoch // 16) == 0 or step == 1:
                        print('Step[{:7d}] Cost[{:3.5f}] LR[{:.6f}] L2-SPEC[{:3.5f}] SNR[{:3.5f}] SDR[{:3.5}] SIR[{:3.5}] SAR[{:3.5}]' \
                            .format(step, cost / config.batch_size, lr, l2_spec_batch.mean(), snr_batch.mean(), sdr_batch.mean(), sir_batch.mean(), sar_batch.mean()))
                    
                    if step % 500 == 0 or step % n_steps_epoch == 0:
                        # Save model variables.
                        save_path = saver.save(sess, os.path.join(checkpoints_dir, 'model_epoch_' + str(epoch_counter)
                                                                  + '_' + str(step) + '.ckpt'))
                        print('Model saved.')
                    
                    # Print and write reports at the end of each epoch.
                    if (step % n_steps_epoch == 0) and (step != 0):
                        print('Completed epoch {:d} at step {:d} --> Train cost: {:.6f} - L2-SPEC: {:.6f} - SNR: {:.6f} - SDR: {:.6f} - SIR: {:.6f} - SAR: {:.6f}' \
                              .format(epoch_counter, step, epoch_cost / num_examples_train, epoch_l2_spec.mean(),
                                      epoch_snr.mean(), epoch_sdr.mean(), epoch_sir.mean(), epoch_sar.mean()))
                        print('Epoch training time (seconds) = {:.6f}'.format(time() - epoch_start_time))
                        
                        # Compute validation loss and summaries.
                        val_cost, val_enhanced_audio, summaries = sess.run(fetches=[model.loss, model.enhanced_sources, model.summaries],
                                                                           feed_dict={
                                                                               input_video: val_video_feature,
                                                                               sequence_lengths: val_length,
                                                                               tbm: val_tbm,
                                                                               input_mixed_specs: val_mixed_specs,
                                                                               mixed_sources: val_mixed_audio,
                                                                               target_sources: val_base_audio,
                                                                               keep_prob: 1.0})

                        val_snr = snr_batch_eval(val_base_audio, val_enhanced_audio, sequence_lengths=val_length)
                        val_sdr, val_sir, val_sar = sdr_batch_eval(val_base_audio, val_other_audio, val_enhanced_audio, sequence_lengths=val_length)
                        val_l2_spec = l2_batch_eval(val_base_audio, val_enhanced_audio, sequence_lengths=val_length)

                        print('Validation cost: {:.6f} - L2-SPEC: {:.6f} - SNR: {:.6f} - SDR: {:.6f} - SIR: {:.6f} - SAR: {:.6f}' \
                              .format(val_cost / num_examples_val, val_l2_spec.mean(), val_snr.mean(), val_sdr.mean(), val_sir.mean(), val_sar.mean()))
                        
                        # Write Tensorboard summaries.
                        tb_summary = tf.Summary()
                        tb_summary.value.add(tag='Training loss', simple_value=epoch_cost / num_examples_train)
                        tb_summary.value.add(tag='Training L2 spectrogram', simple_value=epoch_l2_spec.mean())
                        tb_summary.value.add(tag='Training SNR', simple_value=epoch_snr.mean())
                        tb_summary.value.add(tag='Training SDR', simple_value=epoch_sdr.mean())
                        tb_summary.value.add(tag='Training SIR', simple_value=epoch_sdr.mean())
                        tb_summary.value.add(tag='Training SAR', simple_value=epoch_sdr.mean())
                        tb_summary.value.add(tag='Validation loss', simple_value=val_cost / num_examples_val)
                        tb_summary.value.add(tag='Validation L2 spectrogram', simple_value=val_l2_spec.mean())
                        tb_summary.value.add(tag='Validation SNR', simple_value=val_snr.mean())
                        tb_summary.value.add(tag='Validation SDR', simple_value=val_sdr.mean())
                        tb_summary.value.add(tag='Validation SIR', simple_value=val_sir.mean())
                        tb_summary.value.add(tag='Validation SAR', simple_value=val_sar.mean())
                        
                        tb_writer.add_summary(tb_summary, epoch_counter)

                        tb_writer.add_summary(summaries, epoch_counter)
                        tb_writer.flush()
                        
                        # Write trainining log file.
                        trainingLogFile.write('{:d}\t[{:.6f}][{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}]\t[{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}]\n' \
                                              .format(epoch_counter, lr, epoch_cost / num_examples_train, epoch_l2_spec.mean(), epoch_snr.mean(), epoch_sdr.mean(), epoch_sir.mean(), epoch_sar.mean(),
                                                      val_cost / num_examples_val, val_l2_spec.mean(), val_snr.mean(), val_sdr.mean(), val_sir.mean(), val_sar.mean()))
                       
                        trainingLogFile.flush()

                        print('')
                        epoch_counter += 1
                        epoch_cost = 0
                        epoch_start_time = time()
                 
            except tf.errors.OutOfRangeError:
               print('Done training for %d epochs, %d steps: validation cost = %.5f' % (config.num_epochs, step, val_cost / num_examples_val))
import tensorflow as tf, math
from audio_processing import compute_stft, get_oracle_iam, get_sources

class VL2M(object):
    """
    VL2M
    Input: video features (face landmarks).
    Model: stacked BLSTM.
    Output: Target Binary Mask (TBM).
    Loss: binary cross entropy
    """

    def __init__(self, input_video, sequence_lengths, tbm, mixed_sources, target_sources, keep_prob, config):
        self.input_video = input_video
        self.sequence_lengths = sequence_lengths
        self.tbm = tbm
        self.audio_frame_size = config.audio_feat_dim
        self.video_frame_size = config.video_feat_dim
        self.num_audio_samples = config.num_audio_samples
        self.mixed_sources = mixed_sources
        self.target_sources = target_sources
        self.mixed_specs = compute_stft(mixed_sources, out_shape=tf.shape(tbm))
        self.num_units = config.n_hidden
        self.num_layers = config.num_layers
        self.optimizer_choice = config.optimizer_choice
        self.initial_learning_rate = config.learning_rate
        self.learning_rate = config.learning_rate
        self.updating_step = config.updating_step
        self.learning_decay = config.learning_decay
        self.keep_prob = keep_prob
        self.regularization = config.reg
        self.mask_threshold = config.mask_thresh
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._summaries = None
        self._global_step = None

    def create_graph(self):
        """
        Create the Graph of the model.
        """
        self.inference
        self.loss
        self.train_op
        self.prediction
        self.enhanced_sources
        self.summaries

    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            with tf.variable_scope('forward'):
                forward_cells = []
                for i in range(self.num_layers):
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    forward_cells.append(lstm_cell)

            with tf.variable_scope('backward'):
                backward_cells = []
                for i in range(self.num_layers):
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    backward_cells.append(lstm_cell)

            with tf.variable_scope('Bi_LSTM'):
                rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=forward_cells,
                  cells_bw=backward_cells,
                  inputs=self.input_video,
                  dtype=tf.float64,
                  sequence_length=self.sequence_lengths)
            rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.num_units * 2]), keep_prob=self.keep_prob)
            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.num_units * 2, self.audio_frame_size], stddev=1.0 / math.sqrt(float(self.num_units * 2)),
                  dtype=tf.float64),
                  name='weights')
                biases = tf.Variable(tf.zeros([self.audio_frame_size], dtype=tf.float64), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [-1, max_sequence_length, self.audio_frame_size], name='inference')
        return self._inference

    @property
    def prediction(self):
        if self._prediction is None:
            self._prediction = tf.sigmoid(self.inference, name='prediction')
        return self._prediction

    @property
    def loss(self):
        """
        Calculate the loss from logits and the labels.
        """
        if self._loss is None:
            tbm_cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tbm, logits=self.inference)) * (tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float64), axis=2))
            func_loss = tf.reduce_sum(tbm_cross_entropy, name='binary_mask_loss')
            if self.regularization:
                l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='reg_loss')
            else:
                l2_loss = 0
            self._loss = tf.add(func_loss, self.regularization * l2_loss, name='loss')
        return self._loss

    @property
    def train_op(self):
        if self._train_op is None:
            with tf.name_scope('optimizer'):
                if self.optimizer_choice == 'adam':
                    self.learning_rate = tf.constant(self.initial_learning_rate)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate)
                else:
                    self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                                    self.updating_step, self.learning_decay, staircase=True)
                    if self.optimizer_choice == 'sgd':
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.optimizer_choice == 'momentum':
                            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)  
                    else:
                        print('Optimizer must be either sgd, momentum or adam. Closing...')
                        sys.exit(1)
                
                self._train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')
        return self._train_op

    @property
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            if self.mask_threshold == -1:
                masks = self.prediction
            else:
                masks = self.prediction > self.mask_threshold
            masked_mag_specs = tf.abs(self.mixed_specs) * (tf.cast(masks, dtype=tf.float32))
            self._enhanced_sources = get_sources(masked_mag_specs, tf.angle(self.mixed_specs), num_samples=self.num_audio_samples)
        return tf.identity(self._enhanced_sources, name='enhanced_sources')

    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                oracle_tbm = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.tbm, [0, 2, 1]), axis=3))
                estimated_tbm = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                estimated_tbm_round = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.round(self.prediction), [0, 2, 1]), axis=3))
                tf.summary.image('Oracle TBM', oracle_tbm, max_outputs=n_samples)
                tf.summary.image('Estimated TBM', estimated_tbm, max_outputs=n_samples)
                tf.summary.image('Estimated TBM (rounded)', estimated_tbm_round, max_outputs=n_samples)
                mixed_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.mixed_sources), axis=1), (-1,
                                                                                                   1))
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1,
                                                                                                     1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1,
                                                                                                         1))
                tf.summary.audio('Mixed audio', self.mixed_sources / mixed_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Target audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Masked audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

    @property
    def global_step(self):
        if self._global_step is None:
            self._global_step = tf.Variable(0, trainable=False)
        return self._global_step

    @property
    def weights(self):
        if self._weights is None:
            self._weights = tf.trainable_variables()
        return self._weights


class VL2MRef(object):
    """
    VL2M_ref
    Input: audio features and TBM masks.
    Model: two input BLSTM (audio and TBM) + FC layer (AV fusion) + BLSTM.
    Output: Ideal Amplitude Mask (IAM).
    Loss: L2 loss (target_spec - enhanced_spec)
    """

    def __init__(self, input_video, sequence_lengths, tbm, mixed_sources, target_sources, input_mixed_specs, keep_prob, config):
        self.input_video = input_video
        self.sequence_lengths = sequence_lengths
        self.tbm = tbm
        self.audio_frame_size = config.audio_feat_dim
        self.video_frame_size = config.video_feat_dim
        self.num_audio_samples = config.num_audio_samples
        self.mixed_sources = mixed_sources
        self.target_sources = target_sources
        self.mixed_specs = compute_stft(mixed_sources, out_shape=tf.shape(input_mixed_specs))
        self.target_specs = compute_stft(target_sources, out_shape=tf.shape(input_mixed_specs))
        self.input_mixed_specs = input_mixed_specs
        self.num_units = config.n_hidden
        self.num_layers = config.num_layers
        self.optimizer_choice = config.optimizer_choice
        self.initial_learning_rate = config.learning_rate
        self.learning_rate = config.learning_rate
        self.updating_step = config.updating_step
        self.learning_decay = config.learning_decay
        self.keep_prob = keep_prob
        self.regularization = config.reg
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._oracle_iam = None
        self._summaries = None
        self._global_step = None
        self._weights = None
        self.func_loss = None
        self.reg_loss = None

    def create_graph(self):
        """
        Create the Graph of the model.
        """
        self.inference
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.oracle_iam
        self.summaries

    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            with tf.variable_scope('input_lstm'):
                with tf.variable_scope('mask'):
                    mask_lstm_outputs = self._VL2MRef__bi_lstm_block(self.tbm, self.num_layers, self.num_units)
                with tf.variable_scope('mixed'):
                    mix_lstm_outputs = self._VL2MRef__bi_lstm_block(self.input_mixed_specs, self.num_layers, self.num_units)
            with tf.variable_scope('input_lstm_projection'):
                mask_lstm_outputs_res = tf.nn.dropout(tf.reshape(mask_lstm_outputs, [-1, self.num_units * 2]), keep_prob=self.keep_prob)
                mix_lstm_outputs_res = tf.nn.dropout(tf.reshape(mix_lstm_outputs, [-1, self.num_units * 2]), keep_prob=self.keep_prob)
                weights_mask = tf.Variable(tf.truncated_normal([self.num_units * 2, self.num_units], stddev=1.0 / math.sqrt(float(self.num_units * 2)),
                  dtype=tf.float64),
                  name='weights_mask')
                weights_mix = tf.Variable(tf.truncated_normal([self.num_units * 2, self.num_units], stddev=1.0 / math.sqrt(float(self.num_units * 2)),
                  dtype=tf.float64),
                  name='weights_mix')
                biases_projection = tf.Variable(tf.zeros([self.num_units], dtype=tf.float64), name='biases')
                input_projection_res = tf.matmul(mask_lstm_outputs_res, weights_mask) + tf.matmul(mix_lstm_outputs_res, weights_mix) + biases_projection
                input_projection = tf.reshape(input_projection_res, [-1, max_sequence_length, self.num_units])
            with tf.variable_scope('mask_lstm'):
                lstm_outputs = self._VL2MRef__bi_lstm_block(input_projection, self.num_layers, self.num_units)
            with tf.variable_scope('logits'):
                lstm_outputs_res = tf.nn.dropout(tf.reshape(lstm_outputs, [-1, self.num_units * 2]), keep_prob=self.keep_prob)
                weights_out = tf.Variable(tf.truncated_normal([self.num_units * 2, self.audio_frame_size], stddev=1.0 / math.sqrt(float(self.num_units * 2)),
                  dtype=tf.float64),
                  name='weights')
                biases_out = tf.Variable(tf.zeros([self.audio_frame_size], dtype=tf.float64), name='biases')
                logits = tf.matmul(lstm_outputs_res, weights_out) + biases_out
            logits_res = tf.reshape(logits, [-1, max_sequence_length, self.audio_frame_size], name='inference')
            self._inference = logits_res
        return self._inference

    def __bi_lstm_block(self, inputs, num_layers, num_units):
        with tf.variable_scope('forward'):
            forward_cells = []
            for i in range(num_layers):
                lstm_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                forward_cells.append(lstm_cell)

        with tf.variable_scope('backward'):
            backward_cells = []
            for i in range(num_layers):
                lstm_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                backward_cells.append(lstm_cell)

        with tf.variable_scope('Bi_LSTM'):
            rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=forward_cells,
              cells_bw=backward_cells,
              inputs=inputs,
              dtype=tf.float64,
              sequence_length=self.sequence_lengths)
        return rnn_outputs

    @property
    def prediction(self):
        if self._prediction is None:
            prediction = tf.sigmoid(self.inference) * 10
            prediction = (tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float64), axis=2)) * prediction
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            target_specs_mag = tf.cast(tf.abs(self.target_specs) ** 0.3, tf.float64)
            mixed_specs_mag = tf.abs(self.mixed_specs) ** 0.3
            predicted_specs = self.prediction * tf.cast(mixed_specs_mag, tf.float64)
            self.func_loss = tf.nn.l2_loss(target_specs_mag - predicted_specs, name='func_loss')
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.weights], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float64)
            self._loss = tf.add(self.func_loss, self.regularization * self.reg_loss, name='loss')
        return self._loss

    @property
    def train_op(self):
        if self._train_op is None:
            with tf.name_scope('optimizer'):
                if self.optimizer_choice == 'adam':
                    self.learning_rate = tf.constant(self.initial_learning_rate)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate)
                else:
                    self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                                    self.updating_step, self.learning_decay, staircase=True)
                    if self.optimizer_choice == 'sgd':
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.optimizer_choice == 'momentum':
                            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)  
                    else:
                        print('Optimizer must be either sgd, momentum or adam. Closing...')
                        sys.exit(1)
                
                self._train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')
        return self._train_op

    @property
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            mixed_mag_specs = tf.abs(self.mixed_specs) ** 0.3
            masked_mag_specs = (mixed_mag_specs * tf.cast(self.prediction, tf.float32)) ** 3.3333333333333335
            self._enhanced_sources = get_sources(masked_mag_specs, tf.angle(self.mixed_specs), num_samples=self.num_audio_samples)
        return tf.identity(self._enhanced_sources, name='enhanced_sources')

    @property
    def oracle_iam(self):
        if self._oracle_iam is None:
            self._oracle_iam = get_oracle_iam(self.target_specs, self.mixed_specs)
        return tf.identity(self._oracle_iam, name='oracle_iam')

    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                tbm = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.tbm, [0, 2, 1]), axis=3))
                oracle_iam = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.oracle_iam, [0, 2, 1]), axis=3))
                estimated_iam = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                mag_mixed_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.mixed_specs) ** 0.3, [0, 2, 1]), axis=3))
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.target_specs) ** 0.3, [0, 2, 1]), axis=3))
                mag_masked_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.mixed_specs) ** 0.3 * tf.cast(self.prediction, tf.float32), [0, 2, 1]), axis=3))
                tf.summary.image('TBM', tbm, max_outputs=n_samples)
                tf.summary.image('Oracle IAM', oracle_iam, max_outputs=n_samples)
                tf.summary.image('Estimated IAM', estimated_iam, max_outputs=n_samples)
                tf.summary.image('Mixed spectrogram', mag_mixed_specs, max_outputs=n_samples)
                tf.summary.image('Target spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Masked spectrogram', mag_masked_specs, max_outputs=n_samples)
                mixed_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.mixed_sources), axis=1), (-1,
                                                                                                   1))
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1,
                                                                                                     1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1,
                                                                                                         1))
                tf.summary.audio('Mixed audio', self.mixed_sources / mixed_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Target audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Masked audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

    @property
    def global_step(self):
        if self._global_step is None:
            self._global_step = tf.Variable(0, trainable=False)
        return self._global_step

    @property
    def weights(self):
        if self._weights is None:
            self._weights = tf.trainable_variables()
        return self._weights


class AudioVisualConcatMask(object):
    """
    Audio-Visual concat model (with masking)
    Input: frame-level concatenation of audio and video (face landmarks) features.
    Model: stacked BLSTM.
    Output: Ideal Amplitude Mask (IAM).
    Loss: L2 (target_spec - enhanced_spec)
    """

    def __init__(self, input_video, sequence_lengths, tbm, mixed_sources, target_sources, input_mixed_specs, keep_prob, config):
        self.input_video = input_video
        self.sequence_lengths = sequence_lengths
        self.audio_frame_size = config.audio_feat_dim
        self.video_frame_size = config.video_feat_dim
        self.num_audio_samples = config.num_audio_samples
        self.mixed_sources = mixed_sources
        self.target_sources = target_sources
        self.mixed_specs = compute_stft(mixed_sources, out_shape=tf.shape(input_mixed_specs))
        self.target_specs = compute_stft(target_sources, out_shape=tf.shape(input_mixed_specs))
        self.input_mixed_specs = input_mixed_specs
        self.num_units = config.n_hidden
        self.num_layers = config.num_layers
        self.optimizer_choice = config.optimizer_choice
        self.initial_learning_rate = config.learning_rate
        self.learning_rate = config.learning_rate
        self.updating_step = config.updating_step
        self.learning_decay = config.learning_decay
        self.keep_prob = keep_prob
        self.regularization = config.reg
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._oracle_iam = None
        self._summaries = None
        self._global_step = None
        self._weights = None
        self.func_loss = None
        self.reg_loss = None

    def create_graph(self):
        """
        Create the Graph of the model.
        """
        self.inference
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.oracle_iam
        self.summaries

    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            net_input = tf.concat([self.input_video, self.input_mixed_specs], axis=2)
            with tf.variable_scope('forward'):
                forward_cells = []
                for i in range(self.num_layers):
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    forward_cells.append(lstm_cell)

            with tf.variable_scope('backward'):
                backward_cells = []
                for i in range(self.num_layers):
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    backward_cells.append(lstm_cell)

            with tf.variable_scope('Bi_LSTM'):
                rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=forward_cells,
                  cells_bw=backward_cells,
                  inputs=net_input,
                  dtype=tf.float64,
                  sequence_length=self.sequence_lengths)
            rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.num_units * 2]), keep_prob=self.keep_prob)
            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.num_units * 2, self.audio_frame_size], stddev=1.0 / math.sqrt(float(self.num_units * 2)),
                  dtype=tf.float64),
                  name='weights')
                biases = tf.Variable(tf.zeros([self.audio_frame_size], dtype=tf.float64), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [-1, max_sequence_length, self.audio_frame_size], name='inference')
        return self._inference

    @property
    def prediction(self):
        if self._prediction is None:
            prediction = tf.sigmoid(self.inference) * 10
            prediction = (tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float64), axis=2)) * prediction
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            target_specs_mag = tf.cast(tf.abs(self.target_specs) ** 0.3, tf.float64)
            mixed_specs_mag = tf.abs(self.mixed_specs) ** 0.3
            estimated_specs = self.prediction * tf.cast(mixed_specs_mag, tf.float64)
            self.func_loss = tf.nn.l2_loss(target_specs_mag - estimated_specs, name='func_loss')
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.weights], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float64)
            self._loss = tf.add(self.func_loss, self.regularization * self.reg_loss, name='loss')
        return self._loss

    @property
    def train_op(self):
        if self._train_op is None:
            with tf.name_scope('optimizer'):
                if self.optimizer_choice == 'adam':
                    self.learning_rate = tf.constant(self.initial_learning_rate)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate)
                else:
                    self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                                    self.updating_step, self.learning_decay, staircase=True)
                    if self.optimizer_choice == 'sgd':
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.optimizer_choice == 'momentum':
                            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)  
                    else:
                        print('Optimizer must be either sgd, momentum or adam. Closing...')
                        sys.exit(1)
                
                self._train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')
        return self._train_op

    @property
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            mixed_mag_specs = tf.abs(self.mixed_specs) ** 0.3
            masked_mag_specs = (mixed_mag_specs * tf.cast(self.prediction, tf.float32)) ** 3.3333333333333335
            self._enhanced_sources = get_sources(masked_mag_specs, tf.angle(self.mixed_specs), num_samples=self.num_audio_samples)
        return tf.identity(self._enhanced_sources, name='enhanced_sources')

    @property
    def oracle_iam(self):
        if self._oracle_iam is None:
            self._oracle_iam = get_oracle_iam(self.target_specs, self.mixed_specs)
        return tf.identity(self._oracle_iam, name='oracle_iam')

    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                oracle_iam = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.oracle_iam, [0, 2, 1]), axis=3))
                estimated_iam = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                mag_mixed_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.mixed_specs) ** 0.3, [0, 2, 1]), axis=3))
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.target_specs) ** 0.3, [0, 2, 1]), axis=3))
                mag_masked_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.mixed_specs) ** 0.3 * tf.cast(self.prediction, tf.float32), [0, 2, 1]), axis=3))
                tf.summary.image('Oracle IAM', oracle_iam, max_outputs=n_samples)
                tf.summary.image('Estimated IAM', estimated_iam, max_outputs=n_samples)
                tf.summary.image('Mixed spectrogram', mag_mixed_specs, max_outputs=n_samples)
                tf.summary.image('Target spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Masked spectrogram', mag_masked_specs, max_outputs=n_samples)
                mixed_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.mixed_sources), axis=1), (-1,
                                                                                                   1))
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1,
                                                                                                     1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1,
                                                                                                         1))
                tf.summary.audio('Mixed audio', self.mixed_sources / mixed_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Target audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Masked audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

    @property
    def global_step(self):
        if self._global_step is None:
            self._global_step = tf.Variable(0, trainable=False)
        return self._global_step

    @property
    def weights(self):
        if self._weights is None:
            self._weights = tf.trainable_variables()
        return self._weights


class AudioVisualConcatSpec(object):
    """
    Audio-Visual concat model (w/o masking)
    Input: frame-level concatenation of audio and video (face landmarks) features.
    Model: stacked BLSTM.
    Output: enhanced spectrogram.
    Loss: MSE (target_spec - enhanced_spec)
    """

    def __init__(self, input_video, sequence_lengths, tbm, mixed_sources, target_sources, input_mixed_specs, keep_prob, config):
        self.input_video = input_video
        self.sequence_lengths = sequence_lengths
        self.tbm = tbm
        self.audio_frame_size = config.audio_feat_dim
        self.video_frame_size = config.video_feat_dim
        self.num_audio_samples = config.num_audio_samples
        self.mixed_sources = mixed_sources
        self.target_sources = target_sources
        self.mixed_specs = compute_stft(mixed_sources, out_shape=tf.shape(input_mixed_specs))
        self.target_specs = compute_stft(target_sources, out_shape=tf.shape(input_mixed_specs))
        self.input_mixed_specs = input_mixed_specs
        self.num_units = config.n_hidden
        self.num_layers = config.num_layers
        self.optimizer_choice = config.optimizer_choice
        self.initial_learning_rate = config.learning_rate
        self.learning_rate = config.learning_rate
        self.updating_step = config.updating_step
        self.learning_decay = config.learning_decay
        self.keep_prob = keep_prob
        self.regularization = config.reg
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._summaries = None
        self._global_step = None
        self._weights = None
        self.func_loss = None
        self.reg_loss = None

    def create_graph(self):
        """
        Create the Graph of the model.
        """
        self.inference
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.summaries

    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            net_input = tf.concat([self.input_video, self.input_mixed_specs], axis=2)
            with tf.variable_scope('forward'):
                forward_cells = []
                for i in range(self.num_layers):
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    forward_cells.append(lstm_cell)

            with tf.variable_scope('backward'):
                backward_cells = []
                for i in range(self.num_layers):
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.num_units, use_peepholes=True, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                    backward_cells.append(lstm_cell)

            with tf.variable_scope('Bi_LSTM'):
                rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=forward_cells,
                  cells_bw=backward_cells,
                  inputs=net_input,
                  dtype=tf.float64,
                  sequence_length=self.sequence_lengths)
            rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.num_units * 2]), keep_prob=self.keep_prob)
            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.num_units * 2, self.audio_frame_size], stddev=1.0 / math.sqrt(float(self.num_units * 2)),
                  dtype=tf.float64),
                  name='weights')
                biases = tf.Variable(tf.zeros([self.audio_frame_size], dtype=tf.float64), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [-1, max_sequence_length, self.audio_frame_size], name='inference')
        return self._inference

    @property
    def prediction(self):
        if self._prediction is None:
            prediction = tf.sigmoid(self.inference) * 100
            prediction = (tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float64), axis=2)) * self.inference
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            target_specs_mag = tf.cast(tf.abs(self.target_specs) ** 0.3, dtype=tf.float64)
            self.func_loss = tf.cast(tf.losses.mean_squared_error(labels=target_specs_mag, predictions=self.prediction), dtype=tf.float64, name='func_loss')
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.weights], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float64)
            self._loss = tf.add(self.func_loss, self.regularization * self.reg_loss, name='loss')
        return self._loss

    @property
    def train_op(self):
        if self._train_op is None:
            with tf.name_scope('optimizer'):
                if self.optimizer_choice == 'adam':
                    self.learning_rate = tf.constant(self.initial_learning_rate)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate)
                else:
                    self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                                    self.updating_step, self.learning_decay, staircase=True)
                    if self.optimizer_choice == 'sgd':
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.optimizer_choice == 'momentum':
                            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)  
                    else:
                        print('Optimizer must be either sgd, momentum or adam. Closing...')
                        sys.exit(1)
                
                self._train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')
        return self._train_op


    @property
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            enhanced_mag_specs = tf.cast(self.prediction, tf.float32) ** 3.3333333333333335
            self._enhanced_sources = get_sources(enhanced_mag_specs, tf.angle(self.mixed_specs), num_samples=self.num_audio_samples)
        return tf.identity(self._enhanced_sources, name='enhanced_sources')

    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_mixed_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.mixed_specs) ** 0.3, [0, 2, 1]), axis=3))
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.target_specs) ** 0.3, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(tf.abs(self.mixed_specs) ** 0.3 * tf.cast(self.prediction, tf.float32), [0, 2, 1]), axis=3))
                tf.summary.image('Mixed spectrogram', mag_mixed_specs, max_outputs=n_samples)
                tf.summary.image('Target spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('enh spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                mixed_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.mixed_sources), axis=1), (-1,
                                                                                                   1))
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1,
                                                                                                     1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1,
                                                                                                         1))
                tf.summary.audio('Mixed audio', self.mixed_sources / mixed_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Target audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Masked audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

    @property
    def global_step(self):
        if self._global_step is None:
            self._global_step = tf.Variable(0, trainable=False)
        return self._global_step

    @property
    def weights(self):
        if self._weights is None:
            self._weights = tf.trainable_variables()
        return self._weights
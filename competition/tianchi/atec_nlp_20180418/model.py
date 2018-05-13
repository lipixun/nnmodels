# encoding=utf8

""" The model
    Author: lipixun
    Created Time : 2018-05-13 19:53:57

    File Name: model.py
    Description:

"""

import logging

from os.path import join, dirname, abspath

import tensorflow as tf

from dataset import Dataset

CurrentDir = dirname(abspath(__file__))

class ModelBase(object):
    """The model base
    """
    logger = logging.getLogger("ModelBase")

    def __init__(self, name):
        """Create a new ModelBase
        """
        self._name = name
        # Build dataset & input
        with tf.variable_scope("input"):
            self._inputfile_placeholder = tf.placeholder(tf.string, shape=[None])
            self._dataset = Dataset(tf.data.TFRecordDataset(self._inputfile_placeholder))
            self._input_line_no, self._input_s1, self._input_s2, self._input_label = \
                self._dataset.get_next()

    def train(self, train_filenames, validate_filenames, max_epoch_nums=100):
        """Train the model
        """
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=join(CurrentDir, "outputs", self._name),
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
            ) as session:
            epoch = 0
            while max_epoch_nums <=0 or epoch < max_epoch_nums:
                self.logger.info("Epoch: %s", epoch)
                session.run(self._dataset.initializer)
                self._run_train(session, feeds={self._inputfile_placeholder: train_filenames})
                if validate_filenames:
                    session.run(self._dataset.initializer)
                    self._run_validate(session, feeds={self._inputfile_placeholder: validate_filenames})

    def predict(self, filenames):
        """Predict by current model
        Returns:
            list: A list of tuple(line_no, score)
        """
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=join(CurrentDir, "outputs", self._name),
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
            ) as session:
            session.run(self._dataset.initializer)
            return self._run_predict(session, feeds={self._inputfile_placeholder: filenames})

    def _run_train(self, session, feeds):
        """Run train
        """
        raise NotImplementedError

    def _run_validate(self, session, feeds):
        """Run validate
        """
        raise NotImplementedError

    def _run_predict(self, session, feeds):
        """Run predict
        """
        raise NotImplementedError

#
#
# Bi-LSTM with cosine distance & full connected output layer
#
#

class BiLSTMModel(ModelBase):
    """The bidirectional lstm model
    """
    def __init__(self,
        name,
        word_id_size,
        embedding_size=128,
        lstm_hidden_size=128,
        lstm_layer_num=1,
        output_method="full_connected",
        output_hidden_size=1024,
        ):
        """Create a new BiLSTMModel
        """
        super(BiLSTMModel, self).__init__(name)
        # Build network
        with tf.variable_scope("input_concat"):
            # Concat s1 and s2, and reshape
            inp = tf.concat([self._input_s1, self._input_s2], axis=1)
            inp = inp.reshape(inp, [-1, inp.shape[1] / 2])
        with tf.variable_scope("embedding"):
            # Embedding
            embedding_w = tf.get_variable("W", [word_id_size, embedding_size], dtype=tf.float32)
            embedded_inp = tf.nn.embedding_lookup(embedding_w, inp) # Shape: [batch size, max length, embedding size]
        with tf.variable_scope("lstm"):
            # Bi-LSTM
            lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
            if lstm_layer_num > 1:
                lstm_forward_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_forward_cell]*lstm_layer_num)
            lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
            if lstm_layer_num > 1:
                lstm_backward_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_backward_cell]*lstm_layer_num)

            (lstm_output_forward, lstm_output_backward), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_backward_cell,
                lstm_backward_cell,
                embedded_inp,
                sequence_length=tf.count_nonzero(inp, axis=1),
            )

            # LSTM output, shape: [batch size, max length, lstm_hidden_size*2]
            lstm_output = tf.concat([lstm_output_forward, tf.reverse(lstm_output_backward)], axis=2)
        with tf.variable_scope("seq2vec"):
            # Sequence to vector
            seq_vectors = tf.reduce_mean(lstm_output, axis=1) # Shape: [batch size, lstm_hidden_size*2]
        with tf.variable_scope("output"):
            # Output layer
            if output_method == "full_connected":
                raise NotImplementedError
            elif output_method == "cosine":
                output_s1, output_s2 = tf.split(
                    value=tf.reshape(seq_vectors, [-1, seq_vectors.shape[1]*2]),
                    axis=1,
                    num_or_size_splits=2
                )
                self._score = 1 - tf.losses.cosine_distance(output_s1, output_s2, axis=1)
                self._loss = tf.square(self._score - self._input_label)

    def _run_train(self, session, feeds):
        """Run train
        """
        raise NotImplementedError

    def _run_validate(self, session, feeds):
        """Run validate
        """
        raise NotImplementedError

    def _run_predict(self, session, feeds):
        """Run predict
        """
        raise NotImplementedError

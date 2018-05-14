# encoding=utf8

""" The model
    Author: lipixun
    Created Time : 2018-05-13 19:53:57

    File Name: model.py
    Description:

"""

import sys
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
        # Build internal vars
        with tf.variable_scope("internal"):
            self._global_step = tf.train.get_or_create_global_step()

    def train(self, train_filenames, validate_filenames, max_epoch_nums=100):
        """Train the model
        """
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=join(CurrentDir, "outputs", self._name),
            config=tf.ConfigProto(allow_soft_placement=True),
            ) as session:
            epoch = 0
            while max_epoch_nums <=0 or epoch < max_epoch_nums:
                epoch += 1
                self.logger.info("Epoch: %s", epoch)

                session.run(self._dataset.initializer, feed_dict={self._inputfile_placeholder: train_filenames})
                try:
                    while True:
                        self._run_train(session, feeds={})
                except tf.errors.OutOfRangeError:
                    pass

                if validate_filenames:
                    session.run(self._dataset.initializer, feed_dict={self._inputfile_placeholder: validate_filenames})
                    try:
                        while True:
                            self._run_validate(session, feeds={})
                    except tf.errors.OutOfRangeError:
                        pass

    def predict(self, filenames):
        """Predict by current model
        Returns:
            list: A list of tuple(line_no, score)
        """
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=join(CurrentDir, "outputs", self._name),
            config=tf.ConfigProto(allow_soft_placement=True),
            ) as session:
            session.run(self._dataset.initializer, feed_dict={self._inputfile_placeholder: filenames})
            try:
                while True:
                    for result in self._run_predict(session, feeds={}):
                        yield result
            except tf.errors.OutOfRangeError:
                pass

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

class EmbeddedCosineModel(ModelBase):
    """The cosine with embedding
    """
    def __init__(self, name, word_id_size, embedding_size=128):
        """Create a new EmbeddedCosineModel
        """
        super(EmbeddedCosineModel, self).__init__(name)
        # Build network
        with tf.variable_scope("input_concat"):
            # Concat s1 and s2, and reshape
            inp = tf.concat([self._input_s1, self._input_s2], axis=1)
            inp = tf.reshape(inp, [-1, inp.shape[1].value/2])
        with tf.variable_scope("embedding"):
            # Embedding
            embedding_w = tf.get_variable(
                "W",
                [word_id_size, embedding_size],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-1),
                )
            embedded_inp = tf.nn.embedding_lookup(embedding_w, inp) # Shape: [batch size, max length, embedding size]
        with tf.variable_scope("seq2vec"):
            # Sequence to vector
            seq_vectors = tf.reduce_mean(embedded_inp, axis=1) # Shape: [batch size, embedding size]
        with tf.variable_scope("output"):
            # Output layer
            output_s1, output_s2 = tf.split(
                value=tf.reshape(seq_vectors, [-1, seq_vectors.shape[1].value*2]),
                axis=1,
                num_or_size_splits=2
            )
            score = 1 - tf.reshape(tf.losses.cosine_distance(
                output_s1 / tf.norm(output_s1, keepdims=True), # unit vector
                output_s2 / tf.norm(output_s2, keepdims=True), # unit vector
                axis=1,
                reduction=tf.losses.Reduction.NONE), [-1])
            # Score in [-1, 1], scale to [0, 1]
            self._score = score / 2.0 + 0.5
            self._loss = tf.reduce_mean(tf.square(score - self._input_label))
        with tf.variable_scope("optimize"):
            # Optimize
            self._optimizer = tf.train.AdamOptimizer(1e-3)
            trainable_vars = tf.trainable_variables()
            gradient_and_variables = self._optimizer.compute_gradients(self._loss, trainable_vars)
            clipped_gradients, _ = tf.clip_by_global_norm([grad for grad, _ in gradient_and_variables], 5.0)
            self._optimize_op = self._optimizer.apply_gradients(zip(clipped_gradients, trainable_vars), self._global_step)

    def _run_train(self, session, feeds):
        """Run train
        """
        loss, _ = session.run([self._loss, self._optimize_op], feeds)
        print >>sys.stderr, "Loss:", loss

    def _run_validate(self, session, feeds):
        """Run validate
        """
        raise NotImplementedError

    def _run_predict(self, session, feeds):
        """Run predict
        """
        line_nos, scores = session.run([self._input_line_no, self._score], feeds)
        return zip(line_nos, scores)

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
        output_method="cosine",
        output_hidden_size=1024,
        ):
        """Create a new BiLSTMModel
        """
        super(BiLSTMModel, self).__init__(name)
        # Build network
        with tf.variable_scope("input_concat"):
            # Concat s1 and s2, and reshape
            inp = tf.concat([self._input_s1, self._input_s2], axis=1)
            inp = tf.reshape(inp, [-1, inp.shape[1].value/2])
        with tf.variable_scope("embedding"):
            # Embedding
            embedding_w = tf.get_variable(
                "W",
                [word_id_size, embedding_size],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-1),
                )
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
                lstm_forward_cell,
                lstm_backward_cell,
                embedded_inp,
                dtype=tf.float32,
                sequence_length=tf.count_nonzero(inp, axis=1),
            )

            # LSTM output, shape: [batch size, max length, lstm_hidden_size*2]
            lstm_output = tf.concat([lstm_output_forward, tf.reverse(lstm_output_backward, axis=[1])], axis=2)
        with tf.variable_scope("seq2vec"):
            # Sequence to vector
            seq_vectors = tf.reduce_mean(lstm_output, axis=1) # Shape: [batch size, lstm_hidden_size*2]
        with tf.variable_scope("output"):
            # Output layer
            if output_method == "full_connected":
                raise NotImplementedError
            elif output_method == "cosine":
                output_s1, output_s2 = tf.split(
                    value=tf.reshape(seq_vectors, [-1, seq_vectors.shape[1].value*2]),
                    axis=1,
                    num_or_size_splits=2
                )
                score = 1 - tf.reshape(tf.losses.cosine_distance(
                    output_s1 / tf.norm(output_s1, keepdims=True), # unit vector
                    output_s2 / tf.norm(output_s2, keepdims=True), # unit vector
                    axis=1,
                    reduction=tf.losses.Reduction.NONE), [-1])
                # Score in [-1, 1], scale to [0, 1]
                self._score = score / 2.0 + 0.5
                self._loss = tf.reduce_mean(tf.square(score - self._input_label))
        with tf.variable_scope("optimize"):
            # Optimize
            self._optimizer = tf.train.AdamOptimizer(1e-3)
            trainable_vars = tf.trainable_variables()
            gradient_and_variables = self._optimizer.compute_gradients(self._loss, trainable_vars)
            clipped_gradients, _ = tf.clip_by_global_norm([grad for grad, _ in gradient_and_variables], 5.0)
            self._optimize_op = self._optimizer.apply_gradients(zip(clipped_gradients, trainable_vars), self._global_step)

    def _run_train(self, session, feeds):
        """Run train
        """
        loss, _ = session.run([self._loss, self._optimize_op], feeds)
        print >>sys.stderr, "Loss:", loss

    def _run_validate(self, session, feeds):
        """Run validate
        """
        raise NotImplementedError

    def _run_predict(self, session, feeds):
        """Run predict
        """
        line_nos, scores = session.run([self._input_line_no, self._score], feeds)
        return zip(line_nos, scores)

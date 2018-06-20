# encoding=utf8

""" The model
    Author: lipixun
    Created Time : 2018-05-13 19:53:57

    File Name: model.py
    Description:

"""
from __future__ import print_function

import math
import logging

from collections import OrderedDict

import tensorflow as tf

import tftrainer
from tftrainer import Model, TrainBatchResult, EvaluateBatchResult, PredictBatchResult

from dataset import Dataset, PaddingSize

class ModelBase(Model):
    """The model base
    """
    logger = logging.getLogger("ModelBase")

    def get_dataset(self):
        """Get the dataset
        """
        return self._dataset

    def init_train_epoch(self, session, feeder, params):
        """Initialize training for the epoch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            bool: Succeed or not
        """
        if not hasattr(params, "input_files") or not params.input_files:
            raise ValueError("Require input_files")

        feeder.initialize(session, {self._inputfile: params.input_files})

        return True

    def train_batch(self, session, feeder, params):
        """Run training for one batch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            TrainBatchResult: The train batch result
        """
        raise NotImplementedError

    def init_evaluate_epoch(self, session, feeder, params):
        """Initialize evaluating
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            bool: Succeed or not
        """
        if not hasattr(params, "input_files") or not params.input_files:
            return False

        feeder.initialize(session, {self._inputfile: params.input_files})

        return True

    def evaluate_batch(self, session, feeder, params):
        """Run evaluating for one batch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            EvaluateBatchResult: The evaluate batch result
        """
        raise NotImplementedError

    def init_predict_epoch(self, session, feeder, params):
        """Initialize predict
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            bool: Succeed or not
        """
        if not hasattr(params, "input_files") or not params.input_files:
            return False

        feeder.initialize(session, {self._inputfile: params.input_files})

        return True

    def predict_batch(self, session, feeder, params):
        """Run predicting for one batch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            PredictBatchResult: The predict batch result
        """
        raise NotImplementedError

    def _build_graph(self):
        """Build the graph
        """
        super(ModelBase, self)._build_graph()

        # Build dataset & input
        with tf.variable_scope("input"):
            self._inputfile = tf.placeholder(tf.string, shape=[None])
            self._dataset = Dataset(tf.data.TFRecordDataset(self._inputfile), batch_size=128)
            self._input_line_no, self._input_s1, self._input_s2, self._input_label = \
                self._dataset.get_next()

class BiLSTMModel(ModelBase):
    """The bidirectional lstm model
    """
    def __init__(self,
        word_id_size,
        embedding_size=128,
        train_embedding=True,
        lstm_hidden_size=128,
        lstm_layer_num=1,
        output_hidden_size=1024,
        regularizer_gamma=1e-3,
        attention=False,
        attention_size=256,
        ):
        """Create a new BiLSTMModel
        """
        self._word_id_size = word_id_size
        self._embedding_size = embedding_size
        self._train_embedding = train_embedding
        self._lstm_hidden_size = lstm_hidden_size
        self._lstm_layer_num = lstm_layer_num
        self._output_hidden_size = output_hidden_size
        self._regularizer_gamma = regularizer_gamma
        self._attention = attention
        self._attention_size = attention_size

        super(BiLSTMModel, self).__init__()

    def init_session(self, session, params):
        """Initialize session for training
        Args:
            session(tf.Session): The tensorflow session
            params(Params): The parameters
        Returns:
            bool: Succeed or not
        """
        if not super(BiLSTMModel, self).init_session(session, params):
            return False

        if hasattr(params, "embedding") and params.embedding is not None:
            session.run(self._assign_embedding_op, {self._embedding_placholder: params.embedding})

        return True

    def init_train_epoch(self, session, feeder, params):
        """Initialize training for the epoch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            bool: Succeed or not
        """
        if not super(BiLSTMModel, self).init_train_epoch(session, feeder, params):
            return False

        session.run([self._reset_metric_op])

        return True

    def init_evaluate_epoch(self, session, feeder, params):
        """Initialize evaluating
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            bool: Succeed or not
        """
        if not super(BiLSTMModel, self).init_evaluate_epoch(session, feeder, params):
            return False

        session.run([self._reset_metric_op])

        return True

    def _build_graph(self):
        """Build the graph
        """
        super(BiLSTMModel, self)._build_graph()

        regularizer = tf.contrib.layers.l2_regularizer(scale=self._regularizer_gamma) # pylint: disable=no-member

        with tf.variable_scope("input_concat"):
            # Concat s1 and s2, and reshape
            inp = tf.concat([self._input_s1, self._input_s2], axis=1)
            inp = tf.reshape(inp, [-1, int(inp.shape[1].value/2)])
            self._inp = inp

        with tf.variable_scope("embedding"):
            # Embedding
            embedding_w = tf.get_variable(
                "W",
                [self._word_id_size, self._embedding_size],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-1),
                trainable=self._train_embedding,
                )
            embedded_inp = tf.nn.embedding_lookup(embedding_w, inp) # Shape: [batch size, max length, embedding size]
            self._embedding_placholder = tf.placeholder(tf.float32, shape=[self._word_id_size, self._embedding_size])
            self._assign_embedding_op = tf.assign(embedding_w, self._embedding_placholder)

        with tf.variable_scope("lstm"):
            # Bi-LSTM
            lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_hidden_size)
            if self._lstm_layer_num > 1:
                lstm_forward_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_forward_cell] * self._lstm_layer_num)
            lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_hidden_size)
            if self._lstm_layer_num > 1:
                lstm_backward_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_backward_cell] * self._lstm_layer_num)

            (lstm_output_forward, lstm_output_backward), _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_forward_cell,
                lstm_backward_cell,
                embedded_inp,
                dtype=tf.float32,
                sequence_length=tf.count_nonzero(inp, axis=1),
            )

            # LSTM output, shape: [batch size, max length, lstm_hidden_size*2]
            lstm_output = tf.concat([lstm_output_forward, tf.reverse(lstm_output_backward, axis=[1])], axis=2)
            lstm_output = tf.cond(self.is_training, lambda: tf.nn.dropout(lstm_output, 0.8), lambda: lstm_output)

        if self._attention:
            with tf.variable_scope("attention", initializer=tf.truncated_normal_initializer(stddev=1e-1), regularizer=regularizer):
                # Sequence to vector by self attention
                w = tf.get_variable("W", shape=[self._lstm_hidden_size*2, self._attention_size*3], dtype=tf.float32)
                # Split into q k v
                q, k, v = tf.split(tf.reshape(tf.matmul(tf.reshape(lstm_output, [-1, self._lstm_hidden_size*2]), w), [-1, PaddingSize, self._attention_size*3]), 3, axis=2) # Shape = [batch size, max_length, attention_size] for each tensor
                # Dot product attention
                attention = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / math.sqrt(self._attention_size)) # Shape = [batch size, max_length, max_length]
                seq_vectors = tf.reduce_sum(tf.matmul(attention, v), axis=1)  # Shape = [batch size, attention_size]
        else:
            with tf.variable_scope("seq2vec"):
                # Sequence to vector by sum pooling
                seq_vectors = tf.reduce_mean(lstm_output, axis=1) # Shape: [batch size, lstm_hidden_size*2]

        seq_vectors = tf.cond(self.is_training, lambda: tf.nn.dropout(seq_vectors, 0.8), lambda: seq_vectors)

        with tf.variable_scope("output", initializer=tf.truncated_normal_initializer(stddev=1e-1), regularizer=regularizer):
            # Output layer
            output = tf.reshape(seq_vectors, shape=[-1, int(seq_vectors.shape[1].value*2)])
            with tf.variable_scope("middle"):
                w = tf.get_variable("W", shape=[output.shape[1].value, self._output_hidden_size], dtype=tf.float32)
                b = tf.get_variable("b", shape=[self._output_hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer())
                output = tf.nn.relu(tf.nn.xw_plus_b(output, w , b))
                output = tf.cond(self.is_training, lambda: tf.nn.dropout(output, 0.8), lambda: output)
            with tf.variable_scope("output"):
                w = tf.get_variable("W", shape=[self._output_hidden_size, 1], dtype=tf.float32)
                b = tf.get_variable("b", shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
                logits = tf.nn.xw_plus_b(output, w, b)
            # Output & loss
            self._score = tf.reshape(tf.nn.sigmoid(logits), shape=[-1])
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self._input_label, shape=[-1, 1]), logits=logits))
            l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if l2_losses:
                self._loss += tf.add_n(l2_losses)

        with tf.variable_scope("metric") as scope:
            # Metric
            self._f1_score, self._recall, self._precision, self._update_metric_op = tftrainer.metrics.f1_at_thresholds(
                tf.equal(self._input_label, 1),
                self._score,
                [0.9, 0.7, 0.5],
            )
            self._reset_metric_op = tftrainer.reset_local_variables(scope)

        with tf.variable_scope("optimize"):
            # Optimize
            self._optimizer = tf.train.AdamOptimizer(1e-3)
            trainable_vars = tf.trainable_variables()
            gradient_and_variables = self._optimizer.compute_gradients(self._loss, trainable_vars)
            clipped_gradients, _ = tf.clip_by_global_norm([grad for grad, _ in gradient_and_variables], 5.0)
            self._optimize_op = self._optimizer.apply_gradients(zip(clipped_gradients, trainable_vars), self.global_step)

    def train_batch(self, session, feeder, params):
        """Run training for one batch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            TrainBatchResult: The train batch result
        """
        feed_dict = feeder.feed_dict() or {}

        if params.should_log:
            _, _, loss, f1, recall, precision = \
                session.run([
                    self._optimize_op,
                    self._update_metric_op,
                    self._loss,
                    self._f1_score,
                    self._recall,
                    self._precision,
                    ], feed_dict)

            return TrainBatchResult(
                {"_": loss},
                OrderedDict([
                    ("f1@0.9", f1[0]),
                    ("f1@0.7", f1[1]),
                    ("f1@0.5", f1[2]),
                    ("p@0.9", precision[0]),
                    ("p@0.7", precision[1]),
                    ("p@0.5", precision[2]),
                    ("r@0.9", recall[0]),
                    ("r@0.7", recall[1]),
                    ("r@0.5", recall[2]),
                ]),
            )
        else:
            _, _, loss = session.run([
                self._optimize_op,
                self._update_metric_op,
                self._loss,
                ], feed_dict)

            return TrainBatchResult({"_": loss})

    def evaluate_batch(self, session, feeder, params):
        """Run evaluating for one batch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            EvaluateBatchResult: The evaluate batch result
        """
        feed_dict = feeder.feed_dict() or {}

        _, _, loss, f1, recall, precision = \
            session.run([
                self._optimize_op,
                self._update_metric_op,
                self._loss,
                self._f1_score,
                self._recall,
                self._precision,
                ], feed_dict)

        return EvaluateBatchResult(
            {"_": loss},
            OrderedDict([
                ("f1@0.9", f1[0]),
                ("f1@0.7", f1[1]),
                ("f1@0.5", f1[2]),
                ("p@0.9", precision[0]),
                ("p@0.7", precision[1]),
                ("p@0.5", precision[2]),
                ("r@0.9", recall[0]),
                ("r@0.7", recall[1]),
                ("r@0.5", recall[2]),
            ]))

    def predict_batch(self, session, feeder, params):
        """Run predicting for one batch
        Args:
            session(tf.Session): The tensorflow session
            feeder(DataFeeder): The data feeder
            params(Params): The parameters
        Returns:
            PredictBatchResult: The predict batch result
        """
        feed_dict = feeder.feed_dict() or {}

        line_nos, scores = session.run([self._input_line_no, self._score], feed_dict)

        return PredictBatchResult({"line_no": line_nos, "score": scores})

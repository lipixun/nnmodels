# encoding=utf8

""" The model
    Author: lipixun
    Created Time : 2018-05-13 19:53:57

    File Name: model.py
    Description:

"""

import logging

from collections import OrderedDict

import tensorflow as tf

from tftrainer import Model, TrainBatchResult, EvaluateBatchResult, PredictBatchResult

from dataset import Dataset

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
        if not hasattr(params, "train_files") or not params.train_files:
            raise ValueError("Require train_files")

        feeder.initialize(session, {self._inputfile: params.train_files})

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
        if not hasattr(params, "eval_files") or not params.eval_files:
            return False

        feeder.initialize(session, {self._inputfile: params.eval_files})

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
        if not hasattr(params, "predict_files") or not params.predict_files:
            return False

        feeder.initialize(session, {self._inputfile: params.predict_files})

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
        lstm_hidden_size=128,
        lstm_layer_num=1,
        output_hidden_size=1024,
        ):
        """Create a new BiLSTMModel
        """
        self._word_id_size = word_id_size
        self._embedding_size = embedding_size
        self._lstm_hidden_size = lstm_hidden_size
        self._lstm_layer_num = lstm_layer_num
        self._output_hidden_size = output_hidden_size

        super(BiLSTMModel, self).__init__()

    def _build_graph(self):
        """Build the graph
        """
        super(BiLSTMModel, self)._build_graph()

        with tf.variable_scope("input_concat"):
            # Concat s1 and s2, and reshape
            inp = tf.concat([self._input_s1, self._input_s2], axis=1)
            inp = tf.reshape(inp, [-1, inp.shape[1].value/2])

        with tf.variable_scope("embedding"):
            # Embedding
            embedding_w = tf.get_variable(
                "W",
                [self._word_id_size, self._embedding_size],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-1),
                )
            embedded_inp = tf.nn.embedding_lookup(embedding_w, inp) # Shape: [batch size, max length, embedding size]

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

        with tf.variable_scope("seq2vec"):
            # Sequence to vector
            seq_vectors = tf.reduce_mean(lstm_output, axis=1) # Shape: [batch size, lstm_hidden_size*2]

        with tf.variable_scope("output"):
            # Output layer
            concated_output = tf.reshape(seq_vectors, shape=[-1, seq_vectors.shape[1].value*2])
            with tf.variable_scope("middle"):
                w = tf.get_variable("W",
                    shape=[concated_output.shape[1].value, self._output_hidden_size],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-1),
                    )
                b = tf.get_variable("b", shape=[self._output_hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer())
                output = tf.nn.relu(tf.nn.xw_plus_b(concated_output, w , b))
            with tf.variable_scope("output"):
                w = tf.get_variable("W",
                    shape=[self._output_hidden_size, 1],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-1),
                    )
                b = tf.get_variable("b", shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())
                logits = tf.nn.xw_plus_b(output, w, b)

            self._score = tf.reshape(tf.nn.sigmoid(logits), shape=[-1])
            self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self._input_label, shape=[-1, 1]), logits=logits))

        with tf.variable_scope("metric"):
            # Metric
            self._precision_at_thresholds, self._precision_at_thresholds_update_op = tf.metrics.precision_at_thresholds(
                tf.equal(self._input_label, 1),
                self._score,
                [0.9, 0.7, 0.5],
            )
            self._recall_at_thresholds, self._recall_at_thresholds_update_op = tf.metrics.recall_at_thresholds(
                tf.equal(self._input_label, 1),
                self._score,
                [0.9, 0.7, 0.5],
            )

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
            _, _, _, loss, precision_at_thresholds, recall_at_thresholds = \
                session.run([
                    self._optimize_op,
                    self._precision_at_thresholds_update_op,
                    self._recall_at_thresholds_update_op,
                    self._loss,
                    self._precision_at_thresholds,
                    self._recall_at_thresholds,
                    ], feed_dict)

            return TrainBatchResult(
                {"_": loss},
                OrderedDict([
                    ("p@0.9", precision_at_thresholds[0]),
                    ("p@0.7", precision_at_thresholds[1]),
                    ("p@0.5", precision_at_thresholds[2]),
                    ("r@0.9", recall_at_thresholds[0]),
                    ("r@0.7", recall_at_thresholds[1]),
                    ("r@0.5", recall_at_thresholds[2]),
                ]),
            )
        else:
            _, _, _, loss = session.run([
                self._optimize_op,
                self._precision_at_thresholds_update_op,
                self._recall_at_thresholds_update_op,
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

        _, _, _, loss, precision_at_thresholds, recall_at_thresholds = \
            session.run([
                self._optimize_op,
                self._precision_at_thresholds_update_op,
                self._recall_at_thresholds_update_op,
                self._loss,
                self._precision_at_thresholds,
                self._recall_at_thresholds,
                ], feed_dict)

        return EvaluateBatchResult(
            {"_": loss},
            OrderedDict([
                ("p@0.9", precision_at_thresholds[0]),
                ("p@0.7", precision_at_thresholds[1]),
                ("p@0.5", precision_at_thresholds[2]),
                ("r@0.9", recall_at_thresholds[0]),
                ("r@0.7", recall_at_thresholds[1]),
                ("r@0.5", recall_at_thresholds[2]),
            ]),
        )

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

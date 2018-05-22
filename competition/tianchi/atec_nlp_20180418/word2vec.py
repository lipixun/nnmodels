# encoding=utf8

""" The word(char) to vector model
    Author: lipixun
    Created Time : 2018-05-22 15:27:44

    File Name: word2vec.py
    Description:

        This is used to convert a word or a char to vector

"""

import tensorflow as tf

import tftrainer

from text import TextDictionary
from dataset import Word2VecDataset
from example import Word2VecExampleBuilder

class Word2Vec(object):
    """The word2vec
    """
    def __init__(self, skip_window=5, use_jieba=False):
        """Create a new Word2Vec
        """
        self._skip_window = skip_window
        self._use_jieba = use_jieba

    def gen_skipgram_pairs(self, words):
        """Generate skipgram pairs from words
        Args:
            words(list): A list of word
        Returns:
            list: A list of ([context word], target word)
        """
        pairs = []
        for i, word in enumerate(words):
            pairs.append((
                [words[j] for j in range(max(0, i-self._skip_window), min(i+self._skip_window+1, len(words))) if i != j],
                word,
            ))

        return pairs

    def load_file(self, filename):
        """Load file
        """

    def preprocess(self, input_path, output_path, dict_path, no_fit=False):
        """Preprocess
        """
        text_dict = TextDictionary(use_jieba=self._use_jieba)

        if no_fit:
            # Load dict from file
            text_dict.load(dict_path)
        else:
            # Build the dictionary
            for input_file in tftrainer.path.findfiles(input_path):
                for s in self.load_file(input_file):
                    text_dict.fit(s)

            text_dict.save(dict_path)

        # Build the tensorflow example file
        for input_file, output_file in tftrainer.path.get_input_output_pair(input_path, output_path):
            example_builder = Word2VecExampleBuilder(text_dict, output_file + ".tfrecord", must_have_label=not no_fit)
            for s in self.load_file(input_file):
                words = text_dict.sentence_to_words(s)
                skip_gram_pairs = self.gen_skipgram_pairs(words)
                for context_words, target in skip_gram_pairs:
                    for context_word in context_words:
                        example_builder.write(target, context_word)
            example_builder.close()

class Word2VecModel(tftrainer.Model):
    """Word2vec model
    """
    def __init__(self, id_size, embedding_size=128, sample_size=64):
        """Create a new Word2VecModel
        """
        self._id_size = id_size
        self._embedding_size = embedding_size
        self._sample_size = sample_size

        super(Word2VecModel, self).__init__()

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
        _, loss = session.run([self._optimize_op, self._loss], feeder.feed_dict() or {})

        return tftrainer.TrainBatchResult({"_": loss})

    def get_embedding(self, session):
        """Get the embedding matrix
        """
        return session.run(self._embedding_w)

    def _build_graph(self):
        """Build the graph
        """
        super(Word2VecModel, self)._build_graph()

        # Build dataset & input
        with tf.variable_scope("input"):
            self._inputfile = tf.placeholder(tf.string, shape=[None])
            self._dataset = Word2VecDataset(tf.data.TFRecordDataset(self._inputfile),
                prefetch_size=51200,
                batch_size=5120,
                shuffle_size=51200,
                )
            self._word, self._label = self._dataset.get_next()

        # Embedding
        with tf.variable_scope("embedding"):
            self._embedding_w = tf.get_variable("W", shape=[self._id_size, self._embedding_size], dtype=tf.float32)
            embedded_words = tf.nn.embedding_lookup(self._embedding_w, self._word)

        # Output layer
        with tf.variable_scope("output", initializer=tf.truncated_normal_initializer(stddev=1e-1)):
            w = tf.get_variable("W", shape=[self._id_size, self._embedding_size])
            b = tf.get_variable("b", shape=[self._id_size], initializer=tf.zeros_initializer()),
            self._loss = tf.nn.nce_loss(
                weights=w,
                biases=b,
                labels=tf.reshape(self._label, shape=[-1, 1]),
                inputs=embedded_words,
                num_sampled=self._sample_size,
                num_classes=self._id_size,
                num_true=1,
            )

        # Optimizer
        with tf.variable_scope("optimize"):
            self._optimize_op = tftrainer.optimize_with_gradient_clipping_by_global_norm(
                tf.train.AdamOptimizer(learning_rate=1e-3),
                self._loss,
                global_step=self.global_step,
                )

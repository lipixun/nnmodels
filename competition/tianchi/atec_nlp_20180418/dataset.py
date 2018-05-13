# encoding=utf8

""" The dataset
    Author: lipixun
    Created Time : 2018-05-13 16:10:16

    File Name: dataset.py
    Description:

"""

import tensorflow as tf

PaddingSize = 64

Features = {
    "line_no": tf.FixedLenFeature([], dtype=tf.int64),
    "s1": tf.FixedLenSequenceFeature(shape=[PaddingSize], dtype=tf.int64),
    "s2": tf.FixedLenSequenceFeature(shape=[PaddingSize], dtype=tf.int64),
    "label": tf.FixedLenFeature([], dtype=tf.float32),
}

class Dataset(object):
    """The dataset based on tensorflow
    """
    def __init__(self, dataset, batch_size=128, prefetch_size=128*5, shuffle_size=128*5):
        """Create a new Dataset
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size
        self._shuffle_size = shuffle_size
        # Build the network
        if self._prefetch_size > 0:
            self._dataset = self._dataset.prefetch(prefetch_size)
        if self._shuffle_size > 0:
            self._dataset = self._dataset.shuffle(shuffle_size)
        self._dataset = self._dataset.map(parse_example)
        self._dataset = self._dataset.batch(self._batch_size)
        self._iterator = self._dataset.make_one_shot_iterator()
        self._iterator_handler = self._iterator_handler.string_handle()

    @property
    def output_types(self):
        """Output types
        """
        return self._dataset.output_types

    @property
    def output_shapes(self):
        """Output shapes
        """
        return self._dataset.output_shapes

    @property
    def get_iterator_handler_op(self):
        """The op to get iterator handler
        """
        return self._iterator_handler

    @property
    def initializer(self):
        """Get the initializer
        """
        return self._iterator.initializer

    def get_next(self):
        """Get next
        """
        return self._iterator.get_next()

    def get_iterator_with_input_handler(self):
        """Get the iterator with input handelr
        """
        input_handler = tf.placeholder(tf.string, shape=[])
        return (
            input_handler,
            tf.data.Iterator.from_string_handle(input_handler, self.output_types, self.output_shapes),
        )

def parse_example(example):
    """Parse the example proto to tensors
    """
    parsed_features = tf.parse_single_example(example, Features)
    return (
        parsed_features["line_no"],
        parsed_features["s1"],
        parsed_features["s2"],
        parsed_features["label"],
    )

# encoding=utf8

""" The dataset
    Author: lipixun
    Created Time : 2018-05-13 16:10:16

    File Name: dataset.py
    Description:

"""

import multiprocessing

import tensorflow as tf

from tftrainer import MultiDataset

PaddingSize = 64

Features = {
    "line_no": tf.FixedLenFeature([], dtype=tf.int64),
    "s1": tf.FixedLenFeature([PaddingSize], dtype=tf.int64),
    "s2": tf.FixedLenFeature([PaddingSize], dtype=tf.int64),
    "label": tf.FixedLenFeature([], dtype=tf.float32),
}

Word2VecFeatures = {
    "word": tf.FixedLenFeature([], dtype=tf.int64),
    "label": tf.FixedLenFeature([], dtype=tf.int64),
}

class Dataset(MultiDataset):
    """The dataset based on tensorflow
    """
    def _map_dataset(self, ds):
        """Map the dataset
        """
        return ds.map(parse_example, multiprocessing.cpu_count())

class Word2VecDataset(MultiDataset):
    """The word2vec dataset
    """
    def _map_dataset(self, ds):
        """Map the dataset
        """
        return ds.map(parse_word2vec_example, multiprocessing.cpu_count())

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

def parse_word2vec_example(example):
    """Parse the word2vec example to tensors
    """
    parsed_features = tf.parse_single_example(example, Word2VecFeatures)
    return (
        parsed_features["word"],
        parsed_features["label"],
    )

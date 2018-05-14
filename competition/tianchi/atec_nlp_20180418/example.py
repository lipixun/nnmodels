# encoding=utf8

""" The example
    Author: lipixun
    Created Time : 2018-05-13 17:20:56

    File Name: example.py
    Description:

"""

import tensorflow as tf

from text import parse_text_line
from dataset import PaddingSize

class ExampleBuilder(object):
    """The example builder
    """
    def __init__(self, text_dictionary, must_have_label=False):
        """Create a new ExampleBuilder
        """
        self._text_dictionary = text_dictionary
        self._must_have_label = must_have_label

    def build_single_example(self, line_no, s1, s2, label=0):
        """Build single example
        """
        example = tf.train.Example()
        example.features.feature["line_no"].int64_list.value.append(line_no) # pylint: disable=no-member
        example.features.feature["s1"].int64_list.value.extend(self._pad(self._text_dictionary.to_id(s1, min_word_id_count=2), PaddingSize)) # pylint: disable=no-member
        example.features.feature["s2"].int64_list.value.extend(self._pad(self._text_dictionary.to_id(s2, min_word_id_count=2), PaddingSize)) # pylint: disable=no-member
        example.features.feature["label"].float_list.value.append(label) # pylint: disable=no-member

        return example

    def build_single_example_from_line(self, line):
        """Build a single example object
        """
        line_no, s1, s2, label = parse_text_line(line)
        if label is None:
            if self._must_have_label:
                raise ValueError("Lack of label field")
            label = 0.0

        return self.build_single_example(line_no, s1, s2, label)

    def build_and_write_tf_record_file(self, strs, filename):
        """Build examples and write as tensorflow record format
        """
        writer = tf.python_io.TFRecordWriter(filename)
        for s in strs:
            example = self.build_single_example_from_line(s)
            writer.write(example.SerializeToString())
        writer.close()

    def _pad(self, values, length, pad_value=0):
        """Pad values
        """
        if len(values) > length:
            return values[:length]
        else:
            return values + [pad_value]*(length - len(values))

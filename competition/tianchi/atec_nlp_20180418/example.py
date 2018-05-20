# encoding=utf8

""" The example
    Author: lipixun
    Created Time : 2018-05-13 17:20:56

    File Name: example.py
    Description:

"""

import tensorflow as tf

from dataset import PaddingSize

class ExampleBuilder(object):
    """The example builder
    """
    def __init__(self, text_dict, output_path, must_have_label=False):
        """Create a new ExampleBuilder
        """
        self._text_dict = text_dict
        self._output_path = output_path
        self._must_have_label = must_have_label
        self._writer = tf.python_io.TFRecordWriter(output_path)

    def build_single_example(self, line_no, s1, s2, label):
        """Build single example
        """
        example = tf.train.Example()
        example.features.feature["line_no"].int64_list.value.append(line_no) # pylint: disable=no-member
        example.features.feature["s1"].int64_list.value.extend(self._pad(self._text_dict.to_id(s1, min_id_count=2), PaddingSize)) # pylint: disable=no-member
        example.features.feature["s2"].int64_list.value.extend(self._pad(self._text_dict.to_id(s2, min_id_count=2), PaddingSize)) # pylint: disable=no-member
        example.features.feature["label"].float_list.value.append(label) # pylint: disable=no-member

        return example

    def write(self, line_no, s1, s2, label):
        """Write
        """
        if self._must_have_label and label is None:
            raise ValueError("Lack of label field")

        if label is None:
            label = 0.0

        example = self.build_single_example(line_no, s1, s2, label)
        self._writer.write(example.SerializeToString())

    def close(self):
        """Close
        """
        self._writer.close()

    def _pad(self, values, length, pad_value=0):
        """Pad values
        """
        if len(values) > length:
            return values[:length]
        else:
            return values + [pad_value]*(length - len(values))

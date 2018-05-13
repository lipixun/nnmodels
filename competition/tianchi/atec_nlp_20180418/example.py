# encoding=utf8

""" The example
    Author: lipixun
    Created Time : 2018-05-13 17:20:56

    File Name: example.py
    Description:

"""

import tensorflow as tf

class ExampleBuilder(object):
    """The example builder
    """
    def __init__(self,
        must_have_label=False,
        build_word_dictionary=False,
        text_dictionary=False,
        ):
        """Create a new ExampleBuilder
        """
        self._must_have_label = must_have_label
        self._build_word_dictionary = build_word_dictionary
        self._text_dictionary = text_dictionary

    def build_single_example(self, line_no, s1, s2, label=0):
        """Build single example
        """
        example = tf.train.Example()
        example.features["line_no"].int64_list.value.append(line_no) # pylint: disable=no-member
        example.features["s1"].int64_list.value.extend(self._text_dictionary.to_id(s1)) # pylint: disable=no-member
        example.features["s2"].int64_list.value.extend(self._text_dictionary.to_id(s2)) # pylint: disable=no-member
        example.features["label"].float_value.value.append(label) # pylint: disable=no-member

        return example

    def build_single_example_from_line(self, line):
        """Build a single example object
        """
        parts = []
        for part in line.split("\t"):
            part = part.strip()
            if part:
                parts.append(part)
        line_no, s1, s2, label = None, None, None, None
        if len(parts) < 3 or len(parts) > 4:
            raise ValueError("Malformed line")
        elif len(parts) == 3:
            if self._must_have_label:
                raise ValueError("Lack of label field")
            line_no, s1, s2 = parts # pylint: disable=unbalanced-tuple-unpacking
            line_no = int(line_no)
            label = 0
        else:
            line_no, s1, s2, label = parts # pylint: disable=unbalanced-tuple-unpacking
            line_no = int(line_no)
            label = float(label)

        return self.build_single_example(line_no, s1, s2, label)


    def build_and_write_tf_record_file(self, strs, filename):
        """Build examples and write as tensorflow record format
        """
        writer = tf.python_io.TFRecordWriter(filename)
        for s in strs:
            example = self.build_single_example_from_line(s)
            writer.write(example.SerializeToString())
        writer.close()

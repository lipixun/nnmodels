# encoding=utf8

""" The preprocess
    Author: lipixun
    Created Time : 2018-05-13 20:24:36

    File Name: preproc.py
    Description:

"""

import tftrainer

from text import TextDictionary
from example import ExampleBuilder

class Preprocessor(object):
    """The preprocessor
    """
    def __init__(self, use_jieba=False):
        """Create a new Preprocessor
        """
        self._use_jieba = use_jieba

    def preprocess(self, input_path, output_path, dict_path, no_fit):
        """Preprocess
        """
        text_dict = TextDictionary(use_jieba=self._use_jieba)

        if no_fit:
            # Load dict from file
            text_dict.load(dict_path)
        else:
            # Build the dictionary
            for input_file in tftrainer.path.findfiles(input_path):
                for _, s1, s2, _ in self.load_csv(input_file):
                    text_dict.fit(s1)
                    text_dict.fit(s2)

            text_dict.save(dict_path)

        # Build the tensorflow example file
        for input_file, output_file in tftrainer.path.get_input_output_pair(input_path, output_path):
            example_builder = ExampleBuilder(text_dict, output_file + ".tfrecord", must_have_label=not no_fit)
            for line_no, s1, s2, label in self.load_csv(input_file):
                example_builder.write(line_no, s1, s2, label)
            example_builder.close()

    def load_csv(self, filename):
        """Load csv
        """
        with open(filename, "rb") as fd:
            for line in fd:
                line = line.strip()
                if line:
                    line_no, s1, s2, label = self.parse_text_line(line)
                    yield line_no, s1, s2, label

    def parse_text_line(self, line):
        """Parse a text line
        """
        line = line.replace(b"\xef\xbb\xbf", b"").decode("utf8")

        parts = []
        for part in line.split("\t"):
            part = part.strip()
            if part:
                parts.append(part)

        line_no, s1, s2, label = None, None, None, None
        if len(parts) < 3 or len(parts) > 4:
            raise ValueError("Malformed line [%s]" % line)
        elif len(parts) == 3:
            line_no, s1, s2 = parts # pylint: disable=unbalanced-tuple-unpacking
            line_no = int(line_no)
            label = None
        else:
            line_no, s1, s2, label = parts # pylint: disable=unbalanced-tuple-unpacking
            line_no = int(line_no)
            label = float(label)

        return line_no, s1, s2, label

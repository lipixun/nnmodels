# encoding=utf8

""" The preprocess
    Author: lipixun
    Created Time : 2018-05-13 20:24:36

    File Name: preproc.py
    Description:

"""

import tftrainer

from text import TextDictionary, parse_text_line
from example import ExampleBuilder

def preprocess(input_path, output_path, dictionary_filename, no_fit):
    """Preprocess
    """
    if no_fit:
        text_dict = TextDictionary.load(dictionary_filename)
    else:
        text_dict = TextDictionary()

    if not no_fit:
        # Build the dictionary
        for input_file in tftrainer.path.findfiles(input_path):
            with open(input_file, "rb") as fd:
                for line in fd:
                    line = line.strip()
                    if line:
                        _, s1, s2, _ = parse_text_line(line)
                        text_dict.fit(s1)
                        text_dict.fit(s2)

        text_dict.save(dictionary_filename)

    # Build the tensorflow example file
    example_builder = ExampleBuilder(text_dict)

    for input_file, output_file in tftrainer.path.get_input_output_pair(input_path, output_path):

        def read_input_file():
            """Read input file
            """
            with open(input_file, "rb") as fd:
                for line in fd:
                    line = line.strip()
                    if line:
                        yield line

        example_builder.build_and_write_tf_record_file(read_input_file(), output_file + ".tfrecord")

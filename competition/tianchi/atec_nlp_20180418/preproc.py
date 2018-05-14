# encoding=utf8

""" The preprocess
    Author: lipixun
    Created Time : 2018-05-13 20:24:36

    File Name: preproc.py
    Description:

"""

from os import makedirs
from os.path import isdir, abspath, dirname

from util import get_input_output_pair
from text import TextDictionary, parse_text_line
from example import ExampleBuilder

def preprocess(input_filename, output_filename, dictionary_filename, no_fit):
    """Preprocess
    """
    if no_fit:
        text_dict = TextDictionary.load(dictionary_filename)
    else:
        text_dict = TextDictionary()

    if not no_fit:
        # Build the dictionary
        for inpfile, _ in get_input_output_pair(input_filename, output_filename):
            with open(inpfile, "rb") as fd:
                for line in fd:
                    line = line.strip()
                    if line:
                        _, s1, s2, _ = parse_text_line(line)
                        text_dict.fit(s1)
                        text_dict.fit(s2)

        text_dict.save(dictionary_filename)

    # Build the tensorflow example file
    example_builder = ExampleBuilder(text_dict)

    for inpfile, outfile in get_input_output_pair(input_filename, output_filename):
        outfile = abspath(outfile)
        out_dirname = dirname(outfile)
        if not isdir(out_dirname):
            makedirs(out_dirname)

        def read_inpfile():
            """Read inpfile
            """
            with open(inpfile, "rb") as fd:
                for line in fd:
                    line = line.strip()
                    if line:
                        yield line

        example_builder.build_and_write_tf_record_file(read_inpfile(), outfile + ".tfrecord")

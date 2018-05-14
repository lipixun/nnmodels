# encoding=utf8

""" The utility
    Author: lipixun
    Created Time : 2018-05-13 20:24:36

    File Name: util.py
    Description:

"""

from os import listdir
from os.path import exists, isfile, join, basename

try:
    import simplejson as json # pylint: disable=unused-import
except ImportError:
    import json # pylint: disable=unused-import

def split_into_chunks(iterable_obj, chunk_size):
    """Split into chunks
    """
    chunk = []
    for obj in iterable_obj:
        chunk.append(obj)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def get_input_output_pair(input_filename, output_filename):
    """Get input and output pair
    """
    if exists(output_filename):
        if isfile(input_filename) and not isfile(output_filename):
            output_filename = join(output_filename, basename(input_filename))
        elif not isfile(input_filename) and isfile(output_filename):
            raise ValueError("Input is a dir, output is a file")

    if isfile(input_filename):
        yield (input_filename, output_filename)
        return

    for _input_filename in listdir(input_filename):
        if isfile(join(input_filename, _input_filename)):
            yield (join(input_filename, _input_filename), join(output_filename, _input_filename))

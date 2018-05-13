# encoding=utf8

""" The utility
    Author: lipixun
    Created Time : 2018-05-13 20:24:36

    File Name: util.py
    Description:

"""

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

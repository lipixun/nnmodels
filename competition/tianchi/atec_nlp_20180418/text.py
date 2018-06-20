# encoding=utf8

""" The text
    Author: lipixun
    Created Time : 2018-05-13 17:36:22

    File Name: text.py
    Description:

"""

import re
import six

from collections import Counter

from util import json

try:
    import jieba
except ImportError:
    jieba = None

class TextDictionary(object):
    """The text dictionary

        ID:

            * 0 Default
            * 1 Unknown
            * 2 Numbers
            * 3 <EOS>
            * ...
            * 10 <word>

    """
    ID_Default  = 0
    ID_Unknown  = 1
    ID_Numbers  = 2
    ID_EOS      = 3
    ID_WordStart= 10

    Regex_Number = re.compile(r"^\d+$", re.UNICODE)

    def __init__(self, word_to_ids=None, id_counter=None,  normalize_nums=True, use_jieba=False):
        """Create a new TextDictionary
        """
        self._word_to_ids = word_to_ids or {}
        self._id_counter = id_counter or Counter()
        self._normalize_nums = normalize_nums
        self._use_jieba = use_jieba
        if self._use_jieba and not jieba:
            raise ValueError("Cannot use jieba since it's not imported successfully")

    @property
    def id_size(self):
        """Get the id size
        """
        return len(self._word_to_ids) + self.ID_WordStart

    def lookup_word_id(self, word, min_id_count=0):
        """Lookup word id
        """
        if self.Regex_Number.match(word) and self._normalize_nums:
            return self.ID_Numbers
        _id = self._word_to_ids.get(word)
        if _id and min_id_count and self._id_counter.get(_id) < min_id_count:
            return
        return _id

    def fit(self, s):
        """Fit this dictionary by sentence
        """
        self.to_id(s, fit=True, count=True)

    def to_id(self, s, fit=False, count=False, merge_continguous_nums=True, min_id_count=0, append_eos=True):
        """Convert sentence to ids
        """
        if fit and min_id_count:
            raise ValueError("Cannot set fit and min_id_count at the same time")

        words = self.sentence_to_words(s)
        word_ids = []
        for word in words:
            _id = self.lookup_word_id(word, min_id_count)
            if not _id:
                if fit:
                    # Fit it
                    _id = len(self._word_to_ids) + self.ID_WordStart
                    self._word_to_ids[word] = _id
                else:
                    # Unknown
                    _id = self.ID_Unknown
            if merge_continguous_nums and _id == self.ID_Numbers and word_ids and word_ids[-1] == self.ID_Numbers:
                continue
            if _id == self.ID_Unknown and word_ids and word_ids[-1] == self.ID_Unknown:
                continue
            if count and _id >= self.ID_WordStart:
                self._id_counter[_id] += 1
            word_ids.append(_id)

        if append_eos:
            word_ids.append(self.ID_EOS)

        return word_ids

    def save(self, filename):
        """Save this dictionary to the file
        """
        d = { "word_to_ids": self._word_to_ids, "id_counter": self._id_counter.most_common() }
        with open(filename, "w") as fd:
            json.dump(d, fd)

    def load(self, filename):
        """Load from file
        """
        with open(filename, "rb") as fd:
            d = json.load(fd)
        word_to_ids = d.get("word_to_ids")
        id_counter = d.get("id_counter")
        if id_counter:
            id_counter = Counter({x: y for x, y in id_counter})
        else:
            id_counter = Counter()

        self._word_to_ids = word_to_ids
        self._id_counter = id_counter

    def sentence_to_words(self, s):
        """Convert sentence to words
        """
        if self._use_jieba:
            return [word.strip() for word in jieba.cut(s) if word.strip()]
        else:
            return list(s)

    def subset_by_word_count(self, min_count):
        """Create a subset of this dictionary by the min count of word
        NOTE:
            This will remap the word index
        """
        word_to_ids = {}
        id_counter = Counter()
        # Build a id to word map
        id_to_word = {}
        for word, word_id in six.iteritems(self._word_to_ids):
            id_to_word[word_id] = word
        # Compute the subset and re-assign the ids
        for word_id, count in self._id_counter.most_common():
            if count < min_count:
                break
            word = id_to_word[word_id]
            word_new_id = len(word_to_ids) + self.ID_WordStart
            word_to_ids[word] = word_new_id
            id_counter[word_new_id] = count
        return TextDictionary(word_to_ids, id_counter, self._normalize_nums, self._use_jieba)

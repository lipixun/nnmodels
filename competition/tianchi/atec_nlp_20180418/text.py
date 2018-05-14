# encoding=utf8

""" The text
    Author: lipixun
    Created Time : 2018-05-13 17:36:22

    File Name: text.py
    Description:

"""

from collections import Counter

from util import json

class TextDictionary(object):
    """The text dictionary

        ID:

            * 0 Default
            * 1 Unknown
            * 2 Numbers
            * ...
            * 10 <word>

    """
    ID_Default = 0
    ID_Unknown = 1
    ID_Numbers = 2
    ID_WordStart = 10

    Numbers = ["0","1","2","3","4","5","6","7","8","9"]

    def __init__(self, word_mapping=None, id_counter=None,  normalize_nums=True, use_jieba=False):
        """Create a new TextDictionary
        """
        self._word_mapping = word_mapping or {}
        self._id_counter = id_counter or Counter()
        self._normalize_nums = normalize_nums
        self._use_jieba = use_jieba

    @property
    def id_size(self):
        """Get the id size
        """
        return len(self._word_mapping) + self.ID_WordStart

    def lookup_word_id(self, word, min_word_id_count=0):
        """Lookup word id
        """
        if word in self.Numbers and self._normalize_nums:
            return self.ID_Numbers
        _id = self._word_mapping.get(word)
        if _id and min_word_id_count and self._id_counter.get(_id) < min_word_id_count:
            return
        return _id

    def fit(self, s):
        """Fit this dictionary by sentence
        """
        self.to_id(s, fit=True, count=True)

    def to_id(self, s, fit=False, count=False, merge_continguous_nums=True, min_word_id_count=0):
        """Convert sentence to ids
        """
        if fit and min_word_id_count:
            raise ValueError("Cannot set fit and min_word_id_count at the same time")

        words = self._sentence_to_words(s)
        word_ids = []
        for word in words:
            _id = self.lookup_word_id(word, min_word_id_count)
            if not _id:
                if fit:
                    # Fit it
                    _id = len(self._word_mapping) + self.ID_WordStart
                    self._word_mapping[word] = _id
                else:
                    # Unknown
                    _id = self.ID_Unknown
            if merge_continguous_nums and _id == self.ID_Numbers and word_ids and word_ids[-1] == self.ID_Numbers:
                continue
            if count:
                self._id_counter[_id] += 1
            word_ids.append(_id)

        return word_ids

    def save(self, filename):
        """Save this dictionary to the file
        """
        d = { "word_mapping": self._word_mapping, "id_counter": self._id_counter.most_common() }
        with open(filename, "wb") as fd:
            json.dump(d, fd, ensure_ascii=False)

    @classmethod
    def load(cls, filename, normalize_nums=True, use_jieba=False):
        """Load from file
        """
        with open(filename, "rb") as fd:
            d = json.load(fd)
        counter = d.get("id_counter")
        if counter:
            counter = Counter({x: y for x, y in counter})
        return cls(d.get("word_mapping"), counter, normalize_nums, use_jieba)

    def _sentence_to_words(self, s):
        """Convert sentence to words
        """
        return list(s)

def parse_text_line(line):
    """Parse a text line
    """
    line = unicode(line.replace("\xef\xbb\xbf", ""))

    parts = []
    for part in line.split("\t"):
        part = part.strip()
        if part:
            parts.append(part)
    line_no, s1, s2, label = None, None, None, None
    if len(parts) < 3 or len(parts) > 4:
        raise ValueError("Malformed line")
    elif len(parts) == 3:
        line_no, s1, s2 = parts # pylint: disable=unbalanced-tuple-unpacking
        line_no = int(line_no)
        label = None
    else:
        line_no, s1, s2, label = parts # pylint: disable=unbalanced-tuple-unpacking
        line_no = int(line_no)
        label = float(label)

    return line_no, s1, s2, label

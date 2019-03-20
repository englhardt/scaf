#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import gzip
import os
import tempfile
import unittest

from scaf.data import RawGramCorpus, COMPRESSED_FILE_EXT

TEST_COUNT = 100


class TestRawGramCorpus(unittest.TestCase):
    @staticmethod
    def create_tmp_raw_file(word='a b c d e', compressed=False):
        tmp_file = os.path.join(tempfile.gettempdir(), 'test.corpus')
        line = '{}\t2000\t{}\t1\n'.format(word, TEST_COUNT)
        if not compressed:
            with open(tmp_file, 'w') as f:
                f.write(line)
        else:
            tmp_file += COMPRESSED_FILE_EXT
            with gzip.open(tmp_file, 'wt') as f:
                f.write(line)
        return tmp_file

    def test_too_short_word(self):
        tmp_file = self.create_tmp_raw_file(word='a b c')
        c = RawGramCorpus(tmp_file)
        output = [x for x in c]
        self.assertEqual(len(output), 0)

    def test_sampling(self):
        tmp_file = self.create_tmp_raw_file()
        with open(tmp_file, 'a') as f:
            f.write('{}\t2000\t{}\t1'.format('b c d e f', TEST_COUNT))
        c = RawGramCorpus(tmp_file, sampling=TEST_COUNT)
        output = [x for x in c]
        self.assertEqual(len(output), TEST_COUNT)

    def test_standard_mode(self):
        tmp_file = self.create_tmp_raw_file()
        c = RawGramCorpus(tmp_file)
        output = [x for x in c]
        self.assertEqual(len(output), TEST_COUNT)

    def test_scale_mode(self):
        tmp_file = self.create_tmp_raw_file()
        c = RawGramCorpus(tmp_file, mode='scale')
        output = [x for x in c]
        self.assertEqual(len(output), 3)

    def test_ignore_mode(self):
        tmp_file = self.create_tmp_raw_file()
        c = RawGramCorpus(tmp_file, mode='ignore')
        output = [x for x in c]
        self.assertEqual(len(output), 1)

    def test_compressed_corpus(self):
        tmp_file = self.create_tmp_raw_file(compressed=True)
        c = RawGramCorpus(tmp_file)
        output = [x for x in c]
        self.assertEqual(len(output), TEST_COUNT)


if __name__ == '__main__':
    unittest.main()

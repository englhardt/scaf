#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from __future__ import division

import gzip
import io
import logging
import math
import os

from .data_config import WORD_SPLIT_REGEX, COL_SPLIT_REGEX, COMPRESSED_FILE_EXT

AVAILABLE_MODES = ['standard', 'scale', 'ignore']


class RawGramCorpus(object):
    # Google Raw Gram 2012 format
    TEXT = 0
    YEAR = 1
    MATCH_COUNT = 2
    VOLUME_COUNT = 3

    def __init__(self, corpus_path, min_num_words=4, sampling=None, mode='standard'):
        self.corpus_path = corpus_path
        self.min_num_words = min_num_words
        self.sampling = sampling
        self.mode = mode

    def __iter__(self):
        if not os.path.exists(self.corpus_path):
            logging.info("skipping %s", self.corpus_path)
        else:
            cur_count = 0
            if self.corpus_path.endswith(COMPRESSED_FILE_EXT):
                input_file = gzip.open(self.corpus_path, "rt")
            else:
                input_file = io.open(self.corpus_path, "r", encoding="utf-8")
            for line in input_file:
                if self.sampling is not None and cur_count > self.sampling:
                    input_file.close()
                    return
                parts = line.split(COL_SPLIT_REGEX)
                parts[self.TEXT] = parts[self.TEXT].split(WORD_SPLIT_REGEX)
                if len(parts[self.TEXT]) >= self.min_num_words:
                    if self.mode == 'ignore':
                        yield parts[self.TEXT]
                        cur_count += len(parts[self.TEXT])
                    else:
                        times = int(parts[self.MATCH_COUNT])
                        if self.mode == 'scale':
                            times = math.ceil(times / max(1., math.log(times, 2) ** 2))
                        for i in range(times):
                            yield parts[self.TEXT]
                        cur_count += times * len(parts[self.TEXT])
            input_file.close()

    @staticmethod
    def valid_mode(mode):
        return mode in AVAILABLE_MODES

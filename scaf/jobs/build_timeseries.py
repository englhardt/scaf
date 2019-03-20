#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import argparse
import logging
import os
import re
from builtins import map
from builtins import str

from future.utils import iteritems

from scaf.data import BasicEmbedding
from scaf.jobs.alignment import Alignment, AVAILABLE_MODES
from scaf.utils import expand_path


class BuildTimeseries(object):
    def __init__(self, paths, output_file='timeseries', alignment_mode=None, anchor_term_file=None, fixed_point=False,
                 full_vocab=False, vocab_file=None):
        self.paths = paths
        self.output_file = output_file
        self.alignment_mode = alignment_mode
        self.anchor_term_file = anchor_term_file
        self.fixed_point = fixed_point
        self.full_vocab = full_vocab
        self.vocab_file = vocab_file

    def execute(self):
        self.paths = list(map(expand_path, self.paths))
        self.output_file = expand_path(self.output_file)
        if self.vocab_file is not None:
            self.vocab_file = expand_path(self.vocab_file)
        if not self.validate():
            logging.error("[BUILD_TS] Stopping building time series because of invalid parameters.")
            return

        logging.info("[BUILD_TS] Starting building time series.")
        logging.info("[BUILD_TS] Loading embeddings.")
        embeddings = list([BasicEmbedding(x) for x in self.paths])
        alignment_vocab = BasicEmbedding.common_vocab(embeddings)
        logging.info("[BUILD_TS] Finished loading embeddings.")
        if self.alignment_mode:
            logging.info("[BUILD_TS] Starting alignment.")
            alignment = Alignment(self.alignment_mode)
            anchors = None
            if self.anchor_term_file:
                with open(self.anchor_term_file, 'r') as f:
                    anchors = f.readlines()
                    anchors = [x.strip('\n') for x in anchors]
            alignment.align(embeddings[-1], embeddings[:-1], alignment_vocab, anchors)
            logging.info("[BUILD_TS] Finished alignment.")

        series = dict()

        if self.vocab_file is not None:
            with open(self.vocab_file, 'r') as f:
                vocab = f.readlines()
                vocab = set([x.split(',')[0] for x in vocab])
            if not self.full_vocab:
                vocab = vocab.intersection(alignment_vocab)
        elif self.full_vocab:
            vocab = BasicEmbedding.merged_vocab(embeddings)
        else:
            vocab = alignment_vocab
        for w in vocab:
            similarities = []
            if self.fixed_point:
                for i in range(len(self.paths)):
                    if self.full_vocab and (w not in embeddings[i].vocab() or
                                            w not in embeddings[len(self.paths) - 1].vocab()):
                        sim = 1
                    else:
                        w1 = embeddings[i].represent(w)
                        w2 = embeddings[len(self.paths) - 1].represent(w)
                        sim = w1.dot(w2)
                    similarities.append(sim)
            else:
                for i in range(len(self.paths) - 1):
                    if self.full_vocab and (w not in embeddings[i].vocab() or w not in embeddings[i + 1].vocab()):
                        sim = 1
                    else:
                        w1 = embeddings[i].represent(w)
                        w2 = embeddings[i + 1].represent(w)
                        sim = w1.dot(w2)
                    similarities.append(sim)
            series[w] = similarities

        self.store_time_series(series)
        logging.info("[BUILD_TS] Finished building time series.")

    def store_time_series(self, series):
        if os.path.dirname(self.output_file) != '' and not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))
        logging.info("[BUILD_TS] Storing time series in '{}'".format(self.output_file))
        header = ["word"]
        if any(c.isdigit() for c in self.paths[0]):
            if self.fixed_point:
                for p in self.paths:
                    header.append(re.search(r'\d+', os.path.basename(p)).group())
            else:
                for p in self.paths[1:]:
                    header.append(re.search(r'\d+', os.path.basename(p)).group())
        else:
            if self.fixed_point:
                header.extend(list(map(str, range(1, len(self.paths) + 1))))
            else:
                header.extend(list(map(str, range(2, len(self.paths) + 1))))
        lines = [",".join(str(x) for x in [k] + v) for k, v in iter(sorted(iteritems(series)))]
        lines.insert(0, ",".join(header))
        with open(self.output_file, 'w') as f:
            f.writelines("\n".join(lines))

    def validate(self):
        errors = []
        for p in self.paths:
            if not os.path.isfile(p):
                errors.append("[BUILD_TS] Can not read '{}'".format(p))
        if len(self.paths) < 2:
            errors.append("[BUILD_TS] More than one embedding required to build time series")
        if self.alignment_mode and self.alignment_mode not in AVAILABLE_MODES:
            errors.append("[BUILD_TS] Alignment mode '{}' not available.".format(self.alignment_mode))
        if self.anchor_term_file and not os.path.isfile(self.anchor_term_file):
            errors.append("[BUILD_TS] Anchor term file '{}' not available.".format(self.anchor_term_file))
        if not errors:
            return True
        else:
            for e in errors:
                logging.error(e)
            return False


def main():
    parser = argparse.ArgumentParser(prog='SCAF - Build Time Series')
    parser.add_argument('paths', nargs='+', metavar='EMB', help='Embedding files')
    parser.add_argument('-o', '--output', metavar='OUPUT_FILE', help='Output file for series', default="timeseries")
    parser.add_argument('-m', '--alignment_mode', metavar='ALIGNMENT_MODE', help='Alignment mode', default=None)
    parser.add_argument('-a', '--anchor_term_file', metavar='ANCHOR_TERM_FILE', help='Output file for series',
                        default=None)
    parser.add_argument('--fixed_point', help='Fixed point for similarities', action='store_true', default=False)
    parser.add_argument('--full_vocab', help='Use full vocabulary', action='store_true', default=False)
    args = parser.parse_args()

    evaluation = BuildTimeseries(args.paths, args.output, args.alignment_mode, args.anchor_term_file, args.fixed_point,
                                 args.full_vocab)
    evaluation.execute()


if __name__ == "__main__":
    main()

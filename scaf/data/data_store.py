#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from __future__ import print_function

import argparse
import logging
import os
import pickle as pkl

import numpy as np
import pandas as pd

STORE_FILE_EXT = '.store'

BOOKS_FREQ_START_YEAR = 1800
BOOKS_FREQ_LAST_YEAR = 2010
BOOKS_SIM_START_YEAR = 1757
BOOKS_SIM_LAST_YEAR = 2010


class DataStore(dict):
    __getattr__ = dict.__getattribute__
    __setattr__ = dict.__setattr__

    def load_data(self, similarity_path, frequency_path, mode='wiki', fixed_point=False):
        if not os.path.isfile(similarity_path):
            logging.error("Similarity file '{}' does not exist ".format(similarity_path))
            return
        if not os.path.isfile(frequency_path):
            logging.error("Frequency file '{}' does not exist ".format(frequency_path))
            return

        sim_data = pd.read_csv(similarity_path)
        freq_raw_data = pd.read_csv(frequency_path, header=None)

        logging.info('Start bulding temp dict')
        freq_data = dict()
        for r in freq_raw_data.iterrows():
            freq_data[r[1][0]] = r[1][2:].values
        logging.info('Finished building temp dict')

        logging.info('Start building final store')
        for r in sim_data.iterrows():
            word = r[1]['word']
            if word in freq_data:
                if mode == 'wiki':
                    if fixed_point:
                        sim_arr = r[1][1:].values
                    else:
                        sim_arr = np.concatenate([np.array([np.nan]), r[1][1:].values])
                elif mode == 'books':
                    start_year_modifier = 0 if not fixed_point else -5
                    sim_arr = np.interp(np.arange(BOOKS_FREQ_START_YEAR, BOOKS_FREQ_LAST_YEAR),
                                        np.arange(BOOKS_SIM_START_YEAR + start_year_modifier, BOOKS_SIM_LAST_YEAR, 5),
                                        r[1][1:].values.astype(float))
                elif mode == 'books5':
                    sim_arr = r[1][10:].values
                    tmp = freq_data[word].astype(float).reshape(-1, 5)
                    freq_data[word] = np.append(np.mean(tmp[:-1], axis=1), np.mean(tmp[-1][:-1]))
                else:
                    print('Other modes not yet supported')
                    return
                self[word] = np.array([sim_arr, freq_data[word]])
        logging.info('Finished building final store')

    def to_file(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(path):
        with open(path, 'rb') as f:
            return pkl.load(f)

    def apply_transformation(self, func):
        for k, v in self.items():
            self[k] = func(v)

    def filter(self, vocab):
        for k in list(self.keys()):
            if k not in vocab:
                del self[k]


def build_and_save_store(similarity_path, frequency_path, output_path, mode='wiki', fixed_point=False):
    store = DataStore()
    store.load_data(similarity_path, frequency_path, mode, fixed_point)
    store.to_file(output_path)


def main():
    parser = argparse.ArgumentParser(prog='SCAF - Build data store and write to file')
    parser.add_argument('sim_file', metavar='SIM_FILE', help='Path to similarity series file')
    parser.add_argument('freq_file', metavar='FREQ_FILE', help='Path to frequency file')
    parser.add_argument('output_file', metavar='OUTPUT_FILE', help='Path to output file')
    parser.add_argument('-m', '--mode', help='Load model from binary C file', default='wiki')
    parser.add_argument('--fixed_point', help='Similarity series is built with fixed point', action='store_true',
                        default=False)

    args = parser.parse_args()

    build_and_save_store(args.sim_file, args.freq_file, args.output_file, args.mode, args.fixed_point)


if __name__ == "__main__":
    main()

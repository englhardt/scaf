#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from __future__ import print_function

import argparse
import logging
import os
from functools import partial

from scaf.data import DataStore
from scaf.utils import expand_path, prepare_data, relative_frequency, transform_to_cosdist, percentual_diff, \
    normalize_2d, normalize_2d_global


def prepare_store(params, store_file, output_file, mode='wiki'):
    logging.info('Start loading target store.')
    store = DataStore.load_from_file(store_file)
    logging.info('Finished loading target store.')

    logging.info('Applying transformations.')
    store.apply_transformation(partial(prepare_data, mode=mode))
    store.apply_transformation(partial(relative_frequency, mode=mode))
    if params.get('measure') == 'cosdist':
        store.apply_transformation(transform_to_cosdist)
    if params.get('percentual') == 'True':
        store.apply_transformation(percentual_diff)
    if params.get('normalize') == 'True':
        store.apply_transformation(normalize_2d)
    if params.get('normalize') == 'global':
        normalize_2d_global(store)
    logging.info('Finished applying transformations.')

    logging.info('Starting writing transformed store.')
    store.to_file(output_file)
    logging.info('Finished writing transformed store.')


def main():
    parser = argparse.ArgumentParser(prog='SCAF - Prepare store')
    parser.add_argument('config', metavar='CONFIG', help='Transformations to apply.')
    parser.add_argument('store_file', metavar='STORE_FILE', help='Path to store.')
    parser.add_argument('output_path', metavar='OUTPUT_PATH', help='Path to store modified store.')
    parser.add_argument('-m', '--mode', help='Mode to process store.', default='wiki')
    args = parser.parse_args()
    args.store_file = expand_path(args.store_file)
    args.output_path = expand_path(args.output_path)

    if not os.path.isfile(args.store_file):
        print("Store file '{}' does not exist.".format(args.store_file))
        return
    if not os.path.isdir(args.output_path):
        print("Output path '{}' does not exist.".format(args.output_path))
        return
    output_file = os.path.join(args.output_path, os.path.basename(args.store_file))
    prepare_store(args.params, args.store_file, output_file, args.mode)


if __name__ == "__main__":
    main()

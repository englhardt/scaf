#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from __future__ import division, print_function

import argparse
import json
import logging
import os
import re
import timeit
from collections import OrderedDict
from functools import partial

import pandas as pd

from scaf.change_detection import ChangePointDetection, CusumChangeDetection1d, CusumChangeDetection2d
from scaf.data import ARTIFICIAL_CHANGE_INDEX, DataStore
from scaf.utils import prepare_data, relative_frequency, transform_to_cosdist, transform_to_padded_cosdist, \
    percentual_diff, normalize_2d, normalize_2d_global

CD_EVAL_FILE_EXT = '.cd.eval'
MACHINE = 'desktop'


class ChangeDetectionJob(object):
    def __init__(self, config):
        self.model_file = config['model_file']
        self.tp_file = config.get('tp_file')
        self.output_file = config['output_file']
        self.ts_mode = config.get('ts_mode', 'pairwise')
        self.cd_method = config['cd_method']
        self.cd_params = config.get('cd_params', {})
        self.store_transformations = config.get('store_transformations', {})
        self.eval_mode = config.get('eval_mode', 'perturbed')
        self.prepared_store = config.get('prepared_store', False)
        self.store_rank_list = config.get('store_rank_list', False)

    def execute(self):
        # Check if job was already run
        if os.path.isfile(self.output_file):
            result_df = pd.read_csv(self.output_file, keep_default_na=False)
            result_df = result_df.astype(str)
            line = self.format_line()
            if self.check_line(result_df, pd.Series(line)):
                logging.info('[CDJ] Configuration already run.')
                return

        tp_vocab = load_tp_vocab(self.tp_file)
        store = DataStore.load_from_file(self.model_file)
        if tp_vocab and self.eval_mode != 'full':
            logging.info('[CDJ] Filtering store with {}'.format(self.tp_file))
            store.filter(tp_vocab)
        if not self.prepared_store:
            # Transform store
            logging.info('[CDJ] Prepare store.')
            self.transform_store(store)
            logging.info('[CDJ] Finished preparing store.')
        else:
            logging.info('[CDJ] Store is already prepared.')
        logging.info('[CDJ] Init change detection.')
        cd = self.init_cd()

        if 'fixed_point' in self.model_file:
            target_index = ARTIFICIAL_CHANGE_INDEX - 1
        else:
            target_index = ARTIFICIAL_CHANGE_INDEX - 2
        if self.store_transformations.get('percentual') == 'True':
            target_index -= 1

        logging.info('[CDJ] Starting change detection.')
        start_time = timeit.default_timer()
        score, result, [tp, fp, fn] = process_wiki(store, cd, tp_vocab, target_index,
                                                   self.output_file if self.store_rank_list else None)
        tot_time = timeit.default_timer() - start_time
        logging.info('[CDJ] Finished change detection.')
        results = self.format_full_result(score, tp, fp, fn, tot_time)
        line = self.format_line(results)
        logging.info('[CDJ] Storing result.')
        self.store_results(pd.DataFrame(line, index=[0]))
        logging.info('[CDJ] Finished storing result.')

    def format_line(self, result=None):
        line = OrderedDict()
        line['model'] = re.findall('(cbhs|cbns|sghs|sgns)', self.model_file)[0]
        line['perturb'] = re.findall('_(\d\.\d|slow|mid|fast).store', self.model_file)[0]
        line['align'] = next(iter(re.findall('(inc|procrustes|linear)', self.model_file)), '')
        line['anchors'] = next(iter(re.findall('anchors/(\w+)/', self.model_file)), '')
        line['ts_mode'] = 'fixed_point' if 'fixed_point' in self.model_file else 'pairwise'
        line['cd_method'] = self.cd_method
        line['measure'] = self.store_transformations.get('measure', 'cossim')
        line['percentual'] = self.store_transformations.get('percentual', 'False')
        line['normalize'] = self.store_transformations.get('normalize', 'False')
        line['samples'] = str(self.cd_params.get('samples', ''))
        line['eval_mode'] = self.eval_mode or ''
        line['eps'] = self.cd_params.get('eps', '')

        if result:
            for k, v in result.items():
                line[k] = v
            line['machine'] = MACHINE
        return line

    @staticmethod
    def check_line(eval_data, line):
        last_index = -6
        return len(eval_data[((line == eval_data.iloc[:, :last_index]) |
                              (pd.isnull(eval_data.iloc[:, :last_index]) & pd.isnull(line))).all(1)].any(1)) > 0

    @staticmethod
    def format_full_result(score, tp, fp, fn, time):
        full_results = OrderedDict()
        full_results['score'] = score
        full_results['tp'] = tp
        full_results['fp'] = fp
        full_results['fn'] = fn
        full_results['time'] = time
        return full_results

    def transform_store(self, store):
        prep_mode = 'fixed_point' if 'fixed_point' in self.model_file else 'wiki'
        store.apply_transformation(partial(prepare_data, mode=prep_mode))
        store.apply_transformation(relative_frequency)
        if self.store_transformations.get('measure') == 'cosdist':
            store.apply_transformation(transform_to_cosdist)
        elif self.store_transformations.get('measure') == 'padcosdist':
            store.apply_transformation(transform_to_padded_cosdist)
        if self.store_transformations.get('percentual') == 'True':
            store.apply_transformation(percentual_diff)
        if self.store_transformations.get('normalize') == 'True':
            store.apply_transformation(normalize_2d)
        if self.store_transformations.get('normalize') == 'global':
            normalize_2d_global(store)

    def init_cd(self):
        if self.cd_method == 'cp_package':
            samples = self.cd_params.get('samples')
            eps = self.cd_params.get('eps')
            return ChangePointDetection(float(eps) if eps else None, int(samples) if samples else 5000)
        elif 'cusum' in self.cd_method and '1d' in self.cd_method:
            return CusumChangeDetection1d()
        else:
            return CusumChangeDetection2d()

    def store_results(self, results):
        if os.path.isfile(self.output_file):
            with open(self.output_file, 'a') as f:
                results.to_csv(f, header=False, index=False)
        else:
            with open(self.output_file, 'w') as f:
                results.to_csv(f, index=False)


def load_tp_vocab(tp_file):
    if tp_file:
        with open(tp_file, 'r') as f:
            vocab = f.readlines()
            vocab = set([x.strip('\n') for x in vocab])
            return vocab


def process_wiki(store, cd, tp_vocab, target_index=4, output_file=None):
    index_modifier = ARTIFICIAL_CHANGE_INDEX - target_index
    result = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    tp = 0
    fp = 0
    scores = []

    count = 0
    for (w, d) in store.items():
        if count % 1000 == 0:
            logging.info('[CDJ] Status {:08.4f}%'.format(float(count) / len(store) * 100))
        count += 1
        cp, _ = cd.detect(d)
        if len(cp) == 0:
            scores.append((w, 1, cd.fail_score()))
        else:
            for c in cp:
                result[str(c[0] + index_modifier)] += 1
                if w in tp_vocab:
                    if c[0] == target_index:
                        tp += 1
                        scores.append((w, c[0] + index_modifier, c[1]))
                    else:
                        fp += 1
                        scores.append((w, c[0] + index_modifier, cd.fail_score()))
                else:
                    scores.append((w, c[0] + index_modifier, c[1]))
    logging.info('[CDJ] Status {:08.4f}%'.format(100))
    fn = len([k for k in tp_vocab if k in store]) - tp - fp

    scores_df = pd.DataFrame(scores, columns=['word', 'time', 'score'])
    scores_df['rank'] = scores_df['score'] \
        .rank(method='average', ascending=cd.sort_order_ascending(), na_option='bottom')
    if output_file is not None:
        scores_df.sort_values('score', ascending=cd.sort_order_ascending()) \
            .to_csv(output_file.replace(CD_EVAL_FILE_EXT, '.ranked'), index=False)
    mrr_score = 1.0 / len(tp_vocab) * sum(1. / (scores_df[scores_df['word'].isin(tp_vocab)]['rank']))
    return mrr_score, result, [tp, fp, fn]


def main():
    parser = argparse.ArgumentParser(prog='SCAF - Change detection')
    parser.add_argument('config', metavar='CONFIG', help='Configuration file.')
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        print("Config file '{}' does not exist.".format(args.config))
        return
    with open(args.config, 'r') as f:
        config = json.load(f)
    cdj = ChangeDetectionJob(config)
    cdj.execute()


if __name__ == "__main__":
    main()

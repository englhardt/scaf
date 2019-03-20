#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT


# Replacement of every second 'the' with 'in':
# sed -e "s/\bthe\b/in/g;n" lee_corpus > lee_corpus_modified

import logging
import os
import shutil
import unittest

import pandas as pd

from scaf.data import DataStore
from scaf.jobs import Training, EmbeddingEvaluation, BuildTimeseries, ChangeDetectionJob

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

module_path = os.path.dirname(__file__)
data_file = lambda file_name: os.path.join(module_path, 'test_data', file_name)  # noqa
output_path = os.path.join(module_path, 'test_data', 'output')
output_file = lambda file_name: os.path.join(output_path, file_name)  # noqa
embedding_eval_files = os.path.join(module_path, 'test_data', 'embedding_eval_files')
similarity_file = output_file('sim.ts')
frequency_file = output_file('freq.ts')
store_file = output_file('sgns_procrustes_0.5.store')
changed_vocab_file = data_file('changed_vocab')
cd_eval_file = output_file('result.cd.eval')
cd_ranked_file = output_file('result.ranked')

# Corpora
# Train 5 times lee.ngrams
# Train 5 times lee_modified.ngrams
corpus_files = ['lee.ngrams', 'lee_modified.ngrams']
corpus_freq_files = ['lee.freq', 'lee_modified.freq']
changed_word = 'in'

embedding_config = {
    'input': '',
    'output': '',
    'corpus_building_mode': 'ignore',
    'gensim_params': {
        'size': 25,
        'sg': 1,
        'negative': 5
    }
}

change_detection_config = {
    'model_file': store_file,
    'tp_file': changed_vocab_file,
    'output_file': cd_eval_file,
    'cd_method': 'cusum_2d',
    'store_transformations': {
        'measure': 'padcosdist',
        'percentual': 'True',
        'normalize': 'False'
    },
    'eval_mode': 'full',
    'store_rank_list': True
}


class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        # Clean up old files
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

    @staticmethod
    def train_model(corpus, init_model=None, manual_name=None):
        config = embedding_config.copy()
        config['input'] = data_file(corpus)
        config['output_path'] = output_path
        if init_model is not None:
            config['init_vectors'] = os.path.join(output_path, init_model)
        t = Training(config)
        t.execute()

        if manual_name is not None:
            target_dir = os.path.join(output_path, corpus)
            [os.rename(os.path.join(target_dir, f), os.path.join(target_dir, f.replace(corpus, manual_name)))
             for f in os.listdir(target_dir) if f.startswith(corpus)]
            os.rename(target_dir, os.path.join(output_path, manual_name))
            return os.path.join(output_path, manual_name, '{}_model'.format(manual_name))
        return os.path.join(output_path, corpus, '{}_model'.format(corpus))

    @staticmethod
    def eval_model(model):
        e = EmbeddingEvaluation(model, embedding_eval_files)
        e.execute()

    def incremental_models(self):
        models = []
        model_file = self.train_model(corpus_files[0], manual_name='1')
        self.eval_model(model_file)
        models.append(model_file)
        for i in range(2, 6):
            model_file = self.train_model(corpus_files[0], '{}/{}_model'.format(i - 1, i - 1), '{}'.format(i))
            self.eval_model(model_file)
            models.append(model_file)
        for i in range(6, 11):
            model_file = self.train_model(corpus_files[1], manual_name='{}'.format(i))
            self.eval_model(model_file)
            models.append(model_file)
        return models

    def procrustes_models(self):
        models = []
        for i in range(1, 6):
            model_file = self.train_model(corpus_files[0], manual_name='{}'.format(i))
            self.eval_model(model_file)
            models.append(model_file)
        for i in range(6, 11):
            model_file = self.train_model(corpus_files[1], manual_name='{}'.format(i))
            self.eval_model(model_file)
            models.append(model_file)
        return models

    @staticmethod
    def build_sim_ts(models, alignment_mode=None):
        b = BuildTimeseries(models, output_file=similarity_file, alignment_mode=alignment_mode)
        b.execute()

    @staticmethod
    def build_freq_ts():
        original = pd.read_csv(data_file(corpus_freq_files[0]), sep='\t',
                               names=('word', 'year', 'match_count', 'volume_count'), quoting=3)
        modified = pd.read_csv(data_file(corpus_freq_files[1]), sep='\t',
                               names=('word', 'year', 'match_count', 'volume_count'), quoting=3)
        merged = original[['word', 'match_count']].join(modified[['word', 'match_count']].set_index('word'),
                                                        on='word', rsuffix='_b')
        merged['word_type'] = 'X'
        for i in range(1, 6):
            merged[str(i)] = merged['match_count']
        for i in range(6, 11):
            merged[str(i)] = merged['match_count_b']
        del merged['match_count']
        del merged['match_count_b']
        merged.to_csv(frequency_file, index=False, quoting=3, header=None)

    @staticmethod
    def build_store():
        store = DataStore()
        store.load_data(similarity_file, frequency_file)
        store.to_file(store_file)

    @staticmethod
    def run_change_detection():
        job = ChangeDetectionJob(change_detection_config)
        job.execute()

    def eval_detection(self):
        df = pd.read_csv(cd_ranked_file)
        d = df.iloc[0, :]
        self.assertEqual(d[0], changed_word)
        self.assertEqual(d[1], 6)

    def run_pipeline(self, incremental=False):
        models = self.incremental_models() if incremental else self.procrustes_models()
        self.build_sim_ts(models, alignment_mode=None if incremental else 'procrustes')
        self.build_freq_ts()
        self.build_store()
        self.run_change_detection()
        self.eval_detection()

    def test_incremental(self):
        self.run_pipeline(incremental=True)

    def test_procrustes(self):
        self.run_pipeline(incremental=False)


if __name__ == '__main__':
    unittest.main()

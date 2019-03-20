#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from __future__ import division, print_function

import argparse
import json
import logging
import operator
import os
from enum import Enum

import numpy as np
from gensim.matutils import unitvec
from gensim.models import Word2Vec, KeyedVectors

from scaf.data import COMPRESSED_FILE_EXT
from scaf.data import RawGramCorpus
from scaf.utils import expand_path


class ModelType(Enum):
    __order__ = 'SGNS SGHS CBNS CBHS'
    SGNS = 1
    SGHS = 2
    CBNS = 3
    CBHS = 4


class Training(object):
    def __init__(self, config, save_snaps=False):
        self.input = None
        self.output_path = None
        self.init_vectors = None
        self.corpus_sampling = 1e9
        self.corpus_building_mode = 'standard'
        self.convergence_abort = True
        self.convergence_threshold = 1e-4
        self.save_snaps = save_snaps
        self.gensim_params = {
            'workers': 8,
            'size': 200,
            'iter': 5,
            'sample': 1e-5,
            'max_vocab_size': 2e5,
            'min_count': 1,
            'window': 4,
            'alpha': 0.025,
            'min_alpha': 0.0001,
            'sg': 0,
            'negative': 5,
            'hs': 0
        }
        if 'input' in config:
            self.input = config['input']
        if 'output_path' in config:
            self.output_path = config['output_path']
        if 'init_vectors' in config:
            self.init_vectors = config['init_vectors']
        if 'corpus_sampling' in config:
            self.corpus_sampling = config['corpus_sampling']
        if 'corpus_building_mode' in config:
            self.corpus_building_mode = config['corpus_building_mode']
        if 'convergence_abort' in config:
            self.convergence_abort = config['convergence_abort']
        if 'convergence_threshold' in config:
            self.convergence_threshold = config['convergence_threshold']
        if 'gensim_params' in config:
            for param in config['gensim_params']:
                self.gensim_params[param] = config['gensim_params'][param]

    def execute(self):
        self.input = expand_path(self.input)
        self.output_path = expand_path(self.output_path)
        self.init_vectors = expand_path(self.init_vectors)
        output_file_names = self.generate_output_folders()
        self.setup_train_env(output_file_names)

        if not self.validate():
            logging.error("[TRAIN] Stopping training because configuration is invalid.")
            return

        model = self.train(output_file_names)

        logging.info("[TRAIN] Training finished. Writing output file.")
        model.wv.save_word2vec_format(output_file_names + "_model")
        logging.info("[TRAIN] Export finished.")

    def train(self, output_file_names):
        corpus = RawGramCorpus(self.input, sampling=self.corpus_sampling, mode=self.corpus_building_mode)
        max_epochs = self.gensim_params['iter']
        self.gensim_params['iter'] = 1
        epoch = 0
        previous_vectors = None

        model = Word2Vec(**self.gensim_params)
        self.add_corpus(model, corpus)

        if self.init_vectors is not None:
            logging.info("[TRAIN] Loading init model.")
            init_model = KeyedVectors.load_word2vec_format(self.init_vectors)
            for word in init_model.vocab:
                if word in model.wv.vocab:
                    model.wv.syn0[model.wv.vocab[word].index] = init_model.syn0[init_model.vocab[word].index]
            previous_vectors = np.empty_like(model.wv.syn0)
            previous_vectors[:] = model.wv.syn0
            logging.info("[TRAIN] Initialized vectors from init model.")

        while epoch < max_epochs and (epoch < 1 or not self.check_convergence(model, previous_vectors)):
            epoch += 1
            logging.info("[TRAIN] Training epoch {}.".format(epoch))
            if epoch > 1:
                previous_vectors = np.copy(model.wv.syn0)
            model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
            if self.save_snaps:
                model.save_word2vec_format(output_file_names + "_{}_model".format(epoch))

        return model

    def add_corpus(self, model, corpus):
        logging.info("[TRAIN] Start building corpus.")
        model.min_count = 1
        model.max_vocab_size = 50e6
        model.scan_vocab(corpus)
        sorted_vocab = sorted(list(model.raw_vocab.items()), key=operator.itemgetter(1), reverse=True)
        for k, _ in sorted_vocab[self.gensim_params['max_vocab_size']:]:
            del model.raw_vocab[k]
        model.scale_vocab()
        model.finalize_vocab()

    def check_convergence(self, cur_embedding, previous_vectors):
        if not self.convergence_abort or previous_vectors is None:
            return False
        dist = sum([np.dot(unitvec(cur_embedding.wv.syn0[i]), unitvec(
            previous_vectors[i])) for i in range(len(cur_embedding.wv.vocab))]) / len(cur_embedding.wv.vocab)
        logging.info("[TRAIN] Current distance is {}.".format(dist))
        return dist > 1 - self.convergence_threshold

    def generate_output_folders(self):
        output_filename = os.path.basename(self.input)
        if output_filename.endswith(COMPRESSED_FILE_EXT):
            output_filename = output_filename.replace(COMPRESSED_FILE_EXT, "")
        output_folder = os.path.join(self.output_path, output_filename)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        return os.path.join(output_folder, output_filename)

    def setup_train_env(self, output_file_names):
        log_format = '%(asctime)s : %(levelname)s : %(message)s'
        logging.basicConfig(format=log_format, level=logging.INFO)
        logger = logging.getLogger()
        log_file_name = output_file_names + "_train.log"
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        file_logger = logging.FileHandler(log_file_name)
        file_logger.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_logger)
        with open(output_file_names + "_train.cfg", 'w') as outfile:
            json.dump(self.format_config(), outfile, indent=4)

    def format_config(self):
        return {
            'input': self.input,
            'output_path': self.output_path,
            'init_vectors': self.init_vectors,
            'corpus_sampling': self.corpus_sampling,
            'corpus_building_mode': self.corpus_building_mode,
            'gensim_params': self.gensim_params
        }

    def validate(self):
        errors = []
        if self.input is None:
            errors.append("[PARAM] Input file not specified")
        if not os.path.isfile(self.input):
            errors.append("[PARAM] Input file does not exist ('{}')".format(self.input))
        if self.output_path is None:
            errors.append("[PARAM] Output path not specified")
        if self.init_vectors is not None and not os.path.isfile(self.init_vectors):
            errors.append("[PARAM] Init vectors file does not exist ('{}')".format(self.init_vectors))
        if self.corpus_sampling is not None:
            if isinstance(self.corpus_sampling, float):
                self.corpus_sampling = int(self.corpus_sampling)
            if isinstance(self.corpus_sampling, int):
                if self.corpus_sampling <= 0:
                    errors.append("[PARAM] Corpus sampling value is not positive ('{}')".format(self.corpus_sampling))
            else:
                errors.append("[PARAM] Parameter 'corpus_sampling' is not an integer.")
        if isinstance(self.corpus_building_mode, str):
            if not RawGramCorpus.valid_mode(self.corpus_building_mode):
                errors.append("[PARAM] Corpus building mode unknown ('{}')".format(self.corpus_building_mode))
        else:
            errors.append("[PARAM] Parameter 'corpus_building_mode' is not a string.")
        if not isinstance(self.convergence_abort, bool):
            errors.append("[PARAM] Convergence abort is not a boolean ('{}')".format(self.convergence_abort))
        if self.convergence_abort and isinstance(self.convergence_threshold, (float, int)):
            if self.convergence_threshold <= 0 or self.convergence_threshold >= 1:
                errors.append("[PARAM] Invalid convergence threshold ('{}')".format(self.convergence_threshold))
        else:
            errors.append("[PARAM] Parameter 'convergence_threshold' is not a number.")
        if isinstance(self.gensim_params['size'], int):
            if self.gensim_params['workers'] < 1:
                errors.append("[PARAM] Invalid number of workers ('{}')".format(self.gensim_params['workers']))
        else:
            errors.append("[PARAM] Gensim parameter 'workers' is not an integer.")
        if isinstance(self.gensim_params['size'], int):
            if self.gensim_params['size'] < 1:
                errors.append("[PARAM] Invalid size of word vectors ('{}')".format(self.gensim_params['size']))
        else:
            errors.append("[PARAM] Gensim parameter 'size' is not an integer.")
        if isinstance(self.gensim_params['iter'], int):
            if self.gensim_params['iter'] < 1:
                errors.append("[PARAM] Invalid number of iterations ('{}')".format(self.gensim_params['iter']))
        else:
            errors.append("[PARAM] Gensim parameter 'iter' is not an integer.")
        if isinstance(self.gensim_params['sample'], (float, int)):
            if self.gensim_params['sample'] < 0:
                errors.append("[PARAM] Invalid higher-frequency words down sampling rate ('{}')"
                              .format(self.gensim_params['sample']))
        else:
            errors.append("[PARAM] Gensim parameter 'sample' is not a number.")
        if isinstance(self.gensim_params['max_vocab_size'], float):
            self.gensim_params['max_vocab_size'] = int(self.gensim_params['max_vocab_size'])
        if isinstance(self.gensim_params['max_vocab_size'], int):
            if self.gensim_params['max_vocab_size'] < 1:
                errors.append("[PARAM] Invalid maximum vocabulary size ('{}')"
                              .format(self.gensim_params['max_vocab_size']))
        else:
            errors.append("[PARAM] Gensim parameter 'max_vocab_size' is not an integer.")
        if isinstance(self.gensim_params['window'], int):
            if self.gensim_params['window'] < 1:
                errors.append("[PARAM] Invalid window size ('{}')".format(self.gensim_params['window']))
        else:
            errors.append("[PARAM] Gensim parameter 'window' is not an integer.")
        if isinstance(self.gensim_params['alpha'], (float, int)):
            if self.gensim_params['alpha'] <= 0:
                errors.append("[PARAM] Invalid learning rate value alpha ('{}')".format(self.gensim_params['alpha']))
        else:
            errors.append("[PARAM] Gensim parameter 'alpha' is not a number.")
        if isinstance(self.gensim_params['min_alpha'], (float, int)):
            if self.gensim_params['min_alpha'] <= 0:
                errors.append("[PARAM] Invalid minimum learning rate value alpha ('{}')"
                              .format(self.gensim_params['min_alpha']))
        else:
            errors.append("[PARAM] Gensim parameter 'min_alpha' is not a number.")
        if isinstance(self.gensim_params['sg'], int):
            if self.gensim_params['sg'] != 0 and self.gensim_params['sg'] != 1:
                errors.append("[PARAM] Invalid skip gram configuration value ('{}')".format(self.gensim_params['sg']))
        else:
            errors.append("[PARAM] Gensim parameter 'sg' is not a number.")

        if isinstance(self.gensim_params['negative'], int):
            if self.gensim_params['negative'] < 0:
                errors.append("[PARAM] Invalid negative sampling configuration value ('{}')"
                              .format(self.gensim_params['negative']))
        else:
            errors.append("[PARAM] Gensim parameter 'negative' is not a number.")

        if isinstance(self.gensim_params['hs'], int):
            if self.gensim_params['hs'] != 0 and self.gensim_params['hs'] != 1:
                errors.append("[PARAM] Invalid hierarchical softmax configuration value ('{}')"
                              .format(self.gensim_params['hs']))
        else:
            errors.append("[PARAM] Gensim parameter 'hs' is not a number.")

        if not errors:
            return True
        else:
            for e in errors:
                logging.error(e)
            return False

    @staticmethod
    def get_filled_config(corpus, output_path, init_model=None, corpus_building_mode='standard',
                          model_type=ModelType.SGNS, worker=8):
        config = {
            'input': corpus,
            'output_path': output_path,
            'init_vectors': None,
            'corpus_building_mode': corpus_building_mode,
            'gensim_params': {
                "workers": worker,
            }
        }
        if model_type == ModelType.SGHS:
            config['gensim_params']['sg'] = 1
            config['gensim_params']['negative'] = 0
            config['gensim_params']['hs'] = 1
            config['output_path'] = os.path.join(config['output_path'], "sghs")
            if init_model is not None:
                init_model = init_model.replace('MODEL', 'sghs')
        elif model_type == ModelType.CBHS:
            config['gensim_params']['sg'] = 0
            config['gensim_params']['negative'] = 0
            config['gensim_params']['hs'] = 1
            config['output_path'] = os.path.join(config['output_path'], "cbhs")
            if init_model is not None:
                init_model = init_model.replace('MODEL', 'cbhs')
        elif model_type == ModelType.CBNS:
            config['gensim_params']['sg'] = 0
            config['gensim_params']['negative'] = 5
            config['gensim_params']['hs'] = 0
            config['output_path'] = os.path.join(config['output_path'], "cbns")
            if init_model is not None:
                init_model = init_model.replace('MODEL', 'cbns')
        else:
            config['gensim_params']['sg'] = 1
            config['gensim_params']['negative'] = 5
            config['gensim_params']['hs'] = 0
            config['output_path'] = os.path.join(config['output_path'], "sgns")
            if init_model is not None:
                init_model = init_model.replace('MODEL', 'sgns')

        if init_model is not None:
            config['init_vectors'] = init_model
            config['output_path'] += "_inc"

        return config


def main():
    parser = argparse.ArgumentParser(prog='SCAF - Enqueue Training')
    parser.add_argument('config', metavar='CONFIG', help='Training config to enqueue')
    parser.add_argument('--save_snaps', help='Save each epoch snapshot', action='store_true', default=False)
    args = parser.parse_args()
    config = None
    if os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    if config is None:
        print("Failed loading input config.")
        return

    training = Training(config, args.save_snaps)
    training.execute()


if __name__ == "__main__":
    main()

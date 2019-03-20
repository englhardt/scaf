#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from __future__ import division

import argparse
import logging
import os
from builtins import zip

from scipy.stats.stats import spearmanr

from scaf.data import Embedding
from scaf.utils import expand_path

ANALOGY_FOLDER = "analogy"
WS_FOLDER = "ws"
EMB_EVAL_FILE_EXT = ".emb.eval"


class EmbeddingEvaluation(object):
    def __init__(self, model, eval_file_path, binary=False):
        self.model_path = model
        self.binary = binary
        self.eval_file_path = eval_file_path

    def execute(self):
        self.model_path = expand_path(self.model_path)
        self.eval_file_path = expand_path(self.eval_file_path)
        if not self.validate():
            logging.error("[EVAL] Stopping evaluation because of invalid parameters.")
            return
        results = []
        embedding = Embedding(self.model_path, binary=self.binary)
        logging.info("[EVAL] Starting evaluation.")

        analogy_path = os.path.join(self.eval_file_path, ANALOGY_FOLDER)
        if os.path.isdir(analogy_path) and len(os.listdir(analogy_path)):
            for f in os.listdir(analogy_path):
                cur_result = self.eval_analogy(embedding, os.path.join(analogy_path, f))
                for r in cur_result:
                    results.append("{},{},{},{}\n".format(self.model_path, ANALOGY_FOLDER, f[:-4], r))
                    logging.info("[EVAL] {}".format(results[-1]))

        ws_path = os.path.join(self.eval_file_path, WS_FOLDER)
        if os.path.isdir(ws_path) and len(os.listdir(ws_path)):
            for f in os.listdir(ws_path):
                cur_result = self.eval_ws(embedding, self.load_ws_file(os.path.join(ws_path, f)))
                results.append("{},{},{},{}\n".format(self.model_path, WS_FOLDER, f[:-4], cur_result))
                logging.info("[EVAL] {}".format(results[-1]))

        self.store_evaluation_result(results)
        logging.info("[EVAL] Finished evaluation.")

    def store_evaluation_result(self, results):
        if '_model' in self.model_path:
            eval_file_name = self.model_path.replace("_model", EMB_EVAL_FILE_EXT)
        else:
            eval_file_name = self.model_path + EMB_EVAL_FILE_EXT
        logging.info("[EVAL] Storing evaluation in '{}'".format(eval_file_name))
        with open(eval_file_name, "w") as outfile:
            outfile.writelines(results)

    def validate(self):
        errors = []
        if self.model_path is None:
            errors.append("[PARAM] Model file not specified")
        if not os.path.isfile(self.model_path):
            errors.append("[PARAM] Model file does not exist ('{}')".format(self.model_path))
        if self.eval_file_path is None:
            errors.append("[PARAM] Evaluation file path not specified")
        if not os.path.isdir(self.eval_file_path):
            errors.append("[PARAM] Evaluation file path does not exist ('{}')".format(self.eval_file_path))
        if not errors:
            return True
        else:
            for e in errors:
                logging.error(e)
            return False

    @staticmethod
    def eval_analogy(model, eval_file):
        result = []
        model_result = model.eval_analogy(eval_file)
        for v in model_result:
            if float(len(v['correct']) + len(v['incorrect'])) == 0:
                accuracy = 0
            else:
                accuracy = len(v['correct']) / float(len(v['correct']) + len(v['incorrect']))
            result.append("{},{},0".format(v['section'], accuracy))
        return result

    @staticmethod
    def load_ws_file(path):
        test = []
        with open(path) as f:
            for line in f:
                x, y, sim = line.strip().lower().split()
                test.append(((x.split("-")[0], y.split("-")[0]), float(sim)))
        return test

    @staticmethod
    def eval_ws(embedding, data):
        results = []
        oov = 0
        for (x, y), sim in data:
            if embedding.oov(x) or embedding.oov(y):
                oov += 1
                results.append((0, sim))
            else:
                results.append((embedding.similarity(x, y), sim))
        actual, expected = list(zip(*results))
        correlation = spearmanr(actual, expected)[0]
        return "{},{},{}".format("total", correlation, oov)


def main():
    parser = argparse.ArgumentParser(prog='SCAF - Evaluate Embedding Model')
    parser.add_argument('model', metavar='MODEL', help='Target model to evaluate')
    parser.add_argument('eval_file_path', metavar='EVAL_FILE_PATH', help='Path to folder with evaluation files')
    parser.add_argument('--binary', dest='binary', action='store_true', help='Load model from binary C file')
    args = parser.parse_args()

    evaluation = EmbeddingEvaluation(args.model, args.eval_file_path, args.binary)
    evaluation.execute()


if __name__ == "__main__":
    main()

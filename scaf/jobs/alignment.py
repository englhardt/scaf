#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import logging

import numpy as np
from sklearn.linear_model import LinearRegression

AVAILABLE_MODES = ['linear', 'procrustes']


class Alignment(object):
    def __init__(self, mode='linear', worker=4):
        self.mode = mode
        self.anchors = None
        self.worker = worker

    def align(self, base, others, vocab, anchors=None):
        if not anchors:
            anchors = vocab
        logging.info('[ALIGN] Starting alignment with mode {}'.format(self.mode))
        base_anchor_matrix = self.build_matrix(base, anchors)
        for i, o in enumerate(others):
            logging.info('[ALIGN] Status {}%'.format(float(i) / len(others) * 100))
            o_anchor_matrix = self.build_matrix(o, anchors)
            o_matrix = self.build_matrix(o, vocab)
            if self.mode == 'linear':
                aligned = self.align_linear(base_anchor_matrix, o_anchor_matrix, o_matrix)
            elif self.mode == 'procrustes':
                aligned = self.align_procrustes(base_anchor_matrix, o_anchor_matrix, o_matrix)
            else:
                raise Exception('Alignment mode {} not found'.format(self.mode))
            o.update_model(vocab, aligned)
        logging.info('[ALIGN] Status {}%'.format(100))

    @staticmethod
    def build_matrix(embedding, vocab):
        m = np.empty((len(vocab), embedding.vector_size))
        for i, w in enumerate(vocab):
            m[i] = embedding.represent(w)
        return m

    def align_linear(self, base_anchors, other_anchors, other):
        reg = LinearRegression(n_jobs=self.worker)
        reg.fit(other_anchors, base_anchors)
        return reg.predict(other)

    @staticmethod
    def align_procrustes(base_anchors, other_anchors, other):
        base_mean_sub = base_anchors - base_anchors.mean(0)
        other_mean_sub = other_anchors - other_anchors.mean(0)
        m = other_mean_sub.T.dot(base_mean_sub)
        u, _, v = np.linalg.svd(m)
        r = u.dot(v)
        return other.dot(r)

    @staticmethod
    def valid_mode(mode):
        return mode in AVAILABLE_MODES

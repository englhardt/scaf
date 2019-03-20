#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import logging
import os
from builtins import map

import numpy as np
from gensim.models import KeyedVectors


class Embedding(object):
    def __init__(self, path, binary=False):
        if os.path.isfile(path):
            self.model = KeyedVectors.load_word2vec_format(path, binary=binary)
        else:
            logging.error('Model \'{}\' can not be loaded.'.format(path))
            return
        self.model.init_sims(replace=True)

    def represent(self, word):
        if word in self.model.vocab:
            return self.model.syn0[self.model.index2word.index(word)]
        else:
            return np.zeros(self.model.vector_size)

    def similarity(self, word1, word2):
        """
        Vectors are supposed to be normalized
        """
        return self.represent(word1).dot(self.represent(word2))

    def most_similar(self, positive=(), negative=(), n=10):
        """
        Vectors are supposed to be normalized
        """
        return self.model.most_similar(positive=positive, negative=negative, topn=n)

    def most_similar_to_word(self, word, n=10):
        """
        Vectors are supposed to be normalized
        """
        return self.model.most_similar(positive=[word], topn=n)

    def oov(self, word):
        return word not in self.model.vocab

    def eval_analogy(self, eval_file):
        return self.model.accuracy(eval_file, case_insensitive=True)

    def vocab(self):
        return self.model.vocab

    def model(self):
        return self.model


class BasicEmbedding(object):
    def __init__(self, path):
        if not os.path.isfile(path):
            logging.error('Model \'{}\' can not be loaded.'.format(path))
            return

        self.model = dict()
        self.vector_size = 0
        with open(path) as f:
            self.vector_size = int(f.readline().split()[1])
            for l in f:
                word_splits = l.split()
                word = word_splits[0]
                series = list(map(float, word_splits[1:]))
                series /= np.linalg.norm(series)
                self.model[word] = series

    def update_model(self, vocab, vectors, normalize=False):
        self.model = dict()
        for i, w in enumerate(vocab):
            self.model[w] = vectors[i]
            if normalize:
                self.model[w] /= np.linalg.norm(self.model[w])

    def vocab(self):
        return list(self.model.keys())

    def represent(self, word):
        return self.model.get(word, np.zeros(self.vector_size))

    def vector_size(self):
        return self.vector_size

    @staticmethod
    def common_vocab(embeddings):
        if not embeddings or len(embeddings) == 0:
            return []
        if len(embeddings) == 1:
            return embeddings[0].vocab()
        intersected_vocab = set(embeddings[0].vocab())
        for e in embeddings[1:]:
            intersected_vocab &= set(e.vocab())
        return intersected_vocab

    @staticmethod
    def merged_vocab(embeddings):
        if not embeddings or len(embeddings) == 0:
            return []
        if len(embeddings) == 1:
            return embeddings[0].vocab()
        merged_vocab = set(embeddings[0].vocab())
        for e in embeddings[1:]:
            merged_vocab |= set(e.vocab())
        return merged_vocab

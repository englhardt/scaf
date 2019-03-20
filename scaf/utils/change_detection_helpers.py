#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT
from __future__ import division, print_function

import math

import numpy as np
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d

from scaf.data import WIKI_TOTAL_COUNT, BOOKS_TOTAL_COUNT, BOOKS_AVG_COUNT


def normalize(series):
    """
    Normalize an array
    :param series: imput array
    :return: New normalized array with mean 0 and std 1
    """
    return (series - np.mean(series)) / np.std(series)


def normalize_2d(data):
    """
    Normalize a 2d array by normalizing each row (no side effects)
    :param data: Data 2d array with similarity and frequency (no NaN values)
    :return: New normalized 2d array with rows having mean 0 and std 1
    """
    return (data - np.mean(data, axis=1)[:, np.newaxis]) / np.std(data, axis=1)[:, np.newaxis]


def normalize_2d_global(store):
    """
    Normalizes all data in a store globally (with side effects)
    :param store: Input store to normalize
    :return: Normalized store
    """
    s0 = 0
    s1 = [0, 0]
    s2 = [0, 0]
    for v in store.values():
        if not np.isnan(v).any() and np.isfinite(v).all():
            s0 += len(v[0])
            for i in range(2):
                s1[i] += np.sum(v[i])
                s2[i] += np.sum(v[i] * v[i])

    if s0 == 0:
        print('No finite series or series that contain no NaN')
        return
    sims_mean = s1[0] / s0
    sims_std = math.sqrt((s0 * s2[0] - s1[0] * s1[0]) / (s0 * (s0 - 1)))
    freqs_mean = s1[1] / s0
    freqs_std = math.sqrt((s0 * s2[1] - s1[1] * s1[1]) / (s0 * (s0 - 1)))

    for v in store.values():
        v[0] = (v[0] - sims_mean) / sims_std
        v[1] = (v[1] - freqs_mean) / freqs_std


def prepare_data(data, mode='wiki'):
    """
    Remove NaN from similarity by dropping first feature and convert to float array
    :param data: Data 2d array with similarity and frequency
    :param mode: Process mode
    :return: New normalized 2d array with cleaned first value
    """
    data_cleaned = np.copy(data).astype(float)
    if mode == 'wiki':
        return data_cleaned[:, 1:]
    elif mode == 'books':
        return data_cleaned[:, :-1]
    elif mode == 'books5':
        return data_cleaned
    elif mode == 'fixed_point':
        return data_cleaned
    return None


def smooth_frequency(data, smoothing_value=5):
    """
    Smooth frequency series by averaging over a sliding window
    :param data: Data 2d array with similarity and frequency
    :param smoothing_value: window size on how many elements to look to the left and right respectively
    :return: New normalized 2d array with smoothed frequency
    """
    data_smoothed = np.copy(data)
    data_smoothed[1] = uniform_filter1d(data_smoothed[1], smoothing_value, mode='nearest')
    return data_smoothed


def transform_to_cosdist(data):
    """
    Transform similarity from cossim to cosdist (no side effects)
    :param data: Data 2d array with similarity and frequency (no NaN values)
    :return: New 2d array with cosdist
    """
    data_cosdist = np.copy(data)
    data_cosdist[0] = 1 - data_cosdist[0]
    return data_cosdist


def transform_to_padded_cosdist(data):
    """
    Transform similarity from cossim to padded cosdist (no side effects) with value range [0, 2]
    :param data: Data 2d array with similarity and frequency (no NaN values)
    :return: New 2d array with padded cosdist
    """
    data_cosdist = np.copy(data)
    data_cosdist[0] = 2 - data_cosdist[0]
    return data_cosdist


def relative_frequency(data, mode='wiki'):
    """
    Transform frequency information of a 2d array to relative frequency (no side effects)
    :param data: Data 2d array with similarity and frequency (no NaN values)
    :param mode: Calculation mode ('wiki' and 'books')
    :return: New 2d array with relative frequency
    """
    data_relative_freq = np.copy(data)
    if mode == 'wiki':
        data_relative_freq[1] /= WIKI_TOTAL_COUNT
    elif mode == 'books':
        if len(data[1]) != len(BOOKS_TOTAL_COUNT):
            raise ValueError('Frequency array has wrong length ({} but expted {})'
                             .format(len(data[1]), len(BOOKS_TOTAL_COUNT)))
        data_relative_freq[1] /= BOOKS_TOTAL_COUNT
    elif mode == 'books5':
        if len(data[1]) != len(BOOKS_AVG_COUNT):
            raise ValueError('Frequency array has wrong length ({} but expted {})'
                             .format(len(data[1]), len(BOOKS_AVG_COUNT)))
        data_relative_freq[1] /= BOOKS_AVG_COUNT
    else:
        raise ValueError('Unknown mode for calculating relative frequency')
    return data_relative_freq


def percentual_diff(data):
    """
    Build percentual change of word data (no side effects)
    :param data: Data 2d array with similarity and frequency (no NaN values)
    :return: New percentual change array (+ if increase, - when drop). Array is one element shorter
    """
    data_perct = np.copy(data)
    for i in range(len(data_perct)):
        tmp = pd.Series(data[i])
        data_perct[i] = (tmp.div(tmp.shift(1)) - 1).fillna(0).values
    return data_perct[:, 1:]


def cut_array(data, target_start_year=1800, target_end_year=2008, start_year=1800, mode='standard'):
    """
    Cut data array for books
    :param data: Target data 2d array with similarity and frequency to cut
    :param target_start_year: Target start year
    :param target_end_year: Target end year
    :param start_year: Start year of the data (i.e. what is index 0)
    :param mode: Mode to control properties of series (i.e. books5)
    :return: Cut data
    """
    if mode == 'standard':
        return data[:, (target_start_year - start_year):(target_end_year - start_year + 1)]
    elif mode == 'books5':
        return data[:, (target_start_year - start_year) / 5:(target_end_year - start_year) / 5 + 1]
    else:
        raise ValueError('Unknown mode for cutting the array')


def filter_min_freq(store, min_freq):
    """
    Filter store by min summed frequency
    :param store: Target store to filter
    :param min_freq: Min frequency below which the data is dropped (can be any float depending on the current store
    that may have absolute or relative frequency)
    :return: Filtered store
    """
    for k in store.keys():
        if np.sum(store[k][1]) < min_freq:
            del store[k]

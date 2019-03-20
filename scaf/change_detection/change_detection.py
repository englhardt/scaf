#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

from abc import ABCMeta, abstractmethod

import numpy as np
from changepoint.mean_shift_model import MeanShiftModel


class ChangeDetection:
    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def detect(self, data):
        pass

    @abstractmethod
    def sort_order_ascending(self):
        pass

    def fail_score(self):
        return np.inf if self.sort_order_ascending() else -np.inf


class ChangePointDetection(ChangeDetection):
    def __init__(self, eps=None, samples=5000, target_dim=0):
        super(ChangePointDetection, self).__init__()
        self.eps = eps
        self.samples = samples
        self.target_dim = target_dim

    def detect(self, data):
        if not len(data):
            return [], None
        s = data[self.target_dim]
        model = MeanShiftModel()
        stats_ts, p_vals, _ = model.detect_mean_shift(s, B=self.samples)
        change_index = np.argmin(p_vals)
        if self.eps and stats_ts[change_index] <= self.eps:
            return [], None
        changes = [[change_index + 1, p_vals[change_index]]]
        return changes, [stats_ts, p_vals]

    def sort_order_ascending(self):
        return True


class CusumChangeDetection1d(ChangeDetection):
    def __init__(self, target_dim=0):
        super(CusumChangeDetection1d, self).__init__()
        self.target_dim = target_dim

    def detect(self, data):
        if not np.size(data, 1):
            return [], None
        s_n = np.cumsum(data[self.target_dim])
        change_index = np.argmax(s_n)
        changes = [(change_index, s_n[change_index])]
        return changes, [s_n]

    def sort_order_ascending(self):
        return False


class CusumChangeDetection2d(ChangeDetection):
    def __init__(self):
        super(CusumChangeDetection2d, self).__init__()

    def detect(self, data):
        if not np.size(data, 1):
            return [], None
        s_n = sum(np.cumsum(data, axis=1))
        change_index = np.argmax(s_n)
        changes = [(change_index, s_n[change_index])]
        return changes, [s_n]

    def sort_order_ascending(self):
        return False

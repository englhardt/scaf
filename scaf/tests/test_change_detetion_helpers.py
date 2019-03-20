#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest

import numpy as np

from scaf.utils import normalize, normalize_2d, relative_frequency, percentual_diff, cut_array


class TestChangeDetectionHelpers(unittest.TestCase):
    def test_normalize(self):
        d = np.array(np.random.uniform(0, 10, 100))
        out = normalize(d)
        self.assertAlmostEqual(np.mean(out), 0.)
        self.assertAlmostEqual(np.std(out), 1.)

    def test_normalize_2d(self):
        d = np.array([np.random.uniform(0, 10, 100), np.random.uniform(0, 10, 100)])
        out = normalize_2d(d)
        for i in range(2):
            self.assertAlmostEqual(np.mean(out[i]), 0.)
            self.assertAlmostEqual(np.std(out[i]), 1.)

    def test_relative_frequency(self):
        d = np.array([np.random.uniform(0, 1, 100), np.random.uniform(1000, 1500, 100)])
        out = relative_frequency(d)
        self.assertTrue(np.array_equal(d[0], out[0]))
        self.assertTrue(all(x > 0. for x in out[1]))
        self.assertTrue(all(x < 1. for x in out[1]))

    def test_percentual_diff(self):
        d = np.array([np.random.uniform(0, 1, 100), np.random.uniform(1000, 1500, 100)])
        out = percentual_diff(d)
        for i in range(2):
            self.assertEqual(len(out[0]), len(d[0]) - 1)

    def test_cut_array(self):
        d = np.array([np.random.uniform(0, 1, 100), np.random.uniform(1000, 1500, 100)])
        out = cut_array(d, target_start_year=10, target_end_year=89, start_year=1)
        for i in range(2):
            self.assertEqual(len(out[0]), 80)


if __name__ == '__main__':
    unittest.main()

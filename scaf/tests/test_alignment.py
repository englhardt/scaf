#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest

import numpy as np

from scaf.jobs.alignment import Alignment


class TestChangePointDetection(unittest.TestCase):
    def test_linear_alignment(self):
        a = Alignment(worker=1)
        base = np.ones((2, 10))
        other = np.zeros((2, 10))
        other_aligned = a.align_linear(base, other, other)
        self.assertEqual(other_aligned.shape, base.shape)

    def test_procrustes_alignment(self):
        a = Alignment(worker=1)
        base = np.ones((2, 10))
        other = np.zeros((2, 10))
        other_aligned = a.align_procrustes(base, other, other)
        self.assertEqual(other_aligned.shape, base.shape)


if __name__ == '__main__':
    unittest.main()

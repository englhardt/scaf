#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import unittest

import numpy as np

from scaf.change_detection.change_detection import ChangePointDetection, CusumChangeDetection1d, CusumChangeDetection2d


class TestChangePointDetection(unittest.TestCase):
    def test_basic_change(self):
        data = np.zeros((2, 10))
        data[0, 5] = 1
        changes, output = ChangePointDetection().detect(data)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], 5)
        self.assertTrue(changes[0][1] >= 0.)
        self.assertIsNotNone(output)


class TestCusumChangeDetection1d(unittest.TestCase):
    def test_basic_change(self):
        data = np.zeros((2, 10))
        data[0, 5] = 1
        changes, output = CusumChangeDetection1d().detect(data)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], 5)
        self.assertEqual(changes[0][1], 1)
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 1)
        self.assertTrue(np.array_equal(output[0], [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]))


class TestCusumChangeDetection2d(unittest.TestCase):
    def test_basic_change(self):
        data = np.zeros((2, 10))
        data[0, 5] = 1
        changes, output = CusumChangeDetection2d().detect(data)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], 5)
        self.assertEqual(changes[0][1], 1)
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 1)
        self.assertTrue(np.array_equal(output[0], [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]))

    def test_change_in_second_dim(self):
        data = np.zeros((2, 10))
        data[1, 5] = 1
        changes, output = CusumChangeDetection2d().detect(data)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], 5)
        self.assertEqual(changes[0][1], 1)
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 1)
        self.assertTrue(np.array_equal(output[0], [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]))

    def test_combined_change(self):
        data = np.zeros((2, 10))
        data[1, 5] = 1
        data[0, 6] = 1
        changes, output = CusumChangeDetection2d().detect(data)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], 6)
        self.assertEqual(changes[0][1], 2)
        self.assertIsNotNone(output)
        self.assertEqual(len(output), 1)
        self.assertTrue(np.array_equal(output[0], [0., 0., 0., 0., 0., 1., 2., 2., 2., 2.]))


if __name__ == '__main__':
    unittest.main()

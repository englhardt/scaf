#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import os
import tempfile
import unittest

from scaf.data import DataStore


class TestDataStore(unittest.TestCase):
    @staticmethod
    def setup_int_store():
        store = DataStore()
        store['a'] = 1
        store['b'] = 2
        return store

    def test_filter(self):
        store = self.setup_int_store()
        store.filter(['a'])
        self.assertEqual(len(store), 1)
        self.assertTrue('a' in store)

    def test_apply_transformation(self):
        store = self.setup_int_store()
        store.apply_transformation(lambda x: x + 1)
        self.assertEqual(len(store), 2)
        self.assertTrue('a' in store)
        self.assertEqual(store['a'], 2)
        self.assertTrue('b' in store)
        self.assertEqual(store['b'], 3)

    def test_save_and_load(self):
        store = self.setup_int_store()
        tmp_file = os.path.join(tempfile.gettempdir(), 'test.store')
        store.to_file(tmp_file)
        store_loaded = DataStore.load_from_file(tmp_file)
        self.assertEqual(store_loaded, store)


if __name__ == '__main__':
    unittest.main()

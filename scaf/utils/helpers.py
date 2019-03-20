#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Adrian Englhardt <adrian.englhardt@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT
import os
import re


def expand_path(path):
    if path:
        path = os.path.expandvars(path)
    return path


def natural_sort(arr):
    def key_function(x):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)]

    return sorted(arr, key=key_function)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author:    levondang
@contact:   levondang@163.com
@project:   tra_kdd21_reimplementation
@file:      io.py
@time:      2022-11-13 16:13
@license:   Apache Licence
"""
import pickle
import numpy as np
import os
import json

def save_dict_to_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict_from_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='utf-8')

def load_dict_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        vocab_size = len(vocab) + 1  # for unk 词表总数

    return vocab, vocab_size
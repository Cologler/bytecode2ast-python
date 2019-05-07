# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_func

def test_store_none():
    def func():
        a = None

    assert_func(func)

def test_store_true():
    def func():
        a = True

    assert_func(func)

def test_store_false():
    def func():
        a = False

    assert_func(func)

def test_store_unpack():
    def func():
        a, b = c

    assert_func(func)

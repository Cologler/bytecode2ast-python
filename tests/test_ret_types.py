# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_func

def test_return_none():
    def func():
        return None

    assert_func(func)

def test_return_true():
    def func():
        return True

    assert_func(func)

def test_return_false():
    def func():
        return False

    assert_func(func)

def test_return_int():
    def func():
        return 10

    assert_func(func)

def test_return_str():
    def func():
        return '10'

    assert_func(func)

def test_return_bytes():
    def func():
        return b'10'

    assert_func(func)

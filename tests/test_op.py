# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_func

def test_op_eq():
    def func():
        a == b

    assert_func(func)

def test_op_gt():
    def func():
        a > b

    assert_func(func)

def test_op_lt():
    def func():
        a < b

    assert_func(func)

def test_op_ge():
    def func():
        a >= b

    assert_func(func)

def test_op_le():
    def func():
        a <= b

    assert_func(func)

def test_op_not():
    def func():
        not a

    assert_func(func)

def test_op_is():
    def func():
        a is b

    assert_func(func)

def test_op_is_not():
    def func():
        a is not b

    assert_func(func)

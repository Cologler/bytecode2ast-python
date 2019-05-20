# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_example

def test_func_dynamic_def():
    def func():
        def f():
            pass

    assert_example(func)

def test_func_dynamic_def_with_decorator():
    def func():
        @some
        def f():
            pass

    assert_example(func)

def test_simple_lambda():
    def func():
        func = lambda: {"x": 4}

    assert_example(func)

def test_0001():
    def func():
	    f = lambda: (lambda: 4).__call__()

    assert_example(func)

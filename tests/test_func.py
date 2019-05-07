# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_func

def test_func_pass():
    def func():
        pass

    assert_func(func)

def test_func_posarg_bin_op():
    def func(x):
        return (((((x + 1) - 1) * 1) / 1) % 1)

    assert_func(func)

def test_func_multi_ret():
    def func(x, y, z):
        return x, y, z

    assert_func(func)

def test_some_func():
    def func():
        if a == 1:
            return 2

    assert_func(func)

def test_call_func():
    def func():
        iter(a, b)

    assert_func(func)

def test_call_func_wk():
    def func():
        iter(a, b, c=c, d=d, e=e)

    assert_func(func)

def test_call_func_anyargs():
    def func():
        iter(a, *args, b, c=d, **kwargs)

    assert_func(func)

def test_call_func_generic_args_unpack():
    def func():
        iter(*a, **b)

    assert_func(func)

def test_func_dynamic_def():
    def func():
        def f():
            pass

    # TODO: unable to test eq ?

def test_simple_lambda():
    func = lambda: {"x": 4}

    assert_func(func)

# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import get_instrs_from_b2a, get_instrs

def test_func_pass():
    def func():
        pass

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_func_posarg_bin_op():
    def func(x):
        return (((((x + 1) - 1) * 1) / 1) % 1)

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_func_multi_ret():
    def func(x, y, z):
        return x, y, z

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_some_func():
    def func():
        if a == 1:
            return 2

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_call_func():
    def func():
        iter(a, b)

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_call_func_wk():
    def func():
        iter(a, b, c=c, d=d, e=e)

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_call_func_anyargs():
    def func():
        iter(a, *args, b, c=d, **kwargs)

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_call_func_generic_args_unpack():
    def func():
        iter(*a, **b)

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_func_dynamic_def():
    def func():
        def f():
            pass

    # TODO: unable to test eq ?
    get_instrs_from_b2a(func)

def test_simple_lambda():
    def func():
        func = lambda: {"x": 4}

    # TODO: unable to test eq ?
    get_instrs_from_b2a(func)

def test_0001():
    def func():
	    f = lambda: (lambda: 4).__call__()

    # TODO: unable to test eq ?
    get_instrs_from_b2a(func)

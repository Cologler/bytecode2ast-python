# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import get_instrs_from_b2a, get_instrs

def test_op_eq():
    def func():
        a == b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_op_gt():
    def func():
        a > b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_op_lt():
    def func():
        a < b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_op_ge():
    def func():
        a >= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_op_le():
    def func():
        a <= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_op_not():
    def func():
        not a

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_op_is():
    def func():
        a is b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_op_is_not():
    def func():
        a is not b

    assert get_instrs(func) == get_instrs_from_b2a(func)

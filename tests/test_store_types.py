# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import get_instrs_from_b2a, get_instrs

def test_store_none():
    def func():
        a = None

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_store_true():
    def func():
        a = True

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_store_false():
    def func():
        a = False

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_store_pack():
    def func():
        a = b, c

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_store_unpack():
    def func():
        a, b = c

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_store_multi_assign():
    def func():
        a, b = c, d

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_store_multi_assign_reverse():
    def func():
        x, y = y, x

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_store_chain_assign():
    def func():
        x = y = z = i = j = k

    assert get_instrs(func) == get_instrs_from_b2a(func)

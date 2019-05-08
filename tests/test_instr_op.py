# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import dis

from bytecode2ast import parse_func, create_module

def get_func_from_exec(code, name):
    g = {}
    exec(code, g, g)
    return g[name]

def get_instrs(func):
    return list(dis.Bytecode(func))

def get_instrs_from_b2a(func):
    name = func.__name__
    b2a_ast = parse_func(func)
    module = create_module([b2a_ast])
    new_func = get_func_from_exec(compile(module, '<string>', 'exec'), name)
    return get_instrs(new_func)

def test_unary_op_add():
    def func():
        +a

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_unary_op_negative():
    def func():
        -a

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_unary_op_not():
    def func():
        not a

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_add():
    def func():
        a + b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_add_eq():
    def func():
        a += b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_sub():
    def func():
        a - b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_sub_eq():
    def func():
        a -= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_mul():
    def func():
        a * b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_mul_eq():
    def func():
        a *= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_div():
    def func():
        a / b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_div_eq():
    def func():
        a /= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_floor_div():
    def func():
        a // b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_floor_div_eq():
    def func():
        a //= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_mod():
    def func():
        a % b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_binary_op_mod_eq():
    def func():
        a %= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_eq():
    def func():
        a == b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_gt():
    def func():
        a > b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_lt():
    def func():
        a < b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_ge():
    def func():
        a >= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_le():
    def func():
        a <= b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_is():
    def func():
        a is b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_is_not():
    def func():
        a is not b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_compare_op_in():
    def func():
        a in b

    assert get_instrs(func) == get_instrs_from_b2a(func)
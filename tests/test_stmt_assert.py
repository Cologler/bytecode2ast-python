# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

# Disabling assert rewriting
#   Disable rewriting for a specific module by
#   adding the string PYTEST_DONT_REWRITE to its docstring.
'''
PYTEST_DONT_REWRITE
'''

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

def test_assert():
    def func():
        assert a

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_assert_with_msg():
    def func():
        assert a, b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_assert_with_msg2():
    def func():
        assert a, b()

    assert get_instrs(func) == get_instrs_from_b2a(func)

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

def test_if():
    def func():
        if a == 1:
            b()

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_if_pass():
    def func():
        if a == 1:
            pass

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_if_else_pass():
    def func():
        if a == 1:
            c()
        else:
            pass

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_if_pass_else_pass():
    def func():
        if a == 1:
            pass
        else:
            pass

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_if_not():
    def func():
        if not (__name__ == 'a'):
            return 10
        return None

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_if_complex():
    def func():
        if __name__ == 'a':
            return 10
        elif s + ds == 111:
            if s and f:
                return 1
            else:
                pass
        return None

    assert get_instrs(func) == get_instrs_from_b2a(func)

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

def test_import():
    def func():
        import a

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_import_multi():
    def func():
        import a, b, c

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_import_as():
    def func():
        import a as b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_import_as_multi():
    def func():
        import a as b, c, d as k

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_import_from():
    def func():
        from a import b

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_import_from_mutil():
    def func():
        from a import b, c

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_import_from_as():
    def func():
        from a import b as c

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_import_from_as_multi():
    def func():
        from a import b as c, e as f

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_rel_import_from():
    def func():
        from ...a import b as c

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_build_const_map():
    def func():
        x = {'a': 1, 'b': 2}

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_build_const_tuple():
    def func():
        x = ('a', 'b', 'c')

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_build_const_list():
    def func():
        x = ['a', 'b', 'c']

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_build_const_set():
    def func():
        x = {'a', 'b', 'c'}

    assert get_instrs(func) == get_instrs_from_b2a(func)

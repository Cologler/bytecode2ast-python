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
        if __name__ == 'a':
            return 10
        elif s + ds == 111:
            if s and f:
                return 1
            else:
                pass
        return None

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_white():
    def func():
        while a:
            break

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_white_a_and_b():
    def func():
        while a and b:
            break

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_white_with_if():
    def func():
        while True:
            if a:
                break

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_for():
    def func():
        for z in []:
            a = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_for_else():
    def func():
        for z in []:
            a = 1
        else:
            a = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_for_unpack():
    def func():
        for a, b in []:
            break

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_with():
    def func():
        with a:
            r = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_with_multi():
    def func():
        with a, b, c, d():
            r = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_with_var():
    def func():
        with a as b:
            r = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_with_var_multi():
    def func():
        with a as b, c() as d:
            r = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except():
    def func():
        try:
            a = 1
        except:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_finally():
    def func():
        try:
            a = 1
        except:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_finally():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_type():
    def func():
        try:
            a = 1
        except (TypeError, KeyError):
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_type_finally():
    def func():
        try:
            a = 1
        except (TypeError, KeyError):
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_as():
    def func():
        try:
            a = 1
        except TypeError as e:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_as_finally():
    def func():
        try:
            a = 1
        except TypeError as e:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_as():
    def func():
        try:
            a = 1
        except (TypeError, ValueError) as e:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_as_finally():
    def func():
        try:
            a = 1
        except (TypeError, ValueError) as e:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

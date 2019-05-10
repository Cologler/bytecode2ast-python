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

def get_instrs(func):
    return list(dis.Bytecode(func))

def assert_instrs_equals(instrs1, instrs2):
    starts_line_offset = instrs1[0].starts_line - instrs2[0].starts_line
    new_instrs2 = []
    for i in instrs2:
        if i.starts_line is not None:
            i = i._replace(starts_line=i.starts_line+starts_line_offset)
        new_instrs2.append(i)

    assert instrs1 == new_instrs2

def assert_funcs_instrs_equals(func1, func2):
    assert_instrs_equals(get_instrs(func1), get_instrs(func2))

def test_assert_and_if_not():
    def func1():
        assert name

    def func2():
        if not name: raise AssertionError

    assert_funcs_instrs_equals(func1, func2)
    
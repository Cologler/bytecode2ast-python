# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import dis
import ast
import types
from pathlib import Path
import inspect

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

def get_code_from_func(func):
    return func.__code__

def get_code_from_b2a(func, prefix: str):
    name = func.__name__
    name = prefix + name

    b2a_ast = parse_func(func)
    b2a_ast.name = name
    module = create_module([b2a_ast])
    new_func = get_func_from_exec(compile(module, '<string>', 'exec'), name)
    return get_code_from_func(new_func)

def _get_code_fields(instr):
    return (
        instr.opname,
        instr.opcode,
        instr.arg,
        instr.offset,
        instr.starts_line,
        instr.is_jump_target,
    )

def assert_code_equals(code1, code2):
    instrs1, instrs2 = get_instrs(code1), get_instrs(code2)
    assert len(instrs1) == len(instrs2)
    for a, b in zip(instrs1, instrs2):
        al, bl = _get_code_fields(a), _get_code_fields(b)
        assert al == bl
        if isinstance(a.argval, types.CodeType) and isinstance(b.argval, types.CodeType):
            assert_code_equals(a.argval, b.argval)
        else:
            assert a.argval == b.argval

def assert_example(func):
    funcname = inspect.getouterframes(inspect.currentframe())[1].function
    prefix = f'{funcname}.<locals>.'
    code1 = get_code_from_func(func)
    code2 = get_code_from_b2a(func, prefix)
    assert_code_equals(code1, code2)

# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from .parsers.func_def import from_func

def parse_func(func):
    ''' return a `ast.FunctionDef` object from a function '''
    return from_func(func)

def create_module(ast_objs: list):
    import ast
    return ast.Module(
        body=list(ast_objs)
    )

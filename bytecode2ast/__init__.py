# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from .parsers.func_def import FunctionDefParser

def parse_func(func):
    ''' return a `ast.FunctionDef` object from a function '''
    return FunctionDefParser(func).parse()

def create_module(ast_objs: list):
    import ast
    return ast.Module(
        body=list(ast_objs)
    )

# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from typing import Union, List
import ast

from .parsers.func_def import from_func

def parse_func(func):
    '''
    return a `ast.FunctionDef` object or `ast.Lambda` object from a function
    '''
    return from_func(func)

def create_module(ast_objs: Union[List[ast.AST], ast.AST]):
    '''
    create a `ast.Module` with body `ast_objs`.

    This is helpful like `create_module(parse_func(?))`
    '''

    mod = ast.Module(
        body=[]
    )

    if isinstance(ast_objs, list):
        mod.body.extend(ast_objs)
    elif isinstance(ast_objs, ast.AST):
        mod.body.append(ast_objs)
    else:
        raise TypeError

    return mod

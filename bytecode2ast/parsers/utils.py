# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import ast

def reduce_as_pass(blocks: list) -> list:
    '''
    reduce some stmt blocks as `pass`.
    '''

    if len(blocks) == 1:
        block = blocks[0]
        if isinstance(block, ast.Return):
            if isinstance(block.value, ast.NameConstant):
                if block.value.value is None:
                    return [ast.Pass(lineno=block.lineno)]

    return blocks

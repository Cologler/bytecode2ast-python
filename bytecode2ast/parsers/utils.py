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

class ensure:
    '''
    a check class
    '''

    @staticmethod
    def body_not_empty(body: list, lineno: int):
        '''
        ensure body list is not empty.
        if body list is empty, append `PASS` on it.
        '''
        # try body and except body can not be empty.
        if not body:
            body.append(ast.Pass(lineno=lineno))
        return body

    @staticmethod
    def pack_expr(node):
        '''
        may need this when you visit:
            TypeError: expected some sort of stmt, but got <? object>
        '''
        if not isinstance(node, ast.stmt):
            node = ast.Expr(
                lineno=node.lineno,
                value=node
            )
        return node

    @staticmethod
    def unpack_expr(node):
        '''
        may need this when you visit:
            TypeError: expected some sort of expr, but got <ast.Expr object>
        '''
        if isinstance(node, ast.Expr):
            node = node.value
        return node

class tests:
    '''
    a test class
    '''

    @staticmethod
    def eq(*items):
        ''' test whether all values are equals. '''

        if not items:
            raise ValueError

        value = items[0]
        return all(x == value for x in items[1:])

    @staticmethod
    def endswith_return_none(body: list) -> bool:
        ''' test whether body stmt endswith `return None` '''
        if body:
            block = body[-1]
            if isinstance(block, ast.Return):
                if isinstance(block.value, ast.NameConstant):
                    if block.value.value is None:
                        return True
        return False

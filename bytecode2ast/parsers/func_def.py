# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import dis
import ast
import inspect

from .instr import CodeReader, CodeState
from .utils import reduce_as_pass

def _is_lambda(code):
    return code.co_name == '<lambda>'


class FunctionDefParser:
    def __init__(self, code, func=None):
        if func is not None:
            assert func.__code__ is code
        self._code = code
        self._func = func

    def parse(self):
        func_def = ast.FunctionDef(
            lineno=self._parse_lineno(),
            name=self._parse_name(),
            args=self._parse_args(),
            body=self._parse_body(),
            decorator_list=self._parse_decorator_list(),
            returns=self._parse_returns()
        )
        ast.fix_missing_locations(func_def)
        return func_def

    def _parse_lineno(self):
        return self._code.co_firstlineno

    def _parse_name(self) -> str:
        return self._code.co_name

    def _parse_args(self):
        code = self._code

        lineno = code.co_firstlineno
        co_varnames = list(reversed(code.co_varnames))

        args = []
        for _ in range(code.co_argcount):
            args.append(
                ast.arg(lineno=lineno, arg=co_varnames.pop(), annotation=None)
            )

        kwonlyargs = []
        for _ in range(code.co_kwonlyargcount):
            kwonlyargs.append(
                ast.arg(lineno=lineno, arg=co_varnames.pop(), annotation=None)
            )

        kw_defaults = []

        vararg = None
        if code.co_flags & inspect.CO_VARARGS:
            vararg = ast.arg(lineno=lineno, arg=co_varnames.pop(), annotation=None)

        kwarg = None
        if code.co_flags & inspect.CO_VARKEYWORDS:
            kwarg = ast.arg(lineno=lineno, arg=co_varnames.pop(), annotation=None)

        defaults = []

        return ast.arguments(
            args=args,
            vararg=vararg,
            kwonlyargs=kwonlyargs,
            kw_defaults=kw_defaults,
            kwarg=kwarg,
            defaults=defaults,
        )

    def _parse_body(self):
        instructions = list(dis.Bytecode(self._code))
        reader = CodeReader(instructions)
        body = reader.read_until_end().get_value()
        return reduce_as_pass(body)

    def _parse_decorator_list(self):
        return []

    def _parse_returns(self):
        return None


class LambdaParser(FunctionDefParser):
    def parse(self):
        func_def = ast.Lambda(
            lineno=self._parse_lineno(),
            args=self._parse_args(),
            body=self._parse_body()
        )
        ast.fix_missing_locations(func_def)
        return func_def

    def _parse_body(self):
        body: list = super()._parse_body()
        assert len(body) == 1
        body = body[0]
        if isinstance(body, ast.Return):
            return body.value
        return body


def from_code(code):
    ''' return a `ast.FunctionDef` or `ast.Lambda` from a code object. '''
    if _is_lambda(code):
        return LambdaParser(code).parse()
    else:
        return FunctionDefParser(code).parse()


def from_func(func):
    ''' return a ast.FunctionDef from a function. '''
    if _is_lambda(func.__code__):
        return LambdaParser(func.__code__, func).parse()
    else:
        return FunctionDefParser(func.__code__, func).parse()

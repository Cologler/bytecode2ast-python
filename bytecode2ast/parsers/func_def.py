# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import dis
import ast


class FunctionDefParser:
    def __init__(self, func):
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
        return self._func.__code__.co_firstlineno

    def _parse_name(self) -> str:
        return self._func.__code__.co_name

    def _parse_args(self):
        import inspect
        args = []
        for param in inspect.signature(self._func).parameters.values():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(
                    ast.arg(
                        lineno=self._parse_lineno(),
                        arg=param.name,
                        annotation=None if param.annotation is inspect.Parameter.empty else param.annotation
                    )
                )
            else:
                raise NotImplementedError

        return ast.arguments(
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

    def _parse_body(self):
        body_parser = StatementsParser(list(dis.Bytecode(self._func)))
        return body_parser.parse()

    def _parse_decorator_list(self):
        return []

    def _parse_returns(self):
        return None


from .stmt import StatementsParser

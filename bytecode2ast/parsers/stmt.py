# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import abc
import ast
import dis
from functools import partial
from itertools import repeat

from .instr import CodeReader, CodeState, get_instr_handler, walk


def not_support(instr):
    raise NotImplementedError(instr)


_OPCODE_MAP = {}


def op(opname, opcode, **kwargs):
    def wrapper(func):
        def func_wrapper(self, instr: dis.Instruction):
            real_kwargs = dict((z[0], z[1]()) for z in kwargs.items())
            func(self, instr, **real_kwargs)
        assert opcode not in _OPCODE_MAP
        _OPCODE_MAP[opcode] = func_wrapper
        return func
    return wrapper


class IBlockParser(abc.ABC):
    def __init__(self, reader: CodeReader):
        self._reader = reader
        self._state = CodeState()

    @abc.abstractmethod
    def parse(self):
        raise NotImplementedError

    def get_body(self) -> list:
        return self._state.get_value()

    def _proc_next_instr(self):
        instr = self._reader.pop()
        proc_func =  self._get_proc_func(instr)
        proc_func(instr)

    def _get_proc_func(self, instr: dis.Instruction):
        '''
        lookup proc func by instr.
        '''
        handler = get_instr_handler(instr)
        if handler:
            return partial(handler, self._reader, self._state)


class OffsetedBlockParser(IBlockParser):
    def __init__(self, reader, end_offset):
        super().__init__(reader)
        self._end_offset = end_offset

    def parse(self):
        while self._reader.peek().offset != self._end_offset:
            self._proc_next_instr()

        return self.get_body()


class LoopBlockParser(IBlockParser):
    def __init__(self, reader, instr):
        super().__init__(reader)
        self._is_for_loop = False
        self._iter = None
        self._target = None
        self._lineno = reader.get_lineno()

        self._parser = self
        self._body_parser = None
        self._orelse_parser = OffsetedBlockParser(self._reader, instr.argval)

    def parse(self):
        while self._parser is self:
            self._proc_next_instr()

        self._body_parser.parse()
        self._orelse_parser.parse()

    def get_loop(self):
        body = self.get_body()
        if self._is_for_loop:
            node = ast.For(
                lineno=self._lineno,
                iter=self._iter
            )
            node.target = self._target
            node.body = self._body_parser.get_body()
            node.orelse = self._orelse_parser.get_body()
        else:
            raise NotImplementedError
        return node

    def _on_instr_get_iter(self, instr: dis.Instruction):
        # opcode: 68
        self._is_for_loop = True
        self._iter = self._state.pop_one()

    def _on_instr_for_iter(self, instr: dis.Instruction):
        # opcode: 93
        self._target = self._parse_store_target()
        self._body_parser = OffsetedBlockParser(self._reader, instr.argval)
        self._parser = self._body_parser

    def _on_instr_pop_jump_if_false(self, instr: dis.Instruction):
        # opcode: 114
        pass


class StatementsParser(IBlockParser):

    def parse(self):
        walk(self._reader, self._state)
        return self._state.get_value()

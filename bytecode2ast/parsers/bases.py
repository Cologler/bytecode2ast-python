# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
# some object for parser
# ----------

from typing import List
import enum
import dis
from collections import defaultdict


class ID:
    def __init__(self, name):
        self._name = name # a name use to debug

    def __repr__(self):
        return f'ID({self._name})'

    def __str__(self):
        return repr(self)


class Scope(enum.IntEnum):
    NONE = enum.auto()
    LOOP = enum.auto()
    WITH = enum.auto()
    EXCEPT = enum.auto()
    FINALLY = enum.auto()


class CodeState:
    def __init__(self, *, scope=Scope.NONE):
        self._ast_stack = []
        self._load_stack = []
        self._scope = scope
        self._state: dict = None if scope is Scope.NONE else {}
        self._blocks = [[]] # ensure has last block
        self._instrs = [] # all handled instrs in this state

    def __repr__(self):
        return f'b({self._blocks!r}), l({self._load_stack!r})'

    @property
    def scope(self):
        return self._scope

    # state

    @property
    def state(self):
        return self._state

    def add_state(self, id, value):
        ''' add a state, also ensure it does not exists. '''
        assert id not in self._state
        self._state[id] = value

    # instrs

    def add_instr(self, instr: dis.Instruction):
        ''' add a handled instruction in this state '''
        self._instrs.append(instr)

    def get_instrs(self, key=None) -> List[dis.Instruction]:
        ''' get all instructions by key from this state '''
        if key is None:
            return self._instrs.copy()
        else:
            return [i for i in self._instrs if i.opcode == key or i.opname == key]

    def copy(self):
        ''' copy a `CodeState` '''
        state = CodeState()
        state._load_stack = self._load_stack.copy()
        state._ast_stack = self._ast_stack.copy()
        return state

    def copy_with_load(self, load_count):
        ''' copy a `CodeState` with empty ast stack. '''
        state = CodeState()
        state._load_stack = self._load_stack[-load_count:]
        return state

    def push(self, node):
        ''' push a node into load stack. '''
        self._load_stack.append(node)

    def pop(self):
        ''' pop the top node from load stack. '''
        return self._load_stack.pop()

    def pop_seq(self, count: int) -> list:
        ''' pop a list of top nodes from load stack. '''
        assert count >= 0
        if count > 0:
            items = self._load_stack[-count:]
            self._load_stack = self._load_stack[0:-count]
            return items
        else:
            return []

    def dup_top(self):
        ''' repeat top once. '''
        self._load_stack.append(self._load_stack[-1])

    def store(self, node):
        ''' store a node '''
        self.add_node(node)

    def add_node(self, node):
        ''' add a final node into ast stmt tree '''
        self._blocks[-1].append(node)

    def get_value(self) -> list:
        ''' get stmts from single block. '''

        # ensure all status was handled
        assert not self._state, self._state
        assert not self._load_stack, self._load_stack

        # get value
        assert len(self._blocks) == 1, self._blocks
        return self._blocks[-1]

    def new_block(self):
        ''' make a new stmts block '''
        self._blocks.append([])

    def get_blocks(self) -> list:
        ''' get all stmts blocks. '''

        # ensure all status was handled
        assert not self._state, self._state
        assert not self._load_stack, self._load_stack

        # get value
        return self._blocks

    def get_block_count(self) -> int:
        ''' get count of stmts blocks. '''

        return len(self._blocks)


class CodeReaderIter:
    __slots__ = ('_reader', '_condition')

    def __init__(self, reader, condition):
        self._reader: CodeReader = reader
        self._condition = condition

    def __iter__(self):
        while self._condition():
            yield self._reader.pop()

    def fill_state(self, state: CodeState):
        ''' iter self into the `CodeState` and return it. '''
        for instr in self:
            handler = get_instr_handler(instr)
            handler(self._reader, state, instr)
            state.add_instr(instr)
        return state

    def get_state(self, *, scope=Scope.NONE):
        ''' iter self into a new `CodeState`, return the `CodeState` '''
        state = CodeState(scope=scope)
        return self.fill_state(state)

    def get_value(self, *, scope=Scope.NONE):
        ''' iter self into a new `CodeState`, return value from `CodeState`. '''
        return self.get_state(scope=scope).get_value()

    def get_blocks(self, *, scope=Scope.NONE):
        ''' iter self into a new `CodeState`, return blocks from `CodeState`. '''
        return self.get_state(scope=scope).get_blocks()


class CodeReader:
    def __init__(self, instructions):
        # reversed will fast
        self._instructions = list(reversed(instructions))
        self._lineno = None

    def __bool__(self):
        return bool(self._instructions)

    def __repr__(self):
        return repr(list(reversed(self._instructions)))

    @property
    def co_consts(self):
        return self._co_consts

    def get_instrs_count(self) -> int:
        return len(self._instructions)

    def get_lineno(self) -> int:
        return self._lineno

    def peek(self) -> dis.Instruction:
        ''' peek one instr '''
        if not self._instructions:
            return None
        return self._instructions[-1]

    def pop(self) -> dis.Instruction:
        ''' pop one instr '''
        instr = self._instructions.pop()
        if instr.starts_line is not None:
            self._lineno = instr.starts_line
        return instr

    def pop_assert(self, opcode: int) -> dis.Instruction:
        instr = self.pop()
        assert instr.opcode == opcode
        return instr

    def pop_if(self, opcode: int) -> dis.Instruction:
        if self._instructions and self._instructions[-1].opcode == opcode:
            return self.pop()

    # read methods

    def read_until_end(self):
        ''' read until reader end. '''
        return CodeReaderIter(self, lambda: self)

    def read_until_offset(self, offset: int):
        ''' read until come to the offset '''
        return CodeReaderIter(self, lambda: self.peek().offset != offset)

    def read_until_opcodes(self, *opcodes):
        ''' read until visit some opcodes '''
        return CodeReaderIter(self, lambda: self.peek().opcode not in opcodes)

    def read_until_count(self, count: int):
        ''' read until handled count of instrs '''
        end_count = self.get_instrs_count() - count
        return CodeReaderIter(self, lambda: self.get_instrs_count() > end_count)

    def read_until_scoped_count(self, count: int):
        ''' read until handled count of instrs in current scope. '''
        if count <= 0:
            raise ValueError(count)

        def cond():
            nonlocal count
            count -= 1
            return count >= 0

        return CodeReaderIter(self, cond)


_OPCODE_MAP = {}

def op(opname, opcode, **kwargs):
    def wrapper(func):
        def func_wrapper(reader, state, instr: dis.Instruction):
            func(reader, state, instr, **kwargs)
        assert opcode not in _OPCODE_MAP
        _OPCODE_MAP[(opname, opcode)] = func_wrapper
        return func
    return wrapper

def get_instr_handler(instr):
    '''
    the return function `(reader, state, instr) -> None`
    '''
    k = (instr.opname, instr.opcode)
    try:
        return _OPCODE_MAP[k]
    except KeyError:
        raise NotImplementedError(k, instr)

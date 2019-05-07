# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import ast
import dis
import enum


class Q(enum.IntEnum):
    UNKNOWN = enum.auto()
    LOAD = enum.auto()
    STORE = enum.auto()


class CodeReader:
    def __init__(self, instructions):
        # reversed will fast
        self._instructions = list(reversed(instructions))
        self._lineno = None

    def __bool__(self):
        return bool(self._instructions)

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


class CodeState:
    def __init__(self):
        self._stack = []

    def copy(self, count=None):
        state = CodeState()
        if count is None:
            state._stack = self._stack.copy()
        else:
            state._stack = self._stack[-count:]
        return state

    def copy_and_pop(self, count=None):
        state = CodeState()
        if count is None:
            state._stack = self._stack.copy()
            self._stack = []
        else:
            state._stack = self._stack[-count:]
            self._stack = self._stack[:-count]
        return state

    def push(self, node, q: Q = Q.UNKNOWN):
        self._stack.append(
            (q, node)
        )

    def push_store(self, node):
        return self.push(node, Q.STORE)

    def push_load(self, node):
        return self.push(node, Q.LOAD)

    def pop(self, count: int):
        ''' return a list of node. '''
        if count > 0:
            items = self._stack[-count:]
            self._stack = self._stack[0:-count]
            return [z[1] for z in items]
        else:
            return []

    def pop_one(self):
        ''' return one node. '''
        return self._stack.pop()[1]

    def _pop_one_not(self, q: Q):
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] != q:
                return self._stack.pop(i)[1]
        raise IndexError

    def pop_one_load(self):
        return self._pop_one_not(Q.STORE)

    def pop_one_store(self):
        return self._pop_one_not(Q.LOAD)

    def swap(self, count: int):
        ''' swap top nodes on stack '''
        self._stack[-count:] = reversed(self._stack[-count:])

    def get_value(self) -> list:
        return [z[1] for z in self._stack]


_OPCODE_MAP = {}
def op(opname, opcode, **kwargs):
    def wrapper(func):
        def func_wrapper(reader, state, instr: dis.Instruction):
            real_kwargs = dict((z[0], z[1]()) for z in kwargs.items())
            func(reader, state, instr, **real_kwargs)
        assert opcode not in _OPCODE_MAP
        _OPCODE_MAP[(opname, opcode)] = func_wrapper
        return func
    return wrapper

def get_instr_handler(instr):
    '''
    the return function `(reader, state, instr) -> None`
    '''
    try:
        return _OPCODE_MAP[(instr.opname, instr.opcode)]
    except KeyError:
        raise NotImplementedError(instr)

def walk(reader: CodeReader, state: CodeState):
    ''' walk reader until end '''
    while reader:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)
        continue
        print(instr)
        print(state._stack)

def walk_until_count(reader: CodeReader, state: CodeState, count):
    ''' walk reader until handle count '''
    end_count = reader.get_instrs_count() - count
    while reader.get_instrs_count() > end_count:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)

def walk_until_offset(reader: CodeReader, state: CodeState, offset):
    ''' walk reader until come to offset '''
    while reader.peek().offset != offset:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)

def walk_until_opcodes(reader: CodeReader, state: CodeState, *opcodes):
    ''' walk reader until visit some opcodes '''
    while reader.peek().opcode not in opcodes:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)

def _get_ast_value(reader, value):
    if value is None or value is True or value is False:
        return ast.NameConstant(
            value,
            lineno=reader.get_lineno()
        )
    elif isinstance(value, (int, float)):
        return ast.Num(
            value,
            lineno=reader.get_lineno()
        )
    elif isinstance(value, (str, )):
        return ast.Str(
            value,
            lineno=reader.get_lineno()
        )
    elif isinstance(value, tuple):
        return ast.Tuple(
            elts=list(value),
            lineno=reader.get_lineno(),
            ctx=ast.Load()
        )
    else:
        raise NotImplementedError(value)

@op('POP_TOP', 1)
def on_instr_pop_top(reader: CodeReader, state: CodeState, instr):
    pass

@op('ROT_TWO', 2)
def on_instr_rot(reader: CodeReader, state: CodeState, instr):
    lineno = reader.get_lineno()
    sub_state = state.copy_and_pop(2)
    sub_state.swap(2)
    walk_until_count(reader, sub_state, 2)
    ns = sub_state.get_value()
    assert all(isinstance(z, ast.Assign) for z in ns)
    lineno = min(lineno, min(z.lineno for z in ns))
    target = ast.Tuple(
        lineno=lineno,
        elts=[z.targets[0] for z in ns],
        ctx=ast.Store()
    )
    value = ast.Tuple(
        lineno=lineno,
        elts=[z.value for z in ns],
        ctx=ast.Load()
    )
    state.push_store(
        ast.Assign(
            lineno=lineno,
            targets=[target],
            value=value
        )
    )

@op('DUP_TOP', 4)
def on_instr_dup_top(reader: CodeReader, state: CodeState, instr):
    raise NotImplementedError
    lineno = reader.get_lineno()
    walk_until_count(reader, state, 1)
    assert reader.peek().opname == 'STORE_FAST'

@op('BINARY_MULTIPLY', 20, op=ast.Mult)
@op('BINARY_MODULO', 22, op=ast.Mod)
@op('BINARY_ADD', 23, op=ast.Add)
@op('BINARY_SUBTRACT', 24, op=ast.Sub)
@op('BINARY_TRUE_DIVIDE', 27, op=ast.Div)
def on_instr_binary_op(reader: CodeReader, state: CodeState, instr, op):
    l, r = state.pop(2)
    node = ast.BinOp(
        lineno=reader.get_lineno(),
        left=l, right=r,
        op=op
    )
    state.push(node)

@op('BREAK_LOOP', 80)
def on_instr_break_loop(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    state.push(ast.Break(
        lineno=reader.get_lineno(),
    ))

@op('RETURN_VALUE', 83)
def on_instr_return_value(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = ast.Return(
        lineno=reader.get_lineno(),
        value=state.pop_one()
    )
    state.push(node)

@op('LOAD_CONST', 100)
def on_instr_load_const(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    state.push(_get_ast_value(reader, instr.argval))

@op('BUILD_TUPLE', 102)
def on_instr_build_tuple(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = ast.Tuple(
        lineno=reader.get_lineno(),
        elts=list(state.pop(instr.arg)),
        ctx=ast.Load()
    )
    state.push(node)

@op('BUILD_MAP', 105)
def on_instr_build_map(reader: CodeReader, state: CodeState, instr: dis.Instruction):

    keys = []
    values = []
    count = instr.argval
    while count > 0:
        count -= 1
        k, v = state.pop(2)
        keys.append(k)
        values.append(v)
    keys.reverse()
    values.reverse()

    node = ast.Dict(
        keys = keys,
        values = values
    )
    if keys:
        node.lineno = keys[0].lineno
    else:
        node.lineno =  reader.get_lineno()
    state.push(node)

@op('COMPARE_OP', 107)
def on_instr_compare_op(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    if instr.argval == '==':
        left, right = state.pop(2)
        node = ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])
        state.push(node)
    else:
        raise NotImplementedError(instr)

@op('POP_JUMP_IF_FALSE', 114)
def on_instr_pop_jump_if_false(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = ast.If(
        lineno=reader.get_lineno(),
        test=state.pop_one()
    )
    # body
    body_state = CodeState()
    walk_until_offset(reader, body_state, instr.argval)
    node.body = body_state.get_value()
    # end body

    else_instr = reader.pop_if(110) # JUMP_FORWARD
    if else_instr:
        orelse_state = CodeState()
        walk_until_offset(reader, orelse_state, else_instr.argval)
        node.orelse = orelse_state.get_value()
    else:
        node.orelse = []
    state.push(node)

@op('SETUP_LOOP', 120)
def on_instr_setup_loop(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    raise NotImplementedError
    lineno = reader.get_lineno()
    # loop always load something
    loop_src_state = CodeState()
    walk_until_opcodes(reader, loop_src_state, 114, 93) # 114 for `while-loop`, 93 for `for-loop`
    loop_id = reader.peek()
    if loop_id.opcode == 93:
        # for-loop
        pass
    else:
        # while-loop
        pass
    loop_parser = LoopBlockParser(reader, instr)
    loop_parser.parse()
    node = loop_parser.get_loop()
    state.push(node)

@op('LOAD_GLOBAL', 116)
@op('LOAD_FAST', 124)
def on_instr_load(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    state.push_load(
        ast.Name(id=instr.argval, ctx=ast.Load(), lineno=reader.get_lineno())
    )

@op('STORE_FAST', 125)
def on_instr_store(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    value = state.pop_one_load()
    targets = [
        ast.Name(
            lineno=reader.get_lineno(),
            id=instr.argval,
            ctx=ast.Store()
        )
    ]
    node = ast.Assign(
        lineno=reader.get_lineno(),
        targets=targets,
        value=value,
    )
    state.push_store(node)

@op('POP_BLOCK', 87)
def on_instr_pop_block(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # end of loop block, just ignore
    pass

@op('JUMP_ABSOLUTE', 113)
def on_instr_jump_absolute(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # end of loop block, just ignore
    pass

def _make_func_call(reader: CodeReader, state: CodeState, instr: dis.Instruction, args, kwargs):
    func = state.pop_one()
    state.push(
        ast.Expr(
            lineno=reader.get_lineno(),
            value=ast.Call(
                lineno=reader.get_lineno(),
                func=func,
                args=args,
                keywords=kwargs
            )
        )
    )

@op('CALL_FUNCTION', 131)
def on_instr_call_function(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    if instr.opcode == 141:
        kw_tuple = state.pop_one()
        kw = kw_tuple.elts

    args = state.pop(instr.argval)
    keywords = []
    if instr.opcode == 141:
        keywords = [
            ast.keyword(arg=n, value=v) for (n, v) in zip(kw, args[-len(kw):])
        ]
        args = args[:-len(kw)]

    return _make_func_call(reader, state, instr, args, keywords)

@op('CALL_FUNCTION_KW', 141)
def on_instr_call_function_kw(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    kw_tuple = state.pop_one()
    kw = kw_tuple.elts

    args = state.pop(instr.argval)
    kwargs = [
        ast.keyword(arg=n, value=v) for (n, v) in zip(kw, args[-len(kw):])
    ]
    args = args[:-len(kw)]

    return _make_func_call(reader, state, instr, args, kwargs)

@op('CALL_FUNCTION_EX', 142)
def on_instr_call_function_ex(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    assert instr.argval == 1

    kw = state.pop_one()
    if isinstance(kw, list): # make from `BUILD_MAP_UNPACK_WITH_CALL`
        kwargs = kw
    else:
        kwargs = [ast.keyword(arg=None, value=kw, lineno=reader.get_lineno())]

    a = state.pop_one()
    if isinstance(a, list): # make from `BUILD_TUPLE_UNPACK_WITH_CALL`
        args = a
    else:
        args = [ast.Starred(
            lineno=reader.get_lineno(),
            value=a,
            ctx=ast.Load()
        )]

    return _make_func_call(reader, state, instr, args, kwargs)

def _get_ast_store_name(reader: CodeReader):
    instr = reader.pop_assert(125) # STORE_FAST
    return ast.Name(
        lineno=reader.get_lineno(),
        id=instr.argval,
        ctx=ast.Store()
    )

def _parse_instr_unpack_sequence(reader: CodeReader, state: CodeState, instr):
    target_tuple = ast.Tuple(
        lineno=reader.get_lineno(),
        ctx=ast.Store(),
        elts = []
    )
    for _ in range(instr.argval):
        target_tuple.elts.append(_get_ast_store_name(reader))
    return target_tuple

@op('UNPACK_SEQUENCE', 92)
def on_instr_unpack_sequence(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    target = _parse_instr_unpack_sequence(reader, state, instr)
    value = state.pop_one()
    node = ast.Assign(
        lineno=reader.get_lineno(),
        targets=[target],
        value=value,
    )
    state.push(node)

@op('BUILD_MAP_UNPACK_WITH_CALL', 151)
def on_instr_build_map_unpack_with_call(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    kwargs = []
    for arg in state.pop(instr.argval):
        if isinstance(arg, ast.Name):
            kwargs.append(
                ast.keyword(arg=None, value=arg)
            )
        elif isinstance(arg, ast.Dict):
            for k, v in zip(arg.keys, arg.values):
                kwargs.append(
                    ast.keyword(arg=k.s, value=v),
                )
        else:
            raise NotImplementedError(arg)
    state.push(kwargs)

@op('BUILD_TUPLE_UNPACK_WITH_CALL', 158)
def on_instr_build_tuple_unpack_with_call(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    args = []
    for arg in state.pop(instr.argval):
        if isinstance(arg, ast.Name):
            args.append(
                ast.Starred(
                    lineno=arg.lineno,
                    value=arg,
                    ctx=ast.Load()
                )
            )
        elif isinstance(arg, ast.Tuple):
            args.extend(arg.elts)
        else:
            raise NotImplementedError(arg)
    state.push(args)

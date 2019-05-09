# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import ast
import dis
import itertools
import enum

from .o import ID

class Scope(enum.IntEnum):
    NONE = enum.auto()
    LOOP = enum.auto()
    WITH = enum.auto()
    TRY = enum.auto()
    FINALLY = enum.auto()

CTX_LOAD = ast.Load()
CTX_STORE = ast.Store()

OP_AND = ast.And()
OP_OR = ast.Or()
OP_FLOORDIV = ast.FloorDiv()
OP_DIV = ast.Div()
OP_ADD = ast.Add()
OP_SUB = ast.Sub()
OP_MULT = ast.Mult()
OP_MOD = ast.Mod()
OP_POW = ast.Pow()
OP_LSHIFT = ast.LShift()
OP_RSHIFT = ast.RShift()
OP_BITAND = ast.BitAnd()
OP_BITXOR = ast.BitXor()
OP_BITOR = ast.BitOr()
OP_UADD = ast.UAdd()
OP_USUB = ast.USub()
OP_NOT = ast.Not()
OP_INVERT = ast.Invert()

ID_PADDING = ID('padding')
ID_FOR_ITER = ID('for_iter')
ID_POP_BLOCK = ID('pop_block')

def load(node):
    ''' set `node.ctx` to `ast.Load` and return it '''
    node.ctx = CTX_LOAD
    return node

def store(node):
    ''' set `node.ctx` to `ast.Load` and return it '''
    node.ctx = CTX_STORE
    return node

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


class CodeState:
    def __init__(self, *, scope=Scope.NONE):
        self._ast_stack = []
        self._load_stack = []
        self._scope = scope
        self._state: dict = None if scope is Scope.NONE else {}
        self._blocks = [[]] # ensure has last block

    @property
    def scope(self):
        return self._scope

    @property
    def state(self):
        return self._state

    def __repr__(self):
        return f'b({self._blocks!r}), l({self._load_stack!r})'

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

def walk(reader: CodeReader, state: CodeState):
    ''' walk reader until reader end '''
    while reader:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)
    return state

def walk_until_count(reader: CodeReader, state: CodeState, count: int):
    ''' walk reader until handled count of instrs '''
    end_count = reader.get_instrs_count() - count
    while reader.get_instrs_count() > end_count:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)
    return state

def walk_until_scoped_count(reader: CodeReader, state: CodeState, count: int):
    ''' walk reader until handled count of instrs in current scope. '''
    assert count > 0
    while count:
        count -= 1
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)
    return state

def walk_until_offset(reader: CodeReader, state: CodeState, offset):
    ''' walk reader until come to the offset '''
    while reader.peek().offset != offset:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)
    return state

def walk_until_opcodes(reader: CodeReader, state: CodeState, *opcodes):
    ''' walk reader until visit some opcodes '''
    while reader.peek().opcode not in opcodes:
        instr = reader.pop()
        handler = get_instr_handler(instr)
        handler(reader, state, instr)
    return state

def _get_ast_value(reader, value):

    if value is None or value is True or value is False:
        return ast.NameConstant(
            value,
            lineno=reader.get_lineno()
        )

    if isinstance(value, (int, float, str)):
        if isinstance(value, (int, float)):
            cls = ast.Num
        elif isinstance(value, str):
            cls = ast.Str

        return cls(
            value,
            lineno=reader.get_lineno()
        )

    if isinstance(value, (tuple, list, set)):
        if isinstance(value, tuple):
            value = [_get_ast_value(reader, x) for x in value]
            cls = ast.Tuple
        elif isinstance(value, list):
            cls = ast.List
        elif isinstance(value, set):
            cls = ast.Set

        return cls(
            elts=list(value),
            lineno=reader.get_lineno()
        )

    return ast.Constant(
        value,
        lineno=reader.get_lineno()
    )

def _ensure_stmt(node):
    if not isinstance(node, ast.stmt):
        # for node like `CompareOp` or `UnaryOp` or more
        node = ast.Expr(
            lineno=node.lineno,
            value=node
        )
    return node

@op('POP_TOP', 1)
def on_instr_pop_top(reader: CodeReader, state: CodeState, instr):
    node = _ensure_stmt(state.pop())
    state.add_node(node)

@op('ROT_TWO', 2)
def on_instr_rot(reader: CodeReader, state: CodeState, instr):
    lineno = reader.get_lineno()
    # copy
    sub_state = CodeState()
    sub_state.push(state.pop())
    sub_state.push(state.pop())

    walk_until_count(reader, sub_state, 2)

    # handle values
    ns = sub_state.get_value()
    assert all(isinstance(z, ast.Assign) for z in ns)
    lineno = min(lineno, min(z.lineno for z in ns))
    target = ast.Tuple(
        lineno=lineno,
        elts=[z.targets[0] for z in ns],
        ctx=CTX_STORE
    )
    value = ast.Tuple(
        lineno=lineno,
        elts=[z.value for z in ns],
        ctx=CTX_LOAD
    )
    state.add_node(
        ast.Assign(
            lineno=lineno,
            targets=[target],
            value=value
        )
    )

@op('DUP_TOP', 4)
def on_instr_dup_top(reader: CodeReader, state: CodeState, instr):
    lineno = reader.get_lineno()

    # make sub
    sub_state = CodeState()
    sub_state.push(state.pop())

    # dup top
    sub_state.dup_top()

    # walk
    walk_until_scoped_count(reader, sub_state, 2)

    # handle values
    ns = sub_state.get_value()
    assert all(isinstance(n, ast.Assign) for n in ns)
    assign_value = ns[0].value
    assert all(n.value is assign_value for n in ns)
    lineno = min(lineno, min(z.lineno for z in ns))
    targets = list(itertools.chain(*[n.targets for n in ns]))
    value = assign_value
    state.add_node(
        ast.Assign(
            lineno=lineno,
            targets=targets,
            value=value
        )
    )

@op('UNARY_POSITIVE', 10, op=OP_UADD)
@op('UNARY_NEGATIVE', 11, op=OP_USUB)
@op('UNARY_NOT', 12, op=OP_NOT)
@op('UNARY_INVERT', 15, op=OP_INVERT)
def on_instr_unary_op(reader: CodeReader, state: CodeState, instr: dis.Instruction, op):
    node = ast.UnaryOp(
        lineno=reader.get_lineno(),
        col_offset=4,
        op=op,
        operand=state.pop()
    )
    state.push(node)

@op('BINARY_POWER', 19, op=OP_POW)
@op('BINARY_MULTIPLY', 20, op=OP_MULT)
@op('BINARY_MODULO', 22, op=OP_MOD)
@op('BINARY_ADD', 23, op=OP_ADD)
@op('BINARY_SUBTRACT', 24, op=OP_SUB)
@op('BINARY_FLOOR_DIVIDE', 26, op=OP_FLOORDIV)
@op('BINARY_TRUE_DIVIDE', 27, op=OP_DIV)
@op('BINARY_LSHIFT', 62, op=OP_LSHIFT)
@op('BINARY_RSHIFT', 63, op=OP_RSHIFT)
@op('BINARY_AND', 64, op=OP_BITAND)
@op('BINARY_XOR', 65, op=OP_BITXOR)
@op('BINARY_OR', 66, op=OP_BITOR)
def on_instr_binary_op(reader: CodeReader, state: CodeState, instr: dis.Instruction, op):
    l, r = state.pop_seq(2)
    node = ast.BinOp(
        lineno=reader.get_lineno(),
        left=l, right=r,
        op=op
    )
    state.push(node)

@op('INPLACE_FLOOR_DIVIDE', 28, op=OP_FLOORDIV)
@op('INPLACE_TRUE_DIVIDE', 29, op=OP_DIV)
@op('INPLACE_ADD', 55, op=OP_ADD)
@op('INPLACE_SUBTRACT', 56, op=OP_SUB)
@op('INPLACE_MULTIPLY', 57, op=OP_MULT)
@op('INPLACE_MODULO', 59, op=OP_MOD)
@op('INPLACE_POWER', 67, op=OP_POW)
@op('INPLACE_LSHIFT', 75, op=OP_LSHIFT)
@op('INPLACE_RSHIFT', 76, op=OP_RSHIFT)
@op('INPLACE_AND', 77, op=OP_BITAND)
@op('INPLACE_XOR', 78, op=OP_BITXOR)
@op('INPLACE_OR', 79, op=OP_BITOR)
def on_instr_inplace_op(reader: CodeReader, state: CodeState, instr: dis.Instruction, op):
    target, value = state.pop_seq(2)
    node = ast.AugAssign(
        lineno=reader.get_lineno(),
        target=store(target),
        op=op,
        value=value
    )
    state.push(node)

@op('BINARY_SUBSCR', 25)
def on_instr_subscr(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    slice_value = state.pop()
    target = state.pop()

    if not isinstance(slice_value, ast.slice):
        slice_value = ast.Index(value=slice_value)

    node = ast.Subscript(
        lineno=reader.get_lineno(),
        value=target,
        slice=slice_value
    )
    state.push(load(node))

@op('STORE_SUBSCR', 60)
def on_instr_store_subscr(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    slice_value = state.pop()
    target = state.pop()
    value = state.pop()

    if not isinstance(slice_value, ast.slice):
        slice_value = ast.Index(value=slice_value)

    subscr_node = ast.Subscript(
        lineno=reader.get_lineno(),
        value=target,
        slice=slice_value
    )
    node = ast.Assign(
        lineno=reader.get_lineno(),
        value=value,
        targets=[store(subscr_node)]
    )
    state.store(node)


@op('BREAK_LOOP', 80)
def on_instr_break_loop(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    state.add_node(ast.Break(
        lineno=reader.get_lineno(),
    ))

@op('RETURN_VALUE', 83)
def on_instr_return_value(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = ast.Return(
        lineno=reader.get_lineno(),
        value=state.pop()
    )
    state.add_node(node)

@op('LOAD_CONST', 100)
def on_instr_load_const(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    state.push(load(_get_ast_value(reader, instr.argval)))

@op('BUILD_TUPLE', 102)
def on_instr_build_tuple(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = ast.Tuple(
        lineno=reader.get_lineno(),
        elts=state.pop_seq(instr.arg)
    )
    state.push(load(node))

@op('BUILD_LIST', 103, col_cls=ast.List)
@op('BUILD_SET', 104, col_cls=ast.Set)
def on_instr_build_list(reader: CodeReader, state: CodeState, instr: dis.Instruction, col_cls):
    items = state.pop_seq(instr.argval)
    node = col_cls(
        lineno=reader.get_lineno(),
        elts=items,
    )
    if items:
        node.lineno = items[0].lineno
    state.push(load(node))

@op('BUILD_MAP', 105)
def on_instr_build_map(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    keys = []
    values = []
    for _ in range(instr.argval):
        k, v = state.pop_seq(2)
        keys.append(k)
        values.append(v)
    keys.reverse()
    values.reverse()

    node = ast.Dict(
        lineno=reader.get_lineno(),
        keys=keys,
        values=values
    )
    if keys:
        node.lineno = keys[0].lineno
    state.push(load(node))

@op('BUILD_CONST_KEY_MAP', 156)
def on_instr_build_const_key_map(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    keys = state.pop()
    values = state.pop_seq(instr.argval)
    lineno = min([reader.get_lineno(), keys.lineno] + [v.lineno for v in values])
    node = ast.Dict(
        lineno=lineno,
        keys=keys.elts,
        values=values
    )
    state.push(node)

_OP_CLS = {
    '==': ast.Eq,
    '>': ast.Gt,
    '<': ast.Lt,
    '>=': ast.GtE,
    '<=': ast.LtE,
    'is': ast.Is,
    'is not': ast.IsNot,
    'in': ast.In
}

@op('COMPARE_OP', 107)
def on_instr_compare_op(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    opcls = _OP_CLS.get(instr.argval)
    if opcls:
        lineno = reader.get_lineno()

        left, right = state.pop_seq(2)
        node = ast.Compare(left=left, ops=[opcls()], comparators=[right])

        lineno = min(lineno, min([left.lineno, right.lineno]))
        node.lineno = lineno
        state.push(node)
        return

    raise NotImplementedError(instr)

@op('JUMP_IF_FALSE_OR_POP', 111, op=OP_AND)
@op('JUMP_IF_TRUE_OR_POP', 112, op=OP_OR)
def on_instr_bool_op(reader: CodeReader, state: CodeState, instr: dis.Instruction, op):
    # logic and: `a and b`
    lineno = reader.get_lineno()
    first = state.pop()
    walk_until_scoped_count(reader, state, 1)
    second = state.pop()
    node = ast.BoolOp(
        lineno=lineno,
        op=op,
        values=[first, second]
    )
    state.push(node)

def on_instr_jump_if_false_or_pop(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # logic and: `a and b`
    lineno = reader.get_lineno()
    first = state.pop()
    walk_until_scoped_count(reader, state, 1)
    second = state.pop()
    node = ast.BoolOp(
        lineno=lineno,
        op=ast.And(),
        values=[first, second]
    )
    state.push(node)

@op('POP_JUMP_IF_FALSE', 114)
def on_instr_pop_jump_if_false(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = ast.If(
        lineno=reader.get_lineno(),
        test=state.pop()
    )

    if_body_state = CodeState(scope=state.scope)

    if state.scope == Scope.LOOP:
        if instr.argval < instr.offset:
            # in loop block, this may possible:
            # while True: if a: break
            # so we should read until block ends
            assert state.scope == Scope.LOOP
            walk_until_opcodes(reader, if_body_state, 113) # JUMP_ABSOLUTE
        else:
            walk_until_offset(reader, if_body_state, instr.argval)
    else:
        walk_until_offset(reader, if_body_state, instr.argval)

    node.body = if_body_state.get_value()

    else_instr = reader.pop_if(110) # JUMP_FORWARD
    if else_instr:
        orelse_state = CodeState()
        walk_until_offset(reader, orelse_state, else_instr.argval)
        node.orelse = orelse_state.get_value()
    else:
        node.orelse = []

    # in while-loop,
    # we need to know offset of if-block
    # so we can check the condition belong which one
    node.jump_to = instr.argval

    state.add_node(node)

@op('GET_ITER', 68)
def on_instr_get_iter(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # in for-loop, get iter from src
    assert state.scope == Scope.LOOP
    # simple to store node from stack
    # so parent can get it
    state.add_node(state.pop())

@op('FOR_ITER', 93)
def on_instr_for_iter(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # in for-loop, load item from src
    assert state.scope == Scope.LOOP
    state.state[ID_FOR_ITER] = ID_PADDING # padding

@op('SETUP_LOOP', 120)
def on_instr_setup_loop(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    lineno = reader.get_lineno()

    loop_state = CodeState(scope=Scope.LOOP)
    walk_until_offset(reader, loop_state, instr.argval) # 114 for `while-loop`, 93 for `for-loop`

    for_iter = loop_state.state.pop(ID_FOR_ITER, None)
    pop_block: dis.Instruction = loop_state.state.pop(ID_POP_BLOCK)

    loop_block, orelse_block = loop_state.get_blocks()

    if for_iter:
        # this is a for-loop
        iter_var = loop_block[0]
        loop_body = loop_block[1:]

        if isinstance(for_iter, ast.Expr):
            # unpack func call
            for_iter = for_iter.value

        node = ast.For(
            lineno=lineno,
            target=for_iter,
            iter=iter_var,
            body=loop_body,
            orelse=orelse_block
        )

    else:
        # this is a while-loop
        jump_offset: int = pop_block.offset

        def unpack_while(maybe_while_expr):
            # unpack
            #   while True: if what: ...
            # to
            #   while what: ...

            if len(maybe_while_expr) != 1:
                return None
            if not isinstance(maybe_while_expr[0], ast.If):
                return None
            if_expr: ast.If = maybe_while_expr[0]
            if if_expr.jump_to != jump_offset:
                return None
            assert not if_expr.orelse # should be empty

            while_node: ast.While = unpack_while(if_expr.body)
            if while_node is None:
                while_node = ast.While(
                    lineno=lineno,
                    test=if_expr.test,
                    body=if_expr.body,
                    orelse=orelse_block
                )
            else:
                while_node.test = ast.BoolOp(
                    lineno=lineno,
                    op=OP_AND,
                    values=[if_expr.test, while_node.test]
                )
            return while_node

        node = unpack_while(loop_block)

        if node is None:
            node = ast.While(
                lineno=lineno,
                test=ast.NameConstant(True),
                body=loop_block,
                orelse=orelse_block
            )

    state.add_node(node)

@op('LOAD_GLOBAL', 116)
@op('LOAD_FAST', 124)
def on_instr_load(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    state.push(
        load(ast.Name(id=instr.argval, lineno=reader.get_lineno()))
    )

@op('STORE_FAST', 125)
def on_instr_store(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    if state.scope == Scope.LOOP and state.state.get(ID_FOR_ITER) is ID_PADDING:
        # that mean `for x in ?`
        # so we store instr as ast.Name for parent to get it.
        node = store(ast.Name(lineno=reader.get_lineno(), id=instr.argval))
        state.state[ID_FOR_ITER] = node
        return

    value = state.pop()

    if isinstance(value, ast.ImportFrom):
        value: ast.ImportFrom = value
        value.names[-1].asname = instr.argval
        # should not store ImportFrom
        # ImportFrom will store in instr POP_TOP
        state.push(value)
        return

    elif isinstance(value, ast.Import):
        value: ast.Import = value
        for n in value.names:
            n.asname = instr.argval
        node = value

    elif isinstance(value, ast.stmt):
        node = value

    else:
        targets = [store(
            ast.Name(lineno=reader.get_lineno(), id=instr.argval)
        )]
        node = ast.Assign(
            lineno=reader.get_lineno(),
            targets=targets,
            value=value,
        )

    state.store(node)

@op('DELETE_FAST', 126)
def on_instr_delete_fast(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    if state.scope == Scope.FINALLY:
        if state.get_block_count() == 2:
            # in this block, clean error vars.
            # do nothing here
            pass
    pass

@op('POP_BLOCK', 87)
def on_instr_pop_block(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # end of with/loop
    assert state.scope in (Scope.LOOP, Scope.WITH, Scope.TRY)
    state.new_block()
    assert ID_POP_BLOCK not in state.state
    state.state[ID_POP_BLOCK] = instr


@op('JUMP_ABSOLUTE', 113)
def on_instr_jump_absolute(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # end of loop block, jump to loop start.
    assert state.scope is Scope.LOOP

def _make_func_call(reader: CodeReader, state: CodeState, instr: dis.Instruction, args, kwargs):
    func = state.pop()
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
        kw_tuple = state.pop()
        kw = kw_tuple.elts

    args = state.pop_seq(instr.argval)
    keywords = []
    if instr.opcode == 141:
        keywords = [
            ast.keyword(arg=n, value=v) for (n, v) in zip(kw, args[-len(kw):])
        ]
        args = args[:-len(kw)]

    return _make_func_call(reader, state, instr, args, keywords)

@op('BUILD_SLICE', 133)
def on_instr_call_function(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    args = [None, None, None]
    for i, v in enumerate(state.pop_seq(instr.argval)):
        args[i] = v
    node = ast.Slice(*args)
    state.push(node)

@op('CALL_FUNCTION_KW', 141)
def on_instr_call_function_kw(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    kw_tuple = state.pop()
    kw = kw_tuple.elts

    args = state.pop_seq(instr.argval)
    kwargs = [
        ast.keyword(arg=n.s, value=v) for (n, v) in zip(kw, args[-len(kw):])
    ]
    args = args[:-len(kw)]

    return _make_func_call(reader, state, instr, args, kwargs)

@op('CALL_FUNCTION_EX', 142)
def on_instr_call_function_ex(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    assert instr.argval == 1

    kw = state.pop()
    if isinstance(kw, list): # make from `BUILD_MAP_UNPACK_WITH_CALL`
        kwargs = kw
    else:
        kwargs = [ast.keyword(arg=None, value=kw, lineno=reader.get_lineno())]

    a = state.pop()
    if isinstance(a, list): # make from `BUILD_TUPLE_UNPACK_WITH_CALL`
        args = a
    else:
        args = [ast.Starred(
            lineno=reader.get_lineno(),
            value=a,
            ctx=CTX_LOAD
        )]

    return _make_func_call(reader, state, instr, args, kwargs)

@op('UNPACK_SEQUENCE', 92)
def on_instr_unpack_sequence(reader: CodeReader, state: CodeState, instr: dis.Instruction):

    def get_target():
        target_tuple = ast.Tuple(
            lineno=reader.get_lineno(),
            ctx=CTX_STORE,
            elts = []
        )
        for _ in range(instr.argval):
            store_instr = reader.pop_assert(125) # STORE_FAST
            target_tuple.elts.append(store(ast.Name(
                lineno=reader.get_lineno(),
                id=store_instr.argval
            )))
        return target_tuple

    target = get_target()

    if state.scope == Scope.LOOP and state.state.get(ID_FOR_ITER) is ID_PADDING:
        state.state[ID_FOR_ITER] = target

    else:
        value = state.pop()
        node = ast.Assign(
            lineno=reader.get_lineno(),
            targets=[target],
            value=value,
        )
        state.add_node(node)

@op('IMPORT_NAME', 108)
def on_instr_import_name(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    lineno = reader.get_lineno()

    # pop unused
    level, _b = state.pop_seq(2)

    node = ast.Import(
        lineno=lineno,
        names=[ast.alias(name=instr.argval)]
    )
    node.level = level.n # add level for rel import
    state.push(node)

@op('IMPORT_FROM', 109)
def on_instr_import_from(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = state.pop()
    if isinstance(node, ast.Import):
        assert len(node.names) == 1
        alias: ast.alias = node.names[0]
        node = ast.ImportFrom(
            lineno=reader.get_lineno(),
            module=alias.name,
            names=[],
            level=node.level
        )
    node.names.append(ast.alias(name=instr.argval))
    state.push(node)

@op('BUILD_MAP_UNPACK_WITH_CALL', 151)
def on_instr_build_map_unpack_with_call(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    kwargs = []
    for arg in state.pop_seq(instr.argval):
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
    for arg in state.pop_seq(instr.argval):
        if isinstance(arg, ast.Name):
            args.append(
                ast.Starred(
                    lineno=arg.lineno,
                    value=arg,
                    ctx=CTX_LOAD
                )
            )
        elif isinstance(arg, ast.Tuple):
            args.extend(arg.elts)
        else:
            raise NotImplementedError(arg)
    state.push(args)

@op('MAKE_FUNCTION', 132)
def on_instr_make_function(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    code, name = state.pop_seq(2)
    code = code.value # unpack

    closure     = state.pop() if instr.argval & 8 else None
    annotations = state.pop() if instr.argval & 4 else None
    kwdefaults  = state.pop() if instr.argval & 2 else None
    defaults    = state.pop() if instr.argval & 1 else None

    if annotations or closure:
        raise NotImplementedError

    from .func_def import from_code

    func_def = from_code(code)

    assert len(func_def.args.defaults) == 0
    if defaults:
        for el in defaults.elts:
            func_def.args.defaults.append(_get_ast_value(reader, el))

    assert len(func_def.args.kw_defaults) == 0
    if kwdefaults:
        # k should be ast.Str
        kwdefaults_map = dict((k.s, v) for (k, v) in zip(kwdefaults.keys, kwdefaults.values))
        for kwarg in func_def.args.kwonlyargs:
            func_def.args.kw_defaults.append(
                kwdefaults_map.get(kwarg.arg)
            )

    state.push(func_def)

@op('LOAD_METHOD', 160)
def on_instr_load_method(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    target = state.pop()
    node = ast.Attribute(
        lineno=reader.get_lineno(),
        value=target,
        attr='__call__',
        ctx=CTX_LOAD,
    )
    state.push(node)

@op('CALL_METHOD', 161)
def on_instr_call_method(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    assert instr.argval == 0
    node = ast.Call(
        lineno=reader.get_lineno(),
        func=state.pop(),
        args=[],
        keywords=[],
    )
    state.push(node)

@op('SETUP_WITH', 143)
def on_instr_setup_with(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    lineno = reader.get_lineno()

    ctx = state.pop()
    # unpack
    if isinstance(ctx, ast.Expr):
        ctx = ctx.value

    ctx_state = CodeState(scope=Scope.WITH)
    ctx_state.push(ctx)
    walk_until_offset(reader, ctx_state, instr.argval)

    # pop noop
    reader.pop_assert(81) # WITH_CLEANUP_START
    reader.pop_assert(82) # WITH_CLEANUP_FINISH
    reader.pop_assert(88) # END_FINALLY
    ctx_state.pop() # py emit a 'LOAD_CONST None' and of with stmt
    ctx_state.state.pop(ID_POP_BLOCK)

    with_item = ast.withitem(
        context_expr=load(ctx),
        optional_vars=None
    )

    ctx_body, ctx_orelse = ctx_state.get_blocks()
    assert len(ctx_orelse) == 0 # unused

    ctx_var = ctx_body[0]

    if isinstance(ctx_var, ast.Expr):
        ctx_var = ctx_var.value

    if ctx_var is ctx:
        # if ctx_var is ctx, mean `with ctx: pass`
        # just ignore it
        pass
    else:
        if isinstance(ctx_var, ast.Assign):
            assert len(ctx_var.targets) == 1
            with_item.optional_vars = ctx_var.targets[0]
        else:
            raise NotImplementedError(ctx_var)

    with_body = ctx_body[1:]

    with_stmt = ast.With(
        lineno=lineno,
        items=[with_item],
        body=with_body
    )

    state.add_node(with_stmt)

@op('JUMP_FORWARD', 110)
def on_instr_jump_forward(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    pass

@op('SETUP_EXCEPT', 121)
def on_instr_setup_except(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    def body_not_empty(body: list, lineno: int):
        # try body and except body can not be empty.
        if not body:
            body.append(ast.Pass(lineno=lineno))
        return body

    lineno = reader.get_lineno()

    try_state = CodeState(scope=Scope.TRY)
    walk_until_offset(reader, try_state, instr.argval)

    try_state.state.pop(ID_POP_BLOCK)

    try_blocks = try_state.get_blocks()
    try_body, jump_forward = try_blocks
    try_body = body_not_empty(try_body, reader.get_lineno())

    # begin capture error handlers:
    handlers = []

    while reader.peek().opcode != 88: # END_FINALLY
        catch_lineno = reader.get_lineno()

        handler = ast.ExceptHandler()

        if reader.peek().opcode != 1:
            # except ?: ... <- with some type match
            reader.pop_assert(4) # DUP_TOP
            type_match_state = CodeState()
            walk_until_opcodes(reader, type_match_state, 107) # COMPARE_OP
            _ = reader.pop_assert(107)
            reader.pop_assert(114) # POP_JUMP_IF_FALSE

            assert _.argval == 'exception match'
            handler.type = type_match_state.pop()
            assert not type_match_state.get_value()

        reader.pop_assert(1) # POP_TOP

        maybe_exc_name = reader.pop() # maybe POP_TOP or STORE_FAST
        if maybe_exc_name.opcode == 125: # STORE_FAST
            handler.name = maybe_exc_name.argval
        else:
            handler.name = None

        handler.lineno = reader.get_lineno()

        reader.pop_assert(1) # POP_TOP

        catch_state = CodeState()
        walk_until_opcodes(reader, catch_state, 89) # POP_EXCEPT
        except_body: list = catch_state.get_value()

        if handler.name:
            assert len(except_body) == 1 and isinstance(except_body[0], ast.Try)
            exc_var_cleanup: ast.Try = except_body[0]
            except_body = exc_var_cleanup.body

        reader.pop_assert(89) # POP_EXCEPT will update loneno

        handler.body = body_not_empty(except_body, reader.get_lineno())

        jump_forward = reader.pop_assert(110) # JUMP_FORWARD

        handlers.append(handler)

    reader.pop_assert(88) # END_FINALLY
    orelse_state = CodeState()
    walk_until_offset(reader, orelse_state, jump_forward.argval)
    orelse_body = orelse_state.get_value()

    node = ast.Try(
        lineno=lineno,
        body=try_body,
        handlers=handlers,
        orelse=orelse_body,
        finalbody=[],
    )
    state.add_node(node)

@op('SETUP_FINALLY', 122)
def on_instr_setup_finally(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    lineno = reader.get_lineno()

    try_state = CodeState(scope=Scope.TRY)
    walk_until_offset(reader, try_state, instr.argval)

    # pop noop
    try_state.state.pop(ID_POP_BLOCK)
    _ = try_state.pop()
    assert isinstance(_, ast.NameConstant) and _.value is None

    try_block, _ = try_state.get_blocks()
    assert not _

    finally_state = CodeState(scope=Scope.FINALLY)
    walk_until_opcodes(reader, finally_state, 88) # END_FINALLY
    reader.pop_assert(88) # END_FINALLY

    finally_body = finally_state.get_value()

    if len(try_block) == 1 and isinstance(try_block[0], ast.Try):
        node: ast.Try = try_block[0]
        node.finalbody = finally_body

    else:
        node = ast.Try(
            lineno=lineno,
            body=try_block,
            handlers=[],
            orelse=[],
            finalbody=finally_body,
        )
    state.add_node(node)

@op('END_FINALLY', 88)
def on_instr_end_finally(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    pass

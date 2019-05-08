# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import ast
import dis
import itertools


CTX_LOAD = ast.Load()
CTX_STORE = ast.Store()

def load(node):
    ''' set `node.ctx` to `ast.Load` and return it '''
    node.ctx = CTX_LOAD
    return node

def store(node):
    ''' set `node.ctx` to `ast.Load` and return it '''
    assert isinstance(node, ast.Name), node
    node.ctx = CTX_STORE
    return node

class CodeReader:
    def __init__(self, instructions):
        # reversed will fast
        self._instructions = list(reversed(instructions))
        self._lineno = None

    def __bool__(self):
        return bool(self._instructions)

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
    def __init__(self):
        self._ast_stack = []
        self._load_stack = []

    def __repr__(self):
        return f'a({self._ast_stack!r}), l({self._load_stack!r})'

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
        self._ast_stack.append(node)

    def get_value(self) -> list:
        assert not self._load_stack
        return self._ast_stack.copy()


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

def _get_ast_value(reader, value, ctx_cls=None):
    ctx = None if ctx_cls is None else ctx_cls()

    if value is None or value is True or value is False:
        return ast.NameConstant(
            value,
            lineno=reader.get_lineno(),
            ctx=ctx
        )

    if isinstance(value, (int, float, str)):
        if isinstance(value, (int, float)):
            cls = ast.Num
        elif isinstance(value, str):
            cls = ast.Str

        return cls(
            value,
            lineno=reader.get_lineno(),
            ctx=ctx
        )

    if isinstance(value, (tuple, list, set)):
        if isinstance(value, tuple):
            cls = ast.Tuple
        elif isinstance(value, list):
            cls = ast.List
        elif isinstance(value, set):
            cls = ast.Set

        return cls(
            elts=list(value),
            lineno=reader.get_lineno(),
            ctx=ctx
        )

    return ast.Constant(
        value,
        lineno=reader.get_lineno(),
        ctx=ctx
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
    sub_state = state.copy_with_load(2)
    state.pop_seq(2)
    # rot_two
    for x in reversed(sub_state.pop_seq(2)):
        sub_state.push(x)

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
    sub_state = state.copy_with_load(1)
    state.pop()

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

@op('UNARY_POSITIVE', 10, op=ast.UAdd)
@op('UNARY_NEGATIVE', 11, op=ast.USub)
@op('UNARY_NOT', 12, op=ast.Not)
@op('UNARY_INVERT', 15, op=ast.Invert)
def on_instr_unary_op(reader: CodeReader, state: CodeState, instr: dis.Instruction, op):
    node = ast.UnaryOp(
        lineno=reader.get_lineno(),
        col_offset=4,
        op=op,
        operand=state.pop()
    )
    state.push(node)

@op('BINARY_POWER', 19, op=ast.Pow)
@op('BINARY_MULTIPLY', 20, op=ast.Mult)
@op('BINARY_MODULO', 22, op=ast.Mod)
@op('BINARY_ADD', 23, op=ast.Add)
@op('BINARY_SUBTRACT', 24, op=ast.Sub)
@op('BINARY_FLOOR_DIVIDE', 26, op=ast.FloorDiv)
@op('BINARY_TRUE_DIVIDE', 27, op=ast.Div)
@op('BINARY_LSHIFT', 62, op=ast.LShift)
@op('BINARY_RSHIFT', 63, op=ast.RShift)
@op('BINARY_AND', 64, op=ast.BitAnd)
@op('BINARY_XOR', 65, op=ast.BitXor)
@op('BINARY_OR', 66, op=ast.BitOr)
def on_instr_binary_op(reader: CodeReader, state: CodeState, instr: dis.Instruction, op):
    l, r = state.pop_seq(2)
    node = ast.BinOp(
        lineno=reader.get_lineno(),
        left=l, right=r,
        op=op
    )
    state.push(node)

@op('INPLACE_FLOOR_DIVIDE', 28, op=ast.FloorDiv)
@op('INPLACE_TRUE_DIVIDE', 29, op=ast.Div)
@op('INPLACE_ADD', 55, op=ast.Add)
@op('INPLACE_SUBTRACT', 56, op=ast.Sub)
@op('INPLACE_MULTIPLY', 57, op=ast.Mult)
@op('INPLACE_MODULO', 59, op=ast.Mod)
@op('INPLACE_POWER', 67, op=ast.Pow)
@op('INPLACE_LSHIFT', 75, op=ast.LShift)
@op('INPLACE_RSHIFT', 76, op=ast.RShift)
@op('INPLACE_AND', 77, op=ast.BitAnd)
@op('INPLACE_XOR', 78, op=ast.BitXor)
@op('INPLACE_OR', 79, op=ast.BitOr)
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

@op('BREAK_LOOP', 80)
def on_instr_break_loop(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    state.push(ast.Break(
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
    state.push(_get_ast_value(reader, instr.argval, ctx_cls=ast.Load))

@op('BUILD_TUPLE', 102)
def on_instr_build_tuple(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    node = ast.Tuple(
        lineno=reader.get_lineno(),
        elts=state.pop_seq(instr.arg),
        ctx=CTX_LOAD
    )
    state.push(node)

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
        keys = keys,
        values = values
    )
    if keys:
        node.lineno = keys[0].lineno
    else:
        node.lineno =  reader.get_lineno()
    state.push(node)

@op('BUILD_CONST_KEY_MAP', 156)
def on_instr_build_const_key_map(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    keys = state.pop()
    values = state.pop_seq(instr.argval)
    lineno = min([reader.get_lineno(), keys.lineno] + [v.lineno for v in values])
    node = ast.Dict(
        lineno=lineno,
        keys=[_get_ast_value(reader, k) for k in keys.elts],
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
    if not opcls:
        raise NotImplementedError(instr)

    linenos = [reader.get_lineno()]

    left, right = state.pop_seq(2)
    linenos.append(left.lineno)
    linenos.append(right.lineno)
    node = ast.Compare(left=left, ops=[opcls()], comparators=[right])

    lineno = min(linenos)
    node.lineno = lineno
    state.push(node)

@op('JUMP_IF_FALSE_OR_POP', 111, op=ast.And)
@op('JUMP_IF_TRUE_OR_POP', 112, op=ast.Or)
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
    state.add_node(node)

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
    state.push(
        load(ast.Name(id=instr.argval, lineno=reader.get_lineno()))
    )

@op('STORE_FAST', 125)
def on_instr_store(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    value = state.pop()
    if isinstance(value, ast.stmt):
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

@op('POP_BLOCK', 87)
def on_instr_pop_block(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # end of loop block, just ignore
    pass

@op('JUMP_ABSOLUTE', 113)
def on_instr_jump_absolute(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    # end of loop block, just ignore
    pass

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
        ast.keyword(arg=n, value=v) for (n, v) in zip(kw, args[-len(kw):])
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

def _get_ast_store_name(reader: CodeReader):
    instr = reader.pop_assert(125) # STORE_FAST
    return ast.Name(
        lineno=reader.get_lineno(),
        id=instr.argval,
        ctx=CTX_STORE
    )

def _parse_instr_unpack_sequence(reader: CodeReader, state: CodeState, instr):
    target_tuple = ast.Tuple(
        lineno=reader.get_lineno(),
        ctx=CTX_STORE,
        elts = []
    )
    for _ in range(instr.argval):
        target_tuple.elts.append(_get_ast_store_name(reader))
    return target_tuple

@op('UNPACK_SEQUENCE', 92)
def on_instr_unpack_sequence(reader: CodeReader, state: CodeState, instr: dis.Instruction):
    target = _parse_instr_unpack_sequence(reader, state, instr)
    value = state.pop()
    node = ast.Assign(
        lineno=reader.get_lineno(),
        targets=[target],
        value=value,
    )
    state.add_node(node)

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

# byetcode2ast

A library for turn python byetcode into abstract syntax tree (`ast`).

## Usage

``` py
from bytecode2ast import parse_func

func_def_ast: ast.FunctionDef = parse_func(some_func)
```

## Testing

The decompiled ast node should have the same instructions with original python object.

## Other Libs

### uncompyle6

``` py
import ast
from uncompyle6 import deparse_code2str

src_code = deparse_code2str(some_func.__code__, out=io.StringIO())
module_ast: ast.Module = ast.parse(src_code) # <- but I need FunctionDef!
```

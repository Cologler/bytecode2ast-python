# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

import dis

from bytecode2ast import parse_func, create_module

def get_func_from_exec(code, name):
    g = {}
    exec(code, g, g)
    return g[name]

def get_instrs(func):
    return list(dis.Bytecode(func))

def get_instrs_from_b2a(func):
    name = func.__name__
    b2a_ast = parse_func(func)
    module = create_module([b2a_ast])
    new_func = get_func_from_exec(compile(module, '<string>', 'exec'), name)
    return get_instrs(new_func)

def test_try_except():
    def func():
        try:
            a = 1
        except:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_finally():
    def func():
        try:
            a = 1
        except:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_finally():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_multi_except_type():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1
        except KeyError:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_multi_except_type_finally():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1
        except KeyError:
            f = 1
        finally:
            k = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_except():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1
        except:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_except_finally():
    def func():
        try:
            a = 1
        except TypeError:
            c = 1
        except:
            f = 1
        finally:
            k = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_type():
    def func():
        try:
            a = 1
        except (TypeError, KeyError):
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_type_finally():
    def func():
        try:
            a = 1
        except (TypeError, KeyError):
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_as():
    def func():
        try:
            a = 1
        except TypeError as e:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_type_as_finally():
    def func():
        try:
            a = 1
        except TypeError as e:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_as():
    def func():
        try:
            a = 1
        except (TypeError, ValueError) as e:
            c = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_multi_types_as_finally():
    def func():
        try:
            a = 1
        except (TypeError, ValueError) as e:
            c = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_finally():
    def func():
        try:
            a = 1
        finally:
            f = 6

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_finally_deep():
    def func():
        try:
            try:
                a = 2
            finally:
                b = 3
        finally:
            d = 5

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_full():
    def func():
        try:
            a = 1
        except (KeyError, ValueError) as c:
            t = 1
        except TypeError as e:
            b = 1
        except:
            c = 1
        else:
            d = 1
        finally:
            f = 1

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_empty_except():
    def func():
        try:
            pass
        except:
            d = 4

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_empty():
    def func():
        try:
            a = 1
        except:
            pass

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_deep_try_except():
    def func():
        try:
            try:
                a = 1
            except:
                b = 2
                try:
                    d = 5
                except:
                    pass
            c = 3
        except:
            d = 4

    assert get_instrs(func) == get_instrs_from_b2a(func)

def test_try_except_finally():
    def func():
        try:
            try:
                a()
            except:
                b()
        finally:
            c()

    assert get_instrs(func) == get_instrs_from_b2a(func)

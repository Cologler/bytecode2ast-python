# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_func

def test_if():
    def func():
        if __name__ == 'a':
            return 10
        elif s + ds == 111:
            if s and f:
                return 1
            else:
                pass
        return None

    assert_func(func)

def test_white():
    return # TODO

    def func():
        while a and b:
            if c == 12:
                break
            v = 13
        else:
            return 12

    assert_func(func)

def test_for():
    return # TODO
    
    def func():
        for z in []:
            if c == 12:
                break
            v = 13
        else:
            return 12

    assert_func(func)


# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_example

def test_build_class_empty():
    def func():
        class A:
            pass

    assert_example(func)

def test_build_class_with_methods():
    def func():
        class A:
            def f():
                pass

            def s():
                pass

    assert_example(func)

def test_build_class_with_metaclass():
    def func():
        class A(metaclass=type):
            pass

    assert_example(func)

def test_build_class_with_bases():
    def func():
        class A(C, D, E):
            pass

    assert_example(func)

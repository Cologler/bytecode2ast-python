# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import assert_func

def test_build_const_map():
    def func():
        {'a': 1, 'b': 2}

    assert_func(func)

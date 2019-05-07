# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
#
# ----------

from utils import get_instrs_from_b2a, get_instrs

def test_build_const_map():
    def func():
        {'a': 1, 'b': 2}

    assert get_instrs(func) == get_instrs_from_b2a(func)

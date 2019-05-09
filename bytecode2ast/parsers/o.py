# -*- coding: utf-8 -*-
#
# Copyright (c) 2019~2999 - Cologler <skyoflw@gmail.com>
# ----------
# some object for parser
# ----------

class ID:
    def __init__(self, name):
        self._name = name # a name use to debug

    def __repr__(self):
        return f'ID({self._name})'

    def __str__(self):
        return repr(self)

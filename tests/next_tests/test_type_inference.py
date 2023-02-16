# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next import type_inference as ti


def test_renamer():
    class Foo(ti.Type):
        bar: ti.Type
        baz: ti.Type

    class Bar(ti.Type):
        ...

    r = ti._Renamer()
    actual = [
        (
            ti._Box(value=Foo(bar=ti.TypeVar(idx=0), baz=ti.TypeVar(idx=1))),
            ti._Box(value=ti.TypeVar(idx=0)),
        )
    ]
    src = ti.TypeVar(idx=0)
    dst = ti.TypeVar(idx=1)
    for s, t in actual:
        r.register(s)
        r.register(t)
    r.register(src)
    r.register(dst)
    r.rename(src, dst)
    expected = [
        (
            ti._Box(value=Foo(bar=ti.TypeVar(idx=1), baz=ti.TypeVar(idx=1))),
            ti._Box(value=ti.TypeVar(idx=1)),
        )
    ]
    assert actual == expected


def test_custom_type_inference():
    class Fun(ti.Type):
        arg: ti.Type
        ret: ti.Type

    class Basic(ti.Type):
        name: str

    class SpecialFun(ti.Type):
        arg_and_ret: ti.Type

        def __eq__(self, other):
            if isinstance(other, Fun):
                return self.arg_and_ret == other.arg == other.ret
            return isinstance(other, SpecialFun) and self.arg_and_ret == other.arg_and_ret

        def handle_constraint(self, other, add_constraint):
            if isinstance(other, Fun):
                add_constraint(self.arg_and_ret, other.arg)
                add_constraint(self.arg_and_ret, other.ret)
                return True
            return False

    v = [ti.TypeVar(idx=i) for i in range(5)]
    constraints = {
        (v[0], SpecialFun(arg_and_ret=v[2])),
        (Fun(arg=v[0], ret=v[3]), v[4]),
        (Basic(name="int"), v[1]),
        (v[1], v[2]),
    }
    dtype = v[4]

    expected = Fun(arg=Fun(arg=Basic(name="int"), ret=Basic(name="int")), ret=ti.TypeVar(idx=0))

    actual = ti.reindex_vars(ti.unify(dtype, constraints))
    assert actual == expected

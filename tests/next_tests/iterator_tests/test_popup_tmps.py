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

import pytest

from gt4py.eve.utils import UIDs
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.popup_tmps import PopupTmps


@pytest.fixture
def fresh_uid_sequence():
    UIDs.reset_sequence()


def test_trivial_single_lift(fresh_uid_sequence):
    testee = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="bar_inp")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="foo_inp")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[ir.SymRef(id="foo_inp")],
                                    ),
                                )
                            ],
                        ),
                        args=[ir.SymRef(id="bar_inp")],
                    )
                ],
            ),
        ),
        args=[ir.SymRef(id="inp")],
    )
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="bar_inp"), ir.Sym(id="_lift_1")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[ir.SymRef(id="_lift_1")],
            ),
        ),
        args=[
            ir.SymRef(id="inp"),
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[
                        ir.Lambda(
                            params=[ir.Sym(id="foo_inp")],
                            expr=ir.FunCall(
                                fun=ir.SymRef(id="deref"),
                                args=[ir.SymRef(id="foo_inp")],
                            ),
                        )
                    ],
                ),
                args=[ir.SymRef(id="inp")],
            ),
        ],
    )
    actual = PopupTmps().visit(testee)
    assert actual == expected


def test_trivial_multiple_lifts(fresh_uid_sequence):
    testee = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="baz_inp")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="bar_inp")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[
                                            ir.FunCall(
                                                fun=ir.FunCall(
                                                    fun=ir.SymRef(id="lift"),
                                                    args=[
                                                        ir.Lambda(
                                                            params=[ir.Sym(id="foo_inp")],
                                                            expr=ir.FunCall(
                                                                fun=ir.SymRef(id="deref"),
                                                                args=[ir.SymRef(id="foo_inp")],
                                                            ),
                                                        )
                                                    ],
                                                ),
                                                args=[ir.SymRef(id="bar_inp")],
                                            )
                                        ],
                                    ),
                                )
                            ],
                        ),
                        args=[ir.SymRef(id="baz_inp")],
                    )
                ],
            ),
        ),
        args=[ir.SymRef(id="inp")],
    )
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="baz_inp"), ir.Sym(id="_lift_2")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[ir.SymRef(id="_lift_2")],
            ),
        ),
        args=[
            ir.SymRef(id="inp"),
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[
                        ir.Lambda(
                            params=[
                                ir.Sym(id="bar_inp"),
                                ir.Sym(id="_lift_1"),
                            ],
                            expr=ir.FunCall(
                                fun=ir.SymRef(id="deref"),
                                args=[
                                    ir.SymRef(id="_lift_1"),
                                ],
                            ),
                        )
                    ],
                ),
                args=[
                    ir.SymRef(id="inp"),
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="foo_inp")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[ir.SymRef(id="foo_inp")],
                                    ),
                                )
                            ],
                        ),
                        args=[ir.SymRef(id="inp")],
                    ),
                ],
            ),
        ],
    )
    actual = PopupTmps().visit(testee)
    assert actual == expected


def test_capture(fresh_uid_sequence):
    testee = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="x")],
            expr=ir.FunCall(
                fun=ir.SymRef(id="deref"),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[ir.SymRef(id="x")],
                                    ),
                                )
                            ],
                        ),
                        args=[],
                    )
                ],
            ),
        ),
        args=[ir.SymRef(id="inp")],
    )
    actual = PopupTmps().visit(testee)
    assert actual == testee


def test_crossref():
    testee = ir.FunCall(
        fun=ir.Lambda(
            params=[ir.Sym(id="x")],
            expr=ir.FunCall(
                fun=ir.Lambda(
                    params=[ir.Sym(id="x1")],
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x1")]),
                ),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="x2")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[ir.SymRef(id="x2")],
                                    ),
                                )
                            ],
                        ),
                        args=[
                            ir.FunCall(
                                fun=ir.FunCall(
                                    fun=ir.SymRef(id="lift"),
                                    args=[
                                        ir.Lambda(
                                            params=[ir.Sym(id="x3")],
                                            expr=ir.FunCall(
                                                fun=ir.SymRef(id="deref"),
                                                args=[ir.SymRef(id="x3")],
                                            ),
                                        )
                                    ],
                                ),
                                args=[ir.SymRef(id="x")],
                            )
                        ],
                    )
                ],
            ),
        ),
        args=[ir.SymRef(id="x")],
    )
    expected = ir.FunCall(
        fun=ir.Lambda(
            params=[
                ir.Sym(id="x"),
                ir.Sym(id="_lift_1"),
                ir.Sym(id="_lift_2"),
            ],
            expr=ir.FunCall(
                fun=ir.Lambda(
                    params=[ir.Sym(id="x1")],
                    expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x1")]),
                ),
                args=[ir.SymRef(id="_lift_2")],
            ),
        ),
        args=[
            ir.SymRef(id="x"),
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[
                        ir.Lambda(
                            params=[ir.Sym(id="x3")],
                            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x3")]),
                        )
                    ],
                ),
                args=[ir.SymRef(id="x")],
            ),
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="lift"),
                    args=[
                        ir.Lambda(
                            params=[ir.Sym(id="x2")],
                            expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="x2")]),
                        )
                    ],
                ),
                args=[
                    ir.FunCall(
                        fun=ir.FunCall(
                            fun=ir.SymRef(id="lift"),
                            args=[
                                ir.Lambda(
                                    params=[ir.Sym(id="x3")],
                                    expr=ir.FunCall(
                                        fun=ir.SymRef(id="deref"),
                                        args=[ir.SymRef(id="x3")],
                                    ),
                                )
                            ],
                        ),
                        args=[ir.SymRef(id="x")],
                    )
                ],
            ),
        ],
    )
    actual = PopupTmps().visit(testee)
    assert actual == expected

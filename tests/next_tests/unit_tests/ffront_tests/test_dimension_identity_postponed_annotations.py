# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
A named collection with postponed (string) annotations resolves them in its module namespace.

Its type translation therefore depends on that namespace, not only on the type object: after
only the dimension is redefined (a re-run notebook cell), the same `Pair` names the new class,
and a cached translation must not replay the old one into the edge identity check.
"""

import importlib
import sys
import textwrap

import numpy as np

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.type_system import type_translation


_DIMS_AND_PAIR = textwrap.dedent(
    """
    from __future__ import annotations

    import dataclasses

    import gt4py.next as gtx


    class IDim(gtx.CartesianAxisIndex): ...


    @dataclasses.dataclass
    class Pair:
        a: gtx.Field[gtx.Dims[IDim], gtx.float64]
    """
)

_PROGRAM = textwrap.dedent(
    """
    from __future__ import annotations

    import gt4py.next as gtx

    from postponed_pair_mod import IDim, Pair


    @gtx.field_operator
    def first(p: Pair) -> gtx.Field[gtx.Dims[IDim], gtx.float64]:
        return p.a


    @gtx.program
    def prog(p: Pair, out: gtx.Field[gtx.Dims[IDim], gtx.float64]):
        first(p, out=out)
    """
)


def test_redefined_dimension_in_postponed_annotations(tmp_path, monkeypatch):
    (tmp_path / "postponed_pair_mod.py").write_text(_DIMS_AND_PAIR)
    (tmp_path / "postponed_pair_prog.py").write_text(_PROGRAM)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("postponed_pair_mod", "postponed_pair_prog"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    mod = importlib.import_module("postponed_pair_mod")
    old = mod.IDim
    type_translation.from_type_hint(mod.Pair)  # the first run translates `Pair` once

    exec("class IDim(gtx.CartesianAxisIndex): ...", mod.__dict__)  # only the dimension cell
    new = mod.IDim
    assert new is not old

    with common.capture_dimension_classes() as captured:
        type_translation.from_type_hint(mod.Pair)
    assert [cls for _, cls in captured] == [new]  # what the annotation names *now*

    # a program written now, called with fields of the current class, is valid
    prog = importlib.import_module("postponed_pair_prog").prog
    out = gtx.as_field([new], np.zeros(4))
    prog.with_backend(None)(mod.Pair(gtx.as_field([new], np.ones(4))), out, offset_provider={})
    assert np.allclose(out.asnumpy(), 1.0)

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Classes at the Python edge, refs inside: a redefined dimension is rejected when called.

A notebook cell re-run redefines a dimension with the same kind, so the new class has a
`DimensionRef` *equal* to the old one. Arguments built on the old class must still be rejected
at the argument check, in embedded and compiled mode alike, as when type specifications held
classes. The dimensions come from another module, `_redefinable_dims`, which is reloaded (the
"re-run") after the arguments of the first run were built.
"""

import dataclasses
import importlib

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, errors

from . import _redefinable_dims as dims


_DIFFERENT = "two different classes named"

# --- the first run: data built on the original classes
old = dims
old_classes = {name: getattr(old, name) for name in ("IDim", "KDim", "Vertex", "Edge", "V2EDim")}
old_i = gtx.as_field([old_classes["IDim"]], np.ones(4))
old_k = gtx.as_field([old_classes["KDim"]], np.ones(4))
old_i_half = gtx.as_field([common.Staggered[old_classes["IDim"]]], np.ones(4))

# --- the re-run: new classes, equal refs; everything below is declared on them
dims = importlib.reload(dims)
IDim, KDim, Vertex, Edge, V2EDim = dims.IDim, dims.KDim, dims.Vertex, dims.Edge, dims.V2EDim
V2E = gtx.FieldOffset(V2EDim.tag, source=Edge, target=(Vertex, V2EDim))


def new_i():
    return gtx.as_field([IDim], np.zeros(4))


@gtx.field_operator
def fop(x: gtx.Field[gtx.Dims[IDim], gtx.float64]) -> gtx.Field[gtx.Dims[IDim], gtx.float64]:
    return x + 1.0


@gtx.program
def prog(x: gtx.Field[gtx.Dims[IDim], gtx.float64], y: gtx.Field[gtx.Dims[IDim], gtx.float64]):
    fop(x, out=y)


@gtx.field_operator
def list_form(x: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return x


@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def cumsum(state: float, x: float) -> float:
    return state + x


@dataclasses.dataclass
class Pair:
    a: gtx.Field[gtx.Dims[IDim], gtx.float64]
    b: gtx.Field[gtx.Dims[IDim], gtx.float64]


@gtx.field_operator
def sum_pair(p: Pair) -> gtx.Field[gtx.Dims[IDim], gtx.float64]:
    return p.a + p.b


@gtx.program
def pair_prog(p: Pair, out: gtx.Field[gtx.Dims[IDim], gtx.float64]):
    sum_pair(p, out=out)


@gtx.field_operator
def staggered(
    x: gtx.Field[gtx.Dims[common.Staggered[IDim]], gtx.float64],
) -> gtx.Field[gtx.Dims[common.Staggered[IDim]], gtx.float64]:
    return x


@gtx.field_operator
def _first_neighbor(
    e: gtx.Field[gtx.Dims[Edge], gtx.float64],
) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
    return e(V2E[0])


@gtx.program
def first_neighbor(
    e: gtx.Field[gtx.Dims[Edge], gtx.float64], out: gtx.Field[gtx.Dims[Vertex], gtx.float64]
):
    _first_neighbor(e, out=out)


BACKENDS = pytest.mark.parametrize("backend", [None, gtx.gtfn_cpu], ids=["embedded", "gtfn_cpu"])


def test_redefinition_keeps_an_equal_ref():
    assert old_classes["IDim"] is not IDim
    assert common.dimension_ref(old_classes["IDim"]) == common.dimension_ref(IDim)


@BACKENDS
def test_program(backend):
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        prog.with_backend(backend)(old_i, new_i(), offset_provider={})


@BACKENDS
def test_field_operator(backend):
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        fop.with_backend(backend)(old_i, out=new_i(), offset_provider={})


@BACKENDS
def test_list_form_annotation(backend):
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        list_form.with_backend(backend)(old_i, out=new_i(), offset_provider={})


@BACKENDS
def test_scan_operator(backend):
    # the parameters are scalars: the declared dimension is the scan axis
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        cumsum.with_backend(backend)(
            old_k, out=gtx.as_field([KDim], np.zeros(4)), offset_provider={}
        )


@BACKENDS
def test_named_collection(backend):
    # only the collection's *values* are stale: its annotations name the current class
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        sum_pair.with_backend(backend)(Pair(old_i, old_i), out=new_i(), offset_provider={})
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        pair_prog.with_backend(backend)(Pair(old_i, old_i), new_i(), offset_provider={})


@BACKENDS
def test_staggered(backend):
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        staggered.with_backend(backend)(
            old_i_half,
            out=gtx.as_field([common.Staggered[IDim]], np.zeros(4)),
            offset_provider={},
        )


@BACKENDS
def test_connectivity_codomain(backend):
    e = gtx.as_field([Edge], np.arange(3.0))
    out = gtx.as_field([Vertex], np.zeros(2))
    v2e = gtx.as_connectivity(
        domain={Vertex: 2, V2EDim: 1},
        codomain=old_classes["Edge"],  # only the codomain is stale
        data=np.array([[2], [0]], np.int32),
    )
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        first_neighbor.with_backend(backend)(e, out, offset_provider={V2EDim.tag: v2e})


@BACKENDS
def test_connectivity_local_dimension(backend):
    # no annotation names `V2EDim`: the declaration is the `FieldOffset` the operator uses
    e = gtx.as_field([Edge], np.arange(3.0))
    out = gtx.as_field([Vertex], np.zeros(2))
    v2e = gtx.as_connectivity(
        domain={Vertex: 2, old_classes["V2EDim"]: 1},  # only the local dimension is stale
        codomain=Edge,
        data=np.array([[2], [0]], np.int32),
    )
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        first_neighbor.with_backend(backend)(e, out, offset_provider={V2EDim.tag: v2e})


def test_current_classes_are_accepted_embedded():
    a = gtx.as_field([IDim], np.ones(4))
    out = gtx.as_field([IDim], np.zeros(4))
    prog.with_backend(None)(a, out, offset_provider={})
    assert np.allclose(out.asnumpy(), 2.0)
    sum_pair.with_backend(None)(Pair(a, a), out=out, offset_provider={})
    assert np.allclose(out.asnumpy(), 2.0)

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import set_at, fendef, fundef

from next_tests.unit_tests.conftest import program_processor, run_processor


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim")

# semantics of stencil return that is called from the fencil (after `:` the structure of the output)
# `return a` -> a: field
# `return make_tuple(a)` -> (a,): [field] or (field)
# `return a,b` -> (a,b): [field, field] or (field, field)
# `return make_tuple(a,b)` -> (a,b): [field, field]
# `return make_tuple(a), make_tuple(b)` -> ((a,), (b,)): [(field,), (field,)]
# `return make_tuple(make_tuple(a,b))` -> ((a,b)): [(field,field)]


@fundef
def tuple_output1(inp1, inp2):
    return deref(inp1), deref(inp2)


@fundef
def tuple_output2(inp1, inp2):
    return make_tuple(deref(inp1), deref(inp2))


@pytest.mark.parametrize("stencil", [tuple_output1, tuple_output2])
@pytest.mark.uses_tuple_returns
def test_tuple_output(program_processor, stencil):
    program_processor, validate = program_processor

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))

    out = (
        gtx.as_field([IDim, JDim, KDim], np.zeros(shape)),
        gtx.as_field([IDim, JDim, KDim], np.zeros(shape)),
    )

    dom = {IDim: range(0, shape[0]), JDim: range(0, shape[1]), KDim: range(0, shape[2])}
    run_processor(stencil[dom], program_processor, inp1, inp2, out=out, offset_provider={})
    if validate:
        assert np.allclose(inp1.asnumpy(), out[0].asnumpy())
        assert np.allclose(inp2.asnumpy(), out[1].asnumpy())


@fundef
def tuple_of_tuple_output1(inp1, inp2, inp3, inp4):
    return (deref(inp1), deref(inp2)), (deref(inp3), deref(inp4))


@fundef
def tuple_of_tuple_output2(inp1, inp2, inp3, inp4):
    return make_tuple(deref(inp1), deref(inp2)), make_tuple(deref(inp3), deref(inp4))


@pytest.mark.uses_tuple_returns
def test_tuple_of_tuple_of_field_output(program_processor):
    program_processor, validate = program_processor

    @fundef
    def stencil(inp1, inp2, inp3, inp4):
        return make_tuple(deref(inp1), deref(inp2)), make_tuple(deref(inp3), deref(inp4))

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp3 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp4 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))

    out = (
        (
            gtx.as_field([IDim, JDim, KDim], np.zeros(shape)),
            gtx.as_field([IDim, JDim, KDim], np.zeros(shape)),
        ),
        (
            gtx.as_field([IDim, JDim, KDim], np.zeros(shape)),
            gtx.as_field([IDim, JDim, KDim], np.zeros(shape)),
        ),
    )

    dom = {IDim: range(0, shape[0]), JDim: range(0, shape[1]), KDim: range(0, shape[2])}
    run_processor(
        stencil[dom], program_processor, inp1, inp2, inp3, inp4, out=out, offset_provider={}
    )
    if validate:
        assert np.allclose(inp1.asnumpy(), out[0][0].asnumpy())
        assert np.allclose(inp2.asnumpy(), out[0][1].asnumpy())
        assert np.allclose(inp3.asnumpy(), out[1][0].asnumpy())
        assert np.allclose(inp4.asnumpy(), out[1][1].asnumpy())


@pytest.mark.parametrize("stencil", [tuple_output1, tuple_output2])
def test_tuple_of_field_output_constructed_inside(program_processor, stencil):
    program_processor, validate = program_processor

    @fendef
    def fencil(size0, size1, size2, inp1, inp2, out1, out2):
        domain = cartesian_domain(
            named_range(IDim, 0, size0), named_range(JDim, 0, size1), named_range(KDim, 0, size2)
        )
        set_at(as_fieldop(stencil, domain)(inp1, inp2), domain, make_tuple(out1, out2))

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))

    out1 = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))
    out2 = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))

    run_processor(
        fencil,
        program_processor,
        shape[0],
        shape[1],
        shape[2],
        inp1,
        inp2,
        out1,
        out2,
        offset_provider={},
    )
    if validate:
        assert np.allclose(inp1.asnumpy(), out1.asnumpy())
        assert np.allclose(inp2.asnumpy(), out2.asnumpy())


def test_asymetric_nested_tuple_of_field_output_constructed_inside(program_processor):
    program_processor, validate = program_processor

    @fundef
    def stencil(inp1, inp2, inp3):
        return make_tuple(deref(inp1), deref(inp2)), deref(inp3)

    @fendef
    def fencil(size0, size1, size2, inp1, inp2, inp3, out1, out2, out3):
        domain = cartesian_domain(
            named_range(IDim, 0, size0), named_range(JDim, 0, size1), named_range(KDim, 0, size2)
        )
        set_at(
            as_fieldop(stencil, domain)(inp1, inp2, inp3),
            domain,
            make_tuple(make_tuple(out1, out2), out3),
        )

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp3 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))

    out1 = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))
    out2 = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))
    out3 = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))

    run_processor(
        fencil,
        program_processor,
        shape[0],
        shape[1],
        shape[2],
        inp1,
        inp2,
        inp3,
        out1,
        out2,
        out3,
        offset_provider={},
    )
    if validate:
        assert np.allclose(inp1.asnumpy(), out1.asnumpy())
        assert np.allclose(inp2.asnumpy(), out2.asnumpy())
        assert np.allclose(inp3.asnumpy(), out3.asnumpy())


@pytest.mark.xfail(reason="Implement wrapper for extradim as tuple")
@pytest.mark.parametrize("stencil", [tuple_output1, tuple_output2])
def test_field_of_extra_dim_output(program_processor, stencil):
    program_processor, validate = program_processor

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))

    out_np = np.zeros(shape + [2])
    out = gtx.as_field([IDim, JDim, KDim, None], out_np)

    dom = {IDim: range(0, shape[0]), JDim: range(0, shape[1]), KDim: range(0, shape[2])}
    run_processor(stencil[dom], program_processor, inp1, inp2, out=out, offset_provider={})
    if validate:
        assert np.allclose(inp1, out_np[:, :, :, 0])
        assert np.allclose(inp2, out_np[:, :, :, 1])


@fundef
def tuple_input(inp):
    inp_deref = deref(inp)
    return tuple_get(0, inp_deref) + tuple_get(1, inp_deref)


@pytest.mark.uses_tuple_args
@pytest.mark.uses_tuple_iterator
def test_tuple_field_input(program_processor):
    program_processor, validate = program_processor

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = gtx.as_field(
        [IDim, JDim, KDim],
        rng.normal(
            size=(shape[0], shape[1], shape[2] + 1)
        ),  # TODO(havogt) currently we allow different sizes, needed for icon4py compatibility
    )

    out = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))

    dom = {IDim: range(0, shape[0]), JDim: range(0, shape[1]), KDim: range(0, shape[2])}
    run_processor(tuple_input[dom], program_processor, (inp1, inp2), out=out, offset_provider={})
    if validate:
        assert np.allclose(inp1.asnumpy() + inp2.asnumpy()[:, :, :-1], out.asnumpy())


@pytest.mark.xfail(reason="Implement wrapper for extradim as tuple")
def test_field_of_extra_dim_input(program_processor):
    program_processor, validate = program_processor

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp2 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp = np.stack((inp1, inp2), axis=-1)

    inp = gtx.as_field([IDim, JDim, KDim, None], inp)
    out = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))

    dom = {IDim: range(0, shape[0]), JDim: range(0, shape[1]), KDim: range(0, shape[2])}
    run_processor(tuple_input[dom], program_processor, inp, out=out, offset_provider={})
    if validate:
        assert np.allclose(np.asarray(inp1) + np.asarray(inp2), out)


@fundef
def tuple_tuple_input(inp):
    inp_deref = deref(inp)
    return (
        tuple_get(0, tuple_get(0, inp_deref))
        + tuple_get(1, tuple_get(0, inp_deref))
        + tuple_get(0, tuple_get(1, inp_deref))
        + tuple_get(1, tuple_get(1, inp_deref))
    )


@pytest.mark.uses_tuple_args
@pytest.mark.uses_tuple_iterator
def test_tuple_of_tuple_of_field_input(program_processor):
    program_processor, validate = program_processor

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp3 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))
    inp4 = gtx.as_field([IDim, JDim, KDim], rng.normal(size=(shape[0], shape[1], shape[2])))

    out = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))

    dom = {IDim: range(0, shape[0]), JDim: range(0, shape[1]), KDim: range(0, shape[2])}
    run_processor(
        tuple_tuple_input[dom],
        program_processor,
        ((inp1, inp2), (inp3, inp4)),
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose(
            (inp1.asnumpy() + inp2.asnumpy() + inp3.asnumpy() + inp4.asnumpy()), out.asnumpy()
        )


@pytest.mark.xfail(reason="Implement wrapper for extradim as tuple")
def test_field_of_2_extra_dim_input(program_processor):
    program_processor, validate = program_processor

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp = gtx.as_field(
        [IDim, JDim, KDim, None, None], rng.normal(size=(shape[0], shape[1], shape[2], 2, 2))
    )

    out = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))

    dom = {IDim: range(0, shape[0]), JDim: range(0, shape[1]), KDim: range(0, shape[2])}
    run_processor(tuple_tuple_input[dom], program_processor, inp, out=out, offset_provider={})
    if validate:
        assert np.allclose(np.sum(inp, axis=(3, 4)), out)


@pytest.mark.uses_tuple_args
@pytest.mark.uses_tuple_iterator
def test_scalar_tuple_args(program_processor):
    @fundef
    def stencil(inp):
        inp_deref = deref(inp)
        return (
            tuple_get(0, inp_deref)
            + 2 * tuple_get(0, tuple_get(1, inp_deref))
            + 3 * tuple_get(1, tuple_get(1, inp_deref))
        )

    program_processor, validate = program_processor

    shape = [5]

    out = gtx.as_field([IDim], np.zeros(shape, dtype=np.int32))

    dom = {IDim: range(0, shape[0])}
    run_processor(
        stencil[dom],
        program_processor,
        (1, (2, 3)),
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose((1 + 2 * 2 + 3 * 3), out.asnumpy())


@pytest.mark.uses_tuple_args
@pytest.mark.uses_tuple_iterator
def test_mixed_field_scalar_tuple_arg(program_processor):
    @fundef
    def stencil(inp):
        inp_deref = deref(inp)
        return (
            tuple_get(0, inp_deref)
            + 2.0 * tuple_get(0, tuple_get(1, inp_deref))
            + 3.0 * tuple_get(1, tuple_get(1, inp_deref))
            + 5.0 * tuple_get(2, tuple_get(1, inp_deref))
        )

    program_processor, validate = program_processor

    shape = [5]

    rng = np.random.default_rng()

    inp = gtx.as_field([IDim], rng.normal(size=shape))
    out = gtx.as_field([IDim], np.zeros(shape))

    dom = {IDim: range(0, shape[0])}
    run_processor(
        stencil[dom],
        program_processor,
        (1.0, (2.0, inp, 3.0)),
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose((1 + 2 * 2 + 3 * inp.asnumpy() + 5 * 3), out.asnumpy())

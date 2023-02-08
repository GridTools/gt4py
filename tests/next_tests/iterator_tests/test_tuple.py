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

import numpy as np
import pytest

from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.iterator.runtime import CartesianAxis, closure, fendef, fundef

from .conftest import run_processor


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")

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


@pytest.mark.parametrize(
    "stencil",
    [tuple_output1, tuple_output2],
)
def test_tuple_output(program_processor_no_gtfn_exec, stencil):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out = (
        np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
        np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
    )

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(stencil[dom], program_processor, inp1, inp2, out=out, offset_provider={})
    if validate:
        assert np.allclose(inp1, out[0])
        assert np.allclose(inp2, out[1])


def test_tuple_of_field_of_tuple_output(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    @fundef
    def stencil(inp1, inp2, inp3, inp4):
        return make_tuple(deref(inp1), deref(inp2)), make_tuple(deref(inp3), deref(inp4))

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp3 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp4 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out_np1 = np.zeros(shape, dtype="f8, f8")
    out1 = np_as_located_field(IDim, JDim, KDim)(out_np1)
    out_np2 = np.zeros(shape, dtype="f8, f8")
    out2 = np_as_located_field(IDim, JDim, KDim)(out_np2)
    out = (out1, out2)

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(
        stencil[dom],
        program_processor,
        inp1,
        inp2,
        inp3,
        inp4,
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose(inp1, out_np1[:]["f0"])
        assert np.allclose(inp2, out_np1[:]["f1"])
        assert np.allclose(inp3, out_np2[:]["f0"])
        assert np.allclose(inp4, out_np2[:]["f1"])


def test_tuple_of_tuple_of_field_output(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    @fundef
    def stencil(inp1, inp2, inp3, inp4):
        return make_tuple(deref(inp1), deref(inp2)), make_tuple(deref(inp3), deref(inp4))

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp3 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp4 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out = (
        (
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
        ),
        (
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
        ),
    )

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(
        stencil[dom],
        program_processor,
        inp1,
        inp2,
        inp3,
        inp4,
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose(inp1, out[0][0])
        assert np.allclose(inp2, out[0][1])
        assert np.allclose(inp3, out[1][0])
        assert np.allclose(inp4, out[1][1])


@pytest.mark.parametrize(
    "stencil",
    [tuple_output1, tuple_output2],
)
def test_field_of_tuple_output(program_processor_no_gtfn_exec, stencil):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out_np = np.zeros(shape, dtype="f8, f8")
    out = np_as_located_field(IDim, JDim, KDim)(out_np)

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(stencil[dom], program_processor, inp1, inp2, out=out, offset_provider={})
    if validate:
        assert np.allclose(inp1, out_np[:]["f0"])
        assert np.allclose(inp2, out_np[:]["f1"])


@pytest.mark.parametrize(
    "stencil",
    [tuple_output1, tuple_output2],
)
def test_tuple_of_field_output_constructed_inside(program_processor, stencil):
    program_processor, validate = program_processor

    @fendef
    def fencil(size0, size1, size2, inp1, inp2, out1, out2):
        closure(
            cartesian_domain(
                named_range(IDim, 0, size0),
                named_range(JDim, 0, size1),
                named_range(KDim, 0, size2),
            ),
            stencil,
            make_tuple(out1, out2),
            [inp1, inp2],
        )

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out1 = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))
    out2 = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

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
        assert np.allclose(inp1, out1)
        assert np.allclose(inp2, out2)


def test_asymetric_nested_tuple_of_field_output_constructed_inside(program_processor):
    program_processor, validate = program_processor

    @fundef
    def stencil(inp1, inp2, inp3):
        return make_tuple(deref(inp1), deref(inp2)), deref(inp3)

    @fendef
    def fencil(size0, size1, size2, inp1, inp2, inp3, out1, out2, out3):
        closure(
            cartesian_domain(
                named_range(IDim, 0, size0),
                named_range(JDim, 0, size1),
                named_range(KDim, 0, size2),
            ),
            stencil,
            make_tuple(make_tuple(out1, out2), out3),
            [inp1, inp2, inp3],
        )

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp3 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out1 = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))
    out2 = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))
    out3 = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

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
        assert np.allclose(inp1, out1)
        assert np.allclose(inp2, out2)
        assert np.allclose(inp3, out3)


@pytest.mark.parametrize(
    "stencil",
    [tuple_output1, tuple_output2],
)
def test_field_of_extra_dim_output(program_processor_no_gtfn_exec, stencil):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out_np = np.zeros(shape + [2])
    out = np_as_located_field(IDim, JDim, KDim, None)(out_np)

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(stencil[dom], program_processor, inp1, inp2, out=out, offset_provider={})
    if validate:
        assert np.allclose(inp1, out_np[:, :, :, 0])
        assert np.allclose(inp2, out_np[:, :, :, 1])


@fundef
def tuple_input(inp):
    inp_deref = deref(inp)
    return tuple_get(0, inp_deref) + tuple_get(1, inp_deref)


def test_tuple_field_input(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(tuple_input[dom], program_processor, (inp1, inp2), out=out, offset_provider={})
    if validate:
        assert np.allclose(np.asarray(inp1) + np.asarray(inp2), out)


def test_field_of_tuple_input(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp2 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp = np.zeros(shape, dtype="f8, f8")
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                inp[i, j, k] = (inp1[i, j, k], inp2[i, j, k])

    inp = np_as_located_field(IDim, JDim, KDim)(inp)
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(tuple_input[dom], program_processor, inp, out=out, offset_provider={})
    if validate:
        assert np.allclose(np.asarray(inp1) + np.asarray(inp2), out)


def test_field_of_extra_dim_input(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp2 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp = np.stack((inp1, inp2), axis=-1)

    inp = np_as_located_field(IDim, JDim, KDim, None)(inp)
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
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


def test_tuple_of_field_of_tuple_input(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp2 = rng.normal(size=(shape[0], shape[1], shape[2]))
    inp = np.zeros(shape, dtype="f8, f8")
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                inp[i, j, k] = (inp1[i, j, k], inp2[i, j, k])

    inp = np_as_located_field(IDim, JDim, KDim)(inp)
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(
        tuple_tuple_input[dom],
        program_processor,
        (inp, inp),
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose(2.0 * (np.asarray(inp1) + np.asarray(inp2)), out)


def test_tuple_of_tuple_of_field_input(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = np_as_located_field(IDim, JDim, KDim)(rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = np_as_located_field(IDim, JDim, KDim)(rng.normal(size=(shape[0], shape[1], shape[2])))
    inp3 = np_as_located_field(IDim, JDim, KDim)(rng.normal(size=(shape[0], shape[1], shape[2])))
    inp4 = np_as_located_field(IDim, JDim, KDim)(rng.normal(size=(shape[0], shape[1], shape[2])))

    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(
        tuple_tuple_input[dom],
        program_processor,
        ((inp1, inp2), (inp3, inp4)),
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose(
            (np.asarray(inp1) + np.asarray(inp2) + np.asarray(inp3) + np.asarray(inp4)), out
        )


def test_field_of_2_extra_dim_input(program_processor_no_gtfn_exec):
    program_processor, validate = program_processor_no_gtfn_exec

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp = np_as_located_field(IDim, JDim, KDim, None, None)(
        rng.normal(size=(shape[0], shape[1], shape[2], 2, 2))
    )

    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    run_processor(
        tuple_tuple_input[dom],
        program_processor,
        inp,
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.allclose(np.sum(inp, axis=(3, 4)), out)

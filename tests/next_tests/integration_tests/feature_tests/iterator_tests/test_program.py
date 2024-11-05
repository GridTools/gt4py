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
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.builtins import (
    as_fieldop,
    cartesian_domain,
    deref,
    index,
    named_range,
    shift,
)
from gt4py.next.iterator.runtime import fendef, fundef, set_at

from next_tests.unit_tests.conftest import program_processor, run_processor


I = gtx.Dimension("I")
Ioff = gtx.FieldOffset("Ioff", source=I, target=(I,))


@fundef
def copy_stencil(inp):
    return deref(inp)


@fendef
def copy_program(inp, out, size):
    set_at(
        as_fieldop(copy_stencil, cartesian_domain(named_range(I, 0, size)))(inp),
        cartesian_domain(named_range(I, 0, size)),
        out,
    )


@pytest.mark.starts_from_gtir_program
def test_prog(program_processor):
    program_processor, validate = program_processor

    isize = 10
    inp = gtx.as_field([I], np.arange(0, isize, dtype=np.float64))
    out = gtx.as_field([I], np.zeros(shape=(isize,)))

    run_processor(copy_program, program_processor, inp, out, isize, offset_provider={})
    if validate:
        assert np.allclose(inp.asnumpy(), out.asnumpy())


@fendef
def index_program_simple(out, size):
    set_at(
        as_fieldop(lambda i: deref(i), cartesian_domain(named_range(I, 0, size)))(index(I)),
        cartesian_domain(named_range(I, 0, size)),
        out,
    )


@pytest.mark.starts_from_gtir_program
@pytest.mark.uses_index_builtin
def test_index_builtin(program_processor):
    program_processor, validate = program_processor

    isize = 10
    out = gtx.as_field([I], np.zeros(shape=(isize,)), dtype=getattr(np, itir.INTEGER_INDEX_BUILTIN))

    run_processor(index_program_simple, program_processor, out, isize, offset_provider={})
    if validate:
        assert np.allclose(np.arange(10), out.asnumpy())


@fendef
def index_program_shift(out, size):
    set_at(
        as_fieldop(
            lambda i: deref(i) + deref(shift(Ioff, 1)(i)), cartesian_domain(named_range(I, 0, size))
        )(index(I)),
        cartesian_domain(named_range(I, 0, size)),
        out,
    )


@pytest.mark.starts_from_gtir_program
@pytest.mark.uses_index_builtin
def test_index_builtin_shift(program_processor):
    program_processor, validate = program_processor

    isize = 10
    out = gtx.as_field([I], np.zeros(shape=(isize,)), dtype=getattr(np, itir.INTEGER_INDEX_BUILTIN))

    run_processor(index_program_shift, program_processor, out, isize, offset_provider={"Ioff": I})
    if validate:
        assert np.allclose(np.arange(10) + np.arange(1, 11), out.asnumpy())

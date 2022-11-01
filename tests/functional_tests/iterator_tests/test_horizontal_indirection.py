# (defun calc (p_vn input_on_cell)
#   (do_some_math
#     (deref
#         ((if (less (deref p_vn) 0)
#             (shift e2c 0)
#             (shift e2c 1)
#          )
#          input_on_cell
#         )
#     )
#   )
# )
import numpy as np
import pytest

from functional.common import Dimension
from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import fundef, offset
from functional.program_processors.formatters import type_check
from functional.program_processors.formatters.gtfn import (
    format_sourcecode as gtfn_format_sourcecode,
)
from functional.program_processors.runners.gtfn_cpu import run_gtfn

from .conftest import run_processor


I = offset("I")


@fundef
def compute_shift(cond):
    return if_(deref(cond) < 0, shift(I, -1), shift(I, 1))


@fundef
def conditional_indirection(inp, cond):
    return deref(compute_shift(cond)(inp))


IDim = Dimension("IDim")


def test_simple_indirection(program_processor):
    program_processor, validate = program_processor

    if program_processor == type_check.check:
        pytest.xfail("bug in type inference")
    if program_processor == run_gtfn or program_processor == gtfn_format_sourcecode:
        pytest.xfail("fails in lowering to gtfn_ir")

    shape = [8]
    inp = np_as_located_field(IDim, origin={IDim: 1})(np.asarray(range(shape[0] + 2)))
    rng = np.random.default_rng()
    cond = np_as_located_field(IDim)(rng.normal(size=shape))
    out = np_as_located_field(IDim)(np.zeros(shape))

    ref = np.zeros(shape)
    for i in range(shape[0]):
        ref[i] = inp[i + 1 - 1] if cond[i] < 0 else inp[i + 1 + 1]

    run_processor(
        conditional_indirection[cartesian_domain(named_range(IDim, 0, shape[0]))],
        program_processor,
        inp,
        cond,
        out=out,
        offset_provider={"I": IDim},
    )

    if validate:
        assert np.allclose(ref, out)


@fundef
def direct_indirection(inp, cond):
    return deref(shift(I, deref(cond))(inp))


def test_direct_offset_for_indirection(program_processor):
    program_processor, validate = program_processor

    shape = [4]
    inp = np_as_located_field(IDim)(np.asarray(range(shape[0])))
    cond = np_as_located_field(IDim)(np.asarray([2, 1, -1, -2], dtype=np.int32))
    out = np_as_located_field(IDim)(np.zeros(shape))

    ref = np.zeros(shape)
    for i in range(shape[0]):
        ref[i] = inp[i + cond[i]]

    run_processor(
        direct_indirection[cartesian_domain(named_range(IDim, 0, shape[0]))],
        program_processor,
        inp,
        cond,
        out=out,
        offset_provider={"I": IDim},
    )

    if validate:
        assert np.allclose(ref, out)

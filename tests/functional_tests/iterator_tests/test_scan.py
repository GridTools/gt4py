import numpy as np

from functional.common import Dimension, DimensionKind
from functional.iterator.builtins import cartesian_domain, deref, lift, named_range, scan, shift
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.iterator.runtime import fundef, offset
from functional.program_processors.codegens.gtfn import gtfn_backend
from functional.program_processors.runners import gtfn_cpu, roundtrip

from .conftest import run_processor


def test_scan_in_field_op(program_processor, lift_mode):
    program_processor, validate = program_processor

    isize = 1
    ksize = 3
    IDim = Dimension("I", kind=DimensionKind.HORIZONTAL)
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    Koff = offset("Koff")
    inp = np_as_located_field(IDim, KDim)(np.ones((isize, ksize)))
    out = np_as_located_field(IDim, KDim)(np.zeros((isize, ksize)))

    reference = np.zeros((isize, ksize - 1))
    reference[:, 0] = inp[:, 0] + inp[:, 1]
    for k in range(1, ksize - 1):
        reference[:, k] = reference[:, k - 1] + inp[:, k] + inp[:, k + 1]

    @fundef
    def sum(state, k, kp):
        return state + deref(k) + deref(kp)

    @fundef
    def shifted(inp):
        return deref(shift(Koff, 1)(inp))

    @fundef
    def wrapped(inp):
        return scan(sum, True, 0.0)(inp, lift(shifted)(inp))

    run_processor(
        wrapped[cartesian_domain(named_range(IDim, 0, isize), named_range(KDim, 0, ksize - 1))],
        program_processor,
        inp,
        out=out,
        lift_mode=lift_mode,
        offset_provider={"I": IDim, "K": KDim, "Koff": KDim},
        column_axis=KDim,
    )

    if validate:
        assert np.allclose(out[:, :-1], reference)

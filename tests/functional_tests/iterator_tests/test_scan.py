from functional.common import DimensionKind
from functional.ffront.decorator import field_operator, program, scan_operator
from functional.ffront.fbuiltins import (
    Dimension,
    Field,
    FieldOffset,
    broadcast,
    float64,
    int32,
    int64,
    max_over,
    min_over,
    neighbor_sum,
    where,
)
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.program_processors.runners import gtfn_cpu, roundtrip
from functional.program_processors.codegens.gtfn import gtfn_backend

import numpy as np


def test_scan_in_field_op():
    size = 10
    IDim = Dimension("I", kind=DimensionKind.HORIZONTAL)
    KDim = Dimension("K", kind=DimensionKind.VERTICAL)
    Koff = FieldOffset("Koff", source=KDim, target=(KDim,))
    inp = np_as_located_field(IDim, KDim)(np.ones((size, size)))
    out = np_as_located_field(IDim, KDim)(np.zeros((size, size)))

    @scan_operator(axis=KDim, forward=True, init=0.0)
    def sum(state: float, k: float, kp: float):
        return state + k + kp

    @field_operator
    def shifted(inp: Field[[IDim, KDim], float]):
        return inp(Koff[1])

    @field_operator
    def wrapped(inp: Field[[IDim, KDim], float]):
        return sum(inp, shifted(inp))

    @program
    def prog(inp: Field[[IDim, KDim], float], out: Field[[IDim, KDim], float]):
        wrapped(inp, out=out, domain={IDim: (0, 4), KDim: (0, 4)})

    print(prog.itir)
    print(
        gtfn_backend.generate(
            prog.itir, offset_provider={Koff.value: KDim, "I": IDim}, column_axis=KDim
        )
    )


test_scan_in_field_op()

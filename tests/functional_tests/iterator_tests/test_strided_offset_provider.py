import numpy as np
import pytest

from functional.common import Dimension
from functional.iterator.builtins import deref, lift, named_range, shift, unstructured_domain
from functional.iterator.embedded import StridedNeighborOffsetProvider, np_as_located_field
from functional.iterator.runtime import closure, fendef, fundef, offset

from .conftest import run_processor


LocA = Dimension("LocA")
LocAB = Dimension("LocAB")
LocB = Dimension("LocB")  # unused

LocA2LocAB = offset("O")
LocA2LocAB_offset_provider = StridedNeighborOffsetProvider(
    origin_axis=LocA, neighbor_axis=LocAB, max_neighbors=2, has_skip_values=False
)


@fundef
def foo(inp):
    return deref(shift(LocA2LocAB, 0)(inp)) + deref(shift(LocA2LocAB, 1)(inp))


@fendef(offset_provider={"O": LocA2LocAB_offset_provider})
def fencil(size, out, inp):
    closure(
        unstructured_domain(named_range(LocA, 0, size)),
        foo,
        out,
        [inp],
    )


def test_strided_offset_provider(fencil_processor_no_gtfn_exec):
    fencil_processor, validate = fencil_processor_no_gtfn_exec

    LocA_size = 2
    max_neighbors = LocA2LocAB_offset_provider.max_neighbors
    LocAB_size = LocA_size * max_neighbors

    rng = np.random.default_rng()
    inp = np_as_located_field(LocAB)(
        rng.normal(
            size=(LocAB_size,),
        )
    )
    out = np_as_located_field(LocA)(np.zeros((LocA_size,)))
    ref = np.sum(np.asarray(inp).reshape(LocA_size, max_neighbors), axis=-1)

    run_processor(fencil, fencil_processor, LocA_size, out, inp)

    if validate:
        assert np.allclose(out, ref)

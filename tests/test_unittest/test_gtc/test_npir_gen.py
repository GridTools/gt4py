import re
from typing import Iterator

import pytest

from gt4py.gtc import gtir, common
from gt4py.gtc.python import npir, npir_gen
from gt4py.backend.gtc_backend.stencil_module_builder import parse_node


UNDEFINED_DTYPES = {
    common.DataType.INVALID,
    common.DataType.AUTO,
    common.DataType.DEFAULT
}

DEFINED_DTYPES = set(common.DataType) - UNDEFINED_DTYPES


@pytest.fixture(params=DEFINED_DTYPES)
def defined_dtype(request) -> Iterator[common.DataType]:
    yield request.param


def test_literal(defined_dtype: common.DataType) -> None:
    result = npir_gen.NpirGen.apply(npir.Literal(dtype=defined_dtype, value="42"))
    print(result)
    match = re.match(r'np.(\w*?)\(42\)', result)
    assert match
    assert match.groups()[0] == defined_dtype.name.lower()


def test_parallel_offset() -> None:
    result = npir_gen.NpirGen.apply(npir.ParallelOffset(axis_name="i", offset=3, sign="-"))
    print(result)
    assert result == "i - 3:I - 3"


def test_sequential_offset() -> None:
    result = npir_gen.NpirGen.apply(npir.SequentialOffset(axis_name="K", offset=5, sign="+"))
    print(result)
    assert result == "k_ + 5"

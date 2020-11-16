import re
from typing import Iterator

import pytest

from gt4py.backend.gtc_backend.stencil_module_builder import parse_node
from gt4py.gtc import common, gtir
from gt4py.gtc.python import npir, npir_gen


UNDEFINED_DTYPES = {common.DataType.INVALID, common.DataType.AUTO, common.DataType.DEFAULT}

DEFINED_DTYPES = set(common.DataType) - UNDEFINED_DTYPES


class FieldSliceBuilder:
    def __init__(self, name: str):
        self._name = name
        self._offsets = [0, 0, 0]
        self._parallel_k = False

    def build(self):
        k_maker = npir.ParallelOffset.k if self._parallel_k else npir.SequentialOffset.k
        return npir.FieldSlice(
            name=self._name,
            i_offset=npir.ParallelOffset.i(self._offsets[0]),
            j_offset=npir.ParallelOffset.j(self._offsets[1]),
            k_offset=k_maker(self._offsets[2]),
        )


@pytest.fixture(params=DEFINED_DTYPES)
def defined_dtype(request) -> Iterator[common.DataType]:
    yield request.param


def test_literal(defined_dtype: common.DataType) -> None:
    result = npir_gen.NpirGen.apply(npir.Literal(dtype=defined_dtype, value="42"))
    print(result)
    match = re.match(r"np.(\w*?)\(42\)", result)
    assert match
    assert match.groups()[0] == defined_dtype.name.lower()


def test_parallel_offset() -> None:
    result = npir_gen.NpirGen.apply(npir.ParallelOffset.i(-3))
    print(result)
    assert result == "(i - 3):(I - 3)"


def test_parallel_offset_zero() -> None:
    result = npir_gen.NpirGen.apply(npir.ParallelOffset.j(0))
    print(result)
    assert result == "j:J"


def test_sequential_offset() -> None:
    result = npir_gen.NpirGen.apply(npir.SequentialOffset.k(5))
    print(result)
    assert result == "k_ + 5"


def test_sequential_offset_zero() -> None:
    result = npir_gen.NpirGen.apply(npir.SequentialOffset.k(0))
    print(result)
    assert result == "k_"


def test_field_slice_sequential_k() -> None:
    result = npir_gen.NpirGen().apply(
        npir.FieldSlice(
            name="a_field",
            i_offset=npir.ParallelOffset.i(-1),
            j_offset=npir.ParallelOffset.j(0),
            k_offset=npir.SequentialOffset.k(4),
        )
    )
    assert result == "a_field[(i - 1):(I - 1), j:J, k_ + 4]"


def test_field_slice_parallel_k() -> None:
    result = npir_gen.NpirGen().apply(
        npir.FieldSlice(
            name="another_field",
            i_offset=npir.ParallelOffset.i(0),
            j_offset=npir.ParallelOffset.j(0),
            k_offset=npir.ParallelOffset.k(-3),
        )
    )
    assert result == "another_field[i:I, j:J, (k - 3):(K - 3)]"


def test_vector_assign() -> None:
    result = npir_gen.NpirGen().apply(
        npir.VectorAssign(
            left=FieldSliceBuilder("a").build(),
            right=FieldSliceBuilder("b").build(),
        )
    )
    assert result == "a[i:I, j:J, k_] = b[i:I, j:J, k_]"


def test_vector_binop() -> None:
    result = npir_gen.NpirGen().apply(
        npir.VectorBinOp(
            left=FieldSliceBuilder("a").build(),
            right=FieldSliceBuilder("b").build(),
            op=common.BinaryOperator.ADD,
        )
    )
    assert result == "(a[i:I, j:J, k_] + b[i:I, j:J, k_])"


def test_numerical_offset_pos() -> None:
    result = npir_gen.NpirGen().apply(npir.NumericalOffset(value=1))
    assert result == " + 1"


def test_numerical_offset_neg() -> None:
    result = npir_gen.NpirGen().apply(npir.NumericalOffset(value=-1))
    assert result == " - 1"


def test_numerical_offset_zero() -> None:
    result = npir_gen.NpirGen().apply(npir.NumericalOffset(value=0))
    assert result == ""


def test_vertical_pass_seq() -> None:
    result = npir_gen.NpirGen().apply(
        npir.VerticalPass(
            body=[
                npir.VectorAssign(
                    left=FieldSliceBuilder("a").build(), right=FieldSliceBuilder("b").build()
                ),
                npir.VectorAssign(
                    left=FieldSliceBuilder("c").build(), right=FieldSliceBuilder("d").build()
                ),
            ],
            lower=npir.NumericalOffset(value=1),
            upper=npir.NumericalOffset(value=-2),
            direction=common.LoopOrder.FORWARD,
        )
    )
    print(result)
    match = re.match(
        r"k, K = DOMAIN_k + 1, DOMAIN_K - 2\nfor k_ in range(k, K):\n    a[.*] = b[.*]\n    c[.*] = d[.*]",
        result,
        re.MULTILINE,
    )
    assert match

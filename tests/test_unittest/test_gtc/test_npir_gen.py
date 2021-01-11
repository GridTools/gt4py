import re
from typing import Iterator, Optional, Set

import pytest

from gt4py.gtc import common
from gt4py.gtc.python import npir, npir_gen

from .npir_utils import FieldSliceBuilder


UNDEFINED_DTYPES = {common.DataType.INVALID, common.DataType.AUTO, common.DataType.DEFAULT}

DEFINED_DTYPES: Set[common.DataType] = set(common.DataType) - UNDEFINED_DTYPES  # type: ignore


@pytest.fixture(params=DEFINED_DTYPES)
def defined_dtype(request) -> Iterator[common.DataType]:
    yield request.param


@pytest.fixture()
def other_dtype(defined_dtype) -> Iterator[Optional[common.DataType]]:
    other = None
    for dtype in DEFINED_DTYPES:
        if dtype != defined_dtype:
            other = dtype
            break
    yield other


def test_literal(defined_dtype: common.DataType) -> None:
    result = npir_gen.NpirGen().visit(npir.Literal(dtype=defined_dtype, value="42"))
    print(result)
    match = re.match(r"np.(\w*?)\(42\)", result)
    assert match
    assert match.groups()[0] == defined_dtype.name.lower()


def test_broadcast_literal(defined_dtype: common.DataType) -> None:
    result = npir_gen.NpirGen().visit(
        npir.BroadCastLiteral(literal=npir.Literal(dtype=defined_dtype, value="42"))
    )
    print(result)
    match = re.match(r"np.(\w*?)\(42\)", result)
    assert match
    assert match.groups()[0] == defined_dtype.name.lower()


def test_cast(defined_dtype: common.DataType, other_dtype: common.DataType) -> None:
    result = npir_gen.NpirGen().visit(
        npir.Cast(dtype=other_dtype, expr=npir.Literal(dtype=defined_dtype, value="42"))
    )
    print(result)
    match = re.match(r"np.(\w*?)\(np.(\w*)\(42\)", result)
    assert match
    assert match.groups()[0] == other_dtype.name.lower()
    assert match.groups()[1] == defined_dtype.name.lower()


def test_parallel_offset() -> None:
    result = npir_gen.NpirGen().visit(npir.AxisOffset.i(-3))
    print(result)
    assert result == "(i - 3):(I - 3)"


def test_parallel_offset_zero() -> None:
    result = npir_gen.NpirGen().visit(npir.AxisOffset.j(0))
    print(result)
    assert result == "j:J"


def test_sequential_offset() -> None:
    result = npir_gen.NpirGen().visit(npir.AxisOffset.k(5))
    print(result)
    assert result == "k_ + 5"


def test_sequential_offset_zero() -> None:
    result = npir_gen.NpirGen().visit(npir.AxisOffset.k(0))
    print(result)
    assert result == "k_"


def test_field_slice_sequential_k() -> None:
    result = npir_gen.NpirGen().visit(
        FieldSliceBuilder("a_field", parallel_k=False).offsets(-1, 0, 4).build()
    )
    assert result == "a_field_[(i - 1):(I - 1), j:J, k_ + 4]"


def test_field_slice_parallel_k() -> None:
    result = npir_gen.NpirGen().visit(
        FieldSliceBuilder("another_field", parallel_k=True).offsets(0, 0, -3).build()
    )
    assert result == "another_field_[i:I, j:J, (k - 3):(K - 3)]"


def test_vector_assign() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VectorAssign(
            left=FieldSliceBuilder("a").build(),
            right=FieldSliceBuilder("b").build(),
        )
    )
    assert result == "a_[i:I, j:J, k_] = b_[i:I, j:J, k_]"


def test_vector_arithmetic() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VectorArithmetic(
            left=FieldSliceBuilder("a").build(),
            right=FieldSliceBuilder("b").build(),
            op=common.ArithmeticOperator.ADD,
        )
    )
    assert result == "(a_[i:I, j:J, k_] + b_[i:I, j:J, k_])"


def test_numerical_offset_pos() -> None:
    result = npir_gen.NpirGen().visit(npir.NumericalOffset(value=1))
    assert result == " + 1"


def test_numerical_offset_neg() -> None:
    result = npir_gen.NpirGen().visit(npir.NumericalOffset(value=-1))
    assert result == " - 1"


def test_numerical_offset_zero() -> None:
    result = npir_gen.NpirGen().visit(npir.NumericalOffset(value=0))
    assert result == ""


def test_vertical_pass_seq() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VerticalPass(
            body=[],
            lower=common.AxisBound.from_start(offset=1),
            upper=common.AxisBound.from_end(offset=2),
            direction=common.LoopOrder.FORWARD,
        )
    )
    print(result)
    match = re.match(
        (r"(#.*?\n)?" r"k, K = DOMAIN_k \+ 1, DOMAIN_K - 2\n" r"for k_ in range\(k, K\):\n"),
        result,
        re.MULTILINE,
    )
    assert match


def test_vertical_pass_par() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VerticalPass(
            body=[],
            lower=common.AxisBound.start(),
            upper=common.AxisBound.end(),
            direction=common.LoopOrder.PARALLEL,
        )
    )
    print(result)
    match = re.match(
        (r"(#.*?\n)?" r"k, K = DOMAIN_k, DOMAIN_K\n"),
        result,
        re.MULTILINE,
    )
    assert match


def test_verticall_pass_start_start_forward() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VerticalPass(
            body=[],
            lower=common.AxisBound.start(),
            upper=common.AxisBound.from_start(offset=5),
            direction=common.LoopOrder.FORWARD,
        )
    )
    print(result)
    match = re.match(
        r"(#.*?\n)?k, K = DOMAIN_k, DOMAIN_k \+ 5\nfor k_ in range\(k, K\):\n",
        result,
        re.MULTILINE,
    )
    assert match


def test_verticall_pass_end_end_backward() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VerticalPass(
            body=[],
            lower=common.AxisBound.from_end(offset=4),
            upper=common.AxisBound.from_end(offset=1),
            direction=common.LoopOrder.BACKWARD,
        )
    )
    print(result)
    match = re.match(
        r"(#.*?\n)?k, K = DOMAIN_K \- 4, DOMAIN_K \- 1\nfor k_ in range\(K-1, k-1, -1\):\n",
        result,
        re.MULTILINE,
    )
    assert match


def test_domain_padding() -> None:
    result = npir_gen.NpirGen().visit(
        npir.DomainPadding(
            lower=(1, 2, 0),
            upper=(0, 6, 2),
        )
    )
    print(result)
    match = re.match(
        (
            r"(#.*?\n)?"
            r"i, j, k = \d*?, \d*?, \d*?\n"
            r"_ui, _uj, _uk = \d*?, \d*?, \d*?\n"
            r"_di, _dj, _dk = _domain_\n"
            r"I, J, K = .*?\n"
            r"DOMAIN_k = k\n"
            r"DOMAIN_K = K\n"
            r"(#.*?\n)?"
        ),
        result,
        re.MULTILINE,
    )
    assert match


def test_computation() -> None:
    result = npir_gen.NpirGen().visit(
        npir.Computation(
            domain_padding=npir.DomainPadding(
                lower=(0, 0, 0),
                upper=(0, 0, 0),
            ),
            params=[],
            field_params=[],
            vertical_passes=[],
        )
    )
    print(result)
    match = re.match(
        (r"def run\(\*, _domain_, _origin_\):\n" r"\n?" r"(    .*?\n)*"),
        result,
        re.MULTILINE,
    )
    assert match


def test_full_computation_valid(tmp_path) -> None:
    result = npir_gen.NpirGen.apply(
        npir.Computation(
            domain_padding=npir.DomainPadding(
                lower=(2, 2, 0),
                upper=(0, 3, 1),
            ),
            params=["f1", "f2", "f3", "s1"],
            field_params=["f1", "f2", "f3"],
            vertical_passes=[
                npir.VerticalPass(
                    lower=common.AxisBound.start(),
                    upper=common.AxisBound.end(),
                    direction=common.LoopOrder.PARALLEL,
                    body=[
                        npir.VectorAssign(
                            left=FieldSliceBuilder("f1", parallel_k=True).build(),
                            right=npir.VectorArithmetic(
                                op=common.ArithmeticOperator.MUL,
                                left=FieldSliceBuilder("f2", parallel_k=True)
                                .offsets(-2, -2, 0)
                                .build(),
                                right=FieldSliceBuilder("f3", parallel_k=True)
                                .offsets(0, 3, 1)
                                .build(),
                            ),
                        ),
                    ],
                ),
                npir.VerticalPass(
                    lower=common.AxisBound.from_start(offset=1),
                    upper=common.AxisBound.from_end(offset=3),
                    direction=common.LoopOrder.BACKWARD,
                    body=[
                        npir.VectorAssign(
                            left=FieldSliceBuilder("f2", parallel_k=False).build(),
                            right=npir.VectorArithmetic(
                                op=common.ArithmeticOperator.ADD,
                                left=FieldSliceBuilder("f2", parallel_k=False).build(),
                                right=FieldSliceBuilder("f2", parallel_k=False)
                                .offsets(0, 0, 1)
                                .build(),
                            ),
                        ),
                    ],
                ),
            ],
        )
    )
    print(result)
    mod_path = tmp_path / "npir_gen_1.py"
    mod_path.write_text(result)
    import sys

    sys.path.append(str(tmp_path))
    import npir_gen_1 as mod
    import numpy as np

    f1 = np.zeros((10, 10, 10))
    f2 = np.ones_like(f1) * 3
    f3 = np.ones_like(f1) * 2
    s1 = 5
    mod.run(
        f1=f1,
        f2=f2,
        f3=f3,
        s1=s1,
        _domain_=(8, 5, 9),
        _origin_={"f1": (2, 2, 0), "f2": (2, 2, 0), "f3": (2, 2, 0)},
    )
    assert (f1[2:, 2:-3, 0:-1] == 6).all()
    assert (f1[0:2, :, :] == 0).all()
    assert (f1[:, 0:2, :] == 0).all()
    assert (f1[:, -3:, :] == 0).all()
    assert (f1[:, :, -1:] == 0).all()

    exp_f2 = np.ones((10)) * 3
    # Remember that reversed ranges still include the first (higher) argument and exclude the
    # second. Thus range(-4, 0, -1) contains the same indices as range(1, -3).
    exp_f2[-4:0:-1] = np.cumsum(exp_f2[1:-3])
    assert (f2[3, 3, :] == exp_f2[:]).all()

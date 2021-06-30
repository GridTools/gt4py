# -*- coding: utf-8 -*-
import re
from typing import Iterator, Optional, Set

import pytest

from gt4py.definitions import Extent
from gtc import common
from gtc.python import npir, npir_gen

from .npir_utils import (
    FieldDeclFactory,
    FieldSliceFactory,
    NativeFuncCallFactory,
    VectorAssignFactory,
    VerticalPassFactory,
)


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


def test_datatype() -> None:
    result = npir_gen.NpirGen().visit(common.DataType.FLOAT64)
    print(result)
    match = re.match(r"np.float64", result)
    assert match


def test_literal(defined_dtype: common.DataType) -> None:
    result = npir_gen.NpirGen().visit(npir.Literal(dtype=defined_dtype, value="42"))
    print(result)
    match = re.match(r"np.(\w*?)\(42\)", result)
    assert match
    assert match.groups()[0] == defined_dtype.name.lower()


def test_broadcast_literal(defined_dtype: common.DataType) -> None:
    result = npir_gen.NpirGen().visit(
        npir.BroadCast(expr=npir.Literal(dtype=defined_dtype, value="42"))
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
    match = re.match(r"^np.(\w*?)\(np.(\w*)\(42\), dtype=np.(\w*)\)", result)
    assert match
    assert match.groups()[0] == "array"
    assert match.groups()[1] == defined_dtype.name.lower()
    assert match.groups()[2] == other_dtype.name.lower()


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
        FieldSliceFactory(name="a_field", parallel_k=False, offsets=(-1, 0, 4))
    )
    assert result == "a_field_[(i - 1):(I - 1), j:J, k_ + 4]"


def test_field_slice_parallel_k() -> None:
    result = npir_gen.NpirGen().visit(
        FieldSliceFactory(name="another_field", parallel_k=True, offsets=(0, 0, -3))
    )
    assert result == "another_field_[i:I, j:J, (k - 3):(K - 3)]"


def test_native_function() -> None:
    result = npir_gen.NpirGen().visit(
        NativeFuncCallFactory(
            func=common.NativeFunction.MIN,
            args=[
                FieldSliceFactory(name="a"),
                FieldSliceFactory(name="b"),
            ],
        )
    )
    assert result == "np.minimum(a_[i:I, j:J, k_], b_[i:I, j:J, k_])"


def test_vector_assign() -> None:
    result = npir_gen.NpirGen().visit(
        VectorAssignFactory(
            left__name="a",
            right__name="b",
        )
    )
    assert result == "a_[i:I, j:J, k_] = b_[i:I, j:J, k_]"


def test_temp_definition() -> None:
    result = npir_gen.NpirGen().visit(VectorAssignFactory(temp_init=True, temp_name="a"))
    assert result == "a_ = ShimmedView(np.zeros(_domain_, dtype=np.int64), [0, 0, 0])"


def test_temp_with_extent_definition() -> None:
    result = npir_gen.NpirGen().visit(
        VectorAssignFactory(temp_init=True, temp_name="a"),
        field_extents={"a": Extent((0, 1), (-2, 3))},
    )
    assert (
        result
        == "a_ = ShimmedView(np.zeros((_dI_ + 1, _dJ_ + 5, _dK_), dtype=np.int64), [0, 2, 0])"
    )


def test_vector_arithmetic() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VectorArithmetic(
            left=FieldSliceFactory(name="a"),
            right=FieldSliceFactory(name="b"),
            op=common.ArithmeticOperator.ADD,
        )
    )
    assert result == "(a_[i:I, j:J, k_] + b_[i:I, j:J, k_])"


def test_vector_unary_op() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VectorUnaryOp(
            expr=FieldSliceFactory(name="a"),
            op=common.UnaryOperator.NEG,
        )
    )
    assert result == "(-(a_[i:I, j:J, k_]))"


def test_vector_unary_not() -> None:
    result = npir_gen.NpirGen().visit(
        npir.VectorUnaryOp(
            expr=FieldSliceFactory(name="a"),
            op=common.UnaryOperator.NOT,
        )
    )
    assert result == "(np.bitwise_not(a_[i:I, j:J, k_]))"


def test_numerical_offset_pos() -> None:
    result = npir_gen.NpirGen().visit(npir.NumericalOffset(value=1))
    assert result == " + 1"


def test_numerical_offset_neg() -> None:
    result = npir_gen.NpirGen().visit(npir.NumericalOffset(value=-1))
    assert result == " - 1"


def test_numerical_offset_zero() -> None:
    result = npir_gen.NpirGen().visit(npir.NumericalOffset(value=0))
    assert result == ""


def test_mask_block_slice_mask() -> None:
    result = npir_gen.NpirGen().visit(
        npir.MaskBlock(body=[], mask=FieldSliceFactory(name="mask1"), mask_name="mask1")
    )
    assert result == ""


def test_mask_block_broadcast() -> None:
    result = npir_gen.NpirGen().visit(
        npir.MaskBlock(
            body=[],
            mask=npir.BroadCast(
                expr=npir.Literal(dtype=common.DataType.BOOL, value=common.BuiltInLiteral.TRUE)
            ),
            mask_name="mask1",
        )
    )
    assert result == "mask1_ = np.full((I - i, J - j, K - k), np.bool(True))\n"


def test_mask_block_other() -> None:
    result = npir_gen.NpirGen().visit(
        npir.MaskBlock(
            body=[],
            mask=npir.VectorLogic(
                op=common.LogicalOperator.AND,
                left=FieldSliceFactory(name="a"),
                right=FieldSliceFactory(name="b"),
            ),
            mask_name="mask1",
        )
    )
    assert result.startswith("mask1_ = np.bitwise_and(a_[i:I")


def test_horizontal_block() -> None:
    result = npir_gen.NpirGen().visit(npir.HorizontalBlock(body=[]))
    print(result)
    match = re.match(
        r"(#.*?\n)?i, I = _di_ - 0, _dI_ \+ 0\nj, J = _dj_ - 0, _dJ_ \+ 0\n",
        result,
        re.MULTILINE,
    )
    assert match


def test_vertical_pass_seq() -> None:
    result = npir_gen.NpirGen().visit(
        VerticalPassFactory(
            temp_defs=[],
            body=[],
            lower=common.AxisBound.from_start(offset=1),
            upper=common.AxisBound.from_end(offset=-2),
            direction=common.LoopOrder.FORWARD,
        )
    )
    print(result)
    match = re.match(
        (r"(#.*?\n)?" r"k, K = _dk_ \+ 1, _dK_ - 2\n" r"for k_ in range\(k, K\):\n"),
        result,
        re.MULTILINE,
    )
    assert match


def test_vertical_pass_par() -> None:
    result = npir_gen.NpirGen().visit(VerticalPassFactory(body=[], temp_defs=[]))
    print(result)
    match = re.match(
        (r"(#.*?\n)?" r"k, K = _dk_, _dK_\n"),
        result,
        re.MULTILINE,
    )
    assert match


def test_verticall_pass_start_start_forward() -> None:
    result = npir_gen.NpirGen().visit(
        VerticalPassFactory(
            body=[],
            temp_defs=[],
            upper=common.AxisBound.from_start(offset=5),
            direction=common.LoopOrder.FORWARD,
        )
    )
    print(result)
    match = re.match(
        r"(#.*?\n)?k, K = _dk_, _dk_ \+ 5\nfor k_ in range\(k, K\):\n",
        result,
        re.MULTILINE,
    )
    assert match


def test_verticall_pass_end_end_backward() -> None:
    result = npir_gen.NpirGen().visit(
        VerticalPassFactory(
            body=[],
            temp_defs=[],
            lower=common.AxisBound.from_end(offset=-4),
            upper=common.AxisBound.from_end(offset=-1),
            direction=common.LoopOrder.BACKWARD,
        )
    )
    print(result)
    match = re.match(
        r"(#.*?\n)?k, K = _dK_ \- 4, _dK_ \- 1\nfor k_ in range\(K-1, k-1, -1\):\n",
        result,
        re.MULTILINE,
    )
    assert match


def test_vertical_pass_temp_def() -> None:
    result = npir_gen.NpirGen().visit(
        VerticalPassFactory(
            temp_defs=[
                VectorAssignFactory(temp_init=True, temp_name="a"),
            ],
            body=[],
            lower=common.AxisBound.from_end(offset=-4),
            upper=common.AxisBound.from_end(offset=-1),
            direction=common.LoopOrder.BACKWARD,
        )
    )
    print(result)
    match = re.match(
        r"(#.*?\n)?a_ = ShimmedView\(np.zeros\(_domain_, dtype=np.int64\), \[0, 0, 0\]\)\nk, K = _dK_ \- 4, _dK_ \- 1\nfor k_ in range\(K-1, k-1, -1\):\n",
        result,
        re.MULTILINE,
    )
    assert match


def test_computation() -> None:
    result = npir_gen.NpirGen().visit(
        npir.Computation(
            params=[],
            field_params=[],
            field_decls=[],
            vertical_passes=[],
        ),
        field_extents={},
    )
    print(result)
    match = re.match(
        (
            r"import numpy as np\n\n\n"
            r"def run\(\*, _domain_, _origin_\):\n"
            r"\n?"
            r"(    .*?\n)*"
        ),
        result,
        re.MULTILINE,
    )
    assert match


def test_full_computation_valid(tmp_path) -> None:
    result = npir_gen.NpirGen.apply(
        npir.Computation(
            params=["f1", "f2", "f3", "s1"],
            field_params=["f1", "f2", "f3"],
            field_decls=[
                FieldDeclFactory(name="f1"),
                FieldDeclFactory(name="f2"),
                FieldDeclFactory(name="f3"),
            ],
            vertical_passes=[
                VerticalPassFactory(
                    temp_defs=[],
                    body=[
                        npir.HorizontalBlock(
                            body=[
                                VectorAssignFactory(
                                    left=FieldSliceFactory(name="f1", parallel_k=True),
                                    right=npir.VectorArithmetic(
                                        op=common.ArithmeticOperator.MUL,
                                        left=FieldSliceFactory(
                                            name="f2", parallel_k=True, offsets=(-2, -2, 0)
                                        ),
                                        right=FieldSliceFactory(
                                            name="f3", parallel_k=True, offsets=(0, 3, 1)
                                        ),
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                VerticalPassFactory(
                    lower=common.AxisBound.from_start(offset=1),
                    upper=common.AxisBound.from_end(offset=-3),
                    direction=common.LoopOrder.BACKWARD,
                    temp_defs=[],
                    body=[
                        npir.HorizontalBlock(
                            body=[
                                VectorAssignFactory(
                                    left__name="f2",
                                    right=npir.VectorArithmetic(
                                        op=common.ArithmeticOperator.ADD,
                                        left=FieldSliceFactory(name="f2", parallel_k=False),
                                        right=FieldSliceFactory(
                                            name="f2", parallel_k=False, offsets=(0, 0, 1)
                                        ),
                                    ),
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
        field_extents={
            "f1": Extent([(0, 0), (0, 0)]),
            "f2": Extent([(-2, 0), (-2, 0)]),
            "f3": Extent([(0, 0), (0, 3)]),
        },
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

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re
import sys
from typing import Iterator, Optional, Set

import numpy as np
import pytest

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.numpy import npir
from gt4py.cartesian.gtc.numpy.npir_codegen import NpirCodegen

from .npir_utils import (
    ComputationFactory,
    FieldDeclFactory,
    FieldSliceFactory,
    HorizontalBlockFactory,
    LocalScalarAccessFactory,
    NativeFuncCallFactory,
    ParamAccessFactory,
    ScalarDeclFactory,
    TemporaryDeclFactory,
    VectorArithmeticFactory,
    VectorAssignFactory,
    VerticalPassFactory,
)


UNDEFINED_DTYPES = {common.DataType.INVALID, common.DataType.AUTO, common.DataType.DEFAULT}

DEFINED_DTYPES: Set[common.DataType] = set(common.DataType) - UNDEFINED_DTYPES  # type: ignore


def match_dtype(result, dtype) -> None:
    if dtype not in {common.DataType.BOOL}:
        assert result == f"np.{dtype.name.lower()}"
    else:
        assert result == dtype.name.lower()


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


@pytest.fixture(params=[True, False])
def is_serial(request):
    yield request.param


def test_datatype() -> None:
    result = NpirCodegen().visit(common.DataType.FLOAT64)
    print(result)
    match = re.match(r"np.float64", result)
    assert match


def test_scalarliteral(defined_dtype: common.DataType) -> None:
    result = NpirCodegen().visit(npir.ScalarLiteral(dtype=defined_dtype, value="42"))
    print(result)
    match = re.match(r"(.+?)\(42\)", result)
    assert match
    match_dtype(match.groups()[0], defined_dtype)


def test_broadcast_literal(defined_dtype: common.DataType, is_serial: bool) -> None:
    result = NpirCodegen().visit(
        npir.Broadcast(expr=npir.ScalarLiteral(dtype=defined_dtype, value="42"))
    )
    print(result)
    match = re.match(r"(.*?)\(42\)", result)
    assert match
    match_dtype(match.groups()[0], defined_dtype)


def test_scalar_cast(defined_dtype: common.DataType, other_dtype: common.DataType) -> None:
    result = NpirCodegen().visit(
        npir.ScalarCast(dtype=other_dtype, expr=npir.ScalarLiteral(dtype=defined_dtype, value="42"))
    )
    print(result)
    match = re.match(r"(?P<other_dtype>.+)\((?P<defined_dtype>.+)\(42\)\)", result)
    assert match
    match_dtype(match.group("defined_dtype"), defined_dtype)
    match_dtype(match.group("other_dtype"), other_dtype)


def test_vector_cast(defined_dtype: common.DataType, other_dtype: common.DataType) -> None:
    result = NpirCodegen().visit(
        npir.VectorCast(
            dtype=other_dtype,
            expr=npir.FieldSlice(name="a", i_offset=0, j_offset=0, k_offset=0, dtype=defined_dtype),
        ),
        is_serial=False,
    )
    print(result)
    match = re.match(r"(?P<name>\w+)\[.*]\.astype\((?P<dtype>.+)\)", result)
    assert match
    assert match.group("name") == "a"
    match_dtype(match.group("dtype"), other_dtype)


def test_field_slice(is_serial: bool) -> None:
    i_offset = 0
    j_offset = -2
    k_offset = 4

    def int_to_str(i):
        if i > 0:
            return f"+ {i}"
        elif i == 0:
            return ""
        else:
            return f"- {-i}"

    field_slice = FieldSliceFactory(
        name="a",
        i_offset=i_offset,
        j_offset=j_offset,
        k_offset=k_offset,
        dtype=common.DataType.INT32,
    )
    result = NpirCodegen().visit(field_slice, is_serial=is_serial)
    print(result)
    match = re.match(
        r"(?P<name>\w+)\[i\s*(?P<il>.*):I\s*(?P<iu>.*),\s*j\s*(?P<jl>.*):J\s*(?P<ju>.*),\s*(?P<kl>.*):(?P<ku>.*)\]",
        result,
    )
    assert match
    assert match.group("name") == "a"
    assert match.group("il") == match.group("iu") == int_to_str(i_offset)
    assert match.group("jl") == match.group("ju") == int_to_str(j_offset)

    if is_serial:
        assert match.group("kl") == "k_ " + int_to_str(k_offset)
        assert match.group("ku") == "k_ " + int_to_str(k_offset + 1)
    else:
        assert match.group("kl") == "k " + int_to_str(k_offset)
        assert match.group("ku") == "K " + int_to_str(k_offset)


def test_native_function() -> None:
    result = NpirCodegen().visit(
        NativeFuncCallFactory(
            func=common.NativeFunction.MIN,
            args=[FieldSliceFactory(name="a"), ParamAccessFactory(name="p")],
        ),
        is_serial=False,
    )
    print(result)
    match = re.match(r"ufuncs.minimum\(a\[.*\],\s*p\)", result)
    assert match


@pytest.mark.parametrize(
    "left", (FieldSliceFactory(name="left"), LocalScalarAccessFactory(name="left"))
)
def test_vector_assign(left, is_serial: bool) -> None:
    result = NpirCodegen().visit(
        VectorAssignFactory(left=left, right=FieldSliceFactory(name="right", k_offset=-1)),
        ctx=NpirCodegen.BlockContext(),
        is_serial=is_serial,
    )
    left_str, right_str = result.split(" = ")

    if isinstance(left, npir.FieldSlice):
        k_str_left = "k_:k_ + 1" if is_serial else "k:K"
    else:
        k_str_left = ":" if is_serial else "k:K"
    k_str_right = "k_ - 1:k_" if is_serial else "k - 1:K - 1"

    assert left_str == f"left[i:I, j:J, {k_str_left}]"
    assert right_str == f"right[i:I, j:J, {k_str_right}]"


def test_field_definition() -> None:
    result = NpirCodegen().visit(FieldDeclFactory(name="a", dimensions=(True, True, False)))
    print(result)
    assert result == "a = Field(a, _origin_['a'], (True, True, False))"


def test_temp_definition() -> None:
    result = NpirCodegen().visit(
        TemporaryDeclFactory(
            name="a",
            offset=(1, 2),
            padding=(3, 4),
            dtype=common.DataType.FLOAT32,
            dimensions=(True, True, True),
        )
    )
    print(result)
    assert (
        result
        == "a = Field.empty((_dI_ + 3, _dJ_ + 4, _dK_), np.float32, (1, 2, 0), (True, True, True))"
    )


def test_vector_arithmetic() -> None:
    result = NpirCodegen().visit(
        npir.VectorArithmetic(
            left=FieldSliceFactory(name="a"),
            right=FieldSliceFactory(name="b"),
            op=common.ArithmeticOperator.ADD,
        ),
        is_serial=False,
    )
    assert result == "(a[i:I, j:J, k:K] + b[i:I, j:J, k:K])"


def test_vector_unary_op() -> None:
    result = NpirCodegen().visit(
        npir.VectorUnaryOp(expr=FieldSliceFactory(name="a"), op=common.UnaryOperator.NEG),
        is_serial=False,
    )
    assert result == "(-(a[i:I, j:J, k:K]))"


def test_vector_unary_not() -> None:
    result = NpirCodegen().visit(
        npir.VectorUnaryOp(
            op=common.UnaryOperator.NOT,
            expr=FieldSliceFactory(name="mask", dtype=common.DataType.BOOL),
        ),
        is_serial=False,
    )
    assert result == "(np.bitwise_not(mask[i:I, j:J, k:K]))"


def test_horizontal_block() -> None:
    result = NpirCodegen().visit(HorizontalBlockFactory(), is_serial=False).strip("\n")
    print(result)
    match = re.match(
        r"#.*\n" r"i, I = _di_ - 0, _dI_ \+ 0\n" r"j, J = _dj_ - 0, _dJ_ \+ 0\n",
        result,
        re.MULTILINE,
    )
    assert match


def test_vertical_pass_seq() -> None:
    result = NpirCodegen().visit(
        VerticalPassFactory(
            lower=common.AxisBound.from_start(offset=1),
            upper=common.AxisBound.from_end(offset=-2),
            direction=common.LoopOrder.FORWARD,
        )
    )
    print(result)
    match = re.match(
        (r"#.*\n" r"+k, K = _dk_ \+ 1, _dK_ - 2\n" r"for k_ in range\(k, K\):\n"),
        result,
        re.MULTILINE,
    )
    assert match


def test_vertical_pass_par() -> None:
    result = NpirCodegen().visit(VerticalPassFactory(direction=common.LoopOrder.PARALLEL))
    print(result)
    match = re.match((r"(#.*?\n)?" r"k, K = _dk_, _dK_\n"), result, re.MULTILINE)
    assert match


def test_computation() -> None:
    result = NpirCodegen().visit(
        ComputationFactory(
            vertical_passes__0__body__0__body__0=VectorAssignFactory(
                left__name="a", right__name="b"
            )
        )
    )
    match = re.match(
        (
            r"import numbers\n"
            r"from typing import Tuple\n+"
            r"import numpy as np\n"
            r"from gt4py.cartesian.gtc import ufuncs\n+"
            r"from gt4py.cartesian.utils import Field\n"
            r"(.*\n)+"
            r"def run\(\*, a, b, _domain_, _origin_\):\n"
            r"\n?"
            r"(    .*?\n)*"
        ),
        result,
        re.MULTILINE,
    )
    assert match


def test_full_computation_valid(tmp_path) -> None:
    computation = ComputationFactory(
        vertical_passes__0__body__0__body__0=VectorAssignFactory(
            left__name="a",
            right=VectorArithmeticFactory(
                left__name="b", right=ParamAccessFactory(name="p"), op=common.ArithmeticOperator.ADD
            ),
        ),
        param_decls=[ScalarDeclFactory(name="p")],
    )
    result = NpirCodegen().visit(computation)
    print(result)
    mod_path = tmp_path / "npir_codegen_1.py"
    mod_path.write_text(result)

    sys.path.append(str(tmp_path))
    import npir_codegen_1 as mod

    a = np.zeros((10, 10, 10))
    b = np.ones_like(a) * 3
    p = 2
    mod.run(a=a, b=b, p=p, _domain_=(8, 5, 9), _origin_={"a": (1, 1, 0), "b": (0, 0, 0)})
    assert (a[1:9, 1:6, 0:9] == 5).all()


def test_variable_read_outside_bounds(tmp_path) -> None:
    """While loops can cause variable K reads to go outside the bounds of K.

    This tests whether that is appropriately clipped to support that case by constructing
    `a = b[0, 0, index]` where the read is outside bounds.
    """
    computation = ComputationFactory(
        vertical_passes__0__body__0__body__0=VectorAssignFactory(
            left__name="a",
            right=FieldSliceFactory(
                name="b",
                k_offset=npir.VarKOffset(
                    k=FieldSliceFactory(name="index", dtype=common.DataType.INT32)
                ),
            ),
        )
    )

    result = NpirCodegen().visit(computation)
    print(result)
    mod_path = tmp_path / "npir_codegen_2.py"
    mod_path.write_text(result)

    sys.path.append(str(tmp_path))
    import npir_codegen_2 as mod

    a = np.empty((2, 2, 5))
    b = np.ones_like(a) * 3
    index = np.ones_like(a, dtype=np.int_)

    mod.run(
        a=a,
        b=b,
        index=index,
        _domain_=a.shape,
        _origin_={"a": (0, 0, 0), "b": (0, 0, 0), "index": (0, 0, 0)},
    )

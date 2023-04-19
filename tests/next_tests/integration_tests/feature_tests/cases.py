# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import dataclasses
import functools
import inspect
import types
import typing
from typing import Any, Callable, Optional, Protocol, Type, TypeAlias

import numpy as np
import pytest
from typing_extensions import Self

from gt4py.next import common
from gt4py.next.ffront import decorator, fbuiltins
from gt4py.next.iterator import embedded, ir as itir
from gt4py.next.program_processors import processor_interface as ppi
from gt4py.next.type_system import type_specifications as ts, type_translation

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Edge,
    EdgeOffset,
    IDim,
    Ioff,
    JDim,
    Joff,
    KDim,
    Koff,
    Vertex,
    fieldview_backend,
)


IField: TypeAlias = common.Field[[IDim], int]
IJKField: TypeAlias = common.Field[[IDim, JDim, KDim], int]
IJKFloatField: TypeAlias = common.Field[[IDim, JDim, KDim], np.float64]
VField: TypeAlias = common.Field[[Vertex], int]
EField: TypeAlias = common.Field[[Edge], int]
V2EDim = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)
E2VDim = common.Dimension("E2V", kind=common.DimensionKind.LOCAL)
V2E = fbuiltins.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2V = fbuiltins.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))


def no_backend(program: itir.FencilDefinition, *args, **kwargs) -> None:
    """Temporary default backend to avoid accidentally testing the wrong backend."""
    raise ValueError("No backend selected! Backend selection is mandatory in tests.")


class ScalarInitializer(Protocol):
    def __call__(self, dtype: str) -> int | float:
        ...


class FieldInitializer(Protocol):
    def __call__(
        self, backend: ppi.ProgramProcessor, sizes: dict[common.Dimension, int], dtype: str
    ) -> common.Field:
        ...


def zeros(
    backend: ppi.ProgramProcessor, sizes: dict[common.Dimension, int], dtype: str
) -> common.Field:
    """Initialize a field with all zeros."""
    return embedded.np_as_located_field(*sizes.keys())(np.zeros(tuple(sizes.values()), dtype=dtype))


def unique(
    backend: ppi.ProgramProcessor, sizes: dict[common.Dimension, int], dtype: str
) -> common.Field:
    """Initialize a field with a unique value in each coordinate."""
    svals = tuple(sizes.values())
    return embedded.np_as_located_field(*sizes.keys())(
        np.arange(np.prod(svals), dtype=dtype).reshape(svals)
    )


def scalar5(dtype: str, shape: Optional[tuple[int, ...]]) -> int | float | np.array:
    if shape:
        return np.ones(np.prod(shape), dtype=dtype) * 5
    return np.dtype(dtype).type(5)


@dataclasses.dataclass(frozen=True)
class Builder:
    partial: functools.partial

    def build(self, *args, **kwargs):
        return self.partial(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)


def make_builder(func):
    """Create a fluid inteface for a function with many arguments."""

    def make_setter(argname):
        def setter(self, arg):
            return self.__class__(
                partial=functools.partial(
                    self.partial, *self.partial.args, **(self.partial.keywords | {argname: arg})
                )
            )

        setter.__name__ = argname
        return setter

    argspec = inspect.getfullargspec(func)

    @dataclasses.dataclass(frozen=True)
    class NewBuilder(Builder):
        for argname in argspec.args + argspec.kwonlyargs:
            locals()[argname] = make_setter(argname)

    func_snake_words = func.__name__.split("_")
    func_camel_name = "".join(word.capitalize() for word in func_snake_words)

    NewBuilder.__name__ = f"{func_camel_name}Builder"

    return NewBuilder(functools.partial(func))


def allocate(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    name: str,
) -> Builder:
    return (
        make_builder(allocate_)
        .sizes(case.default_sizes)
        .case(case)
        .fieldview_prog(fieldview_prog)
        .name(name)
    )


def get_param_types(
    fieldview_prog: decorator.FieldOperator | decorator.Program,
) -> dict[str, ts.FieldType | ts.TupleType | ts.ScalarType]:
    if fieldview_prog.definition is None:
        raise ValueError(
            f"test cases do not support {type(fieldview_prog)} with empty .definition attribute (as you would get from .as_program())!"
        )
    annotations = typing.get_type_hints(fieldview_prog.definition)
    return {
        name: type_translation.from_type_hint(type_hint) for name, type_hint in annotations.items()
    }


def extend_sizes(
    sizes: dict[common.Dimension, str],
    extend: Optional[dict[common.Dimension, tuple[int, int]]] = None,
) -> dict[common.Dimension, str]:
    sizes = sizes.copy()
    if extend:
        for dim, (lower, upper) in extend.items():
            sizes[dim] += upper - lower
    return sizes


RETURN = "return"


def allocate_(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    name: str,
    sizes: dict[common.Dimension, int],
    strategy: Optional[FieldInitializer | ScalarInitializer] = None,
    dtype: Optional[str] = None,
    extend: Optional[dict[common.Dimension, tuple[int, int]]] = None,
) -> common.Field | tuple[common.Field, ...] | int | float:
    """Allocate a field for a parameter or return value of a fieldview program or operator."""
    sizes = extend_sizes(case.default_sizes | (sizes or {}), extend)
    arg_type = get_param_types(fieldview_prog)[name]
    if name in ["out", RETURN] and strategy is None:
        strategy = zeros
    return _allocate_for(case=case, arg_type=arg_type, sizes=sizes, dtype=dtype, strategy=strategy)


def _allocate_for(
    case: Case,
    arg_type: ts.FieldType | ts.TupleType,
    sizes: dict[common.Dimension, int],
    strategy: Optional[FieldInitializer | ScalarInitializer] = None,
    dtype: Optional[str] = None,
) -> common.Field | tuple[common.Field, ...] | int | float:
    """Allocate a field based on the field type or a (nested) tuple thereof."""
    match arg_type:
        case ts.FieldType():
            strategy = strategy or unique
            return strategy(
                backend=case.backend,
                sizes={dim: sizes[dim] for dim in arg_type.dims},
                dtype=dtype or arg_type.dtype.kind.name.lower(),
            )
        case ts.ScalarType():
            strategy = strategy or scalar5
            return strategy(dtype=dtype or arg_type.kind.name.lower(), shape=arg_type.shape)
        case ts.TupleType():
            return tuple(
                (
                    _allocate_for(
                        case=case, arg_type=t, sizes=sizes, dtype=dtype, strategy=strategy
                    )
                    for t in arg_type.types
                )
            )
        case _:
            raise TypeError(f"Can not allocate for type {arg_type}")


def run(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    *args: common.Field,
    **kwargs: Any,
) -> None:
    if kwargs.get("offset_provider", None) is None:
        kwargs["offset_provider"] = case.offset_provider
    fieldview_prog.with_backend(case.backend)(*args, **kwargs)


def get_default_data(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
) -> tuple[tuple[common.Field | int | float, ...], dict[str : common.Field | int | float]]:
    param_types = get_param_types(fieldview_prog)
    kwfields: dict[str, Any] = {}
    if param_types.setdefault(RETURN, types.NoneType) is not types.NoneType:
        kwfields = {"out": allocate(case, fieldview_prog, RETURN).strategy(zeros)()}
    param_types.pop(RETURN)
    inps = tuple(
        allocate(case, fieldview_prog, name)()
        for name in get_param_types(fieldview_prog)
        if name != "out" and name != RETURN
    )
    return inps, kwfields


def verify(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    *args: common.Field,
    out: Optional[common.Field] = None,
    nopass_out: Optional[common.Field] = None,
    ref: common.Field,
    offset_provider: Optional[dict[str, common.Connectivty | common.Dimension]] = None,
    comparison: Callable[[Any, Any], bool] = np.allclose,
) -> None:
    """
    Check the result of executing a fieldview program or operator against ref.

    One of `out` or `nopass_out` must be passed.
    If `out` is passed it will be used as an argument to the fieldview program and compared against `ref`.
    Else, `nopass_out` will not be passed and compared to `ref`.
    """
    if out:
        run(
            case,
            fieldview_prog,
            *args,
            out=out,
            offset_provider=offset_provider,
        )
    else:
        run(case, fieldview_prog, *args, offset_provider=offset_provider)

    assert comparison(ref, out or nopass_out)


def verify_with_default_data(
    case: Case,
    fieldop: decorator.FieldOperator,
    ref: Callable,
    comparison: Callable[[Any, Any], bool] = np.allclose,
) -> None:
    inps, kwfields = get_default_data(case, fieldop)
    ref_args = tuple(i.array() if hasattr(i, "array") else i for i in inps)
    verify(
        case,
        fieldop,
        *inps,
        **kwfields,
        ref=ref(*ref_args),
        offset_provider=case.offset_provider,
        comparison=comparison,
    )


@dataclasses.dataclass
class Case:
    backend: ppi.ProgramProcessor
    # default offset provider for verify
    offset_provider: dict[str, common.Connectivty | common.Dimension]
    default_sizes: dict[common.Dimension, int]


@pytest.fixture
def no_default_backend():
    backup_backend = decorator.DEFAULT_BACKEND
    decorator.DEFAULT_BACKEND = no_backend
    yield
    decorator.DEFAULT_BACKEND = backup_backend


@pytest.fixture
def cartesian_case(no_default_backend, fieldview_backend):  # noqa: F811 # fixtures
    yield Case(
        fieldview_backend,
        offset_provider={"Ioff": IDim, "Joff": JDim, "Koff": KDim},
        default_sizes={IDim: 10, JDim: 10, KDim: 10},
    )


@pytest.fixture
def unstructured_case(
    no_default_backend, reduction_setup, fieldview_backend  # noqa: F811 # fixtures
):
    yield Case(
        fieldview_backend,
        offset_provider=reduction_setup.offset_provider,
        default_sizes={
            Vertex: reduction_setup.num_vertices,
            Edge: reduction_setup.num_edges,
        },
    )

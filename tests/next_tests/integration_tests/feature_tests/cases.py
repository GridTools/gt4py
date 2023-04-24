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
from typing import Any, Callable, Optional, Protocol, Sequence, TypeAlias

import numpy as np
import pytest
from gt4py.eve.extended_typing import Self

from gt4py.next import common
from gt4py.next.ffront import decorator, fbuiltins
from gt4py.next.iterator import embedded, ir as itir
from gt4py.next.program_processors import processor_interface as ppi
from gt4py.next.type_system import type_specifications as ts, type_translation

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (  # noqa: F401 #  fixture and aliases
    Edge,
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


class DataInitializer(Protocol):
    def scalar(self, dtype: str, shape: Optional[Sequence[int]]) -> int | float | np.array:
        ...

    def field(
        self, backend: ppi.ProgramProcessor, sizes: dict[common.Dimension, int], dtype: str
    ) -> common.Field:
        ...

    def from_case(
        self,
        case: Case,
        fieldview_prog: decorator.FieldOperator | decorator.Program,
        arg_name: str,
        tuple_position: tuple[int | tuple, ...] = tuple(),
    ) -> Self:
        return self


@dataclasses.dataclass
class ConstInitializer(DataInitializer):
    """Initialize with a given value accross the coordinate space."""

    value: int | float

    def scalar(self, dtype: str, shape: Optional[Sequence[int]]) -> int | float | np.array:
        if shape:
            return np.ones(np.prod(shape), dtype=dtype) * self.value
        return np.dtype(dtype).type(self.value)

    def field(
        self, backend: ppi.ProgramProcessor, sizes: dict[common.Dimension, int], dtype: str
    ) -> common.Field:
        return embedded.np_as_located_field(*sizes.keys())(
            (np.ones(tuple(sizes.values())) * self.value).astype(dtype=dtype)
        )


@dataclasses.dataclass
class ZeroInitializer(DataInitializer):
    """Initialize with zeros."""

    def scalar(self, dtype: str, shape: Optional[Sequence[int]]) -> int | float | np.array:
        if shape:
            return np.zeros(np.prod(shape), dtype=dtype)
        return np.dtype(dtype).type(0)

    def field(
        self, backend: ppi.ProgramProcessor, sizes: dict[common.Dimension, int], dtype: str
    ) -> common.Field:
        return embedded.np_as_located_field(*sizes.keys())(
            np.zeros(tuple(sizes.values()), dtype=dtype)
        )


@dataclasses.dataclass
class UniqueInitializer(DataInitializer):
    start: int = 0

    def scalar(self, dtype: str, shape: Optional[Sequence[int]]) -> int | float | np.array:
        start = self.start
        if shape:
            n_data = int(np.prod(shape))
            self.start += n_data
            return np.arange(start, start + n_data, dtype=dtype)
        self.start += 1
        return np.dtype(dtype).type(self.start)

    def field(
        self, backend: ppi.ProgramProcessor, sizes: dict[common.Dimension, int], dtype: str
    ) -> common.Field:
        """Initialize a field with a unique value in each coordinate."""
        start = self.start
        svals = tuple(sizes.values())
        n_data = int(np.prod(svals))
        self.start += n_data
        return embedded.np_as_located_field(*sizes.keys())(
            np.arange(start, start + n_data, dtype=dtype).reshape(svals)
        )

    def from_case(
        self,
        case: Case,
        fieldview_prog: decorator.FieldOperator | decorator.Program,
        arg_name: str,
    ) -> Self:
        param_types = get_param_types(fieldview_prog)
        param_sizes = [
            get_param_size(param_type, sizes=case.default_sizes)
            for param_type in param_types.values()
        ]
        param_index = list(param_types.keys()).index(arg_name)
        return self.__class__(start=self.start + sum(param_sizes[:param_index]))


@dataclasses.dataclass(frozen=True)
class Builder:
    partial: functools.partial

    def build(self, *args, **kwargs) -> Self:
        return self.partial(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self.build(*args, **kwargs)


def make_builder(
    func: Optional[Callable] = None, **kwargs: dict[str, Any]
) -> Callable[[Callable], Builder] | Builder:
    """Create a fluid inteface for a function with many arguments."""

    def make_builder_inner(func):
        def make_setter(argname):
            def setter(self, arg):
                return self.__class__(
                    partial=functools.partial(
                        self.partial, *self.partial.args, **(self.partial.keywords | {argname: arg})
                    )
                )

            setter.__name__ = argname
            return setter

        def make_flag_setter(flag_name: str, flag_kwargs):
            def setter(self):
                return self.__class__(
                    partial=functools.partial(
                        self.partial, *self.partial.args, **(self.partial.keywords | flag_kwargs)
                    )
                )

            setter.__name__ = flag_name
            return setter

        argspec = inspect.getfullargspec(func)

        @dataclasses.dataclass(frozen=True)
        class NewBuilder(Builder):
            for argname in argspec.args + argspec.kwonlyargs:
                locals()[argname] = make_setter(argname)

            for flag, flag_kwargs in kwargs.items():
                locals()[flag] = make_flag_setter(flag, flag_kwargs)

        func_snake_words = func.__name__.split("_")
        func_camel_name = "".join(word.capitalize() for word in func_snake_words)

        NewBuilder.__name__ = f"{func_camel_name}Builder"

        return NewBuilder(functools.partial(func))

    if func:
        return make_builder_inner(func)
    return make_builder_inner


def allocate(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    name: str,
) -> Builder:
    """Allocate a parameter or return value from a fieldview code object with a fluid interface."""
    return allocate_.sizes(case.default_sizes).case(case).fieldview_prog(fieldview_prog).name(name)


def get_param_types(
    fieldview_prog: decorator.FieldOperator | decorator.Program,
) -> dict[str, ts.TypeSpec]:
    if fieldview_prog.definition is None:
        raise ValueError(
            f"test cases do not support {type(fieldview_prog)} with empty .definition attribute (as you would get from .as_program())!"
        )
    annotations = typing.get_type_hints(fieldview_prog.definition)
    return {
        name: type_translation.from_type_hint(type_hint) for name, type_hint in annotations.items()
    }


def get_param_size(
    param_type: ts.TypeSpec, sizes: dict[common.Dimension, int | tuple[int | tuple]]
) -> int:
    match param_type:
        case ts.FieldType(dims=dims):
            return int(np.prod([sizes[dim] for dim in sizes if dim in dims]))
        case ts.ScalarType(shape=shape):
            return int(np.prod(shape)) if shape else 1
        case ts.TupleType(types):
            return sum([get_param_size(t, sizes=sizes) for t in types])
        case _:
            raise TypeError(f"Can not get size for parameter of type {param_type}")


def extend_sizes(
    sizes: dict[common.Dimension, int],
    extend: Optional[dict[common.Dimension, tuple[int, int]]] = None,
) -> dict[common.Dimension, int]:
    sizes = sizes.copy()
    if extend:
        for dim, (lower, upper) in extend.items():
            sizes[dim] += upper - lower
    return sizes


RETURN = "return"


@make_builder(zeros={"strategy": ZeroInitializer()}, unique={"strategy": UniqueInitializer()})
def allocate_(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    name: str,
    sizes: dict[common.Dimension, int],
    strategy: DataInitializer = UniqueInitializer(),
    dtype: Optional[str] = None,
    extend: Optional[dict[common.Dimension, tuple[int, int]]] = None,
) -> common.Field | tuple[common.Field | int | float | tuple, ...] | int | float:
    """Allocate a field for a parameter or return value of a fieldview program or operator."""
    sizes = extend_sizes(case.default_sizes | (sizes or {}), extend)
    arg_type = get_param_types(fieldview_prog)[name]
    if name in ["out", RETURN] and strategy is None:
        strategy = ZeroInitializer()
    return _allocate_for(
        case=case,
        arg_type=arg_type,
        sizes=sizes,
        dtype=dtype,
        strategy=strategy.from_case(case=case, fieldview_prog=fieldview_prog, arg_name=name),
    )


def _allocate_for(
    case: Case,
    arg_type: ts.TypeSpec,
    sizes: dict[common.Dimension, int],
    strategy: DataInitializer,
    dtype: Optional[str] = None,
    tuple_start: Optional[int] = None,
) -> common.Field | tuple[common.Field | int | float, ...] | int | float:
    """Allocate a field based on the field or scalar type or a (nested) tuple thereof."""
    match arg_type:
        case ts.FieldType(dims=dims, dtype=arg_dtype):
            return strategy.field(
                backend=case.backend,
                sizes={dim: sizes[dim] for dim in dims},
                dtype=dtype or arg_dtype.kind.name.lower(),
            )
        case ts.ScalarType(kind=kind, shape=shape):
            return strategy.scalar(dtype=dtype or kind.name.lower(), shape=shape)
        case ts.TupleType(types=types):
            return tuple(
                (
                    _allocate_for(
                        case=case, arg_type=t, sizes=sizes, dtype=dtype, strategy=strategy
                    )
                    for t in types
                )
            )
        case _:
            raise TypeError(
                f"Can not allocate for type {arg_type} with initializer {strategy or 'default'}"
            )


def run(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    *args: common.Field,
    **kwargs: Any,
) -> None:
    """Run fieldview code in the context of a given test case."""
    if kwargs.get("offset_provider", None) is None:
        kwargs["offset_provider"] = case.offset_provider
    fieldview_prog.with_backend(case.backend)(*args, **kwargs)


def get_default_data(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
) -> tuple[tuple[common.Field | int | float, ...], dict[str : common.Field | int | float]]:
    """
    Allocate default params and return values for a fieldview code object given a test case.

    Meant to reduce boiler plate for simple cases, everything else should rely on `allocate()`.
    """
    param_types = get_param_types(fieldview_prog)
    kwfields: dict[str, Any] = {}
    if RETURN in param_types:
        if not isinstance(param_types[RETURN], types.NoneType):
            kwfields = {"out": allocate(case, fieldview_prog, RETURN).zeros()()}
        param_types.pop(RETURN)
    if "out" in param_types:
        kwfields = {"out": allocate(case, fieldview_prog, "out").zeros()()}
        param_types.pop("out")
    inps = tuple(allocate(case, fieldview_prog, name)() for name in param_types)
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

    out_comp = out or nopass_out
    out_comp_str = str(out_comp)
    assert out_comp is not None
    if hasattr(out_comp, "array"):
        out_comp_str = str(out_comp.array())
    assert comparison(
        ref, out_comp
    ), f"Verification failed:\n\tcomparison={comparison.__name__}(ref, out)\n\tref = {ref}\n\tout = {out_comp_str}"


def verify_with_default_data(
    case: Case,
    fieldop: decorator.FieldOperator,
    ref: Callable,
    comparison: Callable[[Any, Any], bool] = np.allclose,
) -> None:
    """
    Check the result of executing a fieldview code object agains a reference constructed from the inputs.

    This is a convenience function to reduce boiler plate and not meant to hide logic for complex cases.
    The `verify` function allowes more fine grained control for such tests.
    """
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
    """Parametrizable components for single feature integration tests."""

    backend: ppi.ProgramProcessor
    offset_provider: dict[str, common.Connectivty | common.Dimension]
    default_sizes: dict[common.Dimension, int]


@pytest.fixture
def no_default_backend():
    """Temporarily switch off default backend for feature tests."""
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

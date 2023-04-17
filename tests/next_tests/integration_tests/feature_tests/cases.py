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
from typing import Callable, Optional, Protocol, TypeAlias

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


class FieldInitializer(Protocol):
    """Protocol for field initialization strategies."""

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


DEFAULT = unique


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


def allocate(case: Case, fieldop: decorator.FieldOperator, name: str) -> Builder:
    return (
        make_builder(allocate_fieldop)
        .sizes(case.default_sizes)
        .case(case)
        .fieldop(fieldop)
        .name(name)
    )


def allocate_fieldop(
    case: Case,
    fieldop: decorator.FieldOperator,
    name: str,
    sizes: dict[common.Dimension, int],
    strategy: FieldInitializer,
    dtype: Optional[str] = None,
    extend: Optional[dict[common.Dimension, tuple[int, int]]] = None,
) -> common.Field | tuple[common.Field, ...]:
    """Allocate a field for a parameter or return value of a fieldview program or operator."""
    sizes = case.default_sizes | (sizes or {})
    if extend:
        for dim, (lower, upper) in extend.items():
            sizes[dim] += upper - lower
    arg_type = (
        fieldop.foast_node.type.definition.returns if name == "out" else fieldop.param_types[name]
    )
    return _allocate_for(case=case, arg_type=arg_type, sizes=sizes, dtype=dtype, strategy=strategy)


def _allocate_for(
    case: Case,
    arg_type: ts.FieldType | ts.TupleType,
    sizes: dict[common.Dimension, int],
    strategy: FieldInitializer,
    dtype: Optional[str] = None,
) -> common.Field | tuple[common.Field, ...]:
    """Allocate a field based on the field type or a (nested) tuple thereof."""
    match arg_type:
        case ts.FieldType():
            return strategy(
                backend=case.backend,
                sizes={dim: sizes[dim] for dim in arg_type.dims},
                dtype=dtype or arg_type.dtype.kind.name.lower(),
            )
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


def verify(
    case: Case,
    fieldop: decorator.FieldOperator,
    *args: common.Field,
    out: common.Field,  # TODO: make optional
    ref: common.Field,
    offset_provider: Optional[dict[str, common.Connectivty | common.Dimension]] = None,
) -> None:
    """Check the result of executing a fieldview program or operator against ref."""
    offset_provider = offset_provider or case.offset_provider
    fieldop.with_backend(case.backend)(*args, out=out, offset_provider=offset_provider)

    assert np.allclose(ref, out)


def verify_with_default_data(case: Case, fieldop: decorator.FieldOperator, ref: Callable) -> None:
    inps = tuple(allocate(case, fieldop, name).strategy(DEFAULT)() for name in fieldop.param_types)
    out = allocate(case, fieldop, "out").strategy(zeros)()
    print(f"inps:\n{[i.array() for i in inps]}")  # todo: remove
    print(f"out:\n{out}")  # todo: remove
    verify(
        case,
        fieldop,
        *inps,
        out=out,
        ref=ref(*(i.array() for i in inps)),
        offset_provider=case.offset_provider,
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

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import types
import typing
from typing import Any, Callable, Literal, Optional, Mapping, Protocol, TypeAlias

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import Self
from gt4py.next import (
    allocators as next_allocators,
    backend as next_backend,
    common,
    constructors,
    field_utils,
    utils as gt_utils,
)
from gt4py.next.ffront import decorator
from gt4py.next.type_system import type_specifications as ts, type_translation

from next_tests import definitions as test_definitions
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (  # noqa: F401 [unused-import]
    C2E,
    C2V,
    E2V,
    V2E,
    C2EDim,
    C2VDim,
    Cell,
    E2VDim,
    Edge,
    IDim,
    Ioff,
    JDim,
    Joff,
    KDim,
    KHalfDim,
    Koff,
    V2EDim,
    Vertex,
    exec_alloc_descriptor,
    mesh_descriptor,
    MeshDescriptor,
    simple_cartesian_grid,
    CartesianGridDescriptor,
)


# mypy does not accept [IDim, ...] as a type

IField: TypeAlias = gtx.Field[[IDim], np.int32]  # type: ignore [valid-type]
JField: TypeAlias = gtx.Field[[JDim], np.int32]  # type: ignore [valid-type]
IFloatField: TypeAlias = gtx.Field[[IDim], np.float64]  # type: ignore [valid-type]
IBoolField: TypeAlias = gtx.Field[[IDim], bool]  # type: ignore [valid-type]
JField: TypeAlias = gtx.Field[[JDim], np.int32]  # type: ignore [valid-type]
KField: TypeAlias = gtx.Field[[KDim], np.int32]  # type: ignore [valid-type]
IJField: TypeAlias = gtx.Field[[IDim, JDim], np.int32]  # type: ignore [valid-type]
IKField: TypeAlias = gtx.Field[[IDim, KDim], np.int32]  # type: ignore [valid-type]
IKFloatField: TypeAlias = gtx.Field[[IDim, KDim], np.float64]  # type: ignore [valid-type]
IJKField: TypeAlias = gtx.Field[[IDim, JDim, KDim], np.int32]  # type: ignore [valid-type]
IJKFloatField: TypeAlias = gtx.Field[[IDim, JDim, KDim], np.float64]  # type: ignore [valid-type]
VField: TypeAlias = gtx.Field[[Vertex], np.int32]  # type: ignore [valid-type]
VBoolField: TypeAlias = gtx.Field[[Vertex], bool]  # type: ignore [valid-type]
EField: TypeAlias = gtx.Field[[Edge], np.int32]  # type: ignore [valid-type]
CField: TypeAlias = gtx.Field[[Cell], np.int32]  # type: ignore [valid-type]
EmptyField: TypeAlias = gtx.Field[[], np.int32]  # type: ignore [valid-type]

ScalarValue: TypeAlias = core_defs.Scalar
FieldValue: TypeAlias = gtx.Field
FieldViewArg: TypeAlias = FieldValue | ScalarValue | tuple["FieldViewArg", ...]
FieldViewInout: TypeAlias = FieldValue | tuple["FieldViewInout", ...]
ReferenceValue: TypeAlias = (
    gtx.Field | np.typing.NDArray[ScalarValue] | tuple["ReferenceValue", ...]
)
OffsetProvider: TypeAlias = dict[str, common.Connectivity | gtx.Dimension]


#: To allocate the return value of a field operator, we must pass
#: something that is not an argument name. Currently this is  the
#: literal string "return" (because it is read from annotations).
#: This could change if implemented differently. RETURN acts as a
#: proxy to avoid propagating the change to client code.
RETURN = "return"


class DataInitializer(Protocol):
    @property
    def scalar_value(self) -> ScalarValue: ...

    def scalar(self, dtype: np.typing.DTypeLike) -> ScalarValue:
        # some unlikely numpy dtypes are picky about arguments
        return np.dtype(dtype).type(self.scalar_value)  # type: ignore [arg-type]

    def field(
        self,
        allocator: next_allocators.FieldBufferAllocatorProtocol,
        domain: gtx.Domain,
        dtype: np.typing.DTypeLike,
    ) -> FieldValue: ...

    def from_case(
        self: Self,
        case: Case,
        fieldview_prog: decorator.FieldOperator | decorator.Program,
        arg_name: str,
    ) -> Self:
        return self


@dataclasses.dataclass(init=False)
class ConstInitializer(DataInitializer):
    """Initialize with a given value across the coordinate space."""

    value: ScalarValue

    def __init__(self, value: ScalarValue):
        if not core_defs.is_scalar_type(value):
            raise ValueError(
                "'ConstInitializer' can not be used with non-scalars. Use 'Case.as_field' instead."
            )
        self.value = value

    @property
    def scalar_value(self) -> ScalarValue:
        return self.value

    def field(
        self,
        allocator: next_allocators.FieldBufferAllocatorProtocol,
        domain: gtx.Domain,
        dtype: np.typing.DTypeLike,
    ) -> FieldValue:
        return constructors.full(
            domain=domain, fill_value=self.value, dtype=dtype, allocator=allocator
        )


@dataclasses.dataclass(init=False)
class ZeroInitializer(ConstInitializer):
    """Initialize with zeros."""

    def __init__(self):
        self.value = 0


class IndexInitializer(DataInitializer):
    """Initialize a 1d field with the index of the coordinate point."""

    @property
    def scalar_value(self) -> ScalarValue:
        raise AttributeError("'scalar_value' not supported in 'IndexInitializer'.")

    def field(
        self,
        allocator: next_allocators.FieldBufferAllocatorProtocol,
        domain: gtx.Domain,
        dtype: np.typing.DTypeLike,
    ) -> FieldValue:
        if len(domain.dims) > 1:
            raise ValueError(
                f"'IndexInitializer' only supports fields with a single 'Dimension', got {domain}."
            )
        return constructors.as_field(
            domain=domain,
            data=np.arange(domain.ranges[0].start, domain.ranges[0].stop, dtype=dtype),
            allocator=allocator,
        )

    def from_case(
        self: Self,
        case: Case,
        fieldview_prog: decorator.FieldOperator | decorator.Program,
        arg_name: str,
    ) -> Self:
        return self


@dataclasses.dataclass
class UniqueInitializer(DataInitializer):
    """
    Initialize with a unique value in each coordinate point.

    Data initialized with the same instance will also have unique values across
    data containers.
    """

    start: int = 1

    @property
    def scalar_value(self) -> ScalarValue:
        start = self.start
        self.start += 1
        return start

    def field(
        self,
        allocator: next_allocators.FieldBufferAllocatorProtocol,
        domain: common.Domain,
        dtype: np.typing.DTypeLike,
    ) -> FieldValue:
        start = self.start
        assert isinstance(domain.size, int)
        self.start += domain.size
        return constructors.as_field(
            common.domain(domain),
            np.arange(start, self.start, dtype=dtype).reshape(domain.shape),
            allocator=allocator,
        )

    def from_case(
        self: Self,
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.partial(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"No setter for argument '{name}'.")


@typing.overload
def make_builder(*args: Callable) -> Callable[..., Builder]: ...


@typing.overload
def make_builder(
    *args: Literal[None], **kwargs: dict[str, Any]
) -> Callable[[Callable], Callable[..., Builder]]: ...


@typing.overload
def make_builder(
    *args: Optional[Callable], **kwargs: dict[str, Any]
) -> Callable[[Callable], Callable[..., Builder]] | Callable[..., Builder]: ...


# TODO(ricoh): Think about improving the type hints using `typing.ParamSpec`.
def make_builder(
    *args: Optional[Callable], **kwargs: dict[str, Any]
) -> Callable[[Callable], Callable[..., Builder]] | Callable[..., Builder]:
    """Create a fluid interface for a function with many arguments."""

    def make_builder_inner(func: Callable) -> Callable[..., Builder]:
        def make_setter(argname: str) -> Callable[[Builder, Any], Builder]:
            def setter(self: Builder, arg: Any) -> Builder:
                return self.__class__(
                    partial=functools.partial(
                        self.partial.func,
                        *self.partial.args,
                        **(self.partial.keywords | {argname: arg}),
                    )
                )

            setter.__name__ = argname
            return setter

        def make_flag_setter(
            flag_name: str, flag_kwargs: dict[str, Any]
        ) -> Callable[[Builder], Builder]:
            def setter(self: Builder) -> Builder:
                return self.__class__(
                    partial=functools.partial(
                        self.partial.func,
                        *self.partial.args,
                        **(self.partial.keywords | flag_kwargs),
                    )
                )

            setter.__name__ = flag_name
            return setter

        argspec = inspect.getfullargspec(func)

        @dataclasses.dataclass(frozen=True)
        class NewBuilder(Builder): ...

        for argname in argspec.args + argspec.kwonlyargs:
            setattr(NewBuilder, argname, make_setter(argname))

        for flag, flag_kwargs in kwargs.items():
            setattr(NewBuilder, flag, make_flag_setter(flag, flag_kwargs))

        func_snake_words = func.__name__.split("_")
        func_camel_name = "".join(word.capitalize() for word in func_snake_words)

        NewBuilder.__name__ = f"{func_camel_name}Builder"

        return lambda *args, **kwargs: NewBuilder(functools.partial(func, *args, **kwargs))

    if 0 < len(args) <= 1 and args[0] is not None:
        return make_builder_inner(args[0])
    if len(args) > 1:
        raise ValueError(f"make_builder takes only one positional argument, {len(args)} received.")
    return make_builder_inner


@make_builder(zeros={"strategy": ZeroInitializer()}, unique={"strategy": UniqueInitializer()})
def allocate(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    name: str,
    *,
    sizes: Optional[dict[gtx.Dimension, int]] = None,
    domain: Optional[dict[gtx.Dimension, tuple[int, int]] | gtx.Domain] = None,
    strategy: Optional[DataInitializer] = None,
    dtype: Optional[np.typing.DTypeLike] = None,
    extend: Optional[dict[gtx.Dimension, tuple[int, int]]] = None,
) -> FieldViewArg:
    """
    Allocate a parameter or return value from a fieldview code object.

    Args:
        case: The test case.
        fieldview_prog: The field operator or program to be verified.
        name: The name of the input argument to allocate, or ``RETURN``
            for the return value of a field operator.
    Keyword Args:
        sizes: Override for the test case dimension sizes.
            Use with caution.
        strategy: How to initialize the data.
        dtype: Override for the dtype in the argument's type hint.
        extend: Lower and upper size extension per dimension.
            Useful for shifted fields, which must start off bigger
            than the output field in the shifted dimension.
    """
    if sizes:
        assert not domain and all(dim in case.default_sizes for dim in sizes)
        domain = {
            dim: (0, sizes.get(dim, default_size))
            for dim, default_size in case.default_sizes.items()
        }

    domain = domain or {dim: (0, size) for dim, size in case.default_sizes.items()}

    if not isinstance(domain, gtx.Domain):
        domain = gtx.domain(domain)

    domain = extend_domain(
        domain, extend
    )  # TODO: this should take into account the Domain of the allocated field

    arg_type = get_param_types(fieldview_prog)[name]
    if strategy is None:
        if name in ["out", RETURN]:
            strategy = ZeroInitializer()
        else:
            strategy = UniqueInitializer()
    return _allocate_from_type(
        case=case,
        arg_type=arg_type,
        domain=domain,
        dtype=dtype,
        strategy=strategy.from_case(case=case, fieldview_prog=fieldview_prog, arg_name=name),
    )


def run(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    *args: FieldViewArg,
    **kwargs: Any,
) -> None:
    """Run fieldview code in the context of a given test case."""
    if kwargs.get("offset_provider", None) is None:
        kwargs["offset_provider"] = case.offset_provider
    fieldview_prog.with_grid_type(case.grid_type).with_backend(case.backend)(*args, **kwargs)


try:
    RTOL, ATOL, EQUAL_NAN = np.allclose.__wrapped__.__defaults__
except AttributeError:
    RTOL, ATOL, EQUAL_NAN = (1e-05, 1e-08, False)


def tree_mapped_np_allclose(
    ref: np.ndarray,
    data: np.ndarray,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    equal_nan: bool = EQUAL_NAN,
) -> bool:
    """Compare two arrays or nested tuples of arrays using np.allclose."""
    if (is_tuple := isinstance(ref, tuple)) == isinstance(data, tuple):
        allclose_with_tols = functools.partial(
            np.allclose, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
        if is_tuple:
            allclose_results = gt_utils.tree_map(allclose_with_tols)(ref, data)
            return all(gt_utils.flatten_nested_tuple(allclose_results))
        else:
            return allclose_with_tols(ref, data)

    return False


def verify(
    case: Case,
    fieldview_prog: decorator.FieldOperator | decorator.Program,
    *args: FieldViewArg,
    ref: ReferenceValue,
    domain: Optional[dict[common.Dimension, tuple[int, int]]] = None,
    out: Optional[FieldViewInout] = None,
    inout: Optional[FieldViewInout] = None,
    offset_provider: Optional[OffsetProvider] = None,
    comparison: Callable[[Any, Any], bool] = tree_mapped_np_allclose,
) -> None:
    """
    Check the result of executing a fieldview program or operator against ref.

    Args:
        case: The test case.
        fieldview_prog: The field operator or program to be verified.
        *args: positional input arguments to the fieldview code.
    Keyword Args:
        ref: A field or array which will be compared to the results of the
            fieldview code.
        out: If given will be passed to the fieldview code as ``out=`` keyword
            argument. This will hold the results and be used to compare
            to ``ref``.
        inout: If ``out`` is not passed it is assumed that the fieldview code
            does not take an ``out`` keyword argument, so some of the other
            arguments must be in/out parameters. Pass the according field
            or tuple of fields here and they will be compared to ``ref`` under
            the assumption that the fieldview code stores its results in
            them.
        domain: If given will be passed to the fieldview code as ``domain=``
            keyword argument.
        offset_provider: An override for the test case's offset_provider.
            Use with care!
        comparison: A comparison function, which will be called as
            ``comparison(ref, <out | inout>)`` and should return a boolean.

    One of ``out`` or ``inout`` must be passed. If ``out`` is passed it will be
    used as an argument to the fieldview program and compared against ``ref``.
    Else, ``inout`` will not be passed and compared to ``ref``.
    """
    kwargs = {}
    if out:
        kwargs["out"] = out
    if domain:
        kwargs["domain"] = domain

    run(case, fieldview_prog, *args, **kwargs, offset_provider=offset_provider)

    out_comp = out or inout
    assert out_comp is not None
    out_comp_ndarray = field_utils.asnumpy(out_comp)
    ref_ndarray = field_utils.asnumpy(ref)

    assert comparison(ref_ndarray, out_comp_ndarray), (
        f"Verification failed:\n"
        f"\tcomparison={comparison.__name__}(ref, out)\n"
        f"\tref = {ref_ndarray!s}\n\tout = {out_comp_ndarray!s}"
    )


def verify_with_default_data(
    case: Case,
    fieldop: decorator.FieldOperator,
    ref: Callable,
    comparison: Callable[[Any, Any], bool] = tree_mapped_np_allclose,
) -> None:
    """
    Check the fieldview code against a reference calculation.

    This is a convenience function to reduce boiler plate
    and is not meant to hide logic for complex cases. The
    ``verify`` function allows more fine grained control for such tests.

    Args:
        case: The test case.
        fieldview_prog: The field operator or program to be verified.
        ref: A callable which will be called with all the input arguments
            of the fieldview code, after applying ``.ndarray`` on the fields.
        comparison: A comparison function, which will be called as
            ``comparison(ref, <out | inout>)`` and should return a boolean.
    """
    inps, kwfields = get_default_data(case, fieldop)
    ref_args: tuple = gt_utils.tree_map(
        lambda x: x.asnumpy() if isinstance(x, common.Field) else x
    )(inps)
    verify(
        case,
        fieldop,
        *inps,
        **kwfields,
        ref=ref(*ref_args),
        offset_provider=case.offset_provider,
        comparison=comparison,
    )


@pytest.fixture
def cartesian_case_no_backend():
    return Case.from_cartesian_grid_descriptor(simple_cartesian_grid())


@pytest.fixture
def cartesian_case(
    exec_alloc_descriptor: test_definitions.EmbeddedDummyBackend | next_backend.Backend,
):
    return Case.from_cartesian_grid_descriptor(
        simple_cartesian_grid(),
        backend=(
            None
            if isinstance(exec_alloc_descriptor, test_definitions.EmbeddedDummyBackend)
            else exec_alloc_descriptor
        ),
        allocator=exec_alloc_descriptor.allocator,
    )


@pytest.fixture
def unstructured_case_no_backend(mesh_descriptor: MeshDescriptor):
    return Case.from_mesh_descriptor(mesh_descriptor)


@pytest.fixture
def unstructured_case(
    mesh_descriptor: MeshDescriptor,
    exec_alloc_descriptor: test_definitions.EmbeddedDummyBackend | next_backend.Backend,
):
    return Case.from_mesh_descriptor(
        mesh_descriptor,
        backend=(
            None
            if isinstance(exec_alloc_descriptor, test_definitions.EmbeddedDummyBackend)
            else exec_alloc_descriptor
        ),
        allocator=exec_alloc_descriptor.allocator,
    )


@pytest.fixture
def unstructured_case_3d(unstructured_case):
    return dataclasses.replace(
        unstructured_case,
        default_sizes={**unstructured_case.default_sizes, KDim: 10},
        offset_provider={**unstructured_case.offset_provider, "Koff": KDim},
    )


def _allocate_from_type(
    case: Case,
    arg_type: ts.TypeSpec,
    domain: gtx.Domain,
    strategy: DataInitializer,
    dtype: Optional[np.typing.DTypeLike] = None,
    tuple_start: Optional[int] = None,
) -> FieldViewArg:
    """Allocate data based on the type or a (nested) tuple thereof."""
    match arg_type:
        case ts.FieldType(dims=dims, dtype=arg_dtype):
            return strategy.field(
                allocator=case.allocator,
                domain=common.domain(tuple(domain[dim] for dim in dims)),
                dtype=dtype or arg_dtype.kind.name.lower(),
            )
        case ts.ScalarType(kind=kind):
            return strategy.scalar(dtype=dtype or kind.name.lower())
        case ts.TupleType(types=types):
            return tuple(
                (
                    _allocate_from_type(
                        case=case, arg_type=t, domain=domain, dtype=dtype, strategy=strategy
                    )
                    for t in types
                )
            )
        case _:
            raise TypeError(
                f"Can not allocate for type '{arg_type}' with initializer '{strategy or 'default'}'."
            )


def get_param_types(
    fieldview_prog: decorator.FieldOperator | decorator.Program,
) -> dict[str, ts.TypeSpec]:
    if fieldview_prog.definition is None:
        raise ValueError(
            f"test cases do not support '{type(fieldview_prog)}' with empty .definition attribute (as you would get from .as_program())."
        )
    annotations = xtyping.get_type_hints(fieldview_prog.definition)
    return {
        name: type_translation.from_type_hint(type_hint) for name, type_hint in annotations.items()
    }


def get_param_size(param_type: ts.TypeSpec, sizes: dict[gtx.Dimension, int]) -> int:
    match param_type:
        case ts.FieldType(dims=dims):
            return int(np.prod([sizes[dim] for dim in sizes if dim in dims]))
        case ts.ScalarType(shape=shape):
            return int(np.prod(shape)) if shape else 1
        case ts.TupleType(types):
            return sum([get_param_size(t, sizes=sizes) for t in types])
        case _:
            raise TypeError(f"Can not get size for parameter of type '{param_type}'.")


def extend_domain(
    domain: gtx.Domain, extend: Optional[dict[gtx.Dimension, tuple[int, int]]] = None
) -> dict[gtx.Dimension, int]:
    """Calculate the sizes per dimension given a set of extensions."""
    if extend:
        domain = copy.deepcopy(domain)
        for dim, (lower, upper) in extend.items():
            domain = domain.replace(
                dim,
                common.named_range(
                    (
                        dim,
                        (
                            domain[dim].unit_range.start - lower,
                            domain[dim].unit_range.stop + upper,
                        ),
                    )
                ),
            )
    return domain


def get_default_data(
    case: Case, fieldview_prog: decorator.FieldOperator | decorator.Program
) -> tuple[tuple[gtx.Field | ScalarValue | tuple, ...], dict[str, gtx.Field]]:
    """
    Allocate default data for a fieldview code object given a test case.

    Meant to reduce boiler plate for simple cases, everything else
    should rely on ``allocate()``.
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


@dataclasses.dataclass
class Case:
    """Parametrizable components for single feature integration tests."""

    backend: Optional[next_backend.Backend]
    offset_provider: dict[str, common.Connectivity | gtx.Dimension]
    default_sizes: dict[gtx.Dimension, int]
    grid_type: common.GridType
    allocator: next_allocators.FieldBufferAllocatorFactoryProtocol

    @property
    def as_field(self):
        return constructors.as_field.partial(allocator=self.allocator)

    @classmethod
    def from_cartesian_grid_descriptor(
        cls,
        grid_descriptor: CartesianGridDescriptor,
        backend: Optional[next_backend.Backend] = None,
        allocator: Optional[next_allocators.FieldBufferAllocatorFactoryProtocol] = None,
    ) -> Case:
        return cls(
            backend=backend,
            offset_provider=grid_descriptor.offset_provider,
            default_sizes={
                IDim: grid_descriptor.sizes[0],
                JDim: grid_descriptor.sizes[1],
                KDim: grid_descriptor.sizes[2],
                KHalfDim: grid_descriptor.sizes[3],
            },
            grid_type=common.GridType.CARTESIAN,
            allocator=allocator,
        )

    @classmethod
    def from_mesh_descriptor(
        cls,
        mesh_descriptor: MeshDescriptor,
        backend: Optional[next_backend.Backend] = None,
        allocator: Optional[next_allocators.FieldBufferAllocatorFactoryProtocol] = None,
    ) -> Case:
        return cls(
            backend=backend,
            offset_provider=mesh_descriptor.offset_provider,
            default_sizes={
                Vertex: mesh_descriptor.num_vertices,
                Edge: mesh_descriptor.num_edges,
                Cell: mesh_descriptor.num_cells,
            },
            grid_type=common.GridType.UNSTRUCTURED,
            allocator=allocator,
        )

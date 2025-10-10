# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
import enum
import typing
from typing import Any, Generic, Mapping, Optional, final

from typing_extensions import Self

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import MaybeNestedInTuple
from gt4py.next import common, errors
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


DATA_T = typing.TypeVar("DATA_T")
T = typing.TypeVar("T")


def _make_dict_expr(exprs: dict[str, str]) -> str:
    return "{" + ",".join((f"'{k}': {v}" for k, v in exprs.items())) + "}"


class ArgStaticDescriptor(abc.ABC):
    """
    Abstract class to represent, extract, validate compile time information of an argument.

    The information that is available at compile time is extracted from the runtime argument
    (or provided when pre-compiling) is described by a set of (python) expressions returned by the
    `attribute_extractor` class-method. These expressions are evaluated in the context of the
    arguments. We chose expressions here instead of a method taking the actual value such that we
    can code generate a single expression for all argument descriptors only retrieving the necessary
    values without actually constructing the descriptors. That way the cache key computation to the
    compiled is fast.
    """

    def validate(self, name: str, type_: ts.TypeSpec) -> None:  # noqa: B027  # method is not abstract, but just empty when not implemented
        """
        Validate argument descriptor in the context of an actual program.

        This function is called when the type of the argument is available. The name is merely
        given to give good error messages.
        """
        pass

    @classmethod
    @final
    def from_value(cls, value: Any) -> ArgStaticDescriptor:
        attr_exprs = cls.attribute_extractor_exprs("self")
        return cls(**eval(f"""lambda self: {_make_dict_expr(attr_exprs)}""")(value))

    @classmethod
    @abc.abstractmethod
    def attribute_extractor_exprs(cls, arg_expr: str) -> dict[str, str]:
        """
        Return a mapping from the attributes of our descriptor to the expressions to retrieve them.

        E.g. if `arg_expr` would be `myarg` and the result of this function
        `{'value': 'my_arg.value'}` then the descriptor is constructed as
        `ArgumentDescriptor(value=my_arg.value)`. We use an expression here such that we can compute
        a cache key by just hashing `my_arg.value` instead of first constructing the descriptor.
        """
        ...


@dataclasses.dataclass(frozen=True)
class StaticArg(ArgStaticDescriptor, Generic[core_defs.ScalarT]):
    value: MaybeNestedInTuple[core_defs.ScalarT]

    def __post_init__(self) -> None:
        # transform enum value into the actual value
        if isinstance(self.value, enum.Enum):
            object.__setattr__(self, "value", self.value.value)

    def validate(self, name: str, type_: ts.TypeSpec) -> None:
        if not type_info.is_type_or_tuple_of_type(type_, ts.ScalarType):
            raise errors.DSLTypeError(
                message=f"Invalid static argument '{name}' with type '{type_}' (only scalars or (nested) tuples of scalars can be static).",
                location=None,
            )

        actual_type = type_translation.from_value(self.value)
        if actual_type != type_:
            raise errors.DSLTypeError(
                message=f"Invalid static argument '{name}', expected '{type_}', but static value '{self.value}' has type '{actual_type}'.",
                location=None,
            )

    @classmethod
    def attribute_extractor_exprs(cls, arg_expr: str) -> dict[str, str]:
        return {"value": arg_expr}


@dataclasses.dataclass(frozen=True)
class JITArgs:
    """Concrete (runtime) arguments to a GTX program in a format that can be passed into the toolchain."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    @classmethod
    def from_signature(cls, *args: Any, **kwargs: Any) -> Self:
        return cls(args=args, kwargs=kwargs)


@dataclasses.dataclass(frozen=True)
class CompileTimeArgs:
    """Compile-time standins for arguments to a GTX program to be used in ahead-of-time compilation."""

    args: tuple[ts.TypeSpec, ...]
    kwargs: dict[str, ts.TypeSpec]
    offset_provider: common.OffsetProvider  # TODO(havogt): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    column_axis: Optional[common.Dimension]
    #: A mapping from an argument descriptor type to a context containing the actual descriptors.
    #: If an argument or element of an argument has no descriptor, the respective value is `None`.
    #: E.g., for a tuple argument `a` with type `ts.TupleTupe(types=[field_t, int32_t])` a possible
    #  context would be `{"a": (FieldDomainDescriptor(...), None)}`.
    argument_descriptor_contexts: Mapping[
        type[ArgStaticDescriptor],
        dict[str, MaybeNestedInTuple[ArgStaticDescriptor | None]],
    ]

    @property
    def offset_provider_type(self) -> common.OffsetProviderType:
        return common.offset_provider_to_type(self.offset_provider)

    @classmethod
    def from_concrete(cls, *args: Any, **kwargs: Any) -> Self:
        """Convert concrete GTX program arguments into their compile-time counterparts."""
        kwargs_copy = kwargs.copy()
        return cls(
            args=tuple(type_translation.from_value(arg) for arg in args),
            offset_provider=kwargs_copy.pop("offset_provider", {}),
            column_axis=kwargs_copy.pop("column_axis", None),
            kwargs={
                k: type_translation.from_value(v) for k, v in kwargs_copy.items() if v is not None
            },
            argument_descriptor_contexts={},
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(tuple(), {}, {}, None, {})


def jit_to_aot_args(
    inp: JITArgs,
) -> CompileTimeArgs:
    return CompileTimeArgs.from_concrete(*inp.args, **inp.kwargs)


def adapted_jit_to_aot_args_factory() -> workflow.Workflow[
    toolchain.CompilableProgram[DATA_T, JITArgs],
    toolchain.CompilableProgram[DATA_T, CompileTimeArgs],
]:
    """Wrap `jit_to_aot` into a workflow adapter to fit into backend transform workflows."""
    return toolchain.ArgsOnlyAdapter(jit_to_aot_args)

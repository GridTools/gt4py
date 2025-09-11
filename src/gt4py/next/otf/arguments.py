# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import typing
from pickletools import ArgumentDescriptor
from typing import Any, Generic, Optional, TypeAlias

from typing_extensions import Self

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing
from gt4py.next import common, utils, errors
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_specifications as ts, type_translation, type_info
from gt4py.next.type_system.type_info import apply_to_primitive_constituents

DATA_T = typing.TypeVar("DATA_T")
T = typing.TypeVar("T")
TOrTupleOf: TypeAlias = T | tuple["TupleOf[T]", ...]
ArgumentDescriptorT = typing.TypeVar("ArgumentDescriptorT", bound=ArgumentDescriptor)

class PartialValue(Generic[ArgumentDescriptorT]):
    attrs: dict[str, Any]
    items: dict[Any, Any]

    def __init__(self):
        object.__setattr__(self, "attrs", {})
        object.__setattr__(self, "items", {})

    def __setattr__(self, key: str, value: Any) -> None:
        object.__getattribute__(self, "attrs")[key] = value

    def __setitem__(self, key: Any, value: Any) -> None:
        object.__getattribute__(self, "items")[key] = value

    @property
    def empty(self):
        return not self.attrs and not self.items

class ArgumentDescriptor:
    def validate(self, name: str, type_: ts.TypeSpec):
        """
        Validate argument descriptor.

        This function is called when the type of the argument is available. The name is merely
        given to give good error messages.
        """
        pass

    @classmethod
    def attribute_extractor(cls, arg_expr: str) -> dict[str, str]:
        """
        Return a mapping from the attributes of our descriptor to the expressions to retrieve them.

        E.g. if `arg_expr` would be `myarg` and the result of this function
        `{'value': 'my_arg.value'}` then the descriptor is constructed as
        `ArgumentDescriptor(value=my_arg.value)`. We use expression here such that we can compute
        a cache key by just hashing `my_arg.value` instead of first constructing the descriptor.
        """
        ...

@dataclasses.dataclass(frozen=True)
class StaticArg(ArgumentDescriptor, Generic[T]):
    value: TOrTupleOf[core_defs.ScalarT]

    def validate(self, name: str, type_: ts.TypeSpec):
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
    def attribute_extractor(cls, arg_expr: str):
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
    argument_descriptors: dict[type[ArgumentDescriptor], PartialValue[ArgumentDescriptor]]

    @property
    def offset_provider_type(self) -> common.OffsetProviderType:
        return common.offset_provider_to_type(self.offset_provider)

    @classmethod
    def from_concrete(cls, *args: Any, **kwargs: Any) -> Self:
        """Convert concrete GTX program arguments into their compile-time counterparts."""
        kwargs = kwargs.copy()
        offset_provider = kwargs.pop("offset_provider", {})
        column_axis = kwargs.pop("column_axis", None)
        compile_args = tuple(StaticArg.from_value(arg) for arg in args)
        compile_kwargs = {
            k: StaticArg.from_value(v) for k, v in kwargs.items() if v is not None
        }
        return cls(
            args=compile_args,
            kwargs=compile_kwargs,
            offset_provider=offset_provider,
            column_axis=column_axis,
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
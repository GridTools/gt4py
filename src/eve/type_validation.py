# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""Generic interface and implementations of run-time type validation for arbitrary values."""


from __future__ import annotations

import abc
import collections.abc
import dataclasses
import functools

from . import exceptions
from . import extended_typing as xtyping
from . import utils
from .extended_typing import (
    Any,
    Dict,
    Final,
    ForwardRef,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeAnnotation,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)


# Protocols
@runtime_checkable
class TypeValidator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        value: Any,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        required: bool = True,
        **kwargs: Any,
    ) -> None:
        """Protocol for callables checking that ``value`` matches ``type_annotation``.

        Arguments:
            value: value to be checked against the typing annotation.
            type_annotation: a valid typing annotation.
            name: the name of the checked value (used for error messages).

        Keyword Arguments:
            globalns: globals dict used in the evaluation of the annotations.
            localns: locals dict used in the evaluation of the annotations.
            required: if ``True``, raise ``ValueError`` when provided type annotation is not supported.
            **kwargs: arbitrary implementation-defined arguments (e.g. for memoization).

        Raises:
            TypeError: if there is a type mismatch.
            ValueError: if ``required is True`` and ``type_annotation`` is not supported.
        """
        ...


class FixedTypeValidator(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        value: Any,
        **kwargs: Any,
    ) -> None:
        """Protocol for callables checking that ``value`` matches a fixed type_annotation.

        Arguments:
            value: value to be checked against the typing annotation.

        Keyword Arguments:
            **kwargs: arbitrary implementation-defined arguments (e.g. for memoization).

        Raises:
            TypeError: if there is a type mismatch.
        """
        ...


@runtime_checkable
class TypeValidatorFactory(Protocol):
    @overload
    def __call__(
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: Literal[True] = True,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FixedTypeValidator:
        ...

    @overload
    def __call__(  # noqa: F811  # redefinion of unused member
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        ...

    @abc.abstractmethod
    def __call__(  # noqa: F811  # redefinion of unused member
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        """Protocol for :class:`FixedTypeValidator`s.

        The arguments match the specification in :class:`TypeValidator`.

        Raises:
            TypeError: if there is a type mismatch.
            ValueError: if ``required is True`` and ``type_annotation`` is not supported.
        """
        ...


# Implementations
@dataclasses.dataclass(frozen=True)
class SimpleTypeValidatorFactory(TypeValidatorFactory):
    """A simple :class:`TypeValidatorFactory` implementation.

    Check :class:`FixedTypeValidator` and :class:`TypeValidatorFactory` for details.

    Keyword Arguments:
        strict_int (bool): do not accept ``bool`` values as ``int`` (default: ``True``).
    """

    @overload
    def __call__(
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: Literal[True] = True,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FixedTypeValidator:
        ...

    @overload
    def __call__(  # noqa: F811  # redefinion of unused member
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        ...

    def __call__(  # noqa: F811,C901  # redefinion of unused member / complex but well organized in cases
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[Dict[str, Any]] = None,
        localns: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:

        # TODO(egparedes): if a "typing tree" structure is implemented, refactor this code as a tree traversal.
        #
        if name is None:
            name = "<value>"

        make_recursive = functools.partial(
            self.__call__, name=name, globalns=globalns, localns=localns, **kwargs
        )

        try:
            # Non-generic types
            if xtyping.is_actual_type(type_annotation) and not isinstance(
                None, type_annotation  # NoneType is a different case
            ):
                assert not xtyping.get_args(type_annotation)
                if type_annotation is int and kwargs.get("strict_int", True):
                    return self.make_is_instance_of_int(name)
                else:
                    return self.make_is_instance_of(name, type_annotation)

            if isinstance(type_annotation, TypeVar):
                if type_annotation.__bound__:
                    return self.make_is_instance_of(name, type_annotation.__bound__)
                else:
                    return self._make_is_any(name)

            if isinstance(type_annotation, ForwardRef):
                return make_recursive(
                    xtyping.eval_forward_ref(type_annotation, globalns=globalns, localns=localns)
                )

            if type_annotation is Any:
                return self._make_is_any(name)

            # Generic and parametrized type hints
            origin_type = xtyping.get_origin(type_annotation)
            type_args = xtyping.get_args(type_annotation)

            if origin_type is Literal:
                if len(type_args) == 1:
                    return self.make_is_literal(name, type_args[0])
                else:
                    return self.combine_validators_as_or(
                        name,
                        *(self.make_is_literal(name, a) for a in type_args),
                        error_type=TypeError,
                    )

            if origin_type is Union:
                has_none = False
                validators = []
                for t in type_args:
                    if t in (type(None), None):
                        has_none = True
                    else:
                        if (v := make_recursive(t)) is None:
                            raise exceptions.EveValueError(f"{t} type annotation is not supported.")
                        validators.append(v)

                validator = (
                    self.combine_validators_as_or(name, *validators)
                    if len(validators) > 1
                    else validators[0]
                )
                return self.combine_optional(name, validator) if has_none else validator

            if isinstance(origin_type, type):
                # Deal with generic collections
                if issubclass(origin_type, tuple):
                    if len(type_args) == 2 and (type_args[1] is Ellipsis):
                        # Tuple as an immutable sequence type (e.g. Tuple[int, ...])
                        if (member_validator := make_recursive(type_args[0])) is None:
                            raise exceptions.EveValueError(
                                f"{type_args[0]} type annotation is not supported."
                            )

                        return self.make_is_iterable_of(
                            name,
                            member_validator,
                            iterable_validator=self.make_is_instance_of(name, origin_type),
                        )

                    else:
                        # Tuple as a heterogeneous container (e.g. Tuple[int, float])
                        item_validators = []
                        for t in type_args:
                            if (v := make_recursive(t)) is None:
                                raise exceptions.EveValueError(
                                    f"{t} type annotation is not supported."
                                )
                            item_validators.append(v)

                        return self.make_is_tuple_of(name, tuple(item_validators), origin_type)

                if issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
                    assert len(type_args) == 1
                    if (member_validator := make_recursive(type_args[0])) is None:
                        raise exceptions.EveValueError(
                            f"{type_args[0]} type annotation is not supported."
                        )

                    return self.make_is_iterable_of(
                        name,
                        member_validator,
                        iterable_validator=self.make_is_instance_of(name, origin_type),
                    )

                if issubclass(origin_type, collections.abc.Mapping):
                    assert len(type_args) == 2
                    if (key_validator := make_recursive(type_args[0])) is None:
                        raise exceptions.EveValueError(
                            f"{type_args[0]} type annotation is not supported."
                        )
                    if (value_validator := make_recursive(type_args[1])) is None:
                        raise exceptions.EveValueError(
                            f"{type_args[1]} type annotation is not supported."
                        )

                    return self.make_is_mapping_of(
                        name,
                        key_validator,
                        value_validator,
                        mapping_validator=self.make_is_instance_of(name, origin_type),
                    )

                # Custom generic type: any regular (not datamodel) user-defined generic types like:
                #   class Foo(Generic[T]):
                #          ...
                #
                # Since this can be an arbitrary type (not something regular like a collection) there is
                # no way to check if the type parameter is verifed in the actual instance.
                # The only check can be done at run-time is to verify that the value is an instance of
                # the original type, completely ignoring the annotation. Ideally, the static type checker
                # can do a better job to try figure out if the type parameter is ok ...

                return make_recursive(origin_type)

            # TODO(egparedes): add support for signature checking in Callables
            raise exceptions.EveValueError(f"{type_annotation} type annotation is not supported.")

        except exceptions.EveValueError as error:
            if required:
                raise error

        assert bool(required) is False

        return None

    @staticmethod
    def _make_is_any(name: str) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for any type."""

        def _is_any(value: Any, **kwargs: Any) -> None:
            pass

        return _is_any

    @staticmethod
    def make_is_instance_of(name: str, type_: type) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for a specific type."""

        def _is_instance_of(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, type_):
                raise TypeError(
                    f"'{name}' must be {type_} (got '{value}' which is a {type(value)})."
                )

        return _is_instance_of

    @staticmethod
    def make_is_instance_of_int(name: str) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for ``int`` values which fails with ``bool`` values."""

        def _is_instance_of_int(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"'{name}' must be {int} (got '{value}' which is a {type(value)}).")

        return _is_instance_of_int

    @staticmethod
    def make_is_literal(name: str, literal_value: Any) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for a literal value."""
        if isinstance(literal_value, bool):

            def _is_literal(value: Any, **kwargs: Any) -> None:
                if value is not literal_value:
                    raise TypeError(
                        f"Provided value '{value}' for '{name}' does not match literal {literal_value}."
                    )

        else:

            def _is_literal(value: Any, **kwargs: Any) -> None:
                if value != literal_value:
                    raise TypeError(
                        f"Provided value '{value}' for '{name}' does not match literal {literal_value}."
                    )

        return _is_literal

    @staticmethod
    def make_is_tuple_of(
        name: str, item_validators: Sequence[FixedTypeValidator], tuple_type: type[tuple]
    ) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for tuple types."""

        def _is_tuple_of(value: Any, **kwargs: Any) -> None:
            if not isinstance(value, tuple_type):
                raise TypeError(
                    f"In '{name}' validation, got '{value}' which is a {type(value)} instead of {tuple_type}."
                )
            if len(value) != len(item_validators):
                raise TypeError(
                    f"In '{name}' validation, got '{value}' tuple which contains {len(value)} elements instead of {len(item_validators)}."
                )

            _i = None
            item_value = ""
            try:
                for _i, (item_value, item_validator) in enumerate(zip(value, item_validators)):
                    item_validator(item_value)
            except Exception as e:
                raise TypeError(
                    f"In '{name}' validation, tuple '{value}' contains invalid value '{item_value}' at position {_i}."
                ) from e

        return _is_tuple_of

    @staticmethod
    def make_is_iterable_of(
        name: str,
        member_validator: FixedTypeValidator,
        iterable_validator: Optional[FixedTypeValidator] = None,
    ) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for deep checks of typed iterables."""

        def _is_iterable_of(value: Any, **kwargs: Any) -> None:
            if iterable_validator is not None:
                iterable_validator(value, **kwargs)

            for member in value:
                member_validator(member, **kwargs)

        return _is_iterable_of

    @staticmethod
    def make_is_mapping_of(
        name: str,
        key_validator: FixedTypeValidator,
        value_validator: FixedTypeValidator,
        mapping_validator: Optional[FixedTypeValidator] = None,
    ) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for deep checks of typed mappings."""

        def _is_mapping_of(value: Any, **kwargs: Any) -> None:
            if mapping_validator is not None:
                mapping_validator(value, **kwargs)

            for k in value:
                key_validator(k, **kwargs)
                value_validator(value[k], **kwargs)

        return _is_mapping_of

    @staticmethod
    def combine_optional(name: str, actual_validator: FixedTypeValidator) -> FixedTypeValidator:
        """Create an ``FixedTypeValidator`` validator for an optional constraint."""

        def _is_optional(value: Any, **kwargs: Any) -> None:
            if value is not None:
                actual_validator(value, **kwargs)

        return _is_optional

    @staticmethod
    def combine_validators_as_or(
        name: str, *validators: FixedTypeValidator, error_type: Type[Exception] = TypeError
    ) -> FixedTypeValidator:
        def _combined_validator(value: Any, **kwargs: Any) -> None:
            for v in validators:
                try:
                    v(value, **kwargs)
                    break
                except Exception:
                    pass
            else:
                raise error_type(
                    f"In '{name}' validation, provided value '{value}' fails for all the possible validators."
                )

        return _combined_validator


simple_type_validator_factory: Final = cast(
    TypeValidatorFactory,
    utils.optional_lru_cache(SimpleTypeValidatorFactory(), typed=True),
)
"""Public (with optional cache) entry point for :class:`SimpleTypeValidatorFactory`."""


def simple_type_validator(
    value: Any,
    type_annotation: TypeAnnotation,
    name: Optional[str] = None,
    *,
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
    required: bool = True,
    **kwargs: Any,
) -> None:
    """Check that value satisfies a type definition (a simple :class:`TypeValidator` implementation).

    Check :class:`TypeValidator` and :class:`SimpleTypeValidatorFactory` for details.

    Keyword Arguments:
        strict_int (bool): do not accept ``bool`` values as ``int`` (default: ``True``).
    """
    type_validator: Optional[FixedTypeValidator] = simple_type_validator_factory(
        type_annotation, name=name, globalns=globalns, localns=localns, required=required, **kwargs
    )
    if type_validator is not None:
        type_validator(value, **kwargs)


# TODO(egparedes): add other implementations for advanced 3rd-party validators
#   e.g. 'typeguard' and specially 'beartype'

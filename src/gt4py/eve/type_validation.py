# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic interface and implementations of run-time type validation for arbitrary values."""

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import functools
import types
import typing

from . import exceptions, extended_typing as xtyping, utils
from .extended_typing import (
    Any,
    Final,
    ForwardRef,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeAnnotation,
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
        globalns: Optional[dict[str, Any]] = None,
        localns: Optional[dict[str, Any]] = None,
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
    def __call__(self, value: Any, **kwargs: Any) -> None:
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
        globalns: Optional[dict[str, Any]] = None,
        localns: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FixedTypeValidator: ...

    @overload
    def __call__(
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[dict[str, Any]] = None,
        localns: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]: ...

    @abc.abstractmethod
    def __call__(
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[dict[str, Any]] = None,
        localns: Optional[dict[str, Any]] = None,
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
@dataclasses.dataclass
class _DeferredTypeValidator:
    """A :class:`FixedTypeValidator` forwarding to a validator which is not known yet.

    This indirection is what makes recursive type aliases work: the validator of
    ``type NestedTuple[T] = tuple[T | NestedTuple[T], ...]`` cannot be built before the
    alias occurring inside its own definition has one, so the inner occurrence gets this
    placeholder and the real validator is filled in once the definition is processed.

    Unlike ``datamodels.ForwardRefValidator``, which resolves itself lazily on its first
    call, this one is always completed by its creator before any value reaches it.
    """

    name: str
    validator: Optional[FixedTypeValidator] = None

    def __call__(self, value: Any, **kwargs: Any) -> None:
        assert self.validator is not None, (
            f"Validator for '{self.name}' has not been completely defined yet."
        )
        self.validator(value, **kwargs)


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
        globalns: Optional[dict[str, Any]] = None,
        localns: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FixedTypeValidator: ...

    @overload
    def __call__(
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[dict[str, Any]] = None,
        localns: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]: ...

    def __call__(
        self,
        type_annotation: TypeAnnotation,
        name: Optional[str] = None,
        *,
        required: bool = True,
        globalns: Optional[dict[str, Any]] = None,
        localns: Optional[dict[str, Any]] = None,
        _alias_memo: Optional[dict[Any, _DeferredTypeValidator]] = None,
        **kwargs: Any,
    ) -> Optional[FixedTypeValidator]:
        # TODO(egparedes): if a "typing tree" structure is implemented, refactor this code as a tree traversal.
        #
        if name is None:
            name = "<value>"
        if _alias_memo is None:
            # Type aliases whose validator is currently being built, used to break the
            # cycles introduced by recursive aliases.
            _alias_memo = {}

        make_recursive = functools.partial(
            self.__call__,
            name=name,
            globalns=globalns,
            localns=localns,
            _alias_memo=_alias_memo,
            **kwargs,
        )

        try:
            if type_annotation is None:
                type_annotation = type(None)

            if isinstance(
                type_annotation, types.UnionType
            ):  # see https://github.com/python/cpython/issues/105499
                type_annotation = typing.Union[type_annotation.__args__]

            # A 'NameError' is deliberately left to propagate here, so that 'datamodels'
            # can defer; see 'xtyping.eval_type_alias' for both failure modes.
            try:
                resolved_annotation = xtyping.eval_type_alias(type_annotation)
            except TypeError as error:
                # Keep its message: it names the alias and the cause.
                raise exceptions.EveValueError(str(error)) from error
            if resolved_annotation is not type_annotation:
                # A recursive alias reaches this point again, with the very same annotation,
                # while its own validator is still being built. Handing the inner occurrence
                # a placeholder breaks that cycle; it is filled in right below, before any
                # value can be validated.
                if (deferred := _alias_memo.get(type_annotation, None)) is not None:
                    return deferred
                _alias_memo[type_annotation] = deferred = _DeferredTypeValidator(name)
                deferred.validator = validator = make_recursive(resolved_annotation)
                return validator

            # Non-generic types
            if xtyping.is_actual_type(type_annotation):
                assert not xtyping.get_args(type_annotation)
                if type_annotation is int and kwargs.get("strict_int", True):
                    return self.make_is_instance_of_int(name)
                else:
                    return self.make_is_instance_of(name, type_annotation)

            if isinstance(type_annotation, typing.TypeVar):
                if type_annotation.__bound__:
                    return self.make_is_instance_of(name, type_annotation.__bound__)
                else:
                    return self._make_is_any(name)

            if isinstance(type_annotation, ForwardRef):
                return make_recursive(
                    xtyping.eval_forward_ref(type_annotation, globalns=globalns, localns=localns)
                )

            if xtyping.is_Any(type_annotation):
                return self._make_is_any(name)

            # Generic and parametrized type hints
            origin_type = xtyping.get_origin(type_annotation)
            type_args = xtyping.get_args(type_annotation)

            if origin_type in {typing.Literal, xtyping.Literal}:
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

            if origin_type is type:
                # `type[X]`. Without this case the annotation falls through to the
                # generic-collection branch below and degrades to `isinstance(value, type)`,
                # i.e. "is any class at all".
                #
                # Every shape that is not recognized here falls back to that same loose
                # check rather than raising: `type[X]` annotations validated (loosely)
                # before this branch existed, so refusing one now would turn a working
                # downstream datamodel into an error at class-creation time.
                if len(type_args) != 1:
                    # bare `type` / `typing.Type`
                    return self.make_is_instance_of(name, type)

                # A PEP 695 alias nested inside `type[...]` is not resolved by the
                # whole-annotation pass at the top of this function, so resolve it here.
                try:
                    arg = xtyping.eval_type_alias(type_args[0])
                except TypeError:
                    return self.make_is_instance_of(name, type)

                if isinstance(arg, types.UnionType):  # `type[A | B]`
                    arg = typing.Union[arg.__args__]

                if isinstance(arg, typing.TypeVar):
                    # Mirror the plain-`TypeVar` branch above, which honours the bound.
                    if xtyping.is_actual_type(arg.__bound__):
                        return self.make_is_subclass_of(name, arg.__bound__)
                    return self.make_is_instance_of(name, type)

                if xtyping.is_actual_type(arg):
                    return self.make_is_subclass_of(name, arg)

                if xtyping.get_origin(arg) is Union:  # `type[A | B]`, `type[Union[A, B]]`
                    members = xtyping.get_args(arg)
                    if members and all(xtyping.is_actual_type(m) for m in members):
                        return self.combine_validators_as_or(
                            name, *(self.make_is_subclass_of(name, m) for m in members)
                        )

                return self.make_is_instance_of(name, type)

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
                    make_recursive(type_args[0])
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
                # no way to check if the type parameter is verified in the actual instance.
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
    def make_is_subclass_of(name: str, type_: type) -> FixedTypeValidator:
        """Create a ``FixedTypeValidator`` validator for ``type[type_]`` annotations."""

        def _is_subclass_of(value: Any, **kwargs: Any) -> None:
            if not (isinstance(value, type) and issubclass(value, type_)):
                raise TypeError(
                    f"'{name}' must be a subclass of {type_} (got '{value}' which is a {type(value)})."
                )

        return _is_subclass_of

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
        name: str, *validators: FixedTypeValidator, error_type: type[Exception] = TypeError
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
    TypeValidatorFactory, utils.optional_lru_cache(SimpleTypeValidatorFactory(), typed=True)
)
"""Public (with optional cache) entry point for :class:`SimpleTypeValidatorFactory`."""


def simple_type_validator(
    value: Any,
    type_annotation: TypeAnnotation,
    name: Optional[str] = None,
    *,
    globalns: Optional[dict[str, Any]] = None,
    localns: Optional[dict[str, Any]] = None,
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

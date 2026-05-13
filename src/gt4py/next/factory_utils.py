# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Any, Callable, overload

import factory


class Factory(factory.Factory):
    """`
    Factory that defaults all ``factory.Trait`` params to ``False``.

    Ensures that every ``factory.Trait`` declared in ``Params`` is present in
    the keyword arguments passed to ``create``, defaulting to ``False`` when
    not explicitly provided.  This allows trait-dependent declarations to
    always access the trait flag, even when the caller does not mention it.
    """

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        # adjust keyword arguments so that traits options are available even when not given
        # explicitly
        for name, param in cls._meta.parameters.items():
            if isinstance(param, factory.Trait):
                kwargs.setdefault(name, False)
        return super().create(**kwargs)


class DynamicTransformer(factory.declarations.BaseDeclaration):
    CAPTURE_OVERRIDES = True
    UNROLL_CONTEXT_BEFORE_EVALUATION = False

    def __init__(self, default: Any, *, transform: Callable[[Any, Any], Any]) -> None:
        super().__init__()
        self.default = default
        self.transform = transform

    def evaluate_pre(self, instance: Any, step: Any, overrides: dict[str, Any]) -> Any:
        # The call-time value, if present, is set under the "" key.
        value_or_declaration = overrides.pop("", self.default)

        if isinstance(value_or_declaration, factory.Transformer.Force):
            bypass_transform = True
            value_or_declaration = value_or_declaration.forced_value
        else:
            bypass_transform = False

        value = self._unwrap_evaluate_pre(
            value_or_declaration,
            instance=instance,
            step=step,
            overrides=overrides,
        )
        if bypass_transform:
            return value

        transform = self._unwrap_evaluate_pre(
            self.transform,
            instance=instance,
            step=step,
            overrides=overrides,
        )

        return transform(instance, value)


@overload
def dynamic_transformer(func: Callable[[Any, Any], Any], *, default: Any) -> DynamicTransformer: ...


@overload
def dynamic_transformer(
    *, default: Any
) -> Callable[[Callable[[Any, Any], Any]], DynamicTransformer]: ...


def dynamic_transformer(
    func: Callable[[Any, Any], Any] | None = None, *, default: Any
) -> DynamicTransformer | Callable[[Callable[[Any, Any], Any]], DynamicTransformer]:
    """
    Decorator that creates a factory field whose value is always passed through a transform.

    Works like ``factory.Transformer`` but the transform function receives the
    full factory instance, so it can read other parameters/traits.  The
    *default* argument provides the base value (may be a factory declaration
    such as ``factory.SelfAttribute``).  Use ``factory.Transformer.Force`` to
    bypass the transform at call-time.

    Example:
        >>> import dataclasses, factory
        >>> @dataclasses.dataclass
        ... class Person:
        ...     name: str
        ...     nickname: str
        >>> class PersonFactory(factory.Factory):
        ...     class Meta:
        ...         model = Person
        ...
        ...     name = "Joe"
        ...
        ...     @dynamic_transformer(default=factory.SelfAttribute(".name"))
        ...     def nickname(self, nickname):
        ...         return f"{nickname}y"
        >>> PersonFactory().nickname
        'Joey'
        >>> PersonFactory(name="John").nickname
        'Johny'
        >>> PersonFactory(name=factory.Transformer.Force("John")).name
        'John'
    """
    if func is None:
        return functools.partial(dynamic_transformer, default=default)
    return DynamicTransformer(default=default, transform=func)

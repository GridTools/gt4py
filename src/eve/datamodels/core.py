# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-l directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data Model class creation and other utils.

Check :mod:`eve.datamodels` for additional information.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import typing
import warnings

import attr
import attrs


try:
    # For perfomance reasons, try to use cytoolz when possible (using cython)
    import cytoolz as toolz  # type: ignore[import]
except ModuleNotFoundError:
    # Fall back to pure Python toolz
    import toolz  # type: ignore[import] # noqa: F401

from .. import exceptions
from .. import extended_typing as xtyping
from .. import type_validation as type_val
from .. import utils
from ..extended_typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    ForwardRef,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeAnnotation,
    TypeVar,
    Union,
    cast,
    overload,
)
from ..type_definitions import NOTHING, NothingType


# Typing
# TODO(egparedes): these typing definitions are not perfect, but they provide
#   some help until more advanced features are added to typing, like
#   PEP 681 - Data Class Transforms (https://peps.python.org/pep-0681/)
#   or intersection types.
_T = TypeVar("_T")


# TODO(egparedes): since these protocols are used instead of the actual classes
#   for type checking, we assign empty tuples and None values to its members
#   to avoid errors from mypy complaining about instantiation of abstract classes


class _AttrsClassTP(Protocol):
    __attrs_attrs__: ClassVar[Tuple[attr.Attribute, ...]] = ()


Attribute: TypeAlias = attr.Attribute


class DataModelTP(_AttrsClassTP, xtyping.DevToolsPrettyPrintable, Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    __datamodel_fields__: ClassVar[utils.FrozenNamespace[Attribute]] = cast(
        utils.FrozenNamespace[Attribute], None
    )
    __datamodel_params__: ClassVar[utils.FrozenNamespace[Any]] = cast(
        utils.FrozenNamespace[Attribute], None
    )
    __datamodel_root_validators__: ClassVar[
        Tuple[xtyping.NonDataDescriptor["DataModelTP", "BoundRootValidator"], ...]
    ] = ()
    # Optional
    __auto_init__: ClassVar[Callable[..., None]] = cast(Callable[..., None], None)
    __pre_init__: ClassVar[Callable[[DataModelTP], None]] = cast(
        Callable[["DataModelTP"], None], None
    )
    __post_init__: ClassVar[Callable[[DataModelTP], None]] = cast(
        Callable[["DataModelTP"], None], None
    )


DataModelT = TypeVar("DataModelT", bound=DataModelTP)


class GenericDataModelTP(DataModelTP, Protocol):
    __args__: ClassVar[Tuple[Union[Type, TypeVar], ...]] = ()
    __parameters__: ClassVar[Tuple[TypeVar, ...]] = ()

    @classmethod
    def __class_getitem__(
        cls: Type[GenericDataModelTP], args: Union[Type, Tuple[Type, ...]]
    ) -> Union[DataModelTP, GenericDataModelTP]:
        ...


_DM = TypeVar("_DM", bound="DataModel")

GenericDataModelT = TypeVar("GenericDataModelT", bound=GenericDataModelTP)

AttrsValidator = Callable[[Any, Attribute, _T], Any]
FieldValidator = Callable[[_DM, Attribute, _T], None]
BoundFieldValidator = Callable[[Attribute, _T], None]

RootValidator = Callable[[Type[_DM], _DM], None]
BoundRootValidator = Callable[[_DM], None]

FieldTypeValidatorFactory = Callable[[TypeAnnotation, str], FieldValidator]

TypeConverter = Callable[[Any], _T]

# Implementation
_DATAMODEL_TAG: Final = "__DATAMODEL_TAG"
_FIELD_VALIDATOR_TAG: Final = "__DATAMODEL_FIELD_VALIDATOR_TAG"
_ROOT_VALIDATOR_TAG: Final = "__DATAMODEL_ROOT_VALIDATOR_TAG"
_COERCED_TYPE_TAG: Final = "__DATAMODEL_COERCED_TYPE_TAG"
_UNCHECKED_TYPE_TAG: Final = "__DATAMODEL_UNCHECKED_TYPE_TAG"

_DM_OPTS = "__dm_opts"
_GENERIC_DATAMODEL_ROOT_DM_OPT: Final = "_GENERIC_DATAMODEL_ROOT_DM_OPT"

MODEL_FIELD_DEFINITIONS_ATTR: Final = "__datamodel_fields__"
MODEL_PARAM_DEFINITIONS_ATTR: Final = "__datamodel_params__"
MODEL_ROOT_VALIDATORS_ATTR: Final = "__datamodel_root_validators__"


Coerced = xtyping.Annotated[_T, _COERCED_TYPE_TAG]
"""Type hint marker to define fields that should be coerced at initialization."""


Unchecked = xtyping.Annotated[_T, _UNCHECKED_TYPE_TAG]
"""Type hint marker to define fields that should NOT be type-checked at initialization."""


if sys.version_info >= (3, 10):
    _dataclass_opts: Final[dict[str, Any]] = {"slots": True}
else:
    _dataclass_opts: Final[Dict[str, Any]] = {}


@dataclasses.dataclass(**_dataclass_opts)
class ForwardRefValidator:
    """Implementation of ``attrs`` field validator for ``ForwardRef`` typings.

    The first time is called it will update the class' field type annotations
    and then create the actual type validator for the field.
    """

    factory: type_val.TypeValidatorFactory
    """Type factory used to create the actual field validator."""

    validator: Union[type_val.FixedTypeValidator, None, NothingType] = NOTHING
    """Actual type validator created after resolving the forward references."""

    def __call__(self, instance: DataModel, attribute: Attribute, value: Any) -> None:
        if self.validator is NOTHING:
            model_cls = instance.__class__
            update_forward_refs(model_cls)
            self.validator = self.factory(
                getattr(getattr(model_cls, MODEL_FIELD_DEFINITIONS_ATTR), attribute.name).type,
                attribute.name,
            )

        if self.validator:
            self.validator(value)


@dataclasses.dataclass(frozen=True, **_dataclass_opts)
class ValidatorAdapter:
    """Adapter to use :class:`eve.type_validation.FixedTypeValidator`s as field validators."""

    validator: type_val.FixedTypeValidator
    description: str

    def __call__(self, _instance: DataModel, _attribute: Attribute, value: Any) -> None:
        self.validator(value)

    def __repr__(self) -> str:
        return self.description


def field_type_validator_factory(
    factory: type_val.TypeValidatorFactory, *, use_cache: bool = False
) -> FieldTypeValidatorFactory:
    """Create a factory of field type validators from a factory of regular type validators."""
    if use_cache:
        factory = cast(
            type_val.TypeValidatorFactory,
            utils.optional_lru_cache(func=factory),
        )

    def _field_type_validator_factory(
        type_annotation: TypeAnnotation,
        name: str,
    ) -> FieldValidator:
        """Field type validator for datamodels, supporting forward references."""
        if isinstance(type_annotation, ForwardRef):
            return ForwardRefValidator(factory)
        else:
            simple_validator = factory(type_annotation, name, required=True)
            return ValidatorAdapter(
                simple_validator, f"{getattr(simple_validator,'__name__', 'TypeValidator')}"
            )

    return _field_type_validator_factory


simple_type_validator_factory = field_type_validator_factory(type_val.simple_type_validator_factory)

DefaultFieldTypeValidatorFactory: Final[Optional[FieldTypeValidatorFactory]] = (
    simple_type_validator_factory if __debug__ else None
)
"""Default type validator factory used by datamodels classes. `None` by default if running in optimized mode."""

_REPR_DEFAULT: Final = True
_EQ_DEFAULT: Final = True
_ORDER_DEFAULT: Final = False
_UNSAFE_HASH_DEFAULT: Final = False
_FROZEN_DEFAULT: Final = False
_MATCH_ARGS_DEFAULT: Final = True
_KW_ONLY_DEFAULT: Final = False
_SLOTS_DEFAULT: Final = False
_COERCE_DEFAULT: Final = False
_GENERIC_DEFAULT: Final = False

DEFAULT_OPTIONS: Final[utils.FrozenNamespace] = utils.FrozenNamespace(
    repr=_REPR_DEFAULT,
    eq=_EQ_DEFAULT,
    order=_ORDER_DEFAULT,
    unsafe_hash=_UNSAFE_HASH_DEFAULT,
    frozen=_FROZEN_DEFAULT,
    match_args=_MATCH_ARGS_DEFAULT,
    kw_only=_KW_ONLY_DEFAULT,
    slots=_SLOTS_DEFAULT,
    coerce=_COERCE_DEFAULT,
    generic=_GENERIC_DEFAULT,
)
"""Convenient public namespace to expose default values to users."""


@overload
def datamodel(
    cls: Literal[None] = None,
    /,
    *,
    repr: bool = _REPR_DEFAULT,  # noqa: A002  # shadowing 'repr' python builtin
    eq: bool = _EQ_DEFAULT,
    order: bool = _ORDER_DEFAULT,
    unsafe_hash: bool = _UNSAFE_HASH_DEFAULT,
    frozen: bool | Literal["strict"] = _FROZEN_DEFAULT,
    match_args: bool = _MATCH_ARGS_DEFAULT,
    kw_only: bool = _KW_ONLY_DEFAULT,
    slots: bool = _SLOTS_DEFAULT,
    coerce: bool = _COERCE_DEFAULT,
    generic: bool = _GENERIC_DEFAULT,
    type_validation_factory: Optional[FieldTypeValidatorFactory] = DefaultFieldTypeValidatorFactory,
) -> Callable[[Type[_T]], Type[_T]]:
    ...


@overload
def datamodel(  # noqa: F811  # redefinion of unused symbol
    cls: Type[_T],
    /,
    *,
    repr: bool = _REPR_DEFAULT,  # noqa: A002  # shadowing 'repr' python builtin
    eq: bool = _EQ_DEFAULT,
    order: bool = _ORDER_DEFAULT,
    unsafe_hash: bool = _UNSAFE_HASH_DEFAULT,
    frozen: bool | Literal["strict"] = _FROZEN_DEFAULT,
    match_args: bool = _MATCH_ARGS_DEFAULT,
    kw_only: bool = _KW_ONLY_DEFAULT,
    slots: bool = _SLOTS_DEFAULT,
    coerce: bool = _COERCE_DEFAULT,
    generic: bool = _GENERIC_DEFAULT,
    type_validation_factory: Optional[FieldTypeValidatorFactory] = DefaultFieldTypeValidatorFactory,
) -> Type[_T]:
    ...


# TODO(egparedes): Use @dataclass_transform(eq_default=True, field_specifiers=("field",))
def datamodel(  # noqa: F811  # redefinion of unused symbol
    cls: Optional[Type[_T]] = None,
    /,
    *,
    repr: bool = _REPR_DEFAULT,  # noqa: A002  # shadowing 'repr' python builtin
    eq: bool = _EQ_DEFAULT,
    order: bool = _ORDER_DEFAULT,
    unsafe_hash: bool = _UNSAFE_HASH_DEFAULT,
    frozen: bool | Literal["strict"] = _FROZEN_DEFAULT,
    match_args: bool = _MATCH_ARGS_DEFAULT,
    kw_only: bool = _KW_ONLY_DEFAULT,
    slots: bool = _SLOTS_DEFAULT,
    coerce: bool = _COERCE_DEFAULT,
    generic: bool = _GENERIC_DEFAULT,
    type_validation_factory: Optional[FieldTypeValidatorFactory] = DefaultFieldTypeValidatorFactory,
) -> Union[Type[_T], Callable[[Type[_T]], Type[_T]]]:
    """Add generated special methods to classes according to the specified attributes (class decorator).

    It converts the class to an `attrs <https://www.attrs.org/>`_ with some extra features.
    Adding strict type validation functions for the fields is done by means of
    the ``type_validation_factory`` argument, falling back to the default factory
        (:class:`DefaultFieldTypeValidatorFactory`). The generated field type
    validators are generated by using PEP 526 ``__annotations__`` to determine field
    types and creating validation functions for them.

    Arguments:
        cls: Original class definition.

    Keyword Arguments:
        repr: If ``True`` (default), a ``__repr__()`` method will be generated if it does
            not overwrite a custom implementation defined in this class (not inherited).
        eq: If ``True`` (default), ``__eq__()`` and ``__ne__()`` methods will be generated
            if they do not overwrite custom implementations defined in this class (not inherited).
            The generated method compares the class as if it were a tuple of its fields, but both
            instances in the comparison must be of identical type.
        order:  If ``True`` (default is ``False``), add ``__lt__()``, ``__le__()``, ``__gt__()``,
            and ``__ge__()`` methods that behave like `eq` above and allow instances
            to be ordered. If ``None`` mirror value of `eq`.
        unsafe_hash: If ``False``, a ``__hash__()`` method is generated in a safe way
            according to how ``eq`` and ``frozen`` are set, or set to ``None`` (disabled)
            otherwise. If ``True``, a ``__hash__()`` method is generated anyway
            (use with care). See :func:`dataclasses.dataclass` for the complete explanation
            (or other sources like: `<https://hynek.me/articles/hashes-and-equality/>`_).
        frozen: If ``True`` (default is ``False``), assigning to fields will generate an exception.
            This emulates read-only frozen instances. The ``__setattr__()`` and
            ``__delattr__()`` methods should not be defined in the class.
        match_args: If ``True`` (default) and ``__match_args__`` is not already defined in the class,
            set ``__match_args__`` on the class to support PEP 634 (Structural Pattern Matching).
            It is a tuple of all positional-only ``__init__`` parameter names on
            Python 3.10 and later. Ignored on older Python versions.
        kw_only: If ``True`` (default is ``False``), make all fields keyword-only in the generated
            ``__init__`` (if ``init`` is ``False``, this parameter is ignored).
        slots: slots: If ``True`` (the default is ``False``), ``__slots__`` attribute will be generated
            and a new slotted class will be returned instead of the original one.
        coerce: If ``True`` (default is ``False``), an automatic type converter will be generated
            for all fields.
        generic: If ``True`` (default is ``False``) the class should be a ``Generic[]`` class,
            and the generated Data Model will support creation of new Data Model subclasses
            when using concrete types as arguments of ``DataModelClass[some_concret_type]``.
        type_validation_factory: Type validation factory used to build the field type validators.
            If ``None``, type validators will not be generators.

    """
    datamodel_options: Final = {
        "repr": repr,
        "eq": eq,
        "order": order,
        "unsafe_hash": unsafe_hash,
        "frozen": frozen,
        "match_args": match_args,
        "kw_only": kw_only,
        "slots": slots,
        "coerce": coerce,
        "generic": generic,
        "type_validation_factory": type_validation_factory,
    }

    if cls is None:  # called as: @datamodel()
        return functools.partial(_make_datamodel, **datamodel_options)
    else:  # called as: @datamodel
        return _make_datamodel(
            cls,
            _stacklevel_offset=1,
            **datamodel_options,  # type: ignore[arg-type]
        )


class _DataModelDecoratorTP(Protocol[_T]):
    def __call__(
        self,
        cls: Optional[Type[_T]] = None,
        /,
        *,
        repr: bool = _REPR_DEFAULT,  # noqa: A002  # shadowing 'repr' python builtin
        eq: bool = _EQ_DEFAULT,
        order: bool = _ORDER_DEFAULT,
        unsafe_hash: bool = _UNSAFE_HASH_DEFAULT,
        match_args: bool = _MATCH_ARGS_DEFAULT,
        kw_only: bool = _KW_ONLY_DEFAULT,
        slots: bool = _SLOTS_DEFAULT,
        coerce: bool = _COERCE_DEFAULT,
        generic: bool = _GENERIC_DEFAULT,
        type_validation_factory: Optional[
            FieldTypeValidatorFactory
        ] = DefaultFieldTypeValidatorFactory,
    ) -> Union[Type[_T], Callable[[Type[_T]], Type[_T]]]:
        ...


frozenmodel: _DataModelDecoratorTP = functools.partial(datamodel, frozen=True)
"""Data Model definition function using ``frozen=True``."""

frozen_model = frozenmodel


# Typing protocols are used instead of the actual classes for type checks
if xtyping.TYPE_CHECKING:

    class DataModel(DataModelTP):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            ...

        def __pretty__(
            self, fmt: Callable[[Any], Any], **kwargs: Any
        ) -> Generator[Any, None, None]:
            ...

else:

    # TODO(egparedes): use @dataclass_transform(eq_default=True, field_specifiers=("field",))
    class DataModel:
        """Base class to automatically convert any subclass into a Data Model.

        Inheriting from this class is equivalent to apply the :func:`datamodel`
        decorator to a class, except that the ``slots`` option is always ``False``
        (since it generates a new class) and all descendants will be also converted
        automatically in Data Models with the same options of the parent class
        (which does not happen when explicitly applying the decorator).

        See :func:`datamodel` for the description of the parameters.
        """

        __slots__ = ()

        @classmethod
        def __init_subclass__(
            cls,
            /,
            *,
            repr: bool  # noqa: A002  # shadowing 'repr' python builtin
            | None
            | Literal["inherited"] = "inherited",
            eq: bool | None | Literal["inherited"] = "inherited",
            order: bool | None | Literal["inherited"] = "inherited",
            unsafe_hash: bool | None | Literal["inherited"] = "inherited",
            frozen: bool | Literal["strict", "inherited"] = "inherited",
            match_args: bool | Literal["inherited"] = "inherited",
            kw_only: bool | Literal["inherited"] = "inherited",
            coerce: bool | Literal["inherited"] = "inherited",
            type_validation_factory: Optional[FieldTypeValidatorFactory]
            | Literal["inherited"] = "inherited",
            **kwargs: Any,
        ) -> None:
            dm_opts = kwargs.pop(_DM_OPTS, [])
            super(DataModel, cls).__init_subclass__(**kwargs)
            cls_params = getattr(cls, MODEL_PARAM_DEFINITIONS_ATTR, None)

            generic: Final = (
                "True_no_checks"
                if _GENERIC_DATAMODEL_ROOT_DM_OPT in dm_opts
                else getattr(cls_params, "generic", False)
            )

            locals_ = locals()
            datamodel_kwargs = {}
            for arg_name, default_value in [
                ("repr", _REPR_DEFAULT),
                ("eq", _EQ_DEFAULT),
                ("order", _ORDER_DEFAULT),
                ("unsafe_hash", _UNSAFE_HASH_DEFAULT),
                ("frozen", _FROZEN_DEFAULT),
                ("match_args", _MATCH_ARGS_DEFAULT),
                ("kw_only", _KW_ONLY_DEFAULT),
                ("coerce", _COERCE_DEFAULT),
                ("type_validation_factory", DefaultFieldTypeValidatorFactory),
            ]:
                arg_value = locals_[arg_name]
                if arg_value == "inherited":
                    datamodel_kwargs[arg_name] = getattr(cls_params, arg_name, default_value)
                else:
                    datamodel_kwargs[arg_name] = arg_value

            if cls_params is not None and cls_params.frozen and not datamodel_kwargs["frozen"]:
                raise TypeError("Subclasses of a frozen DataModel cannot be unfrozen.")

            _make_datamodel(
                cls,
                slots=False,
                generic=generic,
                **datamodel_kwargs,
                _stacklevel_offset=1,
            )


def field(
    *,
    default: Any = NOTHING,
    default_factory: Optional[Callable[[], Any]] = None,
    init: bool = True,
    repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
    hash: Optional[bool] = None,  # noqa: A002   # shadowing 'hash' python builtin
    compare: bool = True,
    metadata: Optional[Mapping[Any, Any]] = None,
    kw_only: bool = _KW_ONLY_DEFAULT,
    converter: Callable[[Any], Any] | Literal["coerce"] | None = None,
    validator: AttrsValidator
    | FieldValidator
    | Sequence[AttrsValidator | FieldValidator]
    | None = None,
) -> Any:  # attr.s lies in some typings
    """Define a new attribute on a class with advanced options.

    Keyword Arguments:
        default: If provided, this will be the default value for this field.
            This is needed because the ``field()`` call itself replaces the
            normal position of the default value.
        default_factory: If provided, it must be a zero-argument callable that will
            be called when a default value is needed for this field. Among other
            purposes, this can be used to specify fields with mutable default values.
            It is an error to specify both `default` and `default_factory`.
        init: If ``True`` (default), this field is included as a parameter to the
            generated ``__init__()`` method.
        repr: If ``True`` (default), this field is included in the string returned
            by the generated ``__repr__()`` method.
        hash: This can be a ``bool`` or ``None`` (default). If ``True``, this field is
            included in the generated ``__hash__()`` method. If ``None``, use the value
            of `compare`, which would normally be the expected behavior: a field
            should be considered in the `hash` if it is used for comparisons.
            Setting this value to anything other than ``None`` is `discouraged`.
        compare: If ``True`` (default), this field is included in the generated equality and
            comparison methods (__eq__(), __gt__(), et al.).
        metadata: An arbitrary mapping, not used at all by Data Models, and provided
            only as a third-party extension mechanism. Multiple third-parties can each
            have their own key, to use as a namespace in the metadata.
        kw_only: If ``True`` (default is ``False``), make this field keyword-only in the
            generated ``__init__`` (if ``init`` is ``False``, this parameter is ignored).
        converter: Callable that is automatically called to convert attribute’s value.
            It is given the passed-in value, and the returned value will be used as the
            new value of the attribute before being passed to the validator, if any.
            If ``"coerce"`` is passed, a naive coercer converter will be generated.
            The automatic converter basically calls the constructor of the type indicated
            in the type hint, so it is assumed that new instances of this type can be created
            like ``type_name(value)``. For collection types, there is not attempt to convert
            its items, only the collection type.
        validator: FieldValidator or list of FieldValidators to be used with this field.
            (Note that validators can also be set using decorator notation).


    Examples:
        >>> from typing import List
        >>> @datamodel
        ... class C:
        ...     mylist: List[int] = field(default_factory=lambda : [1, 2, 3])
        >>> c = C()
        >>> c.mylist
        [1, 2, 3]

    """
    if default is not NOTHING and default_factory is not None:
        raise ValueError("Cannot specify both 'default' and 'default_factory'.")

    if default is not NOTHING:
        defaults_kwargs = {"default": default}
    elif default_factory is not None:
        defaults_kwargs = {"factory": default_factory}
    else:
        defaults_kwargs = {}

    return attrs.field(
        **defaults_kwargs,
        init=init,
        repr=repr,
        hash=hash,
        eq=compare,
        order=compare,
        metadata=metadata,
        kw_only=kw_only,
        converter=converter,  # type: ignore[arg-type]
        validator=validator,  # type: ignore[arg-type]
    )


coerced_field = functools.partial(field, converter="coerce")
"""Field definition function using ``converter="coerce"``."""


def validator(name: str) -> Callable[[FieldValidator], FieldValidator]:
    """Define a custom field validator for a specific field (decorator function).

    Arguments:
        name: Name of the field to be validated by the decorated function.

    The decorated functions should have the following signature:
    ``def _validator_function(self, attribute, value):``
    where ``self`` will be the model instance being validated, ``attribute``
    the definition information of the attribute (the value of
    ``__datamodel_fields__.field_name``) and ``value`` the actual value
    received for this field.
    """
    assert isinstance(name, str)

    def _field_validator_maker(func: FieldValidator) -> FieldValidator:
        names = getattr(func, _FIELD_VALIDATOR_TAG, ())
        setattr(func, _FIELD_VALIDATOR_TAG, tuple([*names, name]))
        return func

    return _field_validator_maker


_RV = TypeVar("_RV", bound=RootValidator)


def root_validator(cls_method: _RV, /) -> _RV:
    """Define a custom root validator (decorator function).

    The decorated functions should have the following signature:
    ``def _root_validator_function(cls, instance):``
    where ``cls`` will be the class of the model and ``instance`` the
    actual instance being validated.
    """
    setattr(cls_method, _ROOT_VALIDATOR_TAG, None)
    return cls_method


# -- Utils --
def is_datamodel(obj: Any) -> bool:
    """Return True if `obj` is a Data Model class or an instance of a Data Model."""
    cls = obj if isinstance(obj, type) else obj.__class__
    return hasattr(cls, MODEL_FIELD_DEFINITIONS_ATTR)


def is_generic_datamodel_class(cls: Type) -> bool:
    """Return ``True`` if `obj` is a generic Data Model class with type parameters."""
    assert isinstance(cls, type)
    return is_datamodel(cls) and xtyping.has_type_parameters(cls)


def get_fields(model: Union[DataModel, Type[DataModel]]) -> utils.FrozenNamespace:
    """Return the field meta-information of a Data Model.

    Arguments:
        model: A Data Model class or instance.

    Examples:
        >>> from typing import List
        >>> @datamodel
        ... class Model:
        ...     name: str
        ...     amount: int = 1
        ...     numbers: List[float] = field(default_factory=list)
        >>> fields(Model)  # doctest:+ELLIPSIS
        FrozenNamespace(...name=Attribute(name='name', default=NOTHING, ...

    """  # noqa: RST201  # doctest conventions confuse RST validator
    if not is_datamodel(model):
        raise TypeError(f"Invalid datamodel instance or class: '{model}'.")
    if not isinstance(model, type):
        model = model.__class__

    ns = getattr(model, MODEL_FIELD_DEFINITIONS_ATTR)
    assert isinstance(ns, utils.FrozenNamespace)
    return ns


fields = get_fields


def asdict(
    instance: DataModel,
    *,
    value_serializer: Optional[Callable[[Type[DataModel], Attribute, Any], Any]] = None,
) -> Dict[str, Any]:
    """Return the contents of a Data Model instance as a new mapping from field names to values.

    Arguments:
        instance: Data Model instance.

    Keyword Arguments:
        value_serializer: A hook that is called for every attribute or dict key/value and must
            return the (updated) value.

    Examples:
        >>> @datamodel
        ... class C:
        ...     x: int
        ...     y: int
        >>> c = C(x=1, y=2)
        >>> assert asdict(c) == {'x': 1, 'y': 2}
    """  # noqa: RST301  # sphinx.napoleon conventions confuse RST validator
    if not is_datamodel(instance) or isinstance(instance, type):
        raise TypeError(f"Invalid datamodel instance: '{instance}'.")
    return attrs.asdict(instance, value_serializer=value_serializer)


def astuple(instance: DataModel) -> Tuple[Any, ...]:
    """Return the contents of a Data Model instance as a new tuple of field values.

    Arguments:
        instance: Data Model instance.

    Examples:
        >>> @datamodel
        ... class C:
        ...     x: int
        ...     y: int
        >>> c = C(x=1, y=2)
        >>> assert astuple(c) == (1, 2)

    """
    if not is_datamodel(instance) or isinstance(instance, type):
        raise TypeError(f"Invalid datamodel instance: '{instance}'.")
    return attrs.astuple(instance, recurse=True)


evolve = attrs.evolve
"""Create a new instance, based on inst with changes applied."""

validate = attrs.validate
"""Validate all attributes on inst that have a validator."""

_DataModelT = TypeVar("_DataModelT", bound=DataModel)


def update_forward_refs(
    model_cls: Type[_DataModelT],
    localns: Optional[Dict[str, Any]] = None,
) -> Type[_DataModelT]:
    """Update Data Model class meta-information replacing forwarded type annotations with actual types.

    Arguments:
        localns: locals ``dict`` used in the evaluation of the annotations
            (globals are automatically taken from ``model.__module__``).

    Returns:
        The provided class (so it can be used as a decorator too).

    """
    if not (isinstance(model_cls, type) and is_datamodel(model_cls)):
        raise TypeError(f"Invalid datamodel class: '{model_cls}'.")

    # attrs.resolve_types() caches the exact class (in the MRO) whose types have been already resolved
    if getattr(model_cls, "__attrs_types_resolved__", None) != model_cls:
        fields = list(model_cls.__datamodel_fields__.keys())
        current_datamodel_fields = getattr(model_cls, MODEL_FIELD_DEFINITIONS_ATTR)

        for field_name in fields:
            if not hasattr(current_datamodel_fields, field_name):
                raise ValueError(f"{model_cls} does not contain a field named '{field_name}'.")

            field_attr = getattr(current_datamodel_fields, field_name)

            if isinstance(field_attr.type, ForwardRef):
                try:
                    actual_type = xtyping.eval_forward_ref(
                        field_attr.type,
                        sys.modules[model_cls.__module__].__dict__,
                        localns,
                        include_extras=True,
                    )
                    object.__setattr__(field_attr, "type", actual_type)
                except Exception as error:
                    raise TypeError(
                        f"Unexpected error trying to solve '{field_name}' field annotation ('{getattr(field_attr, 'type', None)}')"
                    ) from error

            # update resolved cache as attrs.resolve_types() would do
            model_cls.__attrs_types_resolved__ = model_cls  # type: ignore[attr-defined]  # adding a new class attribute

    return model_cls


def concretize(
    datamodel_cls: Type[GenericDataModelT],
    /,
    *type_args: Type,
    class_name: Optional[str] = None,
    module: Optional[str] = None,
    support_pickling: bool = True,  # noqa
    overwrite_definition: bool = True,
) -> Type[DataModelT]:
    """Generate a new concrete subclass of a generic Data Model.

    Arguments:
        datamodel_cls: Generic Data Model to be subclassed.
        type_args: Type definitions replacing the `TypeVars` in
            ``datamodel_cls.__parameters__``.

    Keyword Arguments:
        class_name: Name of the new concrete class. The default value is the
            same of the generic Data Model replacing the `TypeVars` by the provided
            `type_args` in the name.
        module: Value of the ``__module__`` attribute of the new class.
            The default value is the name of the module containing the generic Data Model.
        support_pickling: If ``True``, support for pickling will be added
            by actually inserting the new class into the target `module`.
        overwrite_definition: If ``True``, a previous definition of the class in
            the target module will be overwritten.

    """  # noqa: RST301  # doctest conventions confuse RST validator
    concrete_cls: Type[DataModelT] = _make_concrete_with_cache(
        datamodel_cls, *type_args, class_name=class_name, module=module
    )
    assert isinstance(concrete_cls, type) and is_datamodel(concrete_cls)

    # For pickling to work, the new class has to be added to the proper module
    if support_pickling:
        class_name = concrete_cls.__name__
        reference_module_globals = sys.modules[concrete_cls.__module__].__dict__
        if not (cls_in_module := (class_name in reference_module_globals)) or overwrite_definition:
            reference_module_globals[class_name] = concrete_cls
        elif cls_in_module and reference_module_globals[class_name] is not concrete_cls:
            warnings.warn(
                RuntimeWarning(
                    f"Existing '{class_name}' symbol in module '{module}' contains a reference"
                    "to a different object."
                )
            )

    return concrete_cls


# -- Helpers --
def _collect_field_validators(cls: Type) -> Dict[str, FieldValidator]:
    result = {}
    for member in cls.__dict__.values():
        if hasattr(member, _FIELD_VALIDATOR_TAG):
            for field_name in getattr(member, _FIELD_VALIDATOR_TAG):
                result[field_name] = member
            delattr(member, _FIELD_VALIDATOR_TAG)

    return result


def _collect_root_validators(cls: Type) -> List[RootValidator]:
    result = []
    for base in reversed(cls.__mro__[1:]):
        for validator in getattr(base, MODEL_ROOT_VALIDATORS_ATTR, []):
            if validator not in result:
                result.append(validator)

    for member in cls.__dict__.values():
        if hasattr(member, _ROOT_VALIDATOR_TAG):
            result.append(member)
            delattr(member, _ROOT_VALIDATOR_TAG)

    return result


def _get_attribute_from_bases(
    name: str, mro: Tuple[Type, ...], annotations: Optional[Dict[str, Any]] = None
) -> Optional[Attribute]:
    for base in mro:
        for base_field_attrib in getattr(base, "__attrs_attrs__", []):
            if base_field_attrib.name == name:
                if annotations is not None:
                    annotations[name] = base.__annotations__[name]
                return cast(Attribute, base_field_attrib)

    return None


def _substitute_typevars(
    type_hint: Type, type_params_map: Mapping[TypeVar, Union[Type, TypeVar]]
) -> Tuple[Union[Type, TypeVar], bool]:
    if isinstance(type_hint, xtyping.TypeVar):
        assert type_hint in type_params_map
        return type_params_map[type_hint], True
    elif getattr(type_hint, "__parameters__", []):
        return type_hint[tuple(type_params_map[tp] for tp in type_hint.__parameters__)], True
        # TODO(egparedes): WIP fix for partial specialization
        # # Type hint is a generic model: replace all the concretized type vars
        # noqa: e800 replaced = False
        # noqa: e800 new_args = []
        # noqa: e800 for tp in type_hint.__parameters__:
        # noqa: e800     if tp in type_params_map:
        # noqa: e800         new_args.append(type_params_map[tp])
        # noqa: e800         replaced = True
        # noqa: e800     else:
        # noqa: e800         new_args.append(type_params_map[tp])
        # noqa: e800 return type_hint[tuple(new_args)], replaced
    else:
        return type_hint, False


def _make_counting_attr_from_attribute(
    field_attrib: Attribute, *, include_type: bool = False, **kwargs: Any
) -> Any:  # attr.s lies a bit in some typing definitons
    args = [
        "default",
        "validator",
        "repr",
        "eq",
        "order",
        "hash",
        "init",
        "metadata",
        "converter",
        "kw_only",
        "on_setattr",
    ]
    if include_type:
        args.append("type")

    result = attr.ib(**{key: getattr(field_attrib, key) for key in args}, **kwargs)
    for key in ("eq_key", "order_key"):
        object.__setattr__(result, key, getattr(field_attrib, key))

    return result


def _make_post_init(has_post_init: bool) -> Callable[[DataModel], None]:
    # Duplicated code to facilitate the source inspection of the generated `__init__()` method
    if has_post_init:

        def __attrs_post_init__(self: DataModel) -> None:
            if attr._config._run_validators is True:  # type: ignore[attr-defined]  # attr._config is not visible for mypy
                for validator in self.__datamodel_root_validators__:
                    validator.__get__(self)(self)

            self.__post_init__()

    else:

        def __attrs_post_init__(self: DataModel) -> None:
            if attr._config._run_validators is True:  # type: ignore[attr-defined]  # attr._config is not visible for mypy
                for validator in type(self).__datamodel_root_validators__:
                    validator.__get__(self)(self)

    setattr(__attrs_post_init__, _DATAMODEL_TAG, True)

    return __attrs_post_init__


def _make_devtools_pretty() -> Callable[
    [DataModel, Callable[[Any], Any]], Generator[Any, None, None]
]:
    def __pretty__(
        self: DataModel, fmt: Callable[[Any], Any], **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Provide a human readable representation for `devtools <https://python-devtools.helpmanual.io/>`_.

        Note:
            Adapted from `pydantic <https://github.com/samuelcolvin/pydantic>`_.
        """
        yield self.__class__.__name__ + "("
        yield 1
        for name in self.__datamodel_fields__.keys():
            yield name + "="
            yield fmt(getattr(self, name))
            yield ","
            yield 0
        yield -1
        yield ")"

    return __pretty__


def _make_data_model_class_getitem() -> classmethod:
    def __class_getitem__(
        cls: Type[GenericDataModelT], args: Union[Type, Tuple[Type]]
    ) -> Type[DataModelT] | Type[GenericDataModelT]:
        """Return an instance compatible with aliases created by :class:`typing.Generic` classes.

        See :class:`GenericDataModelAlias` for further information.
        """
        type_args: Tuple[Type] = args if isinstance(args, tuple) else (args,)
        concrete_cls: Type[DataModelT] = concretize(cls, *type_args)
        res = xtyping.StdGenericAliasType(concrete_cls, type_args)
        if sys.version_info < (3, 9):
            # in Python 3.8, xtyping.StdGenericAliasType (aka typing._GenericAlias)
            # does not copy all required `__dict__` entries, so do it manually
            for k, v in concrete_cls.__dict__.items():
                if k not in res.__dict__:
                    res.__dict__[k] = v
        return res

    return classmethod(__class_getitem__)


def _make_type_converter(type_annotation: TypeAnnotation, name: str) -> TypeConverter[_T]:
    # TODO(egparedes): if a "typing tree" structure is implemented, refactor this code as a tree traversal.
    #
    if xtyping.is_actual_type(type_annotation) and not isinstance(None, type_annotation):
        assert not xtyping.get_args(type_annotation)
        assert isinstance(type_annotation, type)

        def _type_converter(value: Any) -> _T:
            try:
                return (
                    value
                    if isinstance(value, type_annotation)  # type: ignore[arg-type]
                    else type_annotation(value)  # type: ignore
                )
            except Exception as error:
                raise TypeError(
                    f"Error during coertion of given value '{value}' for field '{name}'."
                ) from error

        return _type_converter

    if isinstance(type_annotation, TypeVar):
        return (
            _make_type_converter(type_annotation.__bound__, name)
            if type_annotation.__bound__
            else toolz.identity
        )

    if type_annotation is Any:
        return toolz.identity

    origin_type = xtyping.get_origin(type_annotation)

    if (
        origin_type is xtyping.Union
        and type(None) in (args := xtyping.get_args(type_annotation))
        and len(args) == 2
    ):
        # Optional type
        _inner_type_converter: TypeConverter[_T] = _make_type_converter(args[0], name)

        return cast(TypeConverter[_T], lambda x: x if x is None else _inner_type_converter(x))

    if xtyping.is_actual_type(origin_type):
        return _make_type_converter(origin_type, name)

    raise exceptions.EveTypeError(
        f"Automatic type coertion for {type_annotation} types is not supported."
    )


_KNOWN_MUTABLE_TYPES: Final = (list, dict, set)


def _make_datamodel(  # noqa: C901  # too complex but still readable and documented
    cls: Type[_T],
    *,
    repr: bool,  # noqa: A002   # shadowing 'repr' python builtin
    eq: bool,
    order: bool,
    unsafe_hash: bool,
    frozen: bool | Literal["strict"],
    match_args: bool,
    kw_only: bool,
    slots: bool,
    coerce: bool,
    generic: bool | Literal["True_no_checks"],
    type_validation_factory: Optional[FieldTypeValidatorFactory],
    _stacklevel_offset: int = 0,
) -> Type[_T]:
    """Actual implementation of the Data Model creation.

    See :func:`datamodel` for the description of the parameters.

    """
    mro_bases: Tuple[Type, ...] = cls.__mro__[1:]

    if "__annotations__" not in cls.__dict__:
        cls.__annotations__ = {}
    annotations = cls.__dict__["__annotations__"]
    resolved_annotations = xtyping.get_partial_type_hints(cls)
    annotations_with_extras = xtyping.get_partial_type_hints(cls, include_extras=True)

    frozen, strict_frozen = (True, True) if frozen == "strict" else (frozen, False)

    # Create attrib definitions with automatic type validators and converters
    # for the annotated fields. The keys in the original annotations are used for
    # iteration, since the resolved annotations also contain superclasses' annotations
    for key in annotations:
        type_hint = annotations[key] = resolved_annotations[key]

        # Skip members annotated as class variables
        if type_hint is ClassVar or xtyping.get_origin(type_hint) is ClassVar:
            continue

        if xtyping.get_origin(annotations_with_extras[key]) == xtyping.Annotated:
            _, *type_extras = xtyping.get_args(annotations_with_extras[key])
        else:
            type_extras = []

        coerce_field = coerce or _COERCED_TYPE_TAG in type_extras
        qualified_field_name = f"{cls.__name__}.{key}"

        # Create type validator if validation is enabled
        if type_validation_factory is None or _UNCHECKED_TYPE_TAG in type_extras:
            type_validator = lambda a, b, c: None  # noqa: E731
        else:
            type_validator = type_validation_factory(type_hint, qualified_field_name)

        # Declare an attrs.field() for this field with the right options.
        # There are different cases depending on the current value of the attribute in the class dict.
        attr_value_in_cls = cls.__dict__.get(key, NOTHING)
        if isinstance(attr_value_in_cls, attr._make._CountingAttr):  # type: ignore[attr-defined]  # _make is private
            # A field() function has already been used to customize the definition of the field.
            # In this case, we need to:
            #  - prepend the type validator to the list of provided validators (if any)
            #  - add the converter if the field needs to be converted and another custom converter
            #      has not been defined
            attr_value_in_cls._validator = (
                type_validator
                if attr_value_in_cls._validator is None
                else attr._make.and_(type_validator, attr_value_in_cls._validator)  # type: ignore[attr-defined]  # attr._make is not visible for mypy
            )
            coerce_field = coerce_field or attr_value_in_cls.converter == "coerce"
            if coerce_field:
                if attr_value_in_cls.converter in (None, "coerce"):
                    attr_value_in_cls.converter = _make_type_converter(
                        type_hint, qualified_field_name
                    )
                else:
                    raise TypeError(
                        f"Impossible to add automatic type converter to field '{qualified_field_name}' "
                        "which already defines a custom converter."
                    )

        else:
            # Create field converter if automatic coertion is enabled
            converter: TypeConverter = cast(
                TypeConverter,
                _make_type_converter(type_hint, qualified_field_name) if coerce_field else None,
            )
            if attr_value_in_cls is NOTHING:
                # The field has no definition in the class dict, it's only an annotation
                setattr(
                    cls,
                    key,
                    attrs.field(converter=converter, validator=type_validator),
                )

            else:
                # The field contains the default value in the class dict
                if isinstance(attr_value_in_cls, _KNOWN_MUTABLE_TYPES):
                    warnings.warn(
                        f"'{attr_value_in_cls.__class__.__name__}' value used as default in '{cls.__name__}.{key}'.\n"
                        "Mutable types should not defbe normally used as field defaults (use 'default_factory' instead).",
                        stacklevel=_stacklevel_offset + 2,
                    )
                setattr(
                    cls,
                    key,
                    attrs.field(
                        converter=converter, default=attr_value_in_cls, validator=type_validator
                    ),
                )

    # All fields should be annotated with type hints
    num_attrs = 0
    for key, value in cls.__dict__.items():
        if isinstance(value, attr._make._CountingAttr):  # type: ignore[attr-defined]  # attr._make is not visible for mypy
            num_attrs += 1
            if (
                key not in annotations
                and xtyping.get_origin(resolved_annotations.get(key, None)) is not ClassVar
            ):
                raise TypeError(f"Missing type annotation in '{key}' field.")

    # Validator processing
    root_validators = _collect_root_validators(cls)
    field_validators = _collect_field_validators(cls)

    for qualified_field_name, field_validator in field_validators.items():
        field_c_attr = cls.__dict__.get(qualified_field_name, None)
        if not field_c_attr:
            # Field has not been defined in the current class namespace,
            # look for field definition in the base classes.
            base_field_attr = _get_attribute_from_bases(
                qualified_field_name, mro_bases, annotations
            )
            if base_field_attr:
                # Create a new field in the current class cloning the existing
                # definition and add the new validator (attrs recommendation)
                field_c_attr = _make_counting_attr_from_attribute(
                    base_field_attr,
                )
                setattr(cls, qualified_field_name, field_c_attr)
            else:
                raise TypeError(
                    f"Validator assigned to non existing '{qualified_field_name}' field."
                )

        # Add field validator using field_attr.validator
        assert isinstance(field_c_attr, attr._make._CountingAttr)  # type: ignore[attr-defined]  # attr._make is not visible for mypy
        field_c_attr.validator(field_validator)

    setattr(cls, MODEL_ROOT_VALIDATORS_ATTR, tuple(root_validators))

    # Apply attrs.define() to enhance the class once all datamodels features
    # have been converted into attrs options
    if "__pre_init__" in cls.__dict__:
        if "__attrs_pre_init__" in cls.__dict__:
            raise TypeError(
                f"'{cls.__name__}' class contains custom '__attrs_pre_init__', which conflicts with `__pre_init__` ."
            )
        cls.__attrs_pre_init__ = cls.__pre_init__  # type: ignore[attr-defined]  # adding new attribute

    if "__attrs_post_init__" in cls.__dict__ and not hasattr(
        cls.__attrs_post_init__, _DATAMODEL_TAG  # type: ignore[attr-defined]  # mypy doesn't know about __attr_post_init__
    ):
        raise TypeError(f"'{cls.__name__}' class contains forbidden custom '__attrs_post_init__'.")
    cls.__attrs_post_init__ = _make_post_init(has_post_init="__post_init__" in cls.__dict__)  # type: ignore[attr-defined]  # adding new attribute

    if generic:
        if generic == "True_no_checks":
            # For the root GenericDataModel class, only set the attribute in __datamodel_params__ to True
            generic = True
        else:
            # For any other subclass, add the proper __class_getitem__ method
            if not issubclass(cls, (typing.Generic, xtyping.Generic)):  # type: ignore[arg-type]  # Generic is not considered a type
                raise TypeError(
                    f"'{cls.__name__}' cannot be converted to a GenericDataModel because it is not a generic class."
                )
            cls.__class_getitem__ = _make_data_model_class_getitem()  # type: ignore[attr-defined]  # adding new attribute

    # TODO(egparedes): consider the use of the field_transformer hook available in attrs:
    #   https://www.attrs.org/en/stable/extending.html#automatic-field-transformation-and-modification
    new_cls = attrs.define(  # type: ignore[call-overload]
        cls,
        auto_attribs=True,
        auto_detect=True,
        init=None,
        repr=None if repr else False,
        eq=None if eq else False,
        order=order,
        hash=None if not unsafe_hash else True,
        frozen=frozen,
        match_args=match_args,
        kw_only=kw_only,
        slots=slots,
    )
    assert (new_cls is cls) or slots

    # Final checks and postprocessing
    if strict_frozen:
        unhashable_fields = set()
        for f_attr in new_cls.__attrs_attrs__:
            if is_datamodel(f_attr.type):
                if getattr(f_attr.type, MODEL_PARAM_DEFINITIONS_ATTR).strict_frozen is True:
                    continue
            elif xtyping.is_hashable_type(f_attr.type):
                continue
            unhashable_fields.add(f_attr.name)

        if unhashable_fields:
            raise exceptions.EveTypeError(
                f"Some fields ({unhashable_fields}) can not be considered strictly immutable."
            )

    if "__attrs_init__" in new_cls.__dict__:
        assert "__auto_init__" not in cls.__dict__
        new_cls.__auto_init__ = new_cls.__attrs_init__

    new_cls.__pretty__ = _make_devtools_pretty()
    setattr(
        new_cls,
        MODEL_PARAM_DEFINITIONS_ATTR,
        utils.FrozenNamespace(
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            strict_frozen=strict_frozen,
            match_args=match_args,
            kw_only=kw_only,
            slots=slots,
            coerce=coerce,
            generic=generic,
            type_validation_factory=type_validation_factory,
        ),
    )
    setattr(
        new_cls,
        MODEL_FIELD_DEFINITIONS_ATTR,
        utils.FrozenNamespace(**{f_attr.name: f_attr for f_attr in new_cls.__attrs_attrs__}),
    )
    new_cls.update_forward_refs = classmethod(update_forward_refs)

    return new_cls


@utils.optional_lru_cache(maxsize=None, typed=True)
def _make_concrete_with_cache(
    datamodel_cls: Type[GenericDataModelT],
    *type_args: Type,
    class_name: Optional[str] = None,
    module: Optional[str] = None,
) -> Type[DataModelT]:
    if not is_generic_datamodel_class(datamodel_cls):
        raise TypeError(f"'{datamodel_cls.__name__}' is not a generic model class.")
    for t in type_args:
        if not (
            isinstance(t, (type, type(None), xtyping.StdGenericAliasType))
            or (getattr(type(t), "__module__", None) in ("typing", "typing_extensions"))
        ):
            raise TypeError(
                f"Only 'type' and 'typing' definitions can be passed as arguments "
                f"to instantiate a generic model class (received: {type_args})."
            )
    if len(type_args) != len(datamodel_cls.__parameters__):
        raise TypeError(
            f"Instantiating '{datamodel_cls.__name__}' generic model with a wrong number of parameters "
            f"({len(type_args)} used, {len(datamodel_cls.__parameters__)} expected)."
        )

    # Replace field definitions with the new actual types for generic fields
    type_params_map = dict(zip(datamodel_cls.__parameters__, type_args))
    model_fields = getattr(datamodel_cls, MODEL_FIELD_DEFINITIONS_ATTR)
    new_annotations = {
        # TODO(egparedes): ?
        # noqa: e800 "__args__": "ClassVar[Tuple[Union[Type, TypeVar], ...]]",
        # noqa: e800 "__parameters__": "ClassVar[Tuple[TypeVar, ...]]",
    }

    new_field_c_attrs = {}
    for field_name, field_type in xtyping.get_type_hints(datamodel_cls).items():
        new_annotation, replaced = _substitute_typevars(field_type, type_params_map)
        if replaced:
            new_annotations[field_name] = new_annotation
            field_attrib = getattr(model_fields, field_name)
            new_field_c_attrs[field_name] = _make_counting_attr_from_attribute(field_attrib)

    # Create new concrete class
    if not class_name:
        arg_names = []
        for tp_var in datamodel_cls.__parameters__:
            arg_string = tp_var.__name__
            if tp_var in type_params_map:
                concrete_arg = type_params_map[tp_var]
                if isinstance(concrete_arg, type):
                    arg_string = concrete_arg.__name__
                else:
                    arg_string = utils.slugify(
                        str(concrete_arg).replace("typing.", "").replace("...", "ellipsis")
                    )

            arg_names.append(arg_string)

        class_name = f"{datamodel_cls.__name__}__{'_'.join(arg_names)}"

    namespace = {
        "__annotations__": new_annotations,
        "__module__": module if module else datamodel_cls.__module__,
        **new_field_c_attrs,
    }

    concrete_cls = type(class_name, (datamodel_cls,), namespace)
    assert concrete_cls.__module__ == module or not module

    if MODEL_FIELD_DEFINITIONS_ATTR not in concrete_cls.__dict__:
        # If original model does not inherit from GenericDataModel,
        # _make_datamodel() hasn't been called automatically, so call it now
        params = getattr(datamodel_cls, MODEL_PARAM_DEFINITIONS_ATTR)
        concrete_cls = _make_datamodel(
            concrete_cls,
            **{
                name: getattr(params, name)
                for name in ("repr", "eq", "order", "unsafe_hash", "frozen")
            },
        )

    return concrete_cls


if xtyping.TYPE_CHECKING:
    FrozenModel: TypeAlias = DataModel

else:

    class FrozenModel(DataModel, frozen=True):
        __slots__ = ()


if xtyping.TYPE_CHECKING:

    class GenericDataModel(GenericDataModelTP):
        @classmethod
        def __class_getitem__(
            cls: Type[GenericDataModelTP], args: Union[Type, Tuple[Type, ...]]
        ) -> Union[DataModelTP, GenericDataModelTP]:
            ...

else:

    class GenericDataModel(DataModel, __dm_opts=[_GENERIC_DATAMODEL_ROOT_DM_OPT]):
        __slots__ = ()

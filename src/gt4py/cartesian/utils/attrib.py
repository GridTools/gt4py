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

import typing

import attr


class _TypeDescriptor:
    def __init__(self, name, args, make_validator_func, type_hint):
        self.name = name
        assert args is None or isinstance(args, tuple)
        self.args = args
        self.make_validator_func = make_validator_func
        self.type_hint = type_hint

    def __repr__(self):
        args = self.args if self.args is not None else []
        arg_names = []
        for a in args:
            if isinstance(a, type):
                arg_names.append(a.__name__)
            elif isinstance(a, _TypeDescriptor):
                arg_names.append(repr(a))
        args = "[{}]".format(", ".join(arg_names)) if len(arg_names) > 0 else ""
        return "{}{}".format(self.name, args)

    @property
    def validator(self):
        return self.make_validator_func(self.args)


class _GenericTypeDescriptor:
    def __init__(self, name, n_args, make_validator_func, generic_type_hint):
        self.name = name

        if not isinstance(n_args, tuple):
            n_args = (n_args, n_args)
        assert all(isinstance(n, int) and n > 0 for n in n_args if n is not None)

        self.n_args = n_args

        assert callable(make_validator_func)
        self.make_validator_func = make_validator_func
        self.generic_type_hint = generic_type_hint

    def __getitem__(self, type_list):
        if not isinstance(type_list, tuple):
            type_list = tuple([type_list])
        assert all(isinstance(t, (type, _TypeDescriptor)) for t in type_list)
        assert (self.n_args[0] is None or self.n_args[0] <= len(type_list)) and (
            self.n_args[1] is None or self.n_args[1] <= self.n_args[1]
        )
        hint_list = []
        for t in type_list:
            if isinstance(t, type):
                hint_list.append(t)
            else:
                hint_list.append(getattr(t, "type_hint", typing.Any))

        return _TypeDescriptor(
            self.name,
            args=tuple(type_list),
            make_validator_func=self.make_validator_func,
            type_hint=self.generic_type_hint[tuple(hint_list)],
        )


def _make_type_validator(t):
    assert isinstance(t, (type, _TypeDescriptor))

    if isinstance(t, type):
        return attr.validators.instance_of(t)
    else:
        return t.validator


def _make_sequence_validator(type_list, container_types=(list, tuple)):
    assert type_list is not None and len(type_list) == 1
    item_type = type_list[0]

    item_validator = _make_type_validator(item_type)

    def _is_sequence_of_validator(instance, attribute, value):
        try:
            assert isinstance(value, tuple(container_types))
            assert isinstance([item_validator(instance, attribute, v) for v in value], list)
        except Exception as ex:
            raise ValueError(
                "Expr ({value}) does not match the '{name}' specification".format(
                    value=value, name=attribute.name
                )
            ) from ex

    return _is_sequence_of_validator


def _make_list_validator(type_list):
    return _make_sequence_validator(type_list, container_types=[list])


def _make_set_validator(type_list):
    return _make_sequence_validator(type_list, container_types=[set])


def _make_dict_validator(type_list):
    assert type_list is not None and len(type_list) == 2
    key_validator, value_validator = [_make_type_validator(t) for t in type_list]

    def _is_dict_of_validator(instance, attribute, value):
        try:
            assert isinstance(value, dict)
            assert isinstance([key_validator(instance, attribute, v) for v in value.keys()], list)
            assert isinstance(
                [value_validator(instance, attribute, v) for v in value.values()], list
            )
        except Exception as ex:
            raise ValueError(
                "Expr ({value}) does not match the '{name}' specification".format(
                    value=value, name=attribute.name
                )
            ) from ex

    return _is_dict_of_validator


def _make_tuple_validator(type_list):
    assert type_list is not None and len(type_list) > 0
    validators = [_make_type_validator(t) for t in type_list]

    def _is_tuple_of_validator(instance, attribute, value):
        try:
            assert isinstance(value, tuple)
            assert len(value) == len(validators)
            assert isinstance(
                [
                    validator(instance, attribute, value)
                    for value, validator in zip(value, validators)
                ],
                list,
            )
        except Exception as ex:
            raise ValueError(
                "Expr ({value}) does not match the '{name}' specification".format(
                    value=value, name=attribute.name
                )
            ) from ex

    return _is_tuple_of_validator


def _make_union_validator(type_list):
    assert type_list is not None and len(type_list) > 1
    validators = [_make_type_validator(t) for t in type_list]

    def _is_union_of_validator(instance, attribute, value):
        passed = True
        for validator in validators:
            try:
                validator(instance, attribute, value)
                break

            except Exception:
                pass
        else:
            passed = False

        if not passed:
            raise ValueError(
                "Expr ({value}) does not match the '{name}' specification".format(
                    value=value, name=attribute.name
                )
            )

    return _is_union_of_validator


def _make_any_validator(type_list):
    def _is_any_validator(instance, attribute, value):
        pass

    return _is_any_validator


def _make_nothing_validator(type_list):
    def _is_nothing_validator(instance, attribute, value):
        assert value is None

    return _is_nothing_validator


Any = _TypeDescriptor("Any", None, _make_any_validator, typing.Any)

Sequence = _GenericTypeDescriptor("Sequence", 1, _make_sequence_validator, typing.Sequence)
List = _GenericTypeDescriptor("List", 1, _make_list_validator, typing.List)
Dict = _GenericTypeDescriptor("Dict", 2, _make_dict_validator, typing.Dict)
Set = _GenericTypeDescriptor("Set", 1, _make_set_validator, typing.Set)
Tuple = _GenericTypeDescriptor("Tuple", (1, None), _make_tuple_validator, typing.Tuple)
Union = _GenericTypeDescriptor("Union", (2, None), _make_union_validator, typing.Union)

Optional = _GenericTypeDescriptor(
    "Optional",
    (1, None),
    lambda type_list: _make_union_validator([*type_list, type(None)]),
    typing.Optional,
)


def attribute(of, optional=False, **kwargs):
    if isinstance(of, _TypeDescriptor):
        attr_validator = of.validator
        attr_type_hint = of.type_hint

    elif isinstance(of, type):
        # assert of in (bool, float, str, int, enum.Enum) # noqa: ERA001 [commented-out-code]
        attr_validator = attr.validators.instance_of(of)
        attr_type_hint = of

    else:
        raise ValueError("Invalid attribute type '{}'".format(of))

    if optional:
        attr_validator = attr.validators.optional(attr_validator)
        kwargs.setdefault("default", None)

    return attr.ib(validator=attr_validator, type=attr_type_hint, **kwargs)


class AttributeClassLike:
    def validate(self): ...

    @property
    def attributes(self): ...

    @property
    def as_dict(self): ...


def attribclass(cls_or_none=None, **kwargs):
    """Class decorator to convert a regular class into an `AttribClass`."""

    def validate(self):
        """Validate this instance's data attributes."""
        attr.validate(self)

    def attributes(self):
        """Generate a :class:`list` with this class' attribute names."""
        result = [a.name for a in attr.fields(self.__class__)]
        return result

    def as_dict(self):
        """Generate a :class:`dict` with this instance's data attributes."""
        return attr.asdict(self)

    extra_members = dict(validate=validate, attributes=property(attributes), as_dict=as_dict)

    def _make_attrs_class_wrapper(cls):
        for name, member in extra_members.items():
            if name in cls.__dict__.keys():
                raise ValueError(
                    "Name clashing with a existing '{name}' member"
                    " of the decorated class ".format(name=name)
                )
            setattr(cls, name, member)

        new_cls = attr.s(cls, **kwargs)
        return new_cls

    if cls_or_none is None:
        return _make_attrs_class_wrapper
    else:
        return _make_attrs_class_wrapper(cls_or_none)


def attribkwclass(cls_or_none=None, **kwargs):
    kwargs.setdefault("kw_only", True)
    return attribclass(cls_or_none=cls_or_none, **kwargs)


def attributes_of(an_attribclass):
    return [a.name for a in attr.fields(an_attribclass)]

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
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
from typing import Any, List, cast


class BuiltInTypeMeta(type):
    """
    Metaclass representing types used inside GTScript code.

    For now only a bare minimum of operations on these types is supported, i.e. subclass checks and extraction
    of type arguments.
    """

    class_name: str
    namespace: str
    args: List[Any]

    def __new__(cls, class_name, bases, namespace, args=None):
        if not bases:
            bases = (object,)  # cannonicalization

        assert bases == (object,) or (len(bases) == 1 and issubclass(bases[0], BuiltInType))
        # TODO(tehrengruber): allow only class methods
        # assert all(attr[0:2] == "__" for attr in namespace.keys())  # no custom attributes
        # TODO(tehrengruber): there might be a better way to do this
        instance = type.__new__(cls, class_name, bases, namespace)
        instance.class_name = class_name
        instance.namespace = namespace
        instance.args = args if args else ()
        return instance

    @property
    def body(self):
        """Return type function body (name borrowed from polymorphism theory)."""
        return self.__class__(self.class_name, self.__bases__, self.namespace, args=[])

    def __eq__(self, other) -> bool:
        if (
            isinstance(other, BuiltInTypeMeta)
            and self.namespace == other.namespace
            and self.class_name == other.class_name
            and self.args == other.args
        ):
            return True
        return False

    def __getitem__(
        self, args
    ) -> "BuiltInTypeMeta":  # TODO(tehrengruber): evaluate using __class_getitem__ instead
        if not isinstance(args, tuple):
            args = (args,)
        return self.__class__(self.class_name, self.__bases__, self.namespace, args=args)

    def __instancecheck__(self, instance) -> bool:
        # TODO(tehrengruber): implement
        raise RuntimeError("not implemented")

    def __hash__(self):
        return hash((self.class_name, hash(frozenset(self.namespace.items())), self.args))

    def __subclasscheck__(self, other) -> bool:
        # TODO(tehrengruber): enhance
        if inspect.isclass(other):
            for base in other.__mro__:
                if self == base or (isinstance(base, BuiltInTypeMeta) and self == base.body):
                    return True
        return False


class BuiltInType(metaclass=BuiltInTypeMeta):
    pass


class Connectivity(BuiltInType):
    @classmethod
    def base_connectivity(cls):
        return next(
            t for t in cls.__mro__ if issubclass(t, Connectivity) and t.body == Connectivity
        )

    @classmethod
    def primary_location(cls):
        return cls.base_connectivity().args[0]

    @classmethod
    def secondary_location(cls):
        return cls.base_connectivity().args[1]

    @classmethod
    def max_neighbors(cls):
        return cls.base_connectivity().args[2]

    @classmethod
    def has_skip_values(cls):
        return cls.base_connectivity().args[3]


class Field(BuiltInType):
    pass


class SparseField(BuiltInType):
    pass


class TemporarySparseField(BuiltInType):  # TODO(tehrengruber): make this a subtype of Field
    pass


class TemporaryField(BuiltInType):  # TODO(tehrengruber): make this a subtype of Field
    pass


class Location(BuiltInType):
    pass


class Local(BuiltInType):
    """Used as a type argument to :class:`.Field` representing a Local dimension."""

    pass


class LocalField(BuiltInType):
    pass

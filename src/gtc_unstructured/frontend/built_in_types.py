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

from typing import Any, List, cast


class BuiltInTypeMeta(type):
    """
    Metaclass representing types used inside GTScript code.

    For now only a bare minimum of operations on these types is supported, i.e. (pseudo) subclass checks and extraction
    of type arguments.
    """

    class_name: str
    namespace: str
    args: List[Any]

    def __new__(cls, class_name, bases, namespace, args=None):
        assert bases == () or (len(bases) == 1 and issubclass(bases[0], BuiltInType))
        assert all(attr[0:2] == "__" for attr in namespace.keys())  # no custom attributes
        # TODO(tehrengruber): there might be a better way to do this
        instance = type.__new__(cls, class_name, bases, namespace)
        instance.class_name = class_name
        instance.namespace = namespace
        instance.args = args
        return instance

    def __eq__(self, other) -> bool:
        if (
            isinstance(other, BuiltInTypeMeta)
            and self.namespace == other.namespace
            and self.class_name == other.class_name
        ) and (self.args is None or self.args == other.args):
            return True
        return False

    def __getitem__(
        self, args
    ) -> "BuiltInTypeMeta":  # TODO(tehrengruber): evaluate using __class_getitem__ instead
        if not isinstance(args, tuple):
            args = (args,)
        return BuiltInTypeMeta(self.class_name, (), self.namespace, args=args)

    def __instancecheck__(self, instance) -> bool:
        # TODO(tehrengruber): implement
        raise RuntimeError("not implemented")

    def __subclasscheck__(self, other) -> bool:
        # TODO(tehrengruber): enhance
        return cast(bool, self == other)


class BuiltInType(metaclass=BuiltInTypeMeta):
    pass


class Mesh(BuiltInType):
    pass


class Field(BuiltInType):
    pass


class TemporaryField(BuiltInType):  # TODO(tehrengruber): make this a subtype of Field
    pass


class Location(BuiltInType):
    pass


class Local(BuiltInType):
    """Used as a type argument to :class:`.Field` representing a Local dimension."""

    pass

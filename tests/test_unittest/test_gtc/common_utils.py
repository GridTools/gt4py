# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from typing import Any, Callable, Set

import factory

from eve import concepts, traits, type_definitions, visitors
from gtc import common


def undefined_symbol_list(
    symbol_creator: Callable[[str], Any], *fields_to_collect: str, list_creator=None
) -> factory.LazyAttribute:
    class CollectSymbolsAndRefs(visitors.NodeVisitor):
        def visit_Node(self, node: concepts.Node, *, symbols: Set[str], refs: Set[str]) -> None:
            for name, metadata in node.__node_children__.items():
                type_ = metadata["definition"].type_
                if isinstance(type_, type):
                    if issubclass(type_, type_definitions.SymbolName):
                        symbols.add(getattr(node, name))
                    elif issubclass(type_, type_definitions.SymbolRef):
                        refs.add(getattr(node, name))

            if not isinstance(node, traits.SymbolTableTrait):
                self.generic_visit(node, symbols=symbols, refs=refs)

    def func(obj):
        symbols: Set[str] = set()
        refs: Set[str] = set()
        for field in fields_to_collect:
            CollectSymbolsAndRefs().visit(getattr(obj, field), symbols=symbols, refs=refs)
        res = [symbol_creator(name) for name in refs - symbols]
        if list_creator:
            return list_creator(res)
        return res

    return factory.LazyAttribute(func)


def identifier(cls):
    return factory.Sequence(lambda n: f"val_{cls.__name__}_{n}")


class CartesianOffsetFactory(factory.Factory):
    class Meta:
        model = common.CartesianOffset

    i = 0
    j = 0
    k = 0

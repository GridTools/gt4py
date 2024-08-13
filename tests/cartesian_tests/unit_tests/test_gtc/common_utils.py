# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Set

import factory

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.eve import datamodels


def undefined_symbol_list(
    symbol_creator: Callable[[str], Any], *fields_to_collect: str, list_creator=None
) -> factory.LazyAttribute:
    class CollectSymbolsAndRefs(eve.NodeVisitor):
        def visit_Node(self, node: eve.Node, *, symbols: Set[str], refs: Set[str]) -> None:
            for value in datamodels.astuple(node):
                if issubclass(value.__class__, eve.SymbolName):
                    symbols.add(str(value))
                elif issubclass(value.__class__, eve.SymbolRef):
                    refs.add(str(value))

            self.generic_visit(node, symbols=symbols, refs=refs)

    def func(obj):
        symbols: Set[str] = set()
        refs: Set[str] = set()
        for field in fields_to_collect:
            CollectSymbolsAndRefs().visit(getattr(obj, field), symbols=symbols, refs=refs)
        res = [symbol_creator(name) for name in sorted(refs - symbols)]
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


class HorizontalMaskFactory(factory.Factory):
    class Meta:
        model = common.HorizontalMask

    i = common.HorizontalInterval.full()
    j = common.HorizontalInterval.full()

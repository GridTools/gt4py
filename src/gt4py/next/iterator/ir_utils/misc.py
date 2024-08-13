# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from collections import ChainMap

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im


@dataclasses.dataclass(frozen=True)
class CannonicalizeBoundSymbolNames(eve.NodeTranslator):
    """
    Given an iterator expression cannonicalize all bound symbol names.

    If two such expression are in the same scope and equal so are their values.

    >>> testee1 = im.lambda_("a")(im.plus("a", "b"))
    >>> cannonicalized_testee1 = CannonicalizeBoundSymbolNames.apply(testee1)
    >>> str(cannonicalized_testee1)
    'λ(_csym_1) → _csym_1 + b'

    >>> testee2 = im.lambda_("c")(im.plus("c", "b"))
    >>> cannonicalized_testee2 = CannonicalizeBoundSymbolNames.apply(testee2)
    >>> assert cannonicalized_testee1 == cannonicalized_testee2
    """

    _uids: eve_utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve_utils.UIDGenerator(prefix="_csym")
    )

    @classmethod
    def apply(cls, node: itir.Expr):
        return cls().visit(node, sym_map=ChainMap({}))

    def visit_Lambda(self, node: itir.Lambda, *, sym_map: ChainMap):
        sym_map = sym_map.new_child()
        for param in node.params:
            sym_map[str(param.id)] = self._uids.sequential_id()

        return im.lambda_(*sym_map.values())(self.visit(node.expr, sym_map=sym_map))

    def visit_SymRef(self, node: itir.SymRef, *, sym_map: dict[str, str]):
        return im.ref(sym_map[node.id]) if node.id in sym_map else node


def is_equal(a: itir.Expr, b: itir.Expr):
    """
    Return true if two expressions have provably equal values.

    Be aware that this function might return false even though the two expression have the same
    value.

    >>> testee1 = im.lambda_("a")(im.plus("a", "b"))
    >>> testee2 = im.lambda_("c")(im.plus("c", "b"))
    >>> assert is_equal(testee1, testee2)

    >>> testee1 = im.lambda_("a")(im.plus("a", "b"))
    >>> testee2 = im.lambda_("c")(im.plus("c", "d"))
    >>> assert not is_equal(testee1, testee2)
    """
    # TODO(tehrengruber): Extend this function cover more cases than just those with equal
    #  structure, e.g., by also canonicalization of the structure.
    return a == b or (
        CannonicalizeBoundSymbolNames.apply(a) == CannonicalizeBoundSymbolNames.apply(b)
    )

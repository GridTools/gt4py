# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import TypeVar

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.type_system import type_specifications as ts


_Node_T = TypeVar("_Node_T", bound=itir.Node)


def _pin(branch: itir.Expr) -> itir.Expr:
    if cpm.is_call_to(branch, "make_tuple"):
        pinned_tuple = im.make_tuple(*(_pin(el) for el in branch.args))
        if hasattr(branch.annex, "domain"):
            pinned_tuple.annex.domain = branch.annex.domain
        return pinned_tuple
    domain = getattr(branch.annex, "domain", None)
    if (
        isinstance(branch, itir.SymRef)
        and isinstance(branch.type, ts.FieldType)
        and isinstance(domain, domain_utils.SymbolicDomain)
    ):
        return im.as_fieldop("deref", domain.as_expr())(branch)
    return branch


@dataclasses.dataclass
class PinIfBranchDomains(PreserveLocationVisitor, NodeTranslator):
    """
    Pin the inferred domain of a field reference selected by a field-level `if_` into the IR.

    Each branch of an `if_` outside of a stencil that is a reference to a field, or a tuple of
    such references, is wrapped into an `as_fieldop(deref, ...)` on its domain from
    `annex.domain`:

        if_(c, a, b)  ->  if_(c, as_fieldop(deref, D_a)(a), as_fieldop(deref, D_b)(b))

    An `as_fieldop` keeps its domain as an argument, a reference only in its annex, and two later
    steps replace that annex: `concat_where.transform_to_as_fieldop` turns the restriction of a
    `concat_where` branch into a condition inside a stencil that domain inference can not see, so
    the re-inference in `create_global_tmps` gives the reference the extended domain of its
    consumer; and once a referenced expression is extracted into a temporary, the reference
    carries the domain the temporary is allocated on. In both cases the `if_` is then written on
    the wrong domain.

    Must run after domain inference. Adds one `as_fieldop` per pinned reference.

    Example:
    >>> from gt4py.next import Dimension, common
    >>> IDim = Dimension("IDim")
    >>> field = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    >>> domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
    >>> a = im.ref("a", field)
    >>> a.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
    >>> b = im.as_fieldop("deref", domain)(im.ref("b", field))
    >>> print(PinIfBranchDomains.apply(im.if_("cond", a, b)))
    if   cond
    then as_fieldop(deref, c⟨ IDimₕ: [0, 10[ ⟩)(a)
    else as_fieldop(deref, c⟨ IDimₕ: [0, 10[ ⟩)(b)
    """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls, node: _Node_T) -> _Node_T:
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        if cpm.is_applied_as_fieldop(node):
            # the stencil is in local view; only its arguments can contain field-level `if_`s
            return im.call(node.fun)(*(self.visit(arg) for arg in node.args))
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "if_"):
            cond, true_branch, false_branch = node.args
            node = im.if_(cond, _pin(true_branch), _pin(false_branch))
        return node

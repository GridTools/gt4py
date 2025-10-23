# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
GridTools Intermediate Representation.

GTIR represents a computation with the semantics of the
`GTScript parallel model <https://github.com/GridTools/concepts/wiki/GTScript-Parallel-model>`.

Type constraints and validators narrow the IR as much as reasonable to valid (executable) IR.

Analysis is required to generate valid code (complying with the parallel model)
- extent analysis to define the extended compute domain
- `FieldIfStmt` expansion to comply with the parallel model
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple, Type

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.common import AxisBound, LocNode
from gt4py.eve import datamodels


@eve.utils.noninstantiable
class Expr(common.Expr):
    pass


@eve.utils.noninstantiable
class Stmt(common.Stmt):
    pass


class BlockStmt(common.BlockStmt[Stmt], Stmt):
    pass


class Literal(common.Literal, Expr):
    pass


class VariableKOffset(common.VariableKOffset[Expr]):
    pass


class AbsoluteKIndex(common.AbsoluteKIndex[Expr]):
    """See gtc.common.AbsoluteKIndex"""

    pass


class ScalarAccess(common.ScalarAccess, Expr):
    pass


class FieldAccess(common.FieldAccess[Expr, VariableKOffset], Expr):
    pass


class IteratorAccess(Expr):
    class AxisName(eve.StrEnum):
        I = "I"  # noqa: E741 [ambiguous-variable-name]
        J = "J"
        K = "K"

    name: AxisName
    kind: common.ExprKind = common.ExprKind.SCALAR


class ParAssignStmt(common.AssignStmt[FieldAccess, Expr], Stmt):
    """Parallel assignment.

    R.h.s. is evaluated for all points and the resulting field is assigned
    (GTScript parallel model).
    Scalar variables on the l.h.s. are not allowed,
    as the only scalar variables are read-only stencil parameters.
    """

    @datamodels.validator("left")
    def no_horizontal_offset_in_assignment(
        self, attribute: datamodels.Attribute, value: FieldAccess
    ) -> None:
        offsets = value.offset.to_dict()
        if offsets["i"] != 0 or offsets["j"] != 0:
            raise ValueError("Lhs of assignment must not have a horizontal offset.")

    @datamodels.root_validator
    @classmethod
    def no_write_and_read_with_offset_of_same_field(
        cls: Type[ParAssignStmt], instance: ParAssignStmt
    ) -> None:
        if isinstance(instance.left, FieldAccess):
            offset_reads = (
                (
                    eve.walk_values(instance.right)
                    .filter(_cartesian_fieldaccess)
                    .filter(lambda acc: acc.offset.i != 0 or acc.offset.j != 0)
                    .getattr("name")
                    .to_set()
                )
                | (
                    eve.walk_values(instance.right)
                    .filter(_absolutekindex_fieldaccess)
                    .getattr("name")
                    .to_set()
                )
                | (
                    eve.walk_values(instance.right)
                    .filter(_variablek_fieldaccess)
                    .getattr("name")
                    .to_set()
                )
            )
            if instance.left.name in offset_reads:
                raise ValueError("Self-assignment with offset is illegal.")

    _dtype_validation = common.assign_stmt_dtype_validation(strict=False)


class FieldIfStmt(common.IfStmt[BlockStmt, Expr], Stmt):
    """
    If statement with a field expression as condition.

    - The condition is evaluated for all gridpoints and stored in a mask.
    - Each statement inside the if and else branches is executed according
      to the same rules as statements outside of branches.

    The following restriction applies:

    - Inside the if and else blocks the same field cannot be written to
      and read with an offset in the parallel axes (order does not matter).

    See `parallel model
    <https://github.com/GridTools/concepts/wiki/GTScript-Parallel-model#conditionals-on-field-expressions>`
    """

    @datamodels.validator("cond")
    def verify_scalar_condition(self, attribute: datamodels.Attribute, value: Expr) -> None:
        if value.kind != common.ExprKind.FIELD:
            raise ValueError("Condition is not a field expression")

    # TODO(havogt) add validator for the restriction (it's a pass over the subtrees...)


class ScalarIfStmt(common.IfStmt[BlockStmt, Expr], Stmt):
    """
    If statement with a scalar expression as condition.

    No special rules apply.
    """

    @datamodels.validator("cond")
    def verify_scalar_condition(self, attribute: datamodels.Attribute, value: Expr) -> None:
        if value.kind != common.ExprKind.SCALAR:
            raise ValueError("Condition is not scalar")


class HorizontalRestriction(common.HorizontalRestriction[Stmt], Stmt):
    pass


class While(common.While[Stmt, Expr], Stmt):
    """While loop with a field or scalar expression as condition."""

    @datamodels.validator("body")
    def _no_write_and_read_with_horizontal_offset_all(
        self, attribute: datamodels.Attribute, value: List[Stmt]
    ) -> None:
        """In a while loop all variables must not be written and read with a horizontal offset."""
        if names := _written_and_read_with_offset(value):
            raise ValueError(f"Illegal write and read with horizontal offset detected for {names}.")


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_validator = common.binary_op_dtype_propagation(strict=False)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=False)


class Cast(common.Cast[Expr], Expr):
    pass


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=False)


class Decl(LocNode):  # TODO probably Stmt
    name: eve.Coerced[eve.SymbolName]
    dtype: common.DataType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = eve.field(default_factory=tuple)


class ScalarDecl(Decl):
    pass


class Interval(LocNode):
    start: AxisBound
    end: AxisBound


# TODO(havogt) should vertical loop open a scope?
class VerticalLoop(LocNode):
    interval: Interval
    loop_order: common.LoopOrder
    temporaries: List[FieldDecl]
    body: List[Stmt]

    @datamodels.root_validator
    @classmethod
    def _no_write_and_read_with_horizontal_offset(
        cls: Type[VerticalLoop], instance: VerticalLoop
    ) -> None:
        """
        In the same VerticalLoop a field must not be written and read with a horizontal offset.

        Temporaries don't have this constraint. Backends are required to implement
        them using block-private halos.
        """
        intersec = _written_and_read_with_offset(instance.body)
        non_tmp_fields = {
            acc for acc in intersec if acc not in {tmp.name for tmp in instance.temporaries}
        }
        if len(non_tmp_fields) > 0:
            raise ValueError(
                f"Illegal write and read with horizontal offset detected for {non_tmp_fields}."
            )


class Argument(eve.Node):
    name: str
    is_keyword: bool
    default: str


class Stencil(LocNode, eve.ValidatedSymbolTableTrait):
    name: str
    api_signature: List[Argument]
    params: List[Decl]
    vertical_loops: List[VerticalLoop]
    externals: Dict[str, Literal]
    sources: Dict[str, str]
    docstring: str

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    _validate_lvalue_dims = common.validate_lvalue_dims(VerticalLoop, FieldDecl)


def _cartesian_fieldaccess(node) -> bool:
    return (
        isinstance(node, FieldAccess)
        and not isinstance(node.offset, VariableKOffset)
        and not isinstance(node.offset, AbsoluteKIndex)
    )


def _variablek_fieldaccess(node) -> bool:
    return (
        isinstance(node, FieldAccess)
        and isinstance(node.offset, VariableKOffset)
        and not isinstance(node.offset, AbsoluteKIndex)
    )


def _absolutekindex_fieldaccess(node) -> bool:
    return (
        isinstance(node, FieldAccess)
        and isinstance(node.offset, AbsoluteKIndex)
        and not isinstance(node.offset, VariableKOffset)
    )


# TODO(havogt): either move to eve or will be removed in the attr-based eve if a List[Node] is represented as a CollectionNode
def _written_and_read_with_offset(stmts: List[Stmt]) -> Set[str]:
    """Return a list of names that are written to and read with offset."""

    def _writes(stmts: List[Stmt]) -> Set[str]:
        result = set()
        for left in eve.walk_values(stmts).if_isinstance(ParAssignStmt).getattr("left"):
            result |= eve.walk_values(left).if_isinstance(FieldAccess).getattr("name").to_set()
        return result

    def _reads_with_offset(stmts: List[Stmt]) -> Set[str]:
        return (
            eve.walk_values(stmts)
            .filter(_cartesian_fieldaccess)
            .filter(
                lambda acc: acc.offset.i != 0 or acc.offset.j != 0
            )  # writes always have zero offset
            .getattr("name")
            .to_set()
        )

    writes = _writes(stmts)
    reads_with_offset = _reads_with_offset(stmts)
    return writes & reads_with_offset

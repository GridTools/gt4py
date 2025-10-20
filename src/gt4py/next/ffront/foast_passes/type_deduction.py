# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, TypeAlias, TypeVar, cast

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, NodeVisitor, traits
from gt4py.next import errors, utils
from gt4py.next.common import DimensionKind, promote_dims
from gt4py.next.ffront import (  # noqa
    dialect_ast_enums,
    experimental,
    fbuiltins,
    type_info as ti_ffront,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.foast_passes import utils as foast_utils
from gt4py.next.iterator import builtins
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


def with_altered_scalar_kind(
    type_spec: ts.TypeSpec, new_scalar_kind: ts.ScalarKind
) -> ts.ScalarType | ts.FieldType:
    """
    Given a scalar or field type, return a type with different scalar kind.

    Examples:
    ---------
    >>> from gt4py.next import Dimension
    >>> scalar_t = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    >>> print(with_altered_scalar_kind(scalar_t, ts.ScalarKind.BOOL))
    bool

    >>> field_t = ts.FieldType(
    ...     dims=[Dimension(value="I")], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    ... )
    >>> print(with_altered_scalar_kind(field_t, ts.ScalarKind.FLOAT32))
    Field[[I], float32]
    """
    if isinstance(type_spec, ts.FieldType):
        return ts.FieldType(
            dims=type_spec.dims,
            dtype=with_altered_scalar_kind(type_spec.dtype, new_scalar_kind),
        )
    elif isinstance(type_spec, ts.ScalarType):
        return ts.ScalarType(kind=new_scalar_kind, shape=type_spec.shape)
    else:
        raise ValueError(f"Expected field or scalar type, got '{type_spec}'.")


def construct_tuple_type(
    true_branch_types: list, false_branch_types: list, mask_type: ts.FieldType
) -> list:
    """
    Recursively construct  the return types for the tuple return branch.

    Examples:
    ---------
    >>> from gt4py.next import Dimension
    >>> mask_type = ts.FieldType(
    ...     dims=[Dimension(value="I")], dtype=ts.ScalarType(kind=ts.ScalarKind.BOOL)
    ... )
    >>> true_branch_types = [
    ...     ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
    ...     ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
    ... ]
    >>> false_branch_types = [
    ...     ts.FieldType(
    ...         dims=[Dimension(value="I")], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    ...     ),
    ...     ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
    ... ]
    >>> print(construct_tuple_type(true_branch_types, false_branch_types, mask_type))
    [FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None)), FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None))]
    """
    element_types_new = true_branch_types
    for i, element in enumerate(true_branch_types):
        if isinstance(element, ts.TupleType):
            element_types_new[i] = ts.TupleType(
                types=construct_tuple_type(element.types, false_branch_types[i].types, mask_type)
            )
        else:
            element_types_new[i] = promote_to_mask_type(
                mask_type, type_info.promote(element_types_new[i], false_branch_types[i])
            )
    return element_types_new


def promote_to_mask_type(
    mask_type: ts.FieldType, input_type: ts.FieldType | ts.ScalarType
) -> ts.FieldType:
    """
    Promote mask type with the input type.

    The input type being the result of promoting the left and right types in a conditional clause.

    If the input type is a scalar, the return type takes the dimensions of the mask_type, while retaining the dtype of
    the input type. The behavior is similar when the input type is a field type with fewer dimensions than the mask_type.
    In all other cases, the return type takes the dimensions and dtype of the input type.

    >>> from gt4py.next import Dimension
    >>> I, J = (Dimension(value=dim) for dim in ["I", "J"])
    >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
    >>> dtype = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    >>> promote_to_mask_type(ts.FieldType(dims=[I, J], dtype=bool_type), dtype)
    FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None))
    >>> promote_to_mask_type(
    ...     ts.FieldType(dims=[I, J], dtype=bool_type), ts.FieldType(dims=[I], dtype=dtype)
    ... )
    FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None))
    >>> promote_to_mask_type(
    ...     ts.FieldType(dims=[I], dtype=bool_type), ts.FieldType(dims=[I, J], dtype=dtype)
    ... )
    FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None))
    """
    if isinstance(input_type, ts.ScalarType) or not all(
        item in input_type.dims for item in mask_type.dims
    ):
        return_dtype = input_type.dtype if isinstance(input_type, ts.FieldType) else input_type
        return type_info.promote(input_type, ts.FieldType(dims=mask_type.dims, dtype=return_dtype))  # type: ignore
    else:
        return input_type


def deduce_stmt_return_type(
    node: foast.BlockStmt, *, requires_unconditional_return: bool = True
) -> Optional[ts.TypeSpec]:
    """
    Deduce type of value returned inside a block statement.

    If `requires_unconditional_return` is true the function additionally ensures that the block
    statement unconditionally returns and raises an `AssertionError` if not.
    """
    conditional_return_type: Optional[ts.TypeSpec] = None

    for stmt in node.stmts:
        is_unconditional_return = False
        return_type: Optional[ts.TypeSpec]

        if isinstance(stmt, foast.Return):
            is_unconditional_return = True
            return_type = stmt.value.type
        elif isinstance(stmt, foast.IfStmt):
            return_types = (
                deduce_stmt_return_type(stmt.true_branch, requires_unconditional_return=False),
                deduce_stmt_return_type(stmt.false_branch, requires_unconditional_return=False),
            )
            # if both branches return
            if return_types[0] and return_types[1]:
                if return_types[0] == return_types[1]:
                    is_unconditional_return = True
                else:
                    raise errors.DSLError(
                        stmt.location,
                        "If statement contains return statements with inconsistent types:"
                        f"{return_types[0]} != {return_types[1]}",
                    )
            return_type = return_types[0] or return_types[1]
        elif isinstance(stmt, foast.BlockStmt):
            # just forward to nested BlockStmt
            return_type = deduce_stmt_return_type(
                stmt, requires_unconditional_return=requires_unconditional_return
            )
        elif isinstance(stmt, (foast.Assign, foast.TupleTargetAssign)):
            return_type = None
        else:
            raise AssertionError(f"Nodes of type '{type(stmt).__name__}' not supported.")

        if conditional_return_type and return_type and return_type != conditional_return_type:
            raise errors.DSLError(
                stmt.location,
                "If statement contains return statements with inconsistent types:"
                f"{conditional_return_type} != {conditional_return_type}",
            )

        if is_unconditional_return:  # found a statement that always returns
            assert return_type
            return return_type
        elif return_type:
            conditional_return_type = return_type

    if requires_unconditional_return:
        # If the node was constructed by the foast parsing we should never get here, but instead
        # we should have gotten an error there.
        raise AssertionError(
            "Malformed block statement: expected a return statement in this context, "
            "but none was found. Please submit a bug report."
        )

    return None


class FieldOperatorTypeDeductionCompletnessValidator(NodeVisitor):
    """Validate an FOAST expression is fully typed."""

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> None:
        incomplete_nodes: list[foast.LocatedNode] = []
        cls().visit(node, incomplete_nodes=incomplete_nodes)

        if incomplete_nodes:
            raise AssertionError("'FOAST' expression is not fully typed.")

    def visit_LocatedNode(
        self, node: foast.LocatedNode, *, incomplete_nodes: list[foast.LocatedNode]
    ) -> None:
        self.generic_visit(node, incomplete_nodes=incomplete_nodes)

        if hasattr(node, "type") and not type_info.is_concrete(node.type):
            incomplete_nodes.append(node)


class FieldOperatorTypeDeduction(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Deduce and check types of FOAST expressions and symbols.

    Examples:
    ---------
    >>> import ast
    >>> import typing
    >>> from gt4py.next import Field, Dimension
    >>> from gt4py.next.ffront.source_utils import SourceDefinition, get_closure_vars_from_function
    >>> from gt4py.next.ffront.func_to_foast import FieldOperatorParser
    >>> IDim = Dimension("IDim")
    >>> def example(a: "Field[[IDim], float]", b: "Field[[IDim], float]"):
    ...     return a + b

    >>> source_definition = SourceDefinition.from_function(example)
    >>> closure_vars = get_closure_vars_from_function(example)
    >>> annotations = typing.get_type_hints(example)
    >>> untyped_fieldop = FieldOperatorParser(
    ...     source_definition=source_definition, closure_vars=closure_vars, annotations=annotations
    ... ).visit(ast.parse(source_definition.source).body[0])
    >>> untyped_fieldop.body.stmts[0].value.type
    DeferredType(constraint=None)

    >>> typed_fieldop = FieldOperatorTypeDeduction.apply(untyped_fieldop)
    >>> assert typed_fieldop.body.stmts[0].value.type == ts.FieldType(
    ...     dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64), dims=[IDim]
    ... )
    """

    @classmethod
    def apply(cls, node: OperatorNodeT) -> OperatorNodeT:
        typed_foast_node = cls().visit(node)

        FieldOperatorTypeDeductionCompletnessValidator.apply(typed_foast_node)

        return typed_foast_node

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> foast.FunctionDefinition:
        new_params = self.visit(node.params, **kwargs)
        new_body = self.visit(node.body, **kwargs)
        new_closure_vars = self.visit(node.closure_vars, **kwargs)
        return_type = deduce_stmt_return_type(new_body)
        if not isinstance(return_type, (ts.DataType, ts.DeferredType, ts.VoidType)):
            raise errors.DSLError(
                node.location,
                f"Function must return 'DataType', 'DeferredType', or 'VoidType', got '{return_type}'.",
            )
        new_type = ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={str(new_param.id): new_param.type for new_param in new_params},
            kw_only_args={},
            returns=return_type,
        )
        return foast.FunctionDefinition(
            id=node.id,
            params=new_params,
            body=new_body,
            closure_vars=new_closure_vars,
            type=new_type,
            location=node.location,
        )

    # TODO(tehrengruber): make sure all scalar arguments are lifted to 0-dim field args
    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs: Any) -> foast.FieldOperator:
        new_definition = self.visit(node.definition, **kwargs)
        return foast.FieldOperator(
            id=node.id,
            definition=new_definition,
            location=node.location,
            type=ts_ffront.FieldOperatorType(definition=new_definition.type),
        )

    def visit_ScanOperator(self, node: foast.ScanOperator, **kwargs: Any) -> foast.ScanOperator:
        new_axis = self.visit(node.axis, **kwargs)
        if not isinstance(new_axis.type, ts.DimensionType):
            raise errors.DSLError(
                node.location, f"Argument 'axis' to scan operator '{node.id}' must be a dimension."
            )
        if not new_axis.type.dim.kind == DimensionKind.VERTICAL:
            raise errors.DSLError(
                node.location,
                f"Argument 'axis' to scan operator '{node.id}' must be a vertical dimension.",
            )
        new_forward = self.visit(node.forward, **kwargs)
        if not new_forward.type.kind == ts.ScalarKind.BOOL:
            raise errors.DSLError(
                node.location, f"Argument 'forward' to scan operator '{node.id}' must be a boolean."
            )
        new_init = self.visit(node.init, **kwargs)
        if not all(
            type_info.is_arithmetic(type_) or type_info.is_logical(type_)
            for type_ in type_info.primitive_constituents(new_init.type)
        ):
            raise errors.DSLError(
                node.location,
                f"Argument 'init' to scan operator '{node.id}' must "
                "be an arithmetic type or a logical type or a composite of arithmetic and logical types.",
            )
        new_definition = self.visit(node.definition, **kwargs)
        new_def_type = new_definition.type
        carry_type = next(iter(new_def_type.pos_or_kw_args.values()))
        if new_init.type != new_def_type.returns:
            raise errors.DSLError(
                node.location,
                f"Argument 'init' to scan operator '{node.id}' must have same type as its return: "
                f"expected '{new_def_type.returns}', got '{new_init.type}'.",
            )
        elif new_init.type != carry_type:
            carry_arg_name = next(iter(new_def_type.pos_or_kw_args.keys()))
            raise errors.DSLError(
                node.location,
                f"Argument 'init' to scan operator '{node.id}' must have same type as '{carry_arg_name}' argument: "
                f"expected '{carry_type}', got '{new_init.type}'.",
            )

        new_type = ts_ffront.ScanOperatorType(axis=new_axis.type.dim, definition=new_def_type)
        return foast.ScanOperator(
            id=node.id,
            axis=new_axis,
            forward=new_forward,
            init=new_init,
            definition=new_definition,
            type=new_type,
            location=node.location,
        )

    def visit_Name(self, node: foast.Name, **kwargs: Any) -> foast.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable or symtable[node.id].type is None:
            raise errors.DSLError(node.location, f"Undeclared symbol '{node.id}'.")

        symbol = symtable[node.id]
        return foast.Name(id=node.id, type=symbol.type, location=node.location)

    def visit_Assign(self, node: foast.Assign, **kwargs: Any) -> foast.Assign:
        new_value = node.value
        if not type_info.is_concrete(node.value.type):
            new_value = self.visit(node.value, **kwargs)
        new_target = self.visit(node.target, refine_type=new_value.type, **kwargs)
        return foast.Assign(target=new_target, value=new_value, location=node.location)

    def visit_TupleTargetAssign(
        self, node: foast.TupleTargetAssign, **kwargs: Any
    ) -> foast.TupleTargetAssign:
        TargetType: TypeAlias = list[foast.Starred | foast.Symbol]
        values = self.visit(node.value, **kwargs)

        if isinstance(values.type, ts.TupleType):
            num_elts: int = len(values.type.types)
            targets: TargetType = node.targets
            indices: list[tuple[int, int] | int] = foast_utils.compute_assign_indices(
                targets, num_elts
            )

            if not any(isinstance(i, tuple) for i in indices) and len(targets) != num_elts:
                raise errors.DSLError(
                    node.location, f"Too many values to unpack (expected {len(targets)})."
                )

            new_targets: TargetType = []
            new_type: ts.DataType
            for i, index in enumerate(indices):
                old_target = targets[i]

                if isinstance(index, tuple):
                    lower, upper = index
                    new_type = ts.TupleType(types=[t for t in values.type.types[lower:upper]])
                    new_target = foast.Starred(
                        id=foast.DataSymbol(
                            id=self.visit(old_target.id).id,
                            location=old_target.location,
                            type=new_type,
                        ),
                        type=new_type,
                        location=old_target.location,
                    )
                else:
                    new_type = values.type.types[index]  # type: ignore[assignment] # see check in next line
                    assert isinstance(new_type, ts.DataType)
                    new_target = self.visit(
                        old_target, refine_type=new_type, location=old_target.location, **kwargs
                    )

                new_target = self.visit(
                    new_target, refine_type=new_type, location=old_target.location, **kwargs
                )
                new_targets.append(new_target)
        else:
            raise errors.DSLError(
                node.location, f"Assignment value must be of type tuple, got '{values.type}'."
            )

        return foast.TupleTargetAssign(targets=new_targets, value=values, location=node.location)

    def visit_IfStmt(self, node: foast.IfStmt, **kwargs: Any) -> foast.IfStmt:
        symtable = kwargs["symtable"]

        new_true_branch = self.visit(node.true_branch, **kwargs)
        new_false_branch = self.visit(node.false_branch, **kwargs)
        new_node = foast.IfStmt(
            condition=self.visit(node.condition, **kwargs),
            true_branch=new_true_branch,
            false_branch=new_false_branch,
            location=node.location,
        )

        if not isinstance(new_node.condition.type, ts.ScalarType):
            raise errors.DSLError(
                node.location,
                f"Condition for 'if' must be scalar, got '{new_node.condition.type}' instead.",
            )

        if new_node.condition.type.kind != ts.ScalarKind.BOOL:
            raise errors.DSLError(
                node.location,
                "Condition for 'if' must be of boolean type, "
                f"got '{new_node.condition.type}' instead.",
            )

        for sym in node.annex.propagated_symbols.keys():
            if (true_type := new_true_branch.annex.symtable[sym].type) != (
                false_type := new_false_branch.annex.symtable[sym].type
            ):
                raise errors.DSLError(
                    node.location,
                    f"Inconsistent types between two branches for variable '{sym}': "
                    f"got types '{true_type}' and '{false_type}.",
                )
            # TODO: properly patch symtable (new node?)
            symtable[sym].type = new_node.annex.propagated_symbols[sym].type = (
                new_true_branch.annex.symtable[sym].type
            )

        return new_node

    def visit_Symbol(
        self, node: foast.Symbol, refine_type: Optional[ts.FieldType] = None, **kwargs: Any
    ) -> foast.Symbol:
        symtable = kwargs["symtable"]
        if refine_type:
            if not type_info.is_concretizable(node.type, to_type=refine_type):
                raise errors.DSLError(
                    node.location,
                    (
                        "Type inconsistency: expression was deduced to be "
                        f"of type '{refine_type}', instead of the expected type '{node.type}'."
                    ),
                )
            new_node: foast.Symbol = foast.Symbol(
                id=node.id, type=refine_type, location=node.location
            )
            symtable[new_node.id] = new_node
            return new_node
        return node

    def visit_Attribute(self, node: foast.Attribute, **kwargs: Any) -> foast.Attribute:
        new_value = self.visit(node.value, **kwargs)
        return foast.Attribute(
            value=new_value,
            attr=node.attr,
            location=node.location,
            type=getattr(new_value.type, node.attr),
        )

    def visit_Subscript(self, node: foast.Subscript, **kwargs: Any) -> foast.Subscript:
        new_value = self.visit(node.value, **kwargs)
        new_index = self.visit(node.index, **kwargs)
        new_type: Optional[ts.TypeSpec] = None

        match new_value.type:
            case ts.TupleType(types=types):
                try:
                    index = foast_utils.expr_to_index(node.index)
                except ValueError as ex:
                    raise errors.DSLError(
                        node.location,
                        f"Tuples need to be indexed with literal integers, got '{node.index}'.",
                    ) from ex
                new_type = types[index]
            case ts.OffsetType(source=source, target=(target1, target2)):
                if not target2.kind == DimensionKind.LOCAL:
                    raise errors.DSLError(
                        new_value.location, "Second dimension in offset must be a local dimension."
                    )
                new_type = ts.OffsetType(source=source, target=(target1,))
            case ts.OffsetType(source=source, target=(target,)):
                # for cartesian axes (e.g. I, J) the index of the subscript only
                #  signifies the displacement in the respective dimension,
                #  but does not change the target type.
                if source != target:
                    raise errors.DSLError(
                        new_value.location,
                        "Source and target must be equal for offsets with a single target.",
                    )
                new_type = new_value.type
            case ts.FieldType(dims=dims, dtype=dtype):
                # e.g. `field[LocalDim(42)]`
                new_type = ts.FieldType(
                    dims=[d for d in dims if d != new_index.type.dim],
                    dtype=dtype,
                )
            case _:
                raise errors.DSLError(
                    new_value.location, "Could not deduce type of subscript expression."
                )

        return foast.Subscript(
            value=new_value,
            index=new_index,
            type=new_type,
            location=node.location,
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs: Any) -> foast.BinOp:
        new_left = self.visit(node.left, **kwargs)
        new_right = self.visit(node.right, **kwargs)
        new_type = self._deduce_binop_type(node, left=new_left, right=new_right)
        return foast.BinOp(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs: Any) -> foast.TernaryExpr:
        new_condition = self.visit(node.condition, **kwargs)
        new_true_expr = self.visit(node.true_expr, **kwargs)
        new_false_expr = self.visit(node.false_expr, **kwargs)
        new_type = self._deduce_ternaryexpr_type(
            node, condition=new_condition, true_expr=new_true_expr, false_expr=new_false_expr
        )
        return foast.TernaryExpr(
            condition=new_condition,
            true_expr=new_true_expr,
            false_expr=new_false_expr,
            location=node.location,
            type=new_type,
        )

    def _deduce_ternaryexpr_type(
        self,
        node: foast.TernaryExpr,
        *,
        condition: foast.Expr,
        true_expr: foast.Expr,
        false_expr: foast.Expr,
    ) -> Optional[ts.TypeSpec]:
        if condition.type != ts.ScalarType(kind=ts.ScalarKind.BOOL):
            raise errors.DSLError(
                condition.location,
                f"Condition is of type '{condition.type}', should be of type 'bool'.",
            )

        if true_expr.type != false_expr.type:
            raise errors.DSLError(
                node.location,
                f"Left and right types are not the same: '{true_expr.type}' and '{false_expr.type}'",
            )
        return true_expr.type

    def visit_Compare(self, node: foast.Compare, **kwargs: Any) -> foast.Compare:
        new_left = self.visit(node.left, **kwargs)
        new_right = self.visit(node.right, **kwargs)
        new_type = self._deduce_compare_type(node, left=new_left, right=new_right)
        return foast.Compare(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def _deduce_arithmetic_compare_type(
        self, node: foast.Compare, *, left: foast.Expr, right: foast.Expr, **kwargs: Any
    ) -> Optional[ts.TypeSpec]:
        # e.g. `1 < 2`
        self._check_operand_dtypes_match(node, left=left, right=right)

        try:
            # transform operands to have bool dtype and use regular promotion
            #  mechanism to handle dimension promotion
            return type_info.promote(
                with_altered_scalar_kind(left.type, ts.ScalarKind.BOOL),
                with_altered_scalar_kind(right.type, ts.ScalarKind.BOOL),
            )
        except ValueError as ex:
            raise errors.DSLError(
                node.location,
                f"Could not promote '{left.type}' and '{right.type}' to common type"
                f" in call to '{node.op}'.",
            ) from ex

    def _deduce_dimension_compare_type(
        self, node: foast.Compare, *, left: foast.Expr, right: foast.Expr, **kwargs: Any
    ) -> Optional[ts.TypeSpec]:
        # e.g. `IDim > 1`
        index_type = ts.ScalarType(
            kind=getattr(ts.ScalarKind, builtins.INTEGER_INDEX_BUILTIN.upper())
        )

        def error_msg(left: ts.TypeSpec, right: ts.TypeSpec) -> str:
            return f"Dimension comparison needs to be between a 'Dimension' and index of type '{index_type}', got '{left}' and '{right}'."

        if isinstance(left.type, ts.DimensionType):
            if not right.type == index_type:
                raise errors.DSLError(
                    right.location,
                    error_msg(left.type, right.type),
                )
            return ts.DomainType(dims=[left.type.dim])
        elif isinstance(right.type, ts.DimensionType):
            if not left.type == index_type:
                raise errors.DSLError(
                    left.location,
                    error_msg(left.type, right.type),
                )
            return ts.DomainType(dims=[right.type.dim])
        else:
            raise AssertionError()

    def _deduce_compare_type(
        self, node: foast.Compare, *, left: foast.Expr, right: foast.Expr, **kwargs: Any
    ) -> Optional[ts.TypeSpec]:
        # e.g. `1 < 1`
        if all(type_info.is_arithmetic(arg) for arg in (left.type, right.type)):
            return self._deduce_arithmetic_compare_type(node, left=left, right=right)
        # e.g. `IDim > 1`
        if any(isinstance(arg, ts.DimensionType) for arg in (left.type, right.type)):
            return self._deduce_dimension_compare_type(node, left=left, right=right)

        raise errors.DSLError(
            left.location,
            "Comparison operators can only be used between arithmetic types "
            "(scalars, fields) or between a dimension and an index type "
            "({builtins.INTEGER_INDEX_BUILTIN}).",
        )

    def _deduce_binop_type(
        self, node: foast.BinOp, *, left: foast.Expr, right: foast.Expr, **kwargs: Any
    ) -> Optional[ts.TypeSpec]:
        # e.g. `IDim+1`
        if (
            isinstance(left.type, ts.DimensionType)
            and isinstance(right.type, ts.ScalarType)
            and type_info.is_integral(right.type)
        ):
            return ts.OffsetType(source=left.type.dim, target=(left.type.dim,))
        if isinstance(left.type, ts.OffsetType):
            raise errors.DSLError(
                node.location, f"Type '{left.type}' can not be used in operator '{node.op}'."
            )

        logical_ops = {
            dialect_ast_enums.BinaryOperator.BIT_AND,
            dialect_ast_enums.BinaryOperator.BIT_OR,
            dialect_ast_enums.BinaryOperator.BIT_XOR,
        }

        err_msg = f"Unsupported operand type(s) for {node.op}: '{left.type}' and '{right.type}'."

        if isinstance(left.type, (ts.ScalarType, ts.FieldType)) and isinstance(
            right.type, (ts.ScalarType, ts.FieldType)
        ):
            is_compatible = (
                type_info.is_logical if node.op in logical_ops else type_info.is_arithmetic
            )
            for arg in (left, right):
                if not is_compatible(arg.type):
                    raise errors.DSLError(arg.location, err_msg)

            if node.op == dialect_ast_enums.BinaryOperator.POW:
                return left.type

            if node.op == dialect_ast_enums.BinaryOperator.MOD and not type_info.is_integral(
                right.type
            ):
                raise errors.DSLError(
                    arg.location,
                    f"Type '{right.type}' can not be used in operator '{node.op}', it only accepts 'int'.",
                )

            try:
                return type_info.promote(left.type, right.type)
            except ValueError as ex:
                raise errors.DSLError(
                    node.location,
                    f"Could not promote '{left.type}' and '{right.type}' to common type"
                    f" in call to '{node.op}'.",
                ) from ex
        elif isinstance(left.type, ts.DomainType) and isinstance(right.type, ts.DomainType):
            if node.op not in logical_ops:
                raise errors.DSLError(
                    node.location,
                    f"{err_msg} Operator "
                    f"must be one of {', '.join((str(op) for op in logical_ops))}.",
                )
            return ts.DomainType(dims=promote_dims(left.type.dims, right.type.dims))
        else:
            raise errors.DSLError(node.location, err_msg)

    def _check_operand_dtypes_match(
        self, node: foast.BinOp | foast.Compare, left: foast.Expr, right: foast.Expr
    ) -> None:
        # check dtypes match
        if not type_info.extract_dtype(left.type) == type_info.extract_dtype(right.type):
            raise errors.DSLError(
                node.location,
                f"Incompatible datatypes in operator '{node.op}': '{left.type}' and '{right.type}'.",
            )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs: Any) -> foast.UnaryOp:
        new_operand = self.visit(node.operand, **kwargs)
        is_compatible = (
            type_info.is_logical
            if node.op
            in [dialect_ast_enums.UnaryOperator.NOT, dialect_ast_enums.UnaryOperator.INVERT]
            else type_info.is_arithmetic
        )
        if not is_compatible(new_operand.type):
            raise errors.DSLError(
                node.location,
                f"Incompatible type for unary operator '{node.op}': '{new_operand.type}'.",
            )
        return foast.UnaryOp(
            op=node.op, operand=new_operand, location=node.location, type=new_operand.type
        )

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs: Any) -> foast.TupleExpr:
        new_elts = self.visit(node.elts, **kwargs)
        new_type = ts.TupleType(types=[element.type for element in new_elts])
        return foast.TupleExpr(elts=new_elts, type=new_type, location=node.location)

    def visit_Call(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        new_func = self.visit(node.func, **kwargs)
        new_args = self.visit(node.args, **kwargs)
        new_kwargs = self.visit(node.kwargs, **kwargs)

        func_type = new_func.type
        arg_types = [arg.type for arg in new_args]
        kwarg_types = {name: arg.type for name, arg in new_kwargs.items()}

        if isinstance(
            new_func.type,
            (
                ts.FunctionType,
                ts_ffront.FieldOperatorType,
                ts_ffront.ScanOperatorType,
                ts.ConstructorType,
            ),
        ):
            # Since we use the `id` attribute in the latter part of the toolchain ensure we
            # have the proper format here.
            if not isinstance(
                new_func,
                (foast.FunctionDefinition, foast.FieldOperator, foast.ScanOperator, foast.Name),
            ):
                raise errors.DSLError(node.location, "Functions can only be called directly.")
        elif isinstance(new_func.type, ts.FieldType):
            pass
        elif isinstance(new_func.type, ts.DimensionType):
            assert new_func.type.dim.kind == DimensionKind.LOCAL
            return foast.Call(
                func=new_func,
                args=new_args,
                kwargs=new_kwargs,
                location=node.location,
                type=ts.IndexType(dim=new_func.type.dim),
            )
        else:
            raise errors.DSLError(
                node.location,
                f"Expression of type '{new_func.type}' is not callable, must be a 'Function', 'FieldOperator', 'ScanOperator' or 'Field'.",
            )

        # ensure signature is valid
        try:
            type_info.accepts_args(
                func_type, with_args=arg_types, with_kwargs=kwarg_types, raise_exception=True
            )
        except ValueError as err:
            raise errors.DSLError(
                node.location, f"Invalid argument types in call to '{new_func}'.\n{err}"
            ) from err

        return_type = type_info.return_type(func_type, with_args=arg_types, with_kwargs=kwarg_types)

        new_node = foast.Call(
            func=new_func,
            args=new_args,
            kwargs=new_kwargs,
            location=node.location,
            type=return_type,
        )

        if (
            isinstance(new_func.type, ts.FunctionType)
            and isinstance(new_func, foast.Name)
            and new_func.id in fbuiltins.MATH_BUILTIN_NAMES
        ):
            return self._visit_math_built_in(new_node, **kwargs)
        elif (
            isinstance(new_func.type, ts.FunctionType)
            and not type_info.is_concrete(return_type)
            and isinstance(new_func, foast.Name)
            and new_func.id
            in (fbuiltins.FUN_BUILTIN_NAMES + experimental.EXPERIMENTAL_FUN_BUILTIN_NAMES)
        ):
            visitor = getattr(self, f"_visit_{new_func.id}")
            return visitor(new_node, **kwargs)

        return new_node

    def _visit_math_built_in(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        func_name = cast(foast.Name, node.func).id

        # validate arguments
        error_msg_preamble = f"Incompatible argument in call to '{func_name}'."
        error_msg_for_validator = {
            type_info.is_arithmetic: "an arithmetic",
            type_info.is_floating_point: "a floating point",
        }
        if func_name in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES:
            arg_validator = type_info.is_arithmetic
        elif func_name in fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES:
            arg_validator = type_info.is_floating_point
        elif func_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
            arg_validator = type_info.is_floating_point
        elif func_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
            arg_validator = type_info.is_arithmetic
        else:
            raise AssertionError(f"Unknown math builtin '{func_name}'.")

        error_msgs = []
        for i, arg in enumerate(node.args):
            if not arg_validator(arg.type):
                error_msgs.append(
                    f"Expected {i}-th argument to be {error_msg_for_validator[arg_validator]} type, got '{arg.type}'."
                )
        if error_msgs:
            raise errors.DSLError(
                node.location,
                "\n".join([error_msg_preamble] + [f"  - {error}" for error in error_msgs]),
            )

        if func_name == "power" and all(type_info.is_integral(arg.type) for arg in node.args):
            print(f"Warning: return type of '{func_name}' might be inconsistent (not implemented).")

        # deduce return type
        return_type: Optional[ts.FieldType | ts.ScalarType] = None
        if (
            func_name
            in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES + fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
        ):
            return_type = cast(ts.FieldType | ts.ScalarType, node.args[0].type)
        elif func_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
            return_type = with_altered_scalar_kind(
                cast(ts.FieldType | ts.ScalarType, node.args[0].type), ts.ScalarKind.BOOL
            )
        elif func_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
            try:
                return_type = type_info.promote(
                    *((cast(ts.FieldType | ts.ScalarType, arg.type)) for arg in node.args)
                )
            except ValueError as ex:
                raise errors.DSLError(node.location, error_msg_preamble) from ex
        else:
            raise AssertionError(f"Unknown math builtin '{func_name}'.")

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def _visit_reduction(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        field_type = cast(ts.FieldType, node.args[0].type)
        reduction_dim = cast(ts.DimensionType, node.kwargs["axis"].type).dim
        # TODO: This code does not handle ellipses for dimensions. Fix it.
        assert field_type.dims is not ...
        if reduction_dim not in field_type.dims:
            field_dims_str = ", ".join(str(dim) for dim in field_type.dims)
            raise errors.DSLError(
                node.location,
                f"Incompatible field argument in call to '{node.func!s}'. "
                f"Expected a field with dimension '{reduction_dim}', got "
                f"'{field_dims_str}'.",
            )
        return_type = ts.FieldType(
            dims=[dim for dim in field_type.dims if dim != reduction_dim], dtype=field_type.dtype
        )

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_min_over(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_astype(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        value, new_type_constructor = node.args
        assert isinstance(
            value.type, (ts.FieldType, ts.ScalarType, ts.TupleType)
        )  # already checked using generic mechanism

        # Note: the type to convert to is uniquely identified by its GT4Py type (`ConstructorType`),
        # not by e.g. its name.
        if not (
            isinstance(new_type_constructor.type, ts.ConstructorType)
            and isinstance(new_type_constructor.type.definition.returns, ts.ScalarType)
        ):
            raise errors.DSLError(
                node.location,
                f"Invalid call to 'astype': second argument must be a scalar type, got '{new_type_constructor}'.",
            )

        new_type = new_type_constructor.type.definition.returns

        return_type = type_info.apply_to_primitive_constituents(
            lambda primitive_type: with_altered_scalar_kind(primitive_type, new_type.kind),
            value.type,
        )
        assert isinstance(return_type, (ts.TupleType, ts.ScalarType, ts.FieldType))

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            type=return_type,
            location=node.location,
        )

    def _visit_as_offset(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        arg_0 = node.args[0].type
        arg_1 = node.args[1].type
        assert isinstance(arg_0, ts.OffsetType)
        assert isinstance(arg_1, ts.FieldType)
        if not type_info.is_integral(arg_1):
            raise errors.DSLError(
                node.location,
                f"Incompatible argument in call to '{node.func!s}': "
                f"expected integer for offset field dtype, got '{arg_1.dtype}'. "
                f"{node.location}",
            )

        if arg_0.source not in arg_1.dims:
            raise errors.DSLError(
                node.location,
                f"Incompatible argument in call to '{node.func!s}': "
                f"'{arg_0.source}' not in list of offset field dimensions '{arg_1.dims}'. "
                f"{node.location}",
            )

        return foast.Call(
            func=node.func, args=node.args, kwargs=node.kwargs, type=arg_0, location=node.location
        )

    def _visit_where(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        mask_type = cast(ts.FieldType, node.args[0].type)
        true_branch_type = node.args[1].type
        false_branch_type = node.args[2].type
        return_type: ts.TupleType | ts.FieldType
        if not type_info.is_logical(mask_type):
            raise errors.DSLError(
                node.location,
                f"Incompatible argument in call to '{node.func!s}': expected "
                f"a field with dtype 'bool', got '{mask_type}'.",
            )

        try:
            # TODO(tehrengruber): the construct_tuple_type function doesn't look correct
            if isinstance(true_branch_type, ts.TupleType) and isinstance(
                false_branch_type, ts.TupleType
            ):
                return_type = ts.TupleType(
                    types=construct_tuple_type(
                        true_branch_type.types, false_branch_type.types, mask_type
                    )
                )
            elif isinstance(true_branch_type, ts.TupleType) or isinstance(
                false_branch_type, ts.TupleType
            ):
                raise errors.DSLError(
                    node.location,
                    f"Return arguments need to be of same type in '{node.func!s}', got "
                    f"'{node.args[1].type}' and '{node.args[2].type}'.",
                )
            else:
                true_branch_fieldtype = cast(ts.FieldType, true_branch_type)
                false_branch_fieldtype = cast(ts.FieldType, false_branch_type)
                promoted_type = type_info.promote(true_branch_fieldtype, false_branch_fieldtype)
                return_type = promote_to_mask_type(mask_type, promoted_type)

        except ValueError as ex:
            raise errors.DSLError(
                node.location, f"Incompatible argument in call to '{node.func!s}'."
            ) from ex

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            type=return_type,
            location=node.location,
        )

    def _visit_concat_where(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        cond_type, true_branch_type, false_branch_type = (arg.type for arg in node.args)

        assert isinstance(cond_type, ts.DomainType)
        assert all(
            isinstance(el, (ts.FieldType, ts.ScalarType))
            for arg in (true_branch_type, false_branch_type)
            for el in type_info.primitive_constituents(arg)
        )

        @utils.tree_map(
            collection_type=ts.TupleType,
            result_collection_constructor=lambda _, elts: ts.TupleType(types=list(elts)),
        )
        def deduce_return_type(
            tb: ts.FieldType | ts.ScalarType, fb: ts.FieldType | ts.ScalarType
        ) -> ts.FieldType:
            if (t_dtype := type_info.extract_dtype(tb)) != (f_dtype := type_info.extract_dtype(fb)):
                raise errors.DSLError(
                    node.location,
                    f"Field arguments must be of same dtype, got '{t_dtype}' != '{f_dtype}'.",
                )
            return_dims = promote_dims(
                cond_type.dims, type_info.extract_dims(type_info.promote(tb, fb))
            )
            return_type = ts.FieldType(dims=return_dims, dtype=t_dtype)
            return return_type

        return_type = deduce_return_type(true_branch_type, false_branch_type)

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            type=return_type,
            location=node.location,
        )

    def _visit_broadcast(self, node: foast.Call, **kwargs: Any) -> foast.Call:
        arg_type = cast(ts.FieldType | ts.ScalarType, node.args[0].type)
        broadcast_dims_expr = cast(foast.TupleExpr, node.args[1]).elts

        if any([not (isinstance(elt.type, ts.DimensionType)) for elt in broadcast_dims_expr]):
            raise errors.DSLError(
                node.location,
                f"Incompatible broadcast dimension type in '{node.func!s}': expected "
                f"all broadcast dimensions to be of type 'Dimension'.",
            )

        broadcast_dims = [cast(ts.DimensionType, elt.type).dim for elt in broadcast_dims_expr]

        if not set((arg_dims := type_info.extract_dims(arg_type))).issubset(set(broadcast_dims)):
            raise errors.DSLError(
                node.location,
                f"Incompatible broadcast dimensions in '{node.func!s}': expected "
                f"broadcast dimension(s) '{set(arg_dims).difference(set(broadcast_dims))}' missing",
            )

        return_type = ts.FieldType(dims=broadcast_dims, dtype=type_info.extract_dtype(arg_type))

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def visit_Constant(self, node: foast.Constant, **kwargs: Any) -> foast.Constant:
        try:
            type_ = type_translation.from_value(node.value)
        except ValueError as e:
            raise errors.DSLError(node.location, "Could not deduce type of constant.") from e
        return foast.Constant(value=node.value, location=node.location, type=type_)

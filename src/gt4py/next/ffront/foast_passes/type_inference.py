import typing

from gt4py.next.ffront import field_operator_ast as foast, fbuiltins
from gt4py import eve
from gt4py.next import errors
from gt4py.next.type_system_2 import types as ts2
from gt4py.next.ffront.type_system_2 import types as ts2_f, inference as ti2_f
from gt4py.next.type_system_2 import traits
from typing import Optional, Any
from gt4py.next import common as gtx_common
import dataclasses


@dataclasses.dataclass(frozen=True)
class ClosureVarInferencePass(eve.NodeTranslator, eve.traits.VisitorWithSymbolTableTrait):
    closure_vars: dict[str, Any]

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs) -> foast.FunctionDefinition:
        new_closure_vars: list[foast.Symbol] = []
        for sym in node.closure_vars:
            ty = ti2_f.inferrer.from_instance(self.closure_vars[sym.id])
            if ty is None:
                raise errors.DSLError(sym.location, f"could not infer type of captured variable '{sym.id}'")
            new_symbol: foast.Symbol = foast.Symbol(
                id=sym.id,
                location=sym.location,
                type=sym.type,
                type_2=ty,
            )
            new_closure_vars.append(new_symbol)
        return foast.FunctionDefinition(
            id=node.id,
            params=node.params,
            body=node.body,
            closure_vars=new_closure_vars,
            type_2=node.type_2,
            location=node.location,
        )


class TypeInferencePass(eve.traits.VisitorWithSymbolTableTrait, eve.NodeTranslator):
    result: Optional[ts2.Type]

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> foast.FieldOperator:
        definition: foast.FunctionDefinition = self.visit(node.definition)
        definition_t = definition.type_2
        assert isinstance(definition_t, ts2.FunctionType)
        ty = ts2_f.FieldOperatorType(definition_t.parameters, definition_t.result)
        return foast.FieldOperator(
            id=node.id,
            definition=definition,
            type_2=ty,
            location=node.location,
        )

    def visit_ScanOperator(self, node: foast.ScanOperator, **kwargs) -> foast.ScanOperator:
        definition: foast.FunctionDefinition = self.visit(node.definition)
        axis: foast.Expr = self.visit(node.axis)
        forward: foast.Expr = self.visit(node.forward)
        init: foast.Expr = self.visit(node.init)
        definition_t = definition.type_2
        axis_t = axis.type_2
        assert isinstance(definition_t, ts2.FunctionType)
        assert isinstance(axis_t, ts2_f.DimensionType)
        ty = ts2_f.ScanOperatorType(
            axis_t.dimension,
            definition_t.parameters[0].ty,
            definition_t.parameters[1:],
            definition_t.result
        )
        return foast.ScanOperator(
            id=node.id,
            definition=definition,
            axis=axis,
            forward=forward,
            init=init,
            type_2=ty,
            location=node.location,
        )

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs) -> foast.FunctionDefinition:
        ty = node.type_2
        assert isinstance(ty, ts2.FunctionType)
        self.result = ty.result
        return foast.FunctionDefinition(
            id=node.id,
            params=[self.visit(item, **kwargs) for item in node.params],
            body=self.visit(node.body, **kwargs),
            closure_vars=[self.visit(item, **kwargs) for item in node.closure_vars],
            type_2=node.type_2,
            location=node.location,
        )

    def visit_Call(self, node: foast.Call, **kwargs) -> foast.Call:
        func: foast.Expr = self.visit(node.func, **kwargs)
        positionals: list = self.visit(node.args, **kwargs)
        keywords: dict = self.visit(node.kwargs, **kwargs)

        func_t = func.type_2
        args = [
            *(traits.FunctionArgument(arg.type_2, idx) for idx, arg in enumerate(positionals)),
            *(traits.FunctionArgument(arg.type_2, name) for name, arg in keywords.items())
        ]

        if isinstance(func_t, traits.CallableTrait):
            is_callable, result_or_err = func_t.is_callable(args)
            if not is_callable:
                raise errors.DSLError(node.location, f"invalid arguments to call: {result_or_err}")
            ty = result_or_err
            return foast.Call(func=func, args=positionals, kwargs=keywords, type_2=ty, location=node.location)
        raise errors.DSLError(func.location, f"'{func_t}' is not callable")

    def visit_Return(self, node: foast.Return, **kwargs) -> foast.Return:
        value: foast.Expr = self.visit(node.value, **kwargs)
        value_t = value.type_2
        if value_t != self.result and not traits.is_implicitly_convertible(value_t, self.result):
            message = f"could not implicitly convert '{value.type_2}' to function return type '{self.result}'"
            supplement = ", use an explicit cast" if traits.is_convertible(value.type_2, self.result) else ""
            raise errors.DSLError(node.location, message + supplement)
        return foast.Return(value=value, type_2=value.type_2, location=node.location)

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> foast.Symbol:
        symtable = kwargs["symtable"]
        symtable[node.id] = node
        return node

    def visit_Name(self, node: foast.Name, **kwargs) -> foast.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable:
            raise errors.UndefinedSymbolError(node.location, node.id)
        symbol: foast.Symbol = symtable[node.id]
        assert symbol.type_2 is not None, f"{node.id}: {node.location}"
        return foast.Name(id=node.id, type_2=symbol.type_2, location=node.location)

    def visit_Assign(self, node: foast.Assign, **kwargs) -> foast.Assign:
        value = self.visit(node.value, **kwargs)
        target = self.visit(node.target, **kwargs)
        target.type_2 = value.type_2
        return foast.Assign(target=target, value=value, type_2=value.type_2, location=node.location)

    def visit_TupleTargetAssign(self, node: foast.TupleTargetAssign, **kwargs) -> foast.TupleTargetAssign:
        targets = [self.visit(target, **kwargs) for target in node.targets]
        value: foast.Expr = self.visit(node.value, **kwargs)
        if not isinstance(value.type_2, ts2.TupleType):
            raise errors.DSLError(value.location, "expected a tuple")

        starred_indices = [idx for idx, tar in enumerate(targets) if isinstance(tar, foast.Starred)]
        if len(starred_indices) > 1:
            message = "expected at most one starred expression"
            raise errors.DSLError(targets[starred_indices[1]], message)

        tys = value.type_2.elements
        starred_count = len(tys) - len(targets) + 1 if starred_indices else 0
        if starred_count < 0:
            raise errors.DSLError(node.location, "not enough values to unpack")
        if not starred_indices and len(targets) != len(tys):
            raise errors.DSLError(node.location, "too many values to unpack")

        ty_idx = 0
        for idx, target in enumerate(targets):
            if starred_indices and idx == starred_indices[0]:
                target.id.type_2 = ts2.TupleType(tys[ty_idx:ty_idx+starred_count])
                target.type_2 = target.id.type_2
                ty_idx += starred_count
            else:
                target.type_2 = tys[ty_idx]
                ty_idx += 1
        return foast.TupleTargetAssign(targets=targets, value=value, type_2=value.type_2, location=node.location)

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> foast.TupleExpr:
        elements = self.visit(node.elts, **kwargs)
        ty = ts2.TupleType([element.type_2 for element in elements])
        return foast.TupleExpr(elts=elements, type_2=ty, location=node.location)

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> foast.Subscript:
        value = self.visit(node.value, **kwargs)
        value_t = value.type_2
        if isinstance(value_t, ts2.TupleType):
            ty = value_t.elements[node.index]
            return foast.Subscript(value=value, index=node.index, type_2=ty, location=node.location)
        if isinstance(value_t, ts2_f.FieldOffsetType):
            if len(value_t.field_offset.target) == 2:
                if value_t.field_offset.target[1].kind != gtx_common.DimensionKind.LOCAL:
                    message = "expected a local dimension for FieldOffset.target[1]"
                    raise errors.DSLError(value.location, message)
                name = value_t.field_offset.value
                src = value_t.field_offset.source
                tar = (value_t.field_offset.target[0],)
                conn = value_t.field_offset.connectivity
                fo = fbuiltins.FieldOffset(value=name, source=src, target=tar, connectivity=conn)
                ty = ts2_f.FieldOffsetType(fo)
            elif len(value_t.field_offset.target) == 1:
                if value_t.field_offset.source != value_t.field_offset.target[0]:
                    message = "expected source and target dimensions to be the same for cartesian offsets"
                    raise errors.DSLError(value.location, message)
                ty = value_t
            else:
                message = "invalid field offset"
                raise errors.DSLError(value.location, message)
            return foast.Subscript(value=value, index=node.index, type_2=ty, location=node.location)
        raise errors.DSLError(node.location, f"'{value.type_2}' is not subscriptable")

    def visit_IfStmt(self, node: foast.IfStmt, **kwargs) -> foast.IfStmt:
        symtable = kwargs["symtable"]

        condition = self.visit(node.condition, **kwargs)
        true_branch = self.visit(node.true_branch, **kwargs)
        false_branch = self.visit(node.false_branch, **kwargs)
        result = foast.IfStmt(
            condition=condition,
            true_branch=true_branch,
            false_branch=false_branch,
            location=node.location,
        )

        if not traits.is_implicitly_convertible(condition.type_2, ts2.BoolType()):
            message = f"could not implicitly convert from '{condition.type_2}' to '{ts2.BoolType()}'"
            raise errors.DSLError(condition.location, message)

        for sym in node.annex.propagated_symbols.keys():
            true_branch_ty = true_branch.annex.symtable[sym].type_2
            false_branch_ty = false_branch.annex.symtable[sym].type_2
            if true_branch_ty != false_branch_ty:
                message = (f"'{sym}' has type '{true_branch_ty}' in the 'then' branch"
                           f" but type {false_branch_ty} in the 'else' branch;"
                           f" types must be the same")
                raise errors.DSLError(true_branch.annex.symtable[sym].location, message)
            # TODO: properly patch symtable (new node?)
            symtable[sym].type = result.annex.propagated_symbols[
                sym
            ].type = true_branch.annex.symtable[sym].type

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> foast.UnaryOp:
        from gt4py.next.ffront.dialect_ast_enums import UnaryOperator

        operand: foast.Expr = self.visit(node.operand, **kwargs)
        operand_t = operand.type_2
        if node.op in [UnaryOperator.NOT, UnaryOperator.INVERT]:
            message = f"'{operand_t}' does not support bitwise operations"
            if not isinstance(operand_t, traits.BitwiseTrait) or not operand_t.supports_bitwise():
                raise errors.DSLError(operand.location, message)
        elif node.op == UnaryOperator.USUB:
            if isinstance(operand_t, ts2.IntegerType) and not operand_t.signed:
                required_width = 2 * operand_t.width
                if required_width > 64:
                    message = f"negating a value of type '{operand_t}' may overflow"
                    raise errors.DSLError(node.location, message)
                operand_t = ts2.IntegerType(required_width, True)

        return foast.UnaryOp(operand=operand, op=node.op, type_2=operand_t, location=node.location)

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> foast.BinOp:
        from gt4py.next.ffront.dialect_ast_enums import BinaryOperator

        bitwise_ops = [BinaryOperator.BIT_AND, BinaryOperator.BIT_OR, BinaryOperator.BIT_XOR]

        lhs: foast.Expr = self.visit(node.left, **kwargs)
        rhs: foast.Expr = self.visit(node.right, **kwargs)
        lhs_t = lhs.type_2
        rhs_t = rhs.type_2
        if node.op in bitwise_ops:
            if not isinstance(lhs_t, traits.BitwiseTrait) or not lhs_t.supports_bitwise():
                message = f"'{lhs_t}' does not support bitwise operations"
                raise errors.DSLError(lhs.location, message)
            if not isinstance(rhs_t, traits.BitwiseTrait) or not rhs_t.supports_bitwise():
                message = f"'{rhs_t}' does not support bitwise operations"
                raise errors.DSLError(rhs.location, message)
            ty = traits.common_bitwise_type(lhs_t, rhs_t)
        else:
            if not isinstance(lhs_t, traits.ArithmeticTrait) or not lhs_t.supports_arithmetic():
                message = f"'{lhs_t}' does not support arithmetic operations"
                raise errors.DSLError(lhs.location, message)
            if not isinstance(rhs_t, traits.ArithmeticTrait) or not rhs_t.supports_arithmetic():
                message = f"'{rhs_t}' does not support arithmetic operations"
                raise errors.DSLError(rhs.location, message)
            ty = traits.common_arithmetic_type(lhs_t, rhs_t)
        if ty is None:
            message = f"no matching operator '{node.op}' for operand types '{lhs_t}', '{rhs_t}'"
            raise errors.DSLError(node.location, message)

        return foast.BinOp(left=lhs, right=rhs, op=node.op, type_2=ty, location=node.location)

    def visit_Compare(self, node: foast.Compare, **kwargs) -> foast.Compare:
        lhs: foast.Expr = self.visit(node.left, **kwargs)
        rhs: foast.Expr = self.visit(node.right, **kwargs)
        lhs_t = lhs.type_2
        rhs_t = rhs.type_2

        if not isinstance(lhs_t, traits.ArithmeticTrait) or not lhs_t.supports_arithmetic():
            message = f"'{lhs_t}' does not support comparison operations"
            raise errors.DSLError(lhs.location, message)
        if not isinstance(rhs_t, traits.ArithmeticTrait) or not rhs_t.supports_arithmetic():
            message = f"'{rhs_t}' does not support comparison operations"
            raise errors.DSLError(rhs.location, message)
        common_ty = traits.common_arithmetic_type(lhs_t, rhs_t)
        if common_ty is None:
            message = f"no matching operator '{node.op}' for operand types '{lhs_t}', '{rhs_t}'"
            raise errors.DSLError(node.location, message)
        if isinstance(common_ty, ts2_f.FieldType):
            ty = ts2_f.FieldType(ts2.BoolType(), common_ty.dimensions)
        else:
            ty = ts2.BoolType()

        return foast.Compare(left=lhs, right=rhs, op=node.op, type_2=ty, location=node.location)

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> foast.TernaryExpr:
        condition: foast.Expr = self.visit(node.condition, **kwargs)
        then_expr: foast.Expr = self.visit(node.true_expr, **kwargs)
        else_expr: foast.Expr = self.visit(node.false_expr, **kwargs)
        if not traits.is_implicitly_convertible(condition.type_2, ts2.BoolType()):
            message = f"could not implicitly convert from '{condition.type_2}' to '{ts2.BoolType()}'"
            raise errors.DSLError(condition.location, message)
        ty = traits.common_type(then_expr.type_2, else_expr.type_2)
        if ty is None:
            message = (f"then expression has type {then_expr.type_2}"
                       f" but else expression has type {else_expr.type_2};"
                       f" could not find a common type")
            raise errors.DSLError(node.location, message)
        return foast.TernaryExpr(
            condition=condition,
            true_expr=then_expr,
            false_expr=else_expr,
            type_2=ty,
            location=node.location,
        )

    def visit_Constant(self, node: foast.Constant, **kwargs) -> foast.Constant:
        ty = ti2_f.inferrer.from_instance(node.value)
        if ty is None:
            raise errors.DSLError(node.location, "could not infer type of constant expression")
        return foast.Constant(value=node.value, location=node.location, type_2=ty)

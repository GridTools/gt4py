from gt4py.next.ffront import program_ast as past
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

    def visit_Program(self, node: past.Program, **kwargs) -> past.Program:
        new_closure_vars: list[past.Symbol] = []
        for sym in node.closure_vars:
            ty = ti2_f.inferrer.from_instance(self.closure_vars[sym.id])
            if ty is None:
                raise errors.DSLError(sym.location, f"could not infer type of captured variable '{sym.id}'")
            new_symbol: past.Symbol = past.Symbol(
                id=sym.id,
                location=sym.location,
                type_2=ty,
            )
            new_closure_vars.append(new_symbol)
        return past.Program(
            id=node.id,
            params=node.params,
            body=node.body,
            closure_vars=new_closure_vars,
            type_2=node.type_2,
            location=node.location,
        )


class TypeInferencePass(eve.traits.VisitorWithSymbolTableTrait, eve.NodeTranslator):
    def visit_Program(self, node: past.Program, **kwargs):
        params = self.visit(node.params)

        ty_params = [ts2.FunctionParameter(p.type_2, p.id, True, True) for p in params]
        ty = ts2.FunctionType(ty_params, None)
        return past.Program(
            id=self.visit(node.id, **kwargs),
            type_2=ty,
            params=params,
            body=self.visit(node.body, **kwargs),
            closure_vars=self.visit(node.closure_vars, **kwargs),
            location=node.location,
        )

    def visit_Call(self, node: past.Call, **kwargs):
        func: past.Expr = self.visit(node.func, **kwargs)
        positionals: list = self.visit(node.args, **kwargs)
        keywords: dict = self.visit(node.kwargs, **kwargs)

        func_t = func.type_2
        args = [
            *(traits.FunctionArgument(arg.type_2, idx) for idx, arg in enumerate(positionals)),
            *(
                traits.FunctionArgument(arg.type_2, name)
                for name, arg in keywords.items()
                if name != "out" and name != "domain"
            )
        ]

        if isinstance(func_t, traits.CallableTrait):
            is_callable, result_or_err = func_t.is_callable(args)
            if not is_callable:
                raise errors.DSLError(node.location, f"invalid arguments to call: {result_or_err}")
            ty = result_or_err
            return past.Call(func=func, args=positionals, kwargs=keywords, type_2=ty, location=node.location)
        raise errors.DSLError(func.location, f"'{func_t}' is not callable")

    def visit_Subscript(self, node: past.Subscript, **kwargs):
        value = self.visit(node.value, **kwargs)
        slice_ = self.visit(node.slice_, **kwargs)
        return past.Subscript(
            value=value,
            slice_=slice_,
            type_2=value.type_2,
            location=node.location,
        )

    def visit_TupleExpr(self, node: past.TupleExpr, **kwargs):
        elements = self.visit(node.elts, **kwargs)
        ty = ts2.TupleType([element.type_2 for element in elements])
        return past.TupleExpr(elts=elements, type_2=ty, location=node.location)

    def visit_Name(self, node: past.Name, **kwargs) -> past.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable:
            raise errors.UndefinedSymbolError(node.location, node.id)
        sym = symtable[node.id]
        assert sym.type_2 is not None, f"'{sym.id}': {sym.location}"
        return past.Name(id=node.id, type_2=sym.type_2, location=node.location)

    def visit_BinOp(self, node: past.BinOp, **kwargs) -> past.BinOp:
        from gt4py.next.ffront.dialect_ast_enums import BinaryOperator

        bitwise_ops = [BinaryOperator.BIT_AND, BinaryOperator.BIT_OR, BinaryOperator.BIT_XOR]

        lhs: past.Expr = self.visit(node.left, **kwargs)
        rhs: past.Expr = self.visit(node.right, **kwargs)
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

        return past.BinOp(left=lhs, right=rhs, op=node.op, type_2=ty, location=node.location)

    def visit_Constant(self, node: past.Constant, **kwargs):
        ty = ti2_f.inferrer.from_instance(node.value)
        if ty is None:
            raise errors.DSLError(node.location, "could not infer type of constant expression")
        return past.Constant(value=node.value, location=node.location, type_2=ty)
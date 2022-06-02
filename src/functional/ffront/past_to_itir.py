# GT4Py Project - GridTools Framework
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
from typing import Union

from eve import NodeTranslator, traits
from functional.common import GTTypeError
from functional.ffront import common_types, program_ast as past
from functional.iterator import ir as itir


def _size_arg_from_field(field_name: str, dim: int) -> str:
    return f"__{field_name}_size_{dim}"


class ProgramLowering(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Lower Program AST (PAST) to Iterator IR (ITIR).

    Examples
    --------
    >>> from functional.ffront.func_to_past import ProgramParser
    >>> from functional.iterator.runtime import offset
    >>> from functional.iterator import ir
    >>> from functional.ffront.fbuiltins import Dimension, Field
    >>>
    >>> float64 = float
    >>> IDim = Dimension("IDim")
    >>> Ioff = offset("Ioff")
    >>>
    >>> def fieldop(inp: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
    ...    ...
    >>> def program(inp: Field[[IDim], "float64"], out: Field[[IDim], "float64"]):
    ...    fieldop(inp, out=out)
    >>>
    >>> parsed = ProgramParser.apply_to_function(program)
    >>> fieldop_def = ir.FunctionDefinition(
    ...     id="fieldop",
    ...     params=[ir.Sym(id="inp")],
    ...     expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="inp")])
    ... )
    >>> lowered = ProgramLowering.apply(parsed, [fieldop_def])
    >>> type(lowered)
    <class 'functional.iterator.ir.FencilDefinition'>
    >>> lowered.id
    SymbolName('program')
    >>> lowered.params
    [Sym(id=SymbolName('inp')), Sym(id=SymbolName('out')), Sym(id=SymbolName('__inp_size_0')), Sym(id=SymbolName('__out_size_0'))]
    """

    @classmethod
    def apply(
        cls, node: past.Program, function_definitions: list[itir.FunctionDefinition]
    ) -> itir.FencilDefinition:
        return cls().visit(node, function_definitions=function_definitions)

    def _gen_size_params_from_program(self, node: past.Program):
        """Generate symbols for each field param and dimension."""
        size_params = []
        for param in node.params:
            if isinstance(param.type, common_types.FieldType):
                for dim_idx in range(0, len(param.type.dims)):
                    size_params.append(itir.Sym(id=_size_arg_from_field(param.id, dim_idx)))
        return size_params

    def visit_Program(
        self, node: past.Program, *, symtable, function_definitions, **kwargs
    ) -> itir.FencilDefinition:
        # The ITIR does not support dynamically getting the size of a field. As
        #  a workaround we add additional arguments to the fencil definition
        #  containing the size of all fields. The caller of a program is (e.g.
        #  program decorator) is required to pass these arguments.
        size_params = self._gen_size_params_from_program(node)

        closures: list[itir.StencilClosure] = []
        for stmt in node.body:
            if isinstance(stmt, past.Call) and isinstance(
                symtable[stmt.func.id].type.returns, common_types.FieldType
            ):
                closures.append(self._visit_stencil_call(stmt, **kwargs))
            else:
                raise NotImplementedError(
                    "Only calls to functions returning a Field supported currently."
                )

        return itir.FencilDefinition(
            id=node.id,
            function_definitions=function_definitions,
            params=[itir.Sym(id=inp.id) for inp in node.params] + size_params,
            closures=closures,
        )

    def _visit_stencil_call(self, node: past.Call, **kwargs) -> itir.StencilClosure:
        assert isinstance(node.kwargs["out"].type, common_types.FieldType)

        output, domain = self._visit_stencil_call_out_arg(node.kwargs["out"], **kwargs)

        return itir.StencilClosure(
            domain=domain,
            stencil=itir.SymRef(id=node.func.id),
            inputs=[itir.SymRef(id=self.visit(arg, **kwargs).id) for arg in node.args],
            output=output,
        )

    def _visit_slice_bound(
        self, slice_bound: past.Constant, default_value: itir.Expr, dim_size: itir.Expr, **kwargs
    ) -> itir.Expr:
        if slice_bound is None:
            lowered_bound = default_value
        elif isinstance(slice_bound, past.Constant):
            assert (
                isinstance(slice_bound.type, common_types.ScalarType)
                and slice_bound.type.kind == common_types.ScalarKind.INT
            )
            if slice_bound.value < 0:
                lowered_bound = itir.FunCall(
                    fun=itir.SymRef(id="plus"),
                    args=[dim_size, self.visit(slice_bound, **kwargs)],
                )
            else:
                lowered_bound = self.visit(slice_bound, **kwargs)
        else:
            raise AssertionError("Expected `None` or `past.Constant`.")
        return lowered_bound

    def _visit_stencil_call_out_arg(
        self, node: past.Expr, **kwargs
    ) -> tuple[itir.SymRef, itir.FunCall]:
        # as the ITIR does not support slicing a field we have to do a deeper
        #  inspection of the PAST to emulate the behaviour
        if isinstance(node, past.Subscript):
            out_field_name: past.Name = node.value
            if isinstance(node.slice_, past.TupleExpr) and all(
                isinstance(el, past.Slice) for el in node.slice_.elts
            ):
                out_field_slice_: list[past.Slice] = node.slice_.elts
            elif isinstance(node.slice_, past.Slice):
                out_field_slice_: list[past.Slice] = [node.slice_]
            else:
                raise RuntimeError(
                    "Unexpected `out` argument. Must be tuple of slices or slice expression."
                )
            if len(out_field_slice_) != len(node.type.dims):
                raise GTTypeError(
                    f"Too many indices for field {out_field_name}: field is {len(node.type.dims)}"
                    f"-dimensional, but {len(out_field_slice_)} were indexed."
                )
            domain_args = []
            for dim_i, (dim, slice_) in enumerate(zip(node.type.dims, out_field_slice_)):
                # an expression for the size of a dimension
                dim_size = itir.SymRef(id=_size_arg_from_field(out_field_name.id, dim_i))
                # lower bound
                lower = self._visit_slice_bound(
                    slice_.lower, itir.Literal(value="0", type="int"), dim_size
                )
                upper = self._visit_slice_bound(slice_.upper, dim_size, dim_size)
                if dim.local:
                    raise GTTypeError(f"Dimension {dim.value} must not be local.")
                domain_args.append(
                    itir.FunCall(
                        fun=itir.SymRef(id="named_range"),
                        args=[itir.AxisLiteral(value=dim.value), lower, upper],
                    )
                )

        elif isinstance(node, past.Name):
            out_field_name = node
            domain_args = [
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value=dim.value),
                        itir.Literal(value="0", type="int"),
                        # here we use the artificial size arguments added to the fencil
                        itir.SymRef(id=_size_arg_from_field(out_field_name.id, dim_idx)),
                    ],
                )
                for dim_idx, dim in enumerate(node.type.dims)
            ]
        else:
            raise RuntimeError(
                "Unexpected `out` argument. Must be a `past.Subscript` or `past.Name` node."
            )

        return itir.SymRef(id=self.visit(out_field_name, **kwargs).id), itir.FunCall(
            fun=itir.SymRef(id="domain"), args=domain_args
        )

    def visit_Constant(self, node: past.Constant, **kwargs) -> Union[itir.Literal]:
        if isinstance(node.type, common_types.ScalarType) and node.type.shape is None:
            match node.type.kind:
                case common_types.ScalarKind.STRING:
                    raise NotImplementedError(
                        f"Scalars of kind {node.type.kind} not supported currently."
                    )
            typename = node.type.kind.name.lower()
            if typename.startswith("int"):
                typename = "int"
            return itir.Literal(value=str(node.value), type=typename)

        raise NotImplementedError("Only scalar literals supported currently.")

    def visit_Name(self, node: past.Name, **kwargs) -> itir.Sym:
        return itir.Sym(id=node.id)

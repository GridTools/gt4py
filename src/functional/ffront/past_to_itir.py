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
from __future__ import annotations

from typing import Optional

from eve import NodeTranslator, concepts, traits
from functional.common import DimensionKind, GridType, GTTypeError
from functional.ffront import common_types, program_ast as past, type_info
from functional.iterator import ir as itir


def _size_arg_from_field(field_name: str, dim: int) -> str:
    return f"__{field_name}_size_{dim}"


def _flatten_tuple_expr(
    node: past.Name | past.Subscript | past.TupleExpr,
) -> list[past.Name | past.Subscript]:
    if isinstance(node, (past.Name, past.Subscript)):
        return [node]
    elif isinstance(node, past.TupleExpr):
        result = []
        for e in node.elts:
            result.extend(_flatten_tuple_expr(e))
        return result
    raise AssertionError(
        "Only `past.Name`, `past.Subscript` or `past.TupleExpr`s thereof are allowed."
    )


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
    >>> parsed = ProgramParser.apply_to_function(program)  # doctest: +SKIP
    >>> fieldop_def = ir.FunctionDefinition(
    ...     id="fieldop",
    ...     params=[ir.Sym(id="inp")],
    ...     expr=ir.FunCall(fun=ir.SymRef(id="deref"), args=[ir.SymRef(id="inp")])
    ... )  # doctest: +SKIP
    >>> lowered = ProgramLowering.apply(parsed, [fieldop_def],
    ...     grid_type=GridType.CARTESIAN)  # doctest: +SKIP
    >>> type(lowered)  # doctest: +SKIP
    <class 'functional.iterator.ir.FencilDefinition'>
    >>> lowered.id  # doctest: +SKIP
    SymbolName('program')
    >>> lowered.params  # doctest: +SKIP
    [Sym(id=SymbolName('inp')), Sym(id=SymbolName('out')), Sym(id=SymbolName('__inp_size_0')), Sym(id=SymbolName('__out_size_0'))]
    """

    # TODO(tehrengruber): enable doctests again. For unknown / obscure reasons
    #  the above doctest fails when executed using `pytest --doctest-modules`.

    @classmethod
    def apply(
        cls,
        node: past.Program,
        function_definitions: list[itir.FunctionDefinition],
        grid_type: GridType,
    ) -> itir.FencilDefinition:
        return cls(grid_type=grid_type).visit(node, function_definitions=function_definitions)

    def __init__(self, grid_type):
        self.grid_type = grid_type

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
            closures.append(self._visit_stencil_call(stmt, **kwargs))

        return itir.FencilDefinition(
            id=node.id,
            function_definitions=function_definitions,
            params=[itir.Sym(id=inp.id) for inp in node.params] + size_params,
            closures=closures,
        )

    def _visit_stencil_call(self, node: past.Call, **kwargs) -> itir.StencilClosure:
        assert type_info.is_field_type_or_tuple_of_field_type(node.kwargs["out"].type)

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

    def _construct_itir_out_arg(
        self, node: past.TupleExpr | past.Name | past.Subscript
    ) -> itir.Expr:
        if isinstance(node, past.Name):
            return itir.SymRef(id=node.id)
        elif isinstance(node, past.Subscript):
            return self._construct_itir_out_arg(node.value)
        elif isinstance(node, past.TupleExpr):
            return itir.FunCall(
                fun=itir.SymRef(id="make_tuple"),
                args=[self._construct_itir_out_arg(el) for el in node.elts],
            )
        else:
            raise AssertionError(
                "Unexpected `out` argument. Must be a `past.Name`, `past.Subscript`"
                " or a `past.TupleExpr` of  `past.Name`, `past.Subscript` or`past.TupleExpr` (recursively)."
            )

    def _construct_itir_domain_arg(
        self, out_field: past.Name, slices: Optional[list[past.Slice]] = None
    ) -> itir.FunCall:
        domain_args = []
        for dim_i, dim in enumerate(out_field.type.dims):
            # an expression for the size of a dimension
            dim_size = itir.SymRef(id=_size_arg_from_field(out_field.id, dim_i))
            # bounds
            lower = self._visit_slice_bound(
                slices[dim_i].lower if slices else None,
                itir.Literal(value="0", type="int"),
                dim_size,
            )
            upper = self._visit_slice_bound(
                slices[dim_i].upper if slices else None, dim_size, dim_size
            )
            if dim.kind == DimensionKind.LOCAL:
                raise GTTypeError(f"Dimension {dim.value} must not be local.")
            domain_args.append(
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[itir.AxisLiteral(value=dim.value), lower, upper],
                )
            )

        if self.grid_type == GridType.CARTESIAN:
            domain_builtin = "cartesian_domain"
        elif self.grid_type == GridType.UNSTRUCTURED:
            domain_builtin = "unstructured_domain"
        else:
            raise AssertionError()

        return itir.FunCall(fun=itir.SymRef(id=domain_builtin), args=domain_args)

    @staticmethod
    def _compute_field_slice(node: past.Subscript):
        out_field_name: past.Name = node.value
        if isinstance(node.slice_, past.TupleExpr) and all(
            isinstance(el, past.Slice) for el in node.slice_.elts
        ):
            out_field_slice_: list[past.Slice] = node.slice_.elts
        elif isinstance(node.slice_, past.Slice):
            out_field_slice_: list[past.Slice] = [node.slice_]
        else:
            raise AssertionError(
                "Unexpected `out` argument. Must be tuple of slices or slice expression."
            )
        if len(out_field_slice_) != len(node.type.dims):
            raise GTTypeError(
                f"Too many indices for field {out_field_name}: field is {len(node.type.dims)}"
                f"-dimensional, but {len(out_field_slice_)} were indexed."
            )
        return out_field_slice_

    def _visit_stencil_call_out_arg(
        self, node: past.Expr, **kwargs
    ) -> tuple[itir.SymRef, itir.FunCall]:
        if isinstance(node, past.Subscript):
            # as the ITIR does not support slicing a field we have to do a deeper
            #  inspection of the PAST to emulate the behaviour
            out_field_name: past.Name = node.value
            return (
                self._construct_itir_out_arg(out_field_name),
                self._construct_itir_domain_arg(out_field_name, self._compute_field_slice(node)),
            )
        elif isinstance(node, past.Name):
            return (self._construct_itir_out_arg(node), self._construct_itir_domain_arg(node))
        elif isinstance(node, past.TupleExpr):
            flattened = _flatten_tuple_expr(node)

            first_field = flattened[0]
            assert all(
                field.type.dims == first_field.type.dims for field in flattened
            ), "Incompatible fields in tuple: all fields must have the same dimensions."

            field_slice = None
            if isinstance(first_field, past.Subscript):
                assert all(
                    isinstance(field, past.Subscript) for field in flattened
                ), "Incompatible field in tuple: either all fields or no field must be sliced."
                assert all(
                    concepts.eq_nonlocated(first_field.slice_, field.slice_) for field in flattened
                ), "Incompatible field in tuple: all fields must be sliced in the same way."
                field_slice = self._compute_field_slice(first_field)
                first_field = first_field.value

            return (
                self._construct_itir_out_arg(node),
                self._construct_itir_domain_arg(first_field, field_slice),
            )
        else:
            raise AssertionError(
                "Unexpected `out` argument. Must be a `past.Subscript`, `past.Name` or `past.TupleExpr` node."
            )

    def visit_Constant(self, node: past.Constant, **kwargs) -> itir.Literal:
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

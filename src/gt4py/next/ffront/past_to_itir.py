# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from typing import Any, Generator, Optional, cast

from gt4py.eve import NodeTranslator, concepts, traits
from gt4py.next.common import Dimension, DimensionKind, GridType, GTTypeError
from gt4py.next.ffront import program_ast as past
from gt4py.next.iterator import ir as itir
from gt4py.next.type_system import type_info, type_specifications as ts


def _size_arg_from_field(field_name: str, dim: int) -> str:
    return f"__{field_name}_size_{dim}"


def _flatten_tuple_expr(
    node: past.Expr,
) -> list[past.Name | past.Subscript]:
    if isinstance(node, (past.Name, past.Subscript)):
        return [node]
    elif isinstance(node, past.TupleExpr):
        result = []
        for e in node.elts:
            result.extend(_flatten_tuple_expr(e))
        return result
    raise GTTypeError(
        "Only `past.Name`, `past.Subscript` or `past.TupleExpr`s thereof are allowed."
    )


class ProgramLowering(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Lower Program AST (PAST) to Iterator IR (ITIR).

    Examples
    --------
    >>> from gt4py.next.ffront.func_to_past import ProgramParser
    >>> from gt4py.next.iterator.runtime import offset
    >>> from gt4py.next.iterator import ir
    >>> from gt4py.next.ffront.fbuiltins import Dimension, Field
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
    <class 'gt4py.next.iterator.ir.FencilDefinition'>
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
            if type_info.is_type_or_tuple_of_type(param.type, ts.FieldType):
                fields_dims: list[list[Dimension]] = (
                    type_info.primitive_constituents(param.type).getattr("dims").to_list()
                )
                assert all(field_dims == fields_dims[0] for field_dims in fields_dims)
                for dim_idx in range(len(fields_dims[0])):
                    size_params.append(itir.Sym(id=_size_arg_from_field(param.id, dim_idx)))

        return size_params

    def visit_Program(
        self, node: past.Program, *, function_definitions, **kwargs
    ) -> itir.FencilDefinition:
        # The ITIR does not support dynamically getting the size of a field. As
        #  a workaround we add additional arguments to the fencil definition
        #  containing the size of all fields. The caller of a program is (e.g.
        #  program decorator) is required to pass these arguments.

        params = self.visit(node.params)

        if any("domain" not in body_entry.kwargs for body_entry in node.body):
            params = params + self._gen_size_params_from_program(node)

        closures: list[itir.StencilClosure] = []
        for stmt in node.body:
            closures.append(self._visit_stencil_call(stmt, **kwargs))

        return itir.FencilDefinition(
            id=node.id,
            function_definitions=function_definitions,
            params=params,
            closures=closures,
        )

    def _visit_stencil_call(self, node: past.Call, **kwargs) -> itir.StencilClosure:
        assert isinstance(node.kwargs["out"].type, ts.TypeSpec)
        assert type_info.is_type_or_tuple_of_type(node.kwargs["out"].type, ts.FieldType)

        domain = node.kwargs.get("domain", None)
        output, lowered_domain = self._visit_stencil_call_out_arg(
            node.kwargs["out"], domain, **kwargs
        )

        return itir.StencilClosure(
            domain=lowered_domain,
            stencil=itir.SymRef(id=node.func.id),
            inputs=self.visit(node.args, **kwargs),
            output=output,
        )

    def _visit_slice_bound(
        self,
        slice_bound: Optional[past.Constant],
        default_value: itir.Expr,
        dim_size: itir.Expr,
        **kwargs,
    ):
        if slice_bound is None:
            lowered_bound = default_value
        elif isinstance(slice_bound, past.Constant):
            assert (
                isinstance(slice_bound.type, ts.ScalarType)
                and slice_bound.type.kind == ts.ScalarKind.INT
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

    def _construct_itir_out_arg(self, node: past.Expr) -> itir.Expr:
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
            raise GTTypeError(
                "Unexpected `out` argument. Must be a `past.Name`, `past.Subscript`"
                " or a `past.TupleExpr` thereof."
            )

    def _construct_itir_domain_arg(
        self,
        out_field: past.Name,
        node_domain: Optional[past.Expr],
        slices: Optional[list[past.Slice]] = None,
    ) -> itir.FunCall:
        domain_args = []

        assert isinstance(out_field.type, ts.TypeSpec)
        out_field_types = type_info.primitive_constituents(out_field.type).to_list()
        out_dims = cast(ts.FieldType, out_field_types[0]).dims
        if any(
            not isinstance(out_field_type, ts.FieldType) or out_field_type.dims != out_dims
            for out_field_type in out_field_types
        ):
            raise AssertionError(
                f"Expected constituents of `{out_field.id}` argument to be"
                f" fields defined on the same dimensions. This error should be "
                f" caught in type deduction already."
            )

        for dim_i, dim in enumerate(out_dims):
            # an expression for the size of a dimension
            dim_size = itir.SymRef(id=_size_arg_from_field(out_field.id, dim_i))
            # bounds
            if node_domain is not None:
                assert isinstance(node_domain, past.Dict)
                lower, upper = self._construct_itir_initialized_domain_arg(dim_i, dim, node_domain)
            else:
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

    def _construct_itir_initialized_domain_arg(
        self,
        dim_i: int,
        dim: Dimension,
        node_domain: past.Dict,
    ) -> Generator[Any, None, None]:
        keys_dims_types = cast(ts.DimensionType, node_domain.keys_[dim_i].type).dim
        if keys_dims_types == dim:
            assert len(node_domain.values_[dim_i].elts) == 2
            return (self.visit(bound) for bound in node_domain.values_[dim_i].elts)
        else:
            raise GTTypeError(
                f"Dimensions in out field and field domain are not equivalent"
                f"Expected {dim}, but got {keys_dims_types} "
            )

    @staticmethod
    def _compute_field_slice(node: past.Subscript):
        out_field_name: past.Name = node.value
        out_field_slice_: list[past.Expr]
        if isinstance(node.slice_, past.TupleExpr) and all(
            isinstance(el, past.Slice) for el in node.slice_.elts
        ):
            out_field_slice_ = node.slice_.elts
        elif isinstance(node.slice_, past.Slice):
            out_field_slice_ = [node.slice_]
        else:
            raise AssertionError(
                "Unexpected `out` argument. Must be tuple of slices or slice expression."
            )
        node_dims_ls = cast(ts.FieldType, node.type).dims
        assert isinstance(node_dims_ls, list)
        if isinstance(node.type, ts.FieldType) and len(out_field_slice_) != len(node_dims_ls):
            raise GTTypeError(
                f"Too many indices for field {out_field_name}: field is {len(node_dims_ls)}"
                f"-dimensional, but {len(out_field_slice_)} were indexed."
            )
        return out_field_slice_

    def _visit_stencil_call_out_arg(
        self, out_arg: past.Expr, domain_arg: Optional[past.Expr], **kwargs
    ) -> tuple[itir.Expr, itir.FunCall]:
        if isinstance(out_arg, past.Subscript):
            # as the ITIR does not support slicing a field we have to do a deeper
            #  inspection of the PAST to emulate the behaviour
            out_field_name: past.Name = out_arg.value
            return (
                self._construct_itir_out_arg(out_field_name),
                self._construct_itir_domain_arg(
                    out_field_name, domain_arg, self._compute_field_slice(out_arg)
                ),
            )
        elif isinstance(out_arg, past.Name):
            return (
                self._construct_itir_out_arg(out_arg),
                self._construct_itir_domain_arg(out_arg, domain_arg),
            )
        elif isinstance(out_arg, past.TupleExpr):
            flattened = _flatten_tuple_expr(out_arg)

            first_field = flattened[0]
            assert all(
                self.visit(field.type).dims == self.visit(first_field.type).dims
                for field in flattened
            ), "Incompatible fields in tuple: all fields must have the same dimensions."

            field_slice = None
            if isinstance(first_field, past.Subscript):
                assert all(
                    isinstance(field, past.Subscript) for field in flattened
                ), "Incompatible field in tuple: either all fields or no field must be sliced."
                assert all(
                    concepts.eq_nonlocated(first_field.slice_, field.slice_) for field in flattened  # type: ignore[union-attr] # mypy cannot deduce type
                ), "Incompatible field in tuple: all fields must be sliced in the same way."
                field_slice = self._compute_field_slice(first_field)
                first_field = first_field.value

            return (
                self._construct_itir_out_arg(out_arg),
                self._construct_itir_domain_arg(first_field, domain_arg, field_slice),
            )
        else:
            raise AssertionError(
                "Unexpected `out` argument. Must be a `past.Subscript`, `past.Name` or `past.TupleExpr` node."
            )

    def visit_Constant(self, node: past.Constant, **kwargs) -> itir.Literal:
        if isinstance(node.type, ts.ScalarType) and node.type.shape is None:
            match node.type.kind:
                case ts.ScalarKind.STRING:
                    raise NotImplementedError(
                        f"Scalars of kind {node.type.kind} not supported currently."
                    )
            typename = node.type.kind.name.lower()
            if typename.startswith("int"):
                typename = "int"
            return itir.Literal(value=str(node.value), type=typename)

        raise NotImplementedError("Only scalar literals supported currently.")

    def visit_Name(self, node: past.Name, **kwargs) -> itir.SymRef:
        return itir.SymRef(id=node.id)

    def visit_Symbol(self, node: past.Symbol, **kwargs) -> itir.Sym:
        return itir.Sym(id=node.id)

    def visit_BinOp(self, node: past.BinOp, **kwargs) -> itir.FunCall:
        return itir.FunCall(
            fun=itir.SymRef(id=node.op.value),
            args=[self.visit(node.left, **kwargs), self.visit(node.right, **kwargs)],
        )
